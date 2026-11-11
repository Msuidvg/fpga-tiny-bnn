"""Generate deterministic Python-to-RTL Tiny-BNN comparison vectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bnn.bitexact import infer_bitexact, load_bitexact_model  # noqa: E402
from tools.bnn.train_bnn import DEFAULT_DATASET, load_dataset  # noqa: E402


DEFAULT_RELEASE = ROOT / "models" / "v1"


def _bits_hex(values: np.ndarray) -> str:
    bits = (np.asarray(values).reshape(-1) != 0).astype(np.uint8)
    if bits.size % 8:
        bits = np.pad(bits, (0, 8 - bits.size % 8))
    packed = np.packbits(bits, bitorder="little")
    return bytes(packed[::-1]).hex().upper()


def _packed_integers_hex(values: np.ndarray, width: int) -> str:
    integer = 0
    flat = np.asarray(values).reshape(-1)
    for index, value in enumerate(flat):
        number = int(value)
        if number < 0 or number >= 1 << width:
            raise ValueError(f"value {number} does not fit unsigned {width} bits")
        integer |= number << (index * width)
    digits = math.ceil(flat.size * width / 4)
    return f"{integer:0{digits}X}"


def _packed_signed_hex(values: np.ndarray, width: int) -> str:
    integer = 0
    mask = (1 << width) - 1
    flat = np.asarray(values).reshape(-1)
    for index, value in enumerate(flat):
        integer |= (int(value) & mask) << (index * width)
    digits = math.ceil(flat.size * width / 4)
    return f"{integer:0{digits}X}"


def _select_indices(
    labels: np.ndarray,
    batches: np.ndarray,
    predicted: np.ndarray,
    margins: np.ndarray,
    per_batch: int,
) -> list[int]:
    selected: list[int] = []
    for batch in sorted(str(value) for value in np.unique(batches)):
        candidates = np.flatnonzero(batches == batch)
        chosen: list[int] = []

        correct = candidates[predicted[candidates] == labels[candidates]]
        if correct.size:
            chosen.append(int(correct[np.argmin(margins[correct])]))

        for label in (3, 4, 5):
            matching = candidates[labels[candidates] == label]
            if matching.size:
                index = int(matching[np.argmin(margins[matching])])
                if index not in chosen:
                    chosen.append(index)

        incorrect = candidates[predicted[candidates] != labels[candidates]]
        for index in incorrect[np.argsort(margins[incorrect], kind="stable")]:
            if int(index) not in chosen:
                chosen.append(int(index))
            if sum(predicted[item] != labels[item] for item in chosen) >= 3:
                break

        for index in candidates[np.argsort(margins[candidates], kind="stable")]:
            if int(index) not in chosen:
                chosen.append(int(index))
            if len(chosen) >= per_batch:
                break
        selected.extend(chosen[:per_batch])
    return selected


def generate(
    dataset_path: Path,
    model_path: Path,
    fpga_dir: Path,
    output_dir: Path,
    per_batch: int,
) -> dict[str, object]:
    dataset = load_dataset(dataset_path)
    model = load_bitexact_model(model_path)
    input_bits, hidden1, hidden2, outputs = model.architecture
    l1_popcount_bits = math.ceil(math.log2(input_bits + 1))
    l2_popcount_bits = math.ceil(math.log2(hidden1 + 1))
    l3_popcount_bits = math.ceil(math.log2(hidden2 + 1))
    result = infer_bitexact(dataset.images, model)
    sorted_scores = np.sort(result.scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    selected = _select_indices(
        dataset.labels,
        dataset.batches,
        result.predicted_labels,
        margins,
        per_batch,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "inputs": output_dir / "inputs.mem",
        "labels": output_dir / "expected_labels.mem",
        "confidence": output_dir / "expected_confidence.mem",
        "h1": output_dir / "expected_h1.mem",
        "h2": output_dir / "expected_h2.mem",
        "l1_popcounts": output_dir / "expected_l1_popcounts.mem",
        "l2_popcounts": output_dir / "expected_l2_popcounts.mem",
        "l3_popcounts": output_dir / "expected_l3_popcounts.mem",
        "scores": output_dir / "expected_scores.mem",
    }
    lines: dict[str, list[str]] = {key: [] for key in files}
    metadata: list[dict[str, object]] = []
    for vector_index, dataset_index in enumerate(selected):
        scores = result.scores[dataset_index]
        confidence = min(255, int(margins[dataset_index]))
        lines["inputs"].append(_bits_hex(dataset.images[dataset_index]))
        lines["labels"].append(f"{int(result.predicted_labels[dataset_index]):X}")
        lines["confidence"].append(f"{confidence:02X}")
        lines["h1"].append(_bits_hex(result.h1_bits[dataset_index]))
        lines["h2"].append(_bits_hex(result.h2_bits[dataset_index]))
        lines["l1_popcounts"].append(
            _packed_integers_hex(
                result.h1_popcounts[dataset_index], l1_popcount_bits
            )
        )
        lines["l2_popcounts"].append(
            _packed_integers_hex(
                result.h2_popcounts[dataset_index], l2_popcount_bits
            )
        )
        lines["l3_popcounts"].append(
            _packed_integers_hex(
                result.output_popcounts[dataset_index], l3_popcount_bits
            )
        )
        lines["scores"].append(_packed_signed_hex(scores, 32))
        metadata.append(
            {
                "vector_index": vector_index,
                "dataset_index": int(dataset_index),
                "path": dataset.paths[dataset_index],
                "batch": str(dataset.batches[dataset_index]),
                "true_label": int(dataset.labels[dataset_index]),
                "expected_model_label": int(result.predicted_labels[dataset_index]),
                "model_correct": bool(
                    result.predicted_labels[dataset_index]
                    == dataset.labels[dataset_index]
                ),
                "confidence_margin": int(margins[dataset_index]),
                "scores": [int(value) for value in scores],
            }
        )
    for key, path in files.items():
        path.write_text("\n".join(lines[key]) + "\n", encoding="ascii")

    def project_relative(path: Path) -> str:
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    header = output_dir / "bnn_test_vectors.vh"
    header.write_text(
        "\n".join(
            [
                "`ifndef BNN_TEST_VECTORS_VH",
                "`define BNN_TEST_VECTORS_VH",
                f"`define BNN_VECTOR_COUNT {len(selected)}",
                f"`define BNN_INPUT_BITS {input_bits}",
                f"`define BNN_HIDDEN1_UNITS {hidden1}",
                f"`define BNN_HIDDEN2_UNITS {hidden2}",
                f"`define BNN_OUTPUT_UNITS {outputs}",
                f"`define BNN_L1_POPCOUNT_BITS {l1_popcount_bits}",
                f"`define BNN_L2_POPCOUNT_BITS {l2_popcount_bits}",
                f"`define BNN_L3_POPCOUNT_BITS {l3_popcount_bits}",
                (
                    "`define BNN_INFERENCE_CYCLES "
                    f"{input_bits * hidden1 + hidden1 * hidden2 + hidden2 * outputs + 1}"
                ),
                f'`define BNN_W1_FILE "{project_relative(fpga_dir / "bnn_w1_32.mem")}"',
                f'`define BNN_W2_FILE "{project_relative(fpga_dir / "bnn_w2_32.mem")}"',
                f'`define BNN_W3_FILE "{project_relative(fpga_dir / "bnn_w3_32.mem")}"',
                f'`define BNN_T1_FILE "{project_relative(fpga_dir / "bnn_t1.mem")}"',
                f'`define BNN_T2_FILE "{project_relative(fpga_dir / "bnn_t2.mem")}"',
                f'`define BNN_B3_FILE "{project_relative(fpga_dir / "bnn_b3.mem")}"',
                *[
                    f'`define BNN_VECTOR_{key.upper()}_FILE "{project_relative(path)}"'
                    for key, path in files.items()
                ],
                "`endif",
                "",
            ]
        ),
        encoding="ascii",
    )

    manifest = {
        "schema_version": 1,
        "model": str(model_path.resolve()),
        "architecture": [input_bits, hidden1, hidden2, outputs],
        "popcount_widths": [
            l1_popcount_bits,
            l2_popcount_bits,
            l3_popcount_bits,
        ],
        "dataset": str(dataset_path.resolve()),
        "vectors": len(selected),
        "vectors_per_batch": per_batch,
        "selection": (
            "per batch: lowest-margin correct sample, hardest labels 3/4/5, "
            "up to three errors, then lowest-margin samples"
        ),
        "correct_vectors": sum(item["model_correct"] for item in metadata),
        "incorrect_vectors": sum(not item["model_correct"] for item in metadata),
        "batch_counts": {
            batch: sum(item["batch"] == batch for item in metadata)
            for batch in sorted({str(item["batch"]) for item in metadata})
        },
        "items": metadata,
    }
    (output_dir / "vectors.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_RELEASE / "model.npz")
    parser.add_argument("--fpga-dir", type=Path, default=DEFAULT_RELEASE / "fpga")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "build" / "testvectors"
    )
    parser.add_argument("--per-batch", type=int, default=8)
    args = parser.parse_args()
    if args.per_batch <= 0:
        parser.error("--per-batch must be positive")
    manifest = generate(
        args.dataset, args.model, args.fpga_dir, args.output_dir, args.per_batch
    )
    print(
        json.dumps(
            {
                "vectors": manifest["vectors"],
                "batch_counts": manifest["batch_counts"],
                "correct_vectors": manifest["correct_vectors"],
                "incorrect_vectors": manifest["incorrect_vectors"],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
