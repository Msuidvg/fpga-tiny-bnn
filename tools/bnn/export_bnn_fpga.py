"""Export a verified Tiny-BNN model to deterministic FPGA memory files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bnn.bitexact import (  # noqa: E402
    activation_thresholds,
    load_bitexact_model,
    sha256_file,
)


DEFAULT_MODEL = ROOT / "models" / "v1" / "model.npz"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weight_hex_lines(packed: np.ndarray) -> list[str]:
    return [bytes(row[::-1]).hex().upper() for row in packed]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _unsigned_hex_lines(values: np.ndarray, width: int) -> list[str]:
    digits = math.ceil(width / 4)
    limit = 1 << width
    output: list[str] = []
    for value in np.asarray(values).astype(np.int64):
        integer = int(value)
        if integer < 0 or integer >= limit:
            raise ValueError(f"value {integer} does not fit unsigned {width} bits")
        output.append(f"{integer:0{digits}X}")
    return output


def _signed_hex_lines(values: np.ndarray, width: int) -> list[str]:
    digits = math.ceil(width / 4)
    mask = (1 << width) - 1
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    output: list[str] = []
    for value in np.asarray(values).astype(np.int64):
        integer = int(value)
        if integer < minimum or integer > maximum:
            raise ValueError(f"value {integer} does not fit signed {width} bits")
        output.append(f"{integer & mask:0{digits}X}")
    return output


def _popcount_width(feature_count: int) -> int:
    return max(1, math.ceil(math.log2(feature_count + 1)))


def _signed_width(minimum: int, maximum: int) -> int:
    width = 1
    while minimum < -(1 << (width - 1)) or maximum > (1 << (width - 1)) - 1:
        width += 1
    return width


def _verify_weight_file(path: Path, expected: np.ndarray) -> None:
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) != expected.shape[0]:
        raise ValueError(f"{path}: wrong row count")
    width_bytes = expected.shape[1]
    decoded = np.asarray(
        [
            list(int(line, 16).to_bytes(width_bytes, byteorder="little"))
            for line in lines
        ],
        dtype=np.uint8,
    )
    if not np.array_equal(decoded, expected):
        raise ValueError(f"{path}: weight round-trip mismatch")


def _verify_unsigned_file(path: Path, expected: np.ndarray) -> None:
    decoded = np.asarray(
        [int(line, 16) for line in path.read_text(encoding="ascii").splitlines()],
        dtype=np.int32,
    )
    if not np.array_equal(decoded, np.asarray(expected, dtype=np.int32)):
        raise ValueError(f"{path}: unsigned value round-trip mismatch")


def _verify_signed_file(path: Path, expected: np.ndarray, width: int) -> None:
    sign_bit = 1 << (width - 1)
    modulus = 1 << width
    decoded_values = []
    for line in path.read_text(encoding="ascii").splitlines():
        value = int(line, 16)
        decoded_values.append(value - modulus if value & sign_bit else value)
    decoded = np.asarray(decoded_values, dtype=np.int32)
    if not np.array_equal(decoded, np.asarray(expected, dtype=np.int32)):
        raise ValueError(f"{path}: signed value round-trip mismatch")


def export_model(model_path: Path, output_dir: Path) -> dict[str, object]:
    model_path = model_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_bitexact_model(model_path)
    input_features, hidden1, hidden2, outputs = model.architecture
    threshold1 = activation_thresholds(input_features, model.b1)
    threshold2 = activation_thresholds(hidden1, model.b2)
    if np.any((threshold1 < 0) | (threshold1 > input_features)):
        raise ValueError("layer-1 threshold outside realizable popcount range")
    if np.any((threshold2 < 0) | (threshold2 > hidden1)):
        raise ValueError("layer-2 threshold outside realizable popcount range")
    popcount1_bits = _popcount_width(input_features)
    popcount2_bits = _popcount_width(hidden1)
    popcount3_bits = _popcount_width(hidden2)
    accumulator1_bits = _signed_width(
        -input_features + int(model.b1.min()),
        input_features + int(model.b1.max()),
    )
    accumulator2_bits = _signed_width(
        -hidden1 + int(model.b2.min()), hidden1 + int(model.b2.max())
    )
    score3_bits = _signed_width(
        -hidden2 + int(model.b3.min()), hidden2 + int(model.b3.max())
    )

    paths = {
        "w1": output_dir / "bnn_w1.mem",
        "w2": output_dir / "bnn_w2.mem",
        "w3": output_dir / "bnn_w3.mem",
        "t1": output_dir / "bnn_t1.mem",
        "t2": output_dir / "bnn_t2.mem",
        "b3": output_dir / "bnn_b3.mem",
        "params": output_dir / "bnn_params.vh",
        "readme": output_dir / "README.md",
    }
    _write_lines(paths["w1"], _weight_hex_lines(model.w1_packed))
    _write_lines(paths["w2"], _weight_hex_lines(model.w2_packed))
    _write_lines(paths["w3"], _weight_hex_lines(model.w3_packed))
    _write_lines(paths["t1"], _unsigned_hex_lines(threshold1, popcount1_bits))
    _write_lines(paths["t2"], _unsigned_hex_lines(threshold2, popcount2_bits))
    _write_lines(paths["b3"], _signed_hex_lines(model.b3, 32))

    paths["params"].write_text(
        "\n".join(
            [
                "`ifndef GESTURE_TINY_BNN_V1_PARAMS_VH",
                "`define GESTURE_TINY_BNN_V1_PARAMS_VH",
                f"`define BNN_INPUT_BITS {input_features}",
                f"`define BNN_HIDDEN1_UNITS {hidden1}",
                f"`define BNN_HIDDEN2_UNITS {hidden2}",
                f"`define BNN_OUTPUT_UNITS {outputs}",
                "`define BNN_CLASS_LABEL_BASE 1",
                "`define BNN_W1_MEM \"bnn_w1.mem\"",
                "`define BNN_W2_MEM \"bnn_w2.mem\"",
                "`define BNN_W3_MEM \"bnn_w3.mem\"",
                "`define BNN_T1_MEM \"bnn_t1.mem\"",
                "`define BNN_T2_MEM \"bnn_t2.mem\"",
                "`define BNN_B3_MEM \"bnn_b3.mem\"",
                "`endif",
                "",
            ]
        ),
        encoding="ascii",
    )
    paths["readme"].write_text(
        f"""# Gesture Tiny-BNN V1 FPGA export

Each non-empty line in `bnn_w*.mem` is one output neuron's weight vector.
Load it into a Verilog word `[FEATURES-1:0]`; bit `i` is the weight for input
feature `i`. Bit 1 means +1 and bit 0 means -1. Inputs and hidden activations
use the same encoding, so the signed dot product is
`2 * popcount(~(activation ^ weight)) - FEATURES`.

`bnn_t1.mem` and `bnn_t2.mem` contain unsigned XNOR-popcount thresholds. A
hidden output bit is 1 exactly when `popcount >= threshold`; this already
folds the trained integer bias into the comparator. `bnn_b3.mem` contains five
signed 32-bit two's-complement output biases. Final class scores are
`2 * popcount - {hidden2} + b3`; argmax ties must prefer the lowest index, producing
gesture IDs 1 through 5. There is no BatchNorm or floating-point operation.

The export manifest records file hashes and the exact bit/row ordering. Do not
edit these files by hand; regenerate them from the frozen `.npz` model.
""",
        encoding="utf-8",
    )

    _verify_weight_file(paths["w1"], model.w1_packed)
    _verify_weight_file(paths["w2"], model.w2_packed)
    _verify_weight_file(paths["w3"], model.w3_packed)
    _verify_unsigned_file(paths["t1"], threshold1)
    _verify_unsigned_file(paths["t2"], threshold2)
    _verify_signed_file(paths["b3"], model.b3, 32)

    manifest = {
        "schema_version": 1,
        "source_model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "architecture": [input_features, hidden1, hidden2, outputs],
        "class_labels": [int(value) for value in model.class_labels],
        "encoding": {
            "activation_and_weight": "0 => -1, 1 => +1",
            "weight_row_order": "one output neuron per line",
            "weight_bit_order": "Verilog word bit i corresponds to input feature i",
            "hex_text_order": "most-significant hexadecimal digit first",
            "hidden_rule": "output_bit = (XNOR_popcount >= threshold)",
            "output_score": f"2 * XNOR_popcount - {hidden2} + signed_b3",
            "argmax_tie_break": "lowest class index wins",
        },
        "widths": {
            "layer1_popcount_bits": popcount1_bits,
            "layer1_signed_accumulator_bits": accumulator1_bits,
            "layer2_popcount_bits": popcount2_bits,
            "layer2_signed_accumulator_bits": accumulator2_bits,
            "layer3_popcount_bits": popcount3_bits,
            "layer3_signed_score_bits": score3_bits,
            "exported_b3_bits": 32,
        },
        "thresholds": {
            "layer1_min": int(threshold1.min()),
            "layer1_max": int(threshold1.max()),
            "layer2_min": int(threshold2.min()),
            "layer2_max": int(threshold2.max()),
            "output_bias": model.b3.astype(int).tolist(),
        },
        "round_trip_verified": True,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in paths.values()
        },
    }
    manifest_path = output_dir / "bnn_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.model.with_name(args.model.stem + "_fpga")
    manifest = export_model(args.model, output_dir)
    print(
        json.dumps(
            {
                "source_model_sha256": manifest["source_model"]["sha256"],
                "architecture": manifest["architecture"],
                "round_trip_verified": manifest["round_trip_verified"],
                "output_dir": str(output_dir.resolve()),
                "files": sorted(manifest["files"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
