"""Verify packed XNOR-popcount inference against signed integer inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bnn.bitexact import (
    activation_thresholds,
    infer_bitexact,
    infer_signed_integer_reference,
    load_bitexact_model,
    sha256_file,
)
from tools.bnn.train_bnn import DEFAULT_DATASET, load_dataset
DEFAULT_MODEL = ROOT / "models" / "v1" / "model.npz"
DEFAULT_TRAINING_REPORT = None  # Historical report uses 4579 samples, current archive uses 4567.


def _metrics(
    expected: np.ndarray, predicted: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    matrix = np.zeros((labels.size, labels.size), dtype=np.int64)
    positions = {int(label): index for index, label in enumerate(labels)}
    for truth, guess in zip(expected, predicted):
        matrix[positions[int(truth)], positions[int(guess)]] += 1
    recall = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(labels.size, dtype=np.float64),
        where=matrix.sum(axis=1) != 0,
    )
    return {
        "samples": int(expected.size),
        "accuracy": float(np.mean(expected == predicted)),
        "macro_recall": float(recall.mean()),
        "per_class_recall": {
            str(int(label)): float(value) for label, value in zip(labels, recall)
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": [int(value) for value in labels],
    }


def verify(
    dataset_path: Path,
    model_path: Path,
    training_report_path: Path | None,
) -> dict[str, object]:
    dataset = load_dataset(dataset_path)
    model = load_bitexact_model(model_path)
    bitexact = infer_bitexact(dataset.images, model)
    reference = infer_signed_integer_reference(dataset.images, model)

    mismatches = {
        "hidden1_activation_bits": int(
            np.count_nonzero(bitexact.h1_bits != reference.h1_bits)
        ),
        "hidden2_activation_bits": int(
            np.count_nonzero(bitexact.h2_bits != reference.h2_bits)
        ),
        "output_scores": int(np.count_nonzero(bitexact.scores != reference.scores)),
        "predicted_labels": int(
            np.count_nonzero(bitexact.predicted_labels != reference.predicted_labels)
        ),
    }
    metrics = _metrics(dataset.labels, bitexact.predicted_labels, model.class_labels)
    per_batch = {
        batch: _metrics(
            dataset.labels[dataset.batches == batch],
            bitexact.predicted_labels[dataset.batches == batch],
            model.class_labels,
        )
        for batch in sorted(str(value) for value in np.unique(dataset.batches))
    }

    report_checks: dict[str, object] = {"provided": False}
    if training_report_path is not None:
        training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
        expected = training_report["full_dataset_fit"]
        # V2 mixed-data reports keep the self-captured and auxiliary fits
        # separate.  Bit-exact verification still runs on the self-captured
        # dataset, so compare against its nested local metrics.
        if isinstance(expected, dict) and isinstance(expected.get("local"), dict):
            expected = expected["local"]
        model_hash = sha256_file(model_path)
        report_checks = {
            "provided": True,
            "model_sha256_matches": training_report["model"]["sha256"] == model_hash,
            "sample_count_matches": int(expected["samples"]) == int(metrics["samples"]),
            "accuracy_matches": float(expected["accuracy"]) == float(metrics["accuracy"]),
            "macro_recall_matches": float(expected["macro_recall"])
            == float(metrics["macro_recall"]),
            "confusion_matrix_matches": expected["confusion_matrix"]
            == metrics["confusion_matrix"],
        }

    passed = all(value == 0 for value in mismatches.values()) and all(
        value is True
        for key, value in report_checks.items()
        if key != "provided"
    )
    input_features, hidden1, hidden2, outputs = model.architecture
    return {
        "schema_version": 1,
        "passed": passed,
        "model": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "architecture": [input_features, hidden1, hidden2, outputs],
            "class_labels": [int(value) for value in model.class_labels],
        },
        "dataset": {
            "path": str(dataset_path.resolve()),
            "samples": int(dataset.images.shape[0]),
            "batches": sorted(str(value) for value in np.unique(dataset.batches)),
        },
        "arithmetic": {
            "input_encoding": "bit 0 => -1, bit 1 => +1",
            "weight_encoding": "bit 0 => -1, bit 1 => +1",
            "dot_product": "2 * XNOR_popcount - input_features",
            "hidden_activation": "+1 when dot_product + int32_bias >= 0",
            "argmax_tie_break": "lowest class index wins",
            "hidden1_popcount_thresholds": activation_thresholds(
                input_features, model.b1
            ).tolist(),
            "hidden2_popcount_thresholds": activation_thresholds(
                hidden1, model.b2
            ).tolist(),
            "output_bias": model.b3.astype(int).tolist(),
        },
        "mismatches": mismatches,
        "full_dataset_metrics": metrics,
        "per_batch_full_model_metrics": per_batch,
        "training_report_checks": report_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--training-report", type=Path, default=DEFAULT_TRAINING_REPORT
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.dataset, args.model, args.training_report)
    output = args.output or ROOT / "build" / "bitexact_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        "passed": result["passed"],
        "samples": result["dataset"]["samples"],
        "mismatches": result["mismatches"],
        "accuracy": result["full_dataset_metrics"]["accuracy"],
        "macro_recall": result["full_dataset_metrics"]["macro_recall"],
        "output": str(output.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
