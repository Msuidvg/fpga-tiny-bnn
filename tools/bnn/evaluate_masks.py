"""Evaluate frozen V1 on externally labelled 256x144 binary masks."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .bitexact import infer_bitexact, load_bitexact_model, sha256_file
from .demo import ROOT, classify, read_mask
from .dataset_archive import load_archive
from .train_bnn import _curated_label, _load_curation


CLASSES = (0, 1, 2, 3, 4, 5)


def metrics(matrix: np.ndarray) -> dict:
    hands = int(matrix[1:, :].sum())
    no_hands = int(matrix[0, :].sum())
    correct = int(np.trace(matrix[1:, 1:]))
    return {
        "confusion": matrix.tolist(),
        "hand_samples": hands,
        "correct_hand_predictions": correct,
        "hand_accuracy": correct / hands if hands else None,
        "hand_no_component": int(matrix[1:, 0].sum()),
        "recall": {
            str(label): int(matrix[label, label]) / int(matrix[label].sum())
            if matrix[label].sum() else None
            for label in CLASSES[1:]
        },
        "no_hand_samples": no_hands,
        "no_hand_false_positives": int(matrix[0, 1:].sum()),
        "no_hand_false_positive_rate": float(matrix[0, 1:].sum() / no_hands) if no_hands else None,
    }


def evaluate(manifests: list[Path], *, normalized: bool, curated: bool) -> dict:
    model_path = ROOT / "models" / "v1" / "model.npz"
    model = load_bitexact_model(model_path)
    train = load_archive(ROOT / "data" / "gesture_v1.npz")
    train_masks = {mask.tobytes() for mask in train["images"]}
    groups = {}
    for manifest in manifests:
        matrix = np.zeros((6, 6), dtype=np.int64)
        seen = set()
        overlaps, excluded = 0, 0
        rules, curation = _load_curation(manifest.parent) if curated else ([], None)
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                label = int(row["label"])
                if label not in (CLASSES[1:] if normalized else CLASSES):
                    raise ValueError(f"invalid label {label} in {manifest}")
                relative = Path(row["path" if normalized else "raw_s5_path"])
                sample = (manifest.parent / relative).resolve()
                if not sample.is_relative_to(manifest.parent.resolve()) or sample in seen:
                    raise ValueError(f"invalid or repeated mask path in {manifest}: {relative}")
                seen.add(sample)
                checksum = row.get("sha256" if normalized else "raw_s5_sha256")
                if checksum and sha256_file(sample) != checksum:
                    raise ValueError(f"input checksum mismatch: {sample}")
                if rules:
                    label = _curated_label(label, int(sample.stem.rsplit("_", 1)[-1]), rules, sample)
                    if label is None:
                        excluded += 1
                        continue
                if normalized:
                    roi = read_mask(sample, (32, 32))
                    prediction = int(infer_bitexact(roi[np.newaxis], model).predicted_labels[0])
                else:
                    result, roi = classify(read_mask(sample), model)
                    prediction = result["class_id"] or 0
                if roi is not None and roi.tobytes() in train_masks:
                    overlaps += 1
                    continue
                matrix[label, prediction] += 1
        if not seen or manifest.parent.name in groups:
            raise ValueError(f"empty manifest or repeated batch name: {manifest}")
        groups[manifest.parent.name] = {
            **metrics(matrix),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "curation_sha256": sha256_file(Path(curation)) if curation else None,
            "input_samples": len(seen),
            "excluded_by_curation": excluded,
            "excluded_exact_training_mask_overlap": overlaps,
        }
    total = np.sum([group["confusion"] for group in groups.values()], axis=0)
    return {
        "method": "Frozen V1 software replay from normalized 32x32 masks" if normalized
                  else "Frozen V1 software replay from binary 256x144 masks",
        "model_sha256": sha256_file(model_path),
        "labels": list(CLASSES),
        "zero_label": "No hand (true label) or no component (prediction)",
        "batches": groups,
        "total": metrics(total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="+", help="CSV with label and raw_s5_path columns")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--normalized", action="store_true", help="use labelled 32x32 masks from the path column")
    parser.add_argument("--apply-curation", action="store_true", help="apply the manifest folder's curation.json rules")
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest, normalized=args.normalized, curated=args.apply_curation)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
