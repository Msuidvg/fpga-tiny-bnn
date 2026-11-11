"""Train and cross-batch-evaluate a deployable 1-bit Tiny-BNN with NumPy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "gesture_v1.npz"
DEFAULT_MODEL = ROOT / "build" / "trained_model.npz"
DEFAULT_REPORT = ROOT / "build" / "training_report.json"
EXPECTED_LABELS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Dataset:
    images: np.ndarray
    labels: np.ndarray
    batches: np.ndarray
    paths: tuple[str, ...]
    curation_files: tuple[str, ...] = ()
    excluded_batches: tuple[str, ...] = ()


def _load_curation(batch_dir: Path) -> tuple[list[dict[str, object]], str | None]:
    path = batch_dir / "curation.json"
    if not path.is_file():
        return [], None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(
        document.get("rules"), list
    ):
        raise ValueError(f"invalid curation schema: {path}")
    rules: list[dict[str, object]] = []
    for index, rule in enumerate(document["rules"]):
        if not isinstance(rule, dict):
            raise ValueError(f"{path}: rule {index} must be an object")
        action = rule.get("action")
        label = rule.get("label")
        first = rule.get("sample_index_start")
        last = rule.get("sample_index_end")
        if action not in {"exclude", "relabel"}:
            raise ValueError(f"{path}: rule {index} has invalid action")
        if label not in EXPECTED_LABELS:
            raise ValueError(f"{path}: rule {index} has invalid label")
        if not isinstance(first, int) or not isinstance(last, int) or first > last:
            raise ValueError(f"{path}: rule {index} has invalid sample range")
        if action == "relabel" and rule.get("new_label") not in EXPECTED_LABELS:
            raise ValueError(f"{path}: rule {index} has invalid new_label")
        rules.append(rule)
    return rules, str(path.resolve())


def _curated_label(
    label: int, sample_index: int, rules: list[dict[str, object]], source: Path
) -> int | None:
    matches = [
        rule
        for rule in rules
        if int(rule["label"]) == label
        and int(rule["sample_index_start"]) <= sample_index
        and sample_index <= int(rule["sample_index_end"])
    ]
    if len(matches) > 1:
        raise ValueError(f"overlapping curation rules for {source}")
    if not matches:
        return label
    rule = matches[0]
    if rule["action"] == "exclude":
        return None
    return int(rule["new_label"])


def load_dataset(root: Path) -> Dataset:
    if root.is_file() and root.suffix == ".npz":
        from tools.bnn.dataset_archive import load_archive
        arrays = load_archive(root)
        return Dataset(**arrays)
    images: list[np.ndarray] = []
    labels: list[int] = []
    batches: list[str] = []
    paths: list[str] = []
    curation_files: list[str] = []
    excluded_batches: list[str] = []
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"dataset directory does not exist: {root}")

    for batch_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        batch_metadata_path = batch_dir / "batch.json"
        if not batch_metadata_path.is_file():
            continue
        batch_metadata = json.loads(batch_metadata_path.read_text(encoding="utf-8"))
        if batch_metadata.get("training_enabled", True) is False:
            excluded_batches.append(batch_dir.name)
            continue
        rules, curation_path = _load_curation(batch_dir)
        if curation_path is not None:
            curation_files.append(curation_path)
        for label in EXPECTED_LABELS:
            label_dir = batch_dir / str(label)
            if not label_dir.is_dir():
                continue
            for image_path in sorted(label_dir.glob("*.png")):
                try:
                    sample_index = int(image_path.stem.rsplit("_", 1)[-1])
                except ValueError as exc:
                    raise ValueError(
                        f"sample filename has no numeric index: {image_path}"
                    ) from exc
                curated_label = _curated_label(
                    label, sample_index, rules, image_path
                )
                if curated_label is None:
                    continue
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"cannot decode image: {image_path}")
                if image.shape != (32, 32):
                    raise ValueError(
                        f"{image_path}: expected 32x32, got {image.shape}"
                    )
                images.append((image >= 128).astype(np.uint8))
                labels.append(curated_label)
                batches.append(batch_dir.name)
                paths.append(str(image_path.relative_to(root)))

    if not images:
        raise ValueError(f"no 32x32 PNG samples found below {root}")
    return Dataset(
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        batches=np.asarray(batches),
        paths=tuple(paths),
        curation_files=tuple(curation_files),
        excluded_batches=tuple(excluded_batches),
    )


def dataset_summary(dataset: Dataset) -> dict[str, object]:
    unique_batches = sorted(str(value) for value in np.unique(dataset.batches))
    counts = {
        batch: {
            str(label): int(
                np.count_nonzero(
                    (dataset.batches == batch) & (dataset.labels == label)
                )
            )
            for label in EXPECTED_LABELS
        }
        for batch in unique_batches
    }
    digest_batches: dict[str, set[str]] = {}
    for image, batch in zip(dataset.images, dataset.batches):
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        digest_batches.setdefault(digest, set()).add(str(batch))
    cross_batch_duplicates = sum(
        1 for value in digest_batches.values() if len(value) > 1
    )
    return {
        "samples": int(dataset.images.shape[0]),
        "batches": unique_batches,
        "counts": counts,
        "cross_batch_duplicate_images": cross_batch_duplicates,
        "curation_files": list(dataset.curation_files),
        "excluded_batches": list(dataset.excluded_batches),
    }


def _binary_sign(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, 1.0, -1.0).astype(np.float32)


class BinaryMLP:
    """Two hidden-layer BNN: binary input, weights, and hidden activations."""

    def __init__(
        self,
        input_features: int,
        hidden1: int,
        hidden2: int,
        classes: int,
        rng: np.random.Generator,
    ) -> None:
        dimensions = (input_features, hidden1, hidden2, classes)
        self.w1 = rng.normal(0.0, 0.08, (dimensions[0], dimensions[1])).astype(
            np.float32
        )
        self.b1 = np.zeros(dimensions[1], dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.12, (dimensions[1], dimensions[2])).astype(
            np.float32
        )
        self.b2 = np.zeros(dimensions[2], dtype=np.float32)
        self.w3 = rng.normal(0.0, 0.16, (dimensions[2], dimensions[3])).astype(
            np.float32
        )
        self.b3 = np.zeros(dimensions[3], dtype=np.float32)

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3,
        }

    def forward(
        self, inputs: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        x = np.asarray(inputs, dtype=np.float32)
        bw1 = _binary_sign(self.w1)
        bw2 = _binary_sign(self.w2)
        bw3 = _binary_sign(self.w3)
        scale1 = math.sqrt(self.w1.shape[0])
        scale2 = math.sqrt(self.w2.shape[0])
        scale3 = math.sqrt(self.w3.shape[0])
        z1 = (x @ bw1) / scale1 + self.b1
        h1 = _binary_sign(z1)
        z2 = (h1 @ bw2) / scale2 + self.b2
        h2 = _binary_sign(z2)
        logits = (h2 @ bw3) / scale3 + self.b3
        return logits, (x, z1, h1, z2, h2, bw1, bw2, bw3)

    def loss_and_gradients(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        class_weights: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray]]:
        logits, cache = self.forward(inputs)
        x, z1, h1, z2, h2, bw1, bw2, bw3 = cache
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        sample_weights = class_weights[targets].astype(np.float32)
        sample_weights /= max(float(sample_weights.mean()), 1e-8)
        loss = -float(
            np.mean(
                sample_weights
                * np.log(probabilities[np.arange(targets.size), targets] + 1e-9)
            )
        )

        dlogits = probabilities
        dlogits[np.arange(targets.size), targets] -= 1.0
        dlogits *= sample_weights[:, None] / targets.size
        scale1 = math.sqrt(self.w1.shape[0])
        scale2 = math.sqrt(self.w2.shape[0])
        scale3 = math.sqrt(self.w3.shape[0])

        gradients: dict[str, np.ndarray] = {}
        gradients["w3"] = (h2.T @ dlogits) / scale3
        gradients["b3"] = dlogits.sum(axis=0)
        dh2 = (dlogits @ bw3.T) / scale3
        dz2 = dh2 * (np.abs(z2) <= 1.0)
        gradients["w2"] = (h1.T @ dz2) / scale2
        gradients["b2"] = dz2.sum(axis=0)
        dh1 = (dz2 @ bw2.T) / scale2
        dz1 = dh1 * (np.abs(z1) <= 1.0)
        gradients["w1"] = (x.T @ dz1) / scale1
        gradients["b1"] = dz1.sum(axis=0)

        for name in ("w1", "w2", "w3"):
            gradients[name] *= np.abs(self.parameters[name]) <= 1.0
        return loss, gradients

    def deployment_arrays(self) -> dict[str, np.ndarray]:
        """Return the exact signed-one weights and integer biases for inference."""

        return {
            "w1": _binary_sign(self.w1).astype(np.int8),
            "b1": np.rint(self.b1 * math.sqrt(self.w1.shape[0])).astype(
                np.int32
            ),
            "w2": _binary_sign(self.w2).astype(np.int8),
            "b2": np.rint(self.b2 * math.sqrt(self.w2.shape[0])).astype(
                np.int32
            ),
            "w3": _binary_sign(self.w3).astype(np.int8),
            "b3": np.rint(self.b3 * math.sqrt(self.w3.shape[0])).astype(
                np.int32
            ),
        }


class Adam:
    def __init__(self, parameters: dict[str, np.ndarray]) -> None:
        self.m = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.v = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step_count = 0

    def step(
        self,
        parameters: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        self.step_count += 1
        beta1, beta2 = 0.9, 0.999
        for name, parameter in parameters.items():
            gradient = np.clip(gradients[name], -5.0, 5.0)
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * gradient
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * gradient * gradient
            m_hat = self.m[name] / (1.0 - beta1**self.step_count)
            v_hat = self.v[name] / (1.0 - beta2**self.step_count)
            parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            if name.startswith("w"):
                np.clip(parameter, -1.0, 1.0, out=parameter)


def _augment(
    flat_images: np.ndarray,
    rng: np.random.Generator,
    *,
    max_shift: int,
    noise_probability: float,
    max_rotation: float = 0.0,
    scale_jitter: float = 0.0,
    morph_probability: float = 0.0,
    background_noise_probability: float = 0.0,
    max_background_blobs: int = 0,
) -> np.ndarray:
    if (
        max_shift == 0
        and noise_probability == 0.0
        and max_rotation == 0.0
        and scale_jitter == 0.0
        and morph_probability == 0.0
        and background_noise_probability == 0.0
    ):
        return flat_images.copy()
    images = (flat_images.reshape(-1, 32, 32) > 0).astype(np.uint8)
    output = np.zeros_like(images)
    morphology_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for index, image in enumerate(images):
        shift_x = int(rng.integers(-max_shift, max_shift + 1))
        shift_y = int(rng.integers(-max_shift, max_shift + 1))
        if max_rotation == 0.0 and scale_jitter == 0.0:
            augmented = np.zeros_like(image)
            src_x0 = max(0, -shift_x)
            src_y0 = max(0, -shift_y)
            src_x1 = min(32, 32 - shift_x)
            src_y1 = min(32, 32 - shift_y)
            dst_x0 = max(0, shift_x)
            dst_y0 = max(0, shift_y)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            augmented[dst_y0:dst_y1, dst_x0:dst_x1] = image[
                src_y0:src_y1, src_x0:src_x1
            ]
        else:
            angle = float(rng.uniform(-max_rotation, max_rotation))
            scale = float(rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter))
            transform = cv2.getRotationMatrix2D((15.5, 15.5), angle, scale)
            transform[0, 2] += shift_x
            transform[1, 2] += shift_y
            augmented = cv2.warpAffine(
                image,
                transform,
                (32, 32),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        if morph_probability > 0.0 and rng.random() < morph_probability:
            if rng.random() < 0.5:
                augmented = cv2.dilate(augmented, morphology_kernel, iterations=1)
            else:
                augmented = cv2.erode(augmented, morphology_kernel, iterations=1)
        if (
            background_noise_probability > 0.0
            and max_background_blobs > 0
            and rng.random() < background_noise_probability
        ):
            forbidden = cv2.dilate(
                augmented, morphology_kernel, iterations=1
            ).astype(bool)
            fragments = np.zeros_like(augmented)
            blob_count = int(rng.integers(1, max_background_blobs + 1))
            for _ in range(blob_count):
                x = int(rng.integers(0, 32))
                y = int(rng.integers(0, 32))
                shape = int(rng.integers(0, 3))
                if shape == 0:
                    radius = int(rng.integers(1, 3))
                    cv2.circle(fragments, (x, y), radius, 1, -1)
                elif shape == 1:
                    width = int(rng.integers(1, 4))
                    height = int(rng.integers(1, 4))
                    cv2.rectangle(
                        fragments,
                        (x, y),
                        (min(31, x + width), min(31, y + height)),
                        1,
                        -1,
                    )
                else:
                    length = int(rng.integers(2, 7))
                    horizontal = bool(rng.integers(0, 2))
                    endpoint = (
                        (min(31, x + length), y)
                        if horizontal
                        else (x, min(31, y + length))
                    )
                    cv2.line(fragments, (x, y), endpoint, 1, 1)
            fragments[forbidden] = 0
            augmented = np.maximum(augmented, fragments)
        output[index] = augmented
    signed = output.astype(np.float32) * 2.0 - 1.0
    if noise_probability > 0.0:
        flips = rng.random(signed.shape) < noise_probability
        signed[flips] *= -1.0
    return signed.reshape(flat_images.shape)


def prepare_inputs(images: np.ndarray) -> np.ndarray:
    binary = (np.asarray(images) != 0).astype(np.float32)
    return (binary.reshape(binary.shape[0], -1) * 2.0 - 1.0).astype(np.float32)


def _batch_class_balanced_indices(
    targets: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one epoch with equal contributions from every batch/class cell."""

    targets = np.asarray(targets)
    groups = np.asarray(groups)
    if targets.ndim != 1 or groups.ndim != 1 or targets.size != groups.size:
        raise ValueError("targets and groups must be same-length 1-D arrays")
    if targets.size == 0:
        raise ValueError("cannot balance an empty training split")
    unique_targets = np.unique(targets)
    unique_groups = np.unique(groups)
    cell_count = int(unique_targets.size * unique_groups.size)
    samples_per_cell = int(math.ceil(targets.size / cell_count))
    cells: list[np.ndarray] = []
    for group in unique_groups:
        for target in unique_targets:
            candidates = np.flatnonzero((groups == group) & (targets == target))
            if candidates.size == 0:
                raise ValueError(
                    f"batch {group!s} has no samples for class {int(target) + 1}"
                )
            cells.append(
                rng.choice(
                    candidates,
                    size=samples_per_cell,
                    replace=candidates.size < samples_per_cell,
                )
            )
    order = np.concatenate(cells).astype(np.int64, copy=False)
    return order[rng.permutation(order.size)]


def predict_deployment(
    inputs: np.ndarray, arrays: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(inputs, dtype=np.int16)
    h1 = np.where(x @ arrays["w1"].astype(np.int16) + arrays["b1"] >= 0, 1, -1)
    h2 = np.where(h1 @ arrays["w2"].astype(np.int16) + arrays["b2"] >= 0, 1, -1)
    scores = h2 @ arrays["w3"].astype(np.int16) + arrays["b3"]
    return np.argmax(scores, axis=1), scores


def train_model(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    hidden1: int,
    hidden2: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_shift: int,
    noise_probability: float,
    max_rotation: float = 0.0,
    scale_jitter: float = 0.0,
    morph_probability: float = 0.0,
    background_noise_probability: float = 0.0,
    max_background_blobs: int = 0,
    groups: np.ndarray | None = None,
    verbose: bool = True,
) -> BinaryMLP:
    rng = np.random.default_rng(seed)
    class_count = int(targets.max()) + 1
    model = BinaryMLP(inputs.shape[1], hidden1, hidden2, class_count, rng)
    optimizer = Adam(model.parameters)
    counts = np.bincount(targets, minlength=class_count).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("every class must be present in the training split")
    if groups is None:
        class_weights = targets.size / (class_count * counts)
    else:
        groups = np.asarray(groups)
        if groups.ndim != 1 or groups.size != targets.size:
            raise ValueError("groups must match the number of training samples")
        class_weights = np.ones(class_count, dtype=np.float32)

    for epoch in range(epochs):
        order = (
            rng.permutation(targets.size)
            if groups is None
            else _batch_class_balanced_indices(targets, groups, rng)
        )
        epoch_losses: list[float] = []
        epoch_lr = learning_rate * (0.15 + 0.85 * (1.0 + math.cos(math.pi * epoch / epochs)) / 2.0)
        for start in range(0, targets.size, batch_size):
            indices = order[start : start + batch_size]
            batch = _augment(
                inputs[indices],
                rng,
                max_shift=max_shift,
                noise_probability=noise_probability,
                max_rotation=max_rotation,
                scale_jitter=scale_jitter,
                morph_probability=morph_probability,
                background_noise_probability=background_noise_probability,
                max_background_blobs=max_background_blobs,
            )
            loss, gradients = model.loss_and_gradients(
                batch, targets[indices], class_weights
            )
            optimizer.step(model.parameters, gradients, epoch_lr)
            epoch_losses.append(loss)
        if verbose and (
            epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs
        ):
            predicted, _ = predict_deployment(inputs, model.deployment_arrays())
            accuracy = float(np.mean(predicted == targets))
            print(
                f"epoch={epoch + 1:03d}/{epochs} "
                f"loss={np.mean(epoch_losses):.4f} deploy_train={accuracy:.4f}"
            )
    return model


def classification_metrics(
    expected: np.ndarray, predicted: np.ndarray, class_count: int
) -> dict[str, object]:
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for truth, guess in zip(expected, predicted):
        confusion[int(truth), int(guess)] += 1
    recalls = []
    for index in range(class_count):
        denominator = int(confusion[index].sum())
        recalls.append(
            None if denominator == 0 else float(confusion[index, index] / denominator)
        )
    valid_recalls = [value for value in recalls if value is not None]
    return {
        "samples": int(expected.size),
        "accuracy": float(np.mean(expected == predicted)),
        "macro_recall": float(np.mean(valid_recalls)),
        "per_class_recall": {
            str(EXPECTED_LABELS[index]): value for index, value in enumerate(recalls)
        },
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": list(EXPECTED_LABELS[:class_count]),
    }


def _validate_split_labels(labels: np.ndarray, split_name: str) -> None:
    present = set(int(value) for value in np.unique(labels))
    expected = set(EXPECTED_LABELS)
    if present != expected:
        missing = sorted(expected - present)
        raise ValueError(f"{split_name} is missing labels {missing}")


def leave_one_batch_out(
    dataset: Dataset,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    hidden1: int,
    hidden2: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_shift: int,
    noise_probability: float,
    max_rotation: float,
    scale_jitter: float,
    morph_probability: float,
    background_noise_probability: float,
    max_background_blobs: int,
    sampling_mode: str,
) -> dict[str, object]:
    batches = sorted(str(value) for value in np.unique(dataset.batches))
    if len(batches) < 2:
        raise ValueError(
            "leave-one-batch-out requires at least two real capture batches"
        )
    folds: list[dict[str, object]] = []
    all_expected: list[np.ndarray] = []
    all_predicted: list[np.ndarray] = []
    for fold_index, held_out in enumerate(batches):
        test_mask = dataset.batches == held_out
        train_mask = ~test_mask
        _validate_split_labels(dataset.labels[train_mask], f"training split for {held_out}")
        _validate_split_labels(dataset.labels[test_mask], f"held-out batch {held_out}")
        print(
            f"LOBO hold_out={held_out} train={np.count_nonzero(train_mask)} "
            f"test={np.count_nonzero(test_mask)}"
        )
        model = train_model(
            inputs[train_mask],
            targets[train_mask],
            hidden1=hidden1,
            hidden2=hidden2,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + fold_index,
            max_shift=max_shift,
            noise_probability=noise_probability,
            max_rotation=max_rotation,
            scale_jitter=scale_jitter,
            morph_probability=morph_probability,
            background_noise_probability=background_noise_probability,
            max_background_blobs=max_background_blobs,
            groups=(
                dataset.batches[train_mask]
                if sampling_mode == "batch-class-balanced"
                else None
            ),
        )
        predicted, _ = predict_deployment(
            inputs[test_mask], model.deployment_arrays()
        )
        metrics = classification_metrics(targets[test_mask], predicted, 5)
        folds.append({"held_out_batch": held_out, **metrics})
        all_expected.append(targets[test_mask])
        all_predicted.append(predicted)
        print(
            f"held_out={held_out} accuracy={metrics['accuracy']:.4f} "
            f"macro_recall={metrics['macro_recall']:.4f}"
        )
    aggregate = classification_metrics(
        np.concatenate(all_expected), np.concatenate(all_predicted), 5
    )
    return {"folds": folds, "aggregate": aggregate}


def within_batch_blocked_validation(
    dataset: Dataset,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    hidden1: int,
    hidden2: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_shift: int,
    noise_probability: float,
    max_rotation: float,
    scale_jitter: float,
    morph_probability: float,
    background_noise_probability: float,
    max_background_blobs: int,
    sampling_mode: str,
    train_fraction: float = 0.70,
) -> dict[str, object]:
    """Evaluate the two chronological ends of a single capture batch.

    This is intentionally not called cross-batch validation.  It catches
    label transitions and within-session drift without leaking neighboring
    frames through a random split.
    """

    batches = np.unique(dataset.batches)
    if batches.size != 1:
        raise ValueError("blocked validation expects exactly one capture batch")
    directions = ("first_70_train_last_30_test", "last_70_train_first_30_test")
    folds: list[dict[str, object]] = []
    all_expected: list[np.ndarray] = []
    all_predicted: list[np.ndarray] = []
    for fold_index, direction in enumerate(directions):
        train_indices: list[int] = []
        test_indices: list[int] = []
        for target in range(len(EXPECTED_LABELS)):
            indices = np.flatnonzero(targets == target)
            indices = np.asarray(
                sorted(
                    indices,
                    key=lambda index: int(
                        Path(dataset.paths[int(index)]).stem.rsplit("_", 1)[-1]
                    ),
                )
            )
            split = max(1, int(indices.size * train_fraction))
            if indices.size - split < 1:
                raise ValueError(
                    f"label {EXPECTED_LABELS[target]} has too few samples "
                    "for blocked validation"
                )
            if fold_index == 0:
                train_indices.extend(indices[:split])
                test_indices.extend(indices[split:])
            else:
                train_indices.extend(indices[-split:])
                test_indices.extend(indices[:-split])
        train = np.asarray(train_indices, dtype=np.int64)
        test = np.asarray(test_indices, dtype=np.int64)
        print(
            f"blocked direction={direction} train={train.size} test={test.size}"
        )
        model = train_model(
            inputs[train],
            targets[train],
            hidden1=hidden1,
            hidden2=hidden2,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + fold_index,
            max_shift=max_shift,
            noise_probability=noise_probability,
            max_rotation=max_rotation,
            scale_jitter=scale_jitter,
            morph_probability=morph_probability,
            background_noise_probability=background_noise_probability,
            max_background_blobs=max_background_blobs,
            groups=(
                dataset.batches[train]
                if sampling_mode == "batch-class-balanced"
                else None
            ),
        )
        predicted, _ = predict_deployment(
            inputs[test], model.deployment_arrays()
        )
        metrics = classification_metrics(targets[test], predicted, 5)
        folds.append({"direction": direction, **metrics})
        all_expected.append(targets[test])
        all_predicted.append(predicted)
        print(
            f"direction={direction} accuracy={metrics['accuracy']:.4f} "
            f"macro_recall={metrics['macro_recall']:.4f}"
        )
    aggregate = classification_metrics(
        np.concatenate(all_expected), np.concatenate(all_predicted), 5
    )
    return {
        "note": "same-environment temporal blocks; not cross-batch accuracy",
        "train_fraction": train_fraction,
        "folds": folds,
        "aggregate": aggregate,
    }


def save_deployment_model(
    path: Path,
    model: BinaryMLP,
    *,
    labels: Iterable[int] = EXPECTED_LABELS,
) -> dict[str, object]:
    arrays = model.deployment_arrays()
    packed = {
        f"{name}_packed": np.packbits(
            (arrays[name].T > 0).astype(np.uint8), axis=1, bitorder="little"
        )
        for name in ("w1", "w2", "w3")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int32),
        class_labels=np.asarray(tuple(labels), dtype=np.int8),
        input_shape=np.asarray((32, 32), dtype=np.int32),
        **arrays,
        **packed,
    )
    weight_bits = int(sum(arrays[name].size for name in ("w1", "w2", "w3")))
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "architecture": [
            int(arrays["w1"].shape[0]),
            int(arrays["w1"].shape[1]),
            int(arrays["w2"].shape[1]),
            int(arrays["w3"].shape[1]),
        ],
        "binary_weight_bits": weight_bits,
        "binary_weight_bytes": (weight_bits + 7) // 8,
        "hidden_activations": "1-bit signed {-1,+1}",
        "biases": "signed int32 folded comparator/score offsets",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a 32x32 1-bit Tiny-BNN and evaluate real batches."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--hidden1", type=int, default=96)
    parser.add_argument("--hidden2", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-shift", type=int, default=1)
    parser.add_argument("--noise-probability", type=float, default=0.002)
    parser.add_argument("--max-rotation", type=float, default=0.0)
    parser.add_argument("--scale-jitter", type=float, default=0.0)
    parser.add_argument("--morph-probability", type=float, default=0.0)
    parser.add_argument("--background-noise-probability", type=float, default=0.0)
    parser.add_argument("--max-background-blobs", type=int, default=0)
    parser.add_argument(
        "--sampling",
        choices=("batch-class-balanced", "class-weighted"),
        default="batch-class-balanced",
        help="training sampler; validation samples are never resampled",
    )
    parser.add_argument(
        "--skip-lobo",
        action="store_true",
        help="train the final model without cross-batch evaluation",
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="inspect data and exit"
    )
    args = parser.parse_args()

    if min(args.hidden1, args.hidden2, args.epochs, args.batch_size) <= 0:
        parser.error("hidden sizes, epochs and batch size must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 <= args.max_shift <= 8:
        parser.error("--max-shift must be between 0 and 8")
    if not 0.0 <= args.noise_probability < 0.5:
        parser.error("--noise-probability must be in [0, 0.5)")
    if not 0.0 <= args.max_rotation <= 45.0:
        parser.error("--max-rotation must be in [0, 45]")
    if not 0.0 <= args.scale_jitter < 0.5:
        parser.error("--scale-jitter must be in [0, 0.5)")
    if not 0.0 <= args.morph_probability <= 1.0:
        parser.error("--morph-probability must be in [0, 1]")
    if not 0.0 <= args.background_noise_probability <= 1.0:
        parser.error("--background-noise-probability must be in [0, 1]")
    if not 0 <= args.max_background_blobs <= 64:
        parser.error("--max-background-blobs must be in [0, 64]")

    try:
        dataset = load_dataset(args.dataset)
        _validate_split_labels(dataset.labels, "complete dataset")
    except ValueError as exc:
        print(f"Dataset error: {exc}")
        return 2
    summary = dataset_summary(dataset)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["cross_batch_duplicate_images"]:
        print(
            "warning: exact images occur in multiple batches; inspect possible "
            "cross-batch leakage before interpreting accuracy"
        )
    if args.summary_only:
        return 0

    inputs = prepare_inputs(dataset.images)
    targets = dataset.labels - 1
    report: dict[str, object] = {
        "schema_version": 1,
        "dataset": str(args.dataset.resolve()),
        "dataset_summary": summary,
        "seed": args.seed,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_shift": args.max_shift,
            "noise_probability": args.noise_probability,
            "max_rotation": args.max_rotation,
            "scale_jitter": args.scale_jitter,
            "morph_probability": args.morph_probability,
            "background_noise_probability": args.background_noise_probability,
            "max_background_blobs": args.max_background_blobs,
            "sampling": args.sampling,
            "evaluation_inference": "integer 1-bit deployment model",
        },
    }
    try:
        if not args.skip_lobo:
            if len(summary["batches"]) >= 2:
                report["leave_one_batch_out"] = leave_one_batch_out(
                    dataset,
                    inputs,
                    targets,
                    hidden1=args.hidden1,
                    hidden2=args.hidden2,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_shift=args.max_shift,
                    noise_probability=args.noise_probability,
                    max_rotation=args.max_rotation,
                    scale_jitter=args.scale_jitter,
                    morph_probability=args.morph_probability,
                    background_noise_probability=args.background_noise_probability,
                    max_background_blobs=args.max_background_blobs,
                    sampling_mode=args.sampling,
                )
            else:
                print(
                    "only one capture batch: running chronological blocked "
                    "validation (this is not cross-batch accuracy)"
                )
                report["within_batch_blocked"] = within_batch_blocked_validation(
                    dataset,
                    inputs,
                    targets,
                    hidden1=args.hidden1,
                    hidden2=args.hidden2,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_shift=args.max_shift,
                    noise_probability=args.noise_probability,
                    max_rotation=args.max_rotation,
                    scale_jitter=args.scale_jitter,
                    morph_probability=args.morph_probability,
                    background_noise_probability=args.background_noise_probability,
                    max_background_blobs=args.max_background_blobs,
                    sampling_mode=args.sampling,
                )
        print(f"final training samples={inputs.shape[0]}")
        final_model = train_model(
            inputs,
            targets,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed + 1000,
            max_shift=args.max_shift,
            noise_probability=args.noise_probability,
            max_rotation=args.max_rotation,
            scale_jitter=args.scale_jitter,
            morph_probability=args.morph_probability,
            background_noise_probability=args.background_noise_probability,
            max_background_blobs=args.max_background_blobs,
            groups=(
                dataset.batches
                if args.sampling == "batch-class-balanced"
                else None
            ),
        )
    except ValueError as exc:
        print(f"Training error: {exc}")
        return 2

    predicted, _ = predict_deployment(inputs, final_model.deployment_arrays())
    report["full_dataset_fit"] = classification_metrics(targets, predicted, 5)
    report["model"] = save_deployment_model(args.model, final_model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"model={args.model}")
    print(f"report={args.report}")
    if "leave_one_batch_out" in report:
        aggregate = report["leave_one_batch_out"]["aggregate"]
        print(
            f"LOBO_accuracy={aggregate['accuracy']:.4f} "
            f"LOBO_macro_recall={aggregate['macro_recall']:.4f}"
        )
    elif "within_batch_blocked" in report:
        aggregate = report["within_batch_blocked"]["aggregate"]
        print(
            f"blocked_accuracy={aggregate['accuracy']:.4f} "
            f"blocked_macro_recall={aggregate['macro_recall']:.4f} "
            "(not cross-batch)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
