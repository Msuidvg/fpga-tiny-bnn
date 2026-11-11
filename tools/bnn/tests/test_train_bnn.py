from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.bnn.train_bnn import (
    _augment,
    _batch_class_balanced_indices,
    dataset_summary,
    load_dataset,
    predict_deployment,
    prepare_inputs,
    save_deployment_model,
    train_model,
)
from tools.bnn.conv_bnn import BinaryConvNet, predict_deployment as predict_conv


class TinyBnnTests(unittest.TestCase):
    def test_structured_background_noise_preserves_foreground(self) -> None:
        image = np.full((1, 32, 32), -1.0, dtype=np.float32)
        image[:, 12:20, 12:20] = 1.0
        augmented = _augment(
            image.reshape(1, -1),
            np.random.default_rng(29),
            max_shift=0,
            noise_probability=0.0,
            background_noise_probability=1.0,
            max_background_blobs=12,
        ).reshape(1, 32, 32)
        self.assertTrue(np.all(augmented[:, 12:20, 12:20] == 1.0))
        self.assertGreater(np.count_nonzero(augmented > 0), 64)
        self.assertTrue(np.all(np.isin(augmented, (-1.0, 1.0))))

    def test_batch_class_balanced_epoch_equalizes_every_cell(self) -> None:
        targets = np.asarray([0, 0, 0, 1, 0, 1], dtype=np.int64)
        groups = np.asarray(["a", "a", "a", "a", "b", "b"])
        order = _batch_class_balanced_indices(
            targets, groups, np.random.default_rng(23)
        )
        self.assertEqual(order.size, 8)
        counts = {
            (group, target): int(
                np.count_nonzero(
                    (groups[order] == group) & (targets[order] == target)
                )
            )
            for group in ("a", "b")
            for target in (0, 1)
        }
        self.assertEqual(set(counts.values()), {2})

    def test_convolutional_bnn_float_and_integer_paths_agree_at_zero_bias(self) -> None:
        rng = np.random.default_rng(19)
        model = BinaryConvNet(5, rng, pool_size=2)
        inputs = rng.choice((-1.0, 1.0), size=(6, 1024)).astype(np.float32)
        logits, _ = model.forward(inputs)
        predicted, integer_scores = predict_conv(
            inputs, model.deployment_arrays()
        )
        np.testing.assert_array_equal(predicted, np.argmax(logits, axis=1))
        self.assertEqual(integer_scores.shape, (6, 5))
        weight_bits = sum(
            model.deployment_arrays()[name].size for name in ("w1", "w2", "w3")
        )
        self.assertEqual(weight_bits, 12872)

    def test_curation_excludes_and_relabels_without_changing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch_a"
            batch.mkdir()
            (batch / "batch.json").write_text("{}", encoding="utf-8")
            for label in range(1, 6):
                label_dir = batch / str(label)
                label_dir.mkdir()
                image = np.full((32, 32), label * 30, dtype=np.uint8)
                cv2.imwrite(str(label_dir / f"batch_a_{label}_00000{label}.png"), image)
            (batch / "curation.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "rules": [
                            {
                                "label": 1,
                                "sample_index_start": 1,
                                "sample_index_end": 1,
                                "action": "relabel",
                                "new_label": 2,
                            },
                            {
                                "label": 5,
                                "sample_index_start": 5,
                                "sample_index_end": 5,
                                "action": "exclude",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            dataset = load_dataset(root)
            self.assertEqual(dataset.images.shape[0], 4)
            self.assertEqual(dataset.labels.tolist(), [2, 2, 3, 4])
            self.assertEqual(len(dataset.curation_files), 1)
            self.assertEqual(dataset_summary(dataset)["counts"]["batch_a"]["2"], 2)
            self.assertTrue((batch / "1" / "batch_a_1_000001.png").is_file())

    def test_batch_can_be_excluded_without_deleting_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for batch_name, enabled in (("active", True), ("too_dark", False)):
                batch = root / batch_name
                batch.mkdir()
                (batch / "batch.json").write_text(
                    json.dumps({"training_enabled": enabled}), encoding="utf-8"
                )
                for label in range(1, 6):
                    label_dir = batch / str(label)
                    label_dir.mkdir()
                    image = np.full((32, 32), label * 30, dtype=np.uint8)
                    cv2.imwrite(
                        str(label_dir / f"{batch_name}_{label}_00000{label}.png"),
                        image,
                    )

            dataset = load_dataset(root)
            summary = dataset_summary(dataset)
            self.assertEqual(dataset.images.shape[0], 5)
            self.assertEqual(set(dataset.batches.tolist()), {"active"})
            self.assertEqual(summary["excluded_batches"], ["too_dark"])
            self.assertTrue((root / "too_dark" / "1" / "too_dark_1_000001.png").is_file())

    def test_synthetic_patterns_train_and_export(self) -> None:
        rng = np.random.default_rng(7)
        images: list[np.ndarray] = []
        labels: list[int] = []
        for label in range(5):
            for _ in range(24):
                image = np.zeros((32, 32), dtype=np.uint8)
                x0 = 2 + label * 6
                image[5:27, x0 : x0 + 4] = 1
                image ^= (rng.random((32, 32)) < 0.005).astype(np.uint8)
                images.append(image)
                labels.append(label)
        inputs = prepare_inputs(np.asarray(images))
        expected = np.asarray(labels)
        model = train_model(
            inputs,
            expected,
            hidden1=32,
            hidden2=16,
            epochs=20,
            batch_size=64,
            learning_rate=0.01,
            seed=11,
            max_shift=1,
            noise_probability=0.002,
            verbose=False,
        )
        predicted, _ = predict_deployment(inputs, model.deployment_arrays())
        self.assertGreaterEqual(float(np.mean(predicted == expected)), 0.95)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            info = save_deployment_model(path, model)
            self.assertEqual(info["binary_weight_bits"], 33360)
            with np.load(path) as archive:
                self.assertEqual(archive["w1"].shape, (1024, 32))
                self.assertEqual(archive["w1_packed"].shape, (32, 128))
                self.assertTrue(np.all(np.isin(archive["w2"], (-1, 1))))


if __name__ == "__main__":
    unittest.main()
