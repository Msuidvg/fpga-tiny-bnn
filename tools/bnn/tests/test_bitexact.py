from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.bnn.bitexact import (
    activation_thresholds,
    infer_bitexact,
    infer_signed_integer_reference,
    load_bitexact_model,
)
from tools.bnn.export_bnn_fpga import export_model


class BitExactTests(unittest.TestCase):
    @staticmethod
    def _write_model(path: Path) -> np.ndarray:
        rng = np.random.default_rng(73)
        w1 = rng.choice((-1, 1), size=(16, 8)).astype(np.int8)
        w2 = rng.choice((-1, 1), size=(8, 8)).astype(np.int8)
        w3 = rng.choice((-1, 1), size=(8, 3)).astype(np.int8)
        b1 = rng.integers(-2, 3, size=8, dtype=np.int32)
        b2 = rng.integers(-2, 3, size=8, dtype=np.int32)
        b3 = rng.integers(-2, 3, size=3, dtype=np.int32)
        packed = {
            f"{name}_packed": np.packbits(
                (value.T > 0).astype(np.uint8), axis=1, bitorder="little"
            )
            for name, value in (("w1", w1), ("w2", w2), ("w3", w3))
        }
        np.savez_compressed(
            path,
            schema_version=np.asarray(1, dtype=np.int32),
            class_labels=np.asarray((1, 2, 3), dtype=np.int8),
            input_shape=np.asarray((4, 4), dtype=np.int32),
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            w3=w3,
            b3=b3,
            **packed,
        )
        return rng.integers(0, 2, size=(31, 4, 4), dtype=np.uint8)

    def test_xnor_popcount_matches_signed_integer_at_every_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            inputs = self._write_model(path)
            model = load_bitexact_model(path)
            bitexact = infer_bitexact(inputs, model)
            reference = infer_signed_integer_reference(inputs, model)
            np.testing.assert_array_equal(bitexact.h1_bits, reference.h1_bits)
            np.testing.assert_array_equal(bitexact.h2_bits, reference.h2_bits)
            np.testing.assert_array_equal(bitexact.scores, reference.scores)
            np.testing.assert_array_equal(
                bitexact.predicted_labels, reference.predicted_labels
            )

    def test_threshold_rounding_and_fpga_export_round_trip(self) -> None:
        np.testing.assert_array_equal(
            activation_thresholds(8, np.asarray((0, 1, -1), dtype=np.int32)),
            np.asarray((4, 4, 5), dtype=np.int32),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "model.npz"
            self._write_model(model_path)
            manifest = export_model(model_path, root / "fpga")
            self.assertTrue(manifest["round_trip_verified"])
            self.assertEqual(manifest["architecture"], [16, 8, 8, 3])
            self.assertEqual(
                len((root / "fpga" / "bnn_w1.mem").read_text().splitlines()),
                8,
            )


if __name__ == "__main__":
    unittest.main()
