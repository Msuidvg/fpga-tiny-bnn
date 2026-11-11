from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.bnn.export_bnn_eram32 import convert, load_architecture


class ExportBnnEram32Tests(unittest.TestCase):
    def test_architecture_drives_all_layer_widths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bnn_export_manifest.json").write_text(
                json.dumps({"architecture": [64, 40, 24, 5]}),
                encoding="utf-8",
            )
            self.assertEqual(load_architecture(root), (64, 40, 24, 5))

            source = root / "bnn_w2.mem"
            source.write_text("0000000001\n0000000100\n", encoding="ascii")
            result = convert(source, root / "bnn_w2_32.mem", 40, 2)

            self.assertEqual(result["feature_bits"], 40)
            self.assertEqual(result["neuron_rows"], 2)
            self.assertEqual(result["chunks_per_neuron"], 2)
            self.assertEqual(
                (root / "bnn_w2_32.mem").read_text(encoding="ascii").splitlines(),
                ["00000001", "00000000", "00000100", "00000000"],
            )

    def test_rejects_row_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "weights.mem"
            source.write_text("00\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "expected 2"):
                convert(source, root / "weights_32.mem", 8, 2)


if __name__ == "__main__":
    unittest.main()
