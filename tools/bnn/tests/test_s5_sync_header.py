"""Checks for the machine-readable V3 synchronized-S5 frame header."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


BNN_DIR = Path(__file__).resolve().parents[1]
if str(BNN_DIR) not in sys.path:
    sys.path.insert(0, str(BNN_DIR))

from preprocess import (  # noqa: E402
    S5SyncHeaderError,
    decode_s5_bnn_status,
    decode_s5_sync_frame_id,
)


class S5SyncHeaderTests(unittest.TestCase):
    @staticmethod
    def make_frame(
        frame_id: int,
        sync_word: int = 0xA55A,
        gesture: int = 0,
        confidence: int = 0,
    ) -> np.ndarray:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        word = ((frame_id & 0xFFFF) << 16) | (sync_word & 0xFFFF)
        for index in range(32):
            if (word >> index) & 1:
                frame[0:4, index * 4 : index * 4 + 4] = 255
        status = ((confidence & 0xFF) << 3) | (gesture & 0x7)
        for index in range(11):
            if (status >> index) & 1:
                frame[12:16, index * 4 : index * 4 + 4] = 255
        return frame

    def test_decodes_frame_id(self) -> None:
        self.assertEqual(
            decode_s5_sync_frame_id(self.make_frame(0x1234)), 0x1234
        )

    def test_rejects_warmup_or_wrong_bitstream(self) -> None:
        with self.assertRaises(S5SyncHeaderError):
            decode_s5_sync_frame_id(self.make_frame(7, sync_word=0))

    def test_decodes_bnn_status(self) -> None:
        frame = self.make_frame(0x1234, gesture=5, confidence=0xA6)
        self.assertEqual(decode_s5_bnn_status(frame), (5, 0xA6))


if __name__ == "__main__":
    unittest.main()
