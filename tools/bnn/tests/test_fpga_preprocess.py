"""Unit checks for the RGB-to-S5 FPGA reference path."""

from __future__ import annotations

import unittest

import numpy as np

from tools.bnn.fpga_preprocess import (
    BLOCK_SKIN_THRESHOLD,
    REFERENCE_HEIGHT,
    REFERENCE_WIDTH,
    block_vote_4x4,
    cbcr_skin_mask,
    majority_3x3_causal,
    rgb_to_ycbcr_fixed,
)


class FpgaPreprocessTests(unittest.TestCase):
    def test_rgb_to_ycbcr_known_pixels(self) -> None:
        pixels = np.asarray(
            [[[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255]]],
            dtype=np.uint8,
        )
        actual = rgb_to_ycbcr_fixed(pixels)
        expected = np.asarray(
            [[[0, 128, 128], [255, 128, 128], [77, 85, 255], [29, 255, 107]]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_cbcr_limits_are_inclusive(self) -> None:
        values = np.asarray(
            [[[0, 103, 130], [0, 144, 172], [0, 102, 130], [0, 144, 173]]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(
            cbcr_skin_mask(values), np.asarray([[1, 1, 0, 0]], dtype=np.uint8)
        )

    def test_four_by_four_vote_accepts_exact_tie(self) -> None:
        skin = np.zeros((REFERENCE_HEIGHT, REFERENCE_WIDTH), dtype=np.uint8)
        skin[72:74, 128:132] = 1
        self.assertEqual(int(skin[72:76, 128:132].sum()), BLOCK_SKIN_THRESHOLD)
        voted = block_vote_4x4(skin)
        self.assertEqual(int(voted[0, 0]), 1)

    def test_causal_majority_has_two_pixel_zero_border(self) -> None:
        raw = np.zeros((144, 256), dtype=np.uint8)
        raw[0:3, 0:3] = np.asarray(
            [[1, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.uint8
        )
        majority = majority_3x3_causal(raw)
        self.assertEqual(int(majority[2, 2]), 1)
        self.assertEqual(int(majority[:2].sum()), 0)
        self.assertEqual(int(majority[:, :2].sum()), 0)


if __name__ == "__main__":
    unittest.main()
