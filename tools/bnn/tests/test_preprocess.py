from __future__ import annotations

import unittest

import numpy as np

from tools.bnn.preprocess import (
    REFERENCE_HEIGHT,
    REFERENCE_WIDTH,
    SAMPLE_SCALE,
    SAMPLED_ROI_X,
    SAMPLED_ROI_Y,
    decode_s5_mask,
    extract_hand_sample,
    normalize_bbox_roi,
)


def make_s5_frame(mask: np.ndarray) -> np.ndarray:
    frame = np.zeros((REFERENCE_HEIGHT, REFERENCE_WIDTH, 3), dtype=np.uint8)
    expanded = np.repeat(np.repeat(mask * 255, SAMPLE_SCALE, axis=0), SAMPLE_SCALE, axis=1)
    x0 = SAMPLED_ROI_X * SAMPLE_SCALE
    y0 = SAMPLED_ROI_Y * SAMPLE_SCALE
    frame[y0 : y0 + expanded.shape[0], x0 : x0 + expanded.shape[1]] = expanded[:, :, None]
    return frame


class PreprocessTests(unittest.TestCase):
    def test_decode_and_largest_component(self) -> None:
        mask = np.zeros((144, 256), dtype=np.uint8)
        mask[20:25, 10:15] = 1
        mask[35:95, 80:120] = 1
        sample = extract_hand_sample(make_s5_frame(mask), min_area=50, margin=0.0)
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.bbox, (80, 35, 40, 60))
        self.assertEqual(sample.component_area, 2400)
        self.assertEqual(sample.image.shape, (32, 32))
        self.assertEqual(set(np.unique(sample.image)), {0, 1})

    def test_block_center_sampling_ignores_edges(self) -> None:
        mask = np.zeros((144, 256), dtype=np.uint8)
        mask[30:80, 90:140] = 1
        frame = make_s5_frame(mask)
        x0 = SAMPLED_ROI_X * SAMPLE_SCALE
        y0 = SAMPLED_ROI_Y * SAMPLE_SCALE
        roi = frame[y0 : y0 + 576, x0 : x0 + 1024]
        roi[::4, :, :] = 127
        roi[:, ::4, :] = 127
        decoded, _ = decode_s5_mask(frame, validate=False)
        np.testing.assert_array_equal(decoded, mask)

    def test_normalization_preserves_tall_shape(self) -> None:
        mask = np.zeros((144, 256), dtype=np.uint8)
        mask[10:110, 50:70] = 1
        image = normalize_bbox_roi(mask, (50, 10, 20, 100), margin=0.0)
        ys, xs = np.nonzero(image)
        self.assertGreater(int(ys.max() - ys.min()), int(xs.max() - xs.min()))


if __name__ == "__main__":
    unittest.main()
