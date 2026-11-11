"""Bit-oriented Python reference for the deployed RGB-to-BNN input path.

This module mirrors the active RTL arithmetic and causal mask geometry.  It is
separate from ``preprocess.py`` because that older module decodes an already
rendered S5 HDMI debug frame; it does not reproduce RGB-to-S5 processing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tools.bnn.preprocess import largest_component, normalize_bbox_roi


REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 720
DS_SHIFT = 2
BLOCK_SIZE = 1 << DS_SHIFT
ROI_X = 32
ROI_Y = 18
ROI_WIDTH = 256
ROI_HEIGHT = 144
SOURCE_ROI_X = ROI_X << DS_SHIFT
SOURCE_ROI_Y = ROI_Y << DS_SHIFT
SOURCE_ROI_WIDTH = ROI_WIDTH << DS_SHIFT
SOURCE_ROI_HEIGHT = ROI_HEIGHT << DS_SHIFT
BLOCK_SKIN_THRESHOLD = 8
CB_MIN = 103
CB_MAX = 144
CR_MIN = 130
CR_MAX = 172
MAJORITY_THRESHOLD = 5
MIN_COMPONENT_AREA = 80
ROI_MARGIN = 0.10


@dataclass(frozen=True)
class FpgaPreprocessResult:
    ycbcr: np.ndarray
    skin_mask: np.ndarray
    block_mask: np.ndarray
    majority_mask: np.ndarray
    bbox: tuple[int, int, int, int] | None
    component_area: int
    normalized_mask: np.ndarray | None


def rgb_to_ycbcr_fixed(rgb: np.ndarray) -> np.ndarray:
    """Match gesture_rgb_to_ycbcr.v for an RGB uint8 image."""

    source = np.asarray(rgb)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("expected an RGB image with three channels")
    values = source[:, :, :3].astype(np.int32)
    red = values[:, :, 0]
    green = values[:, :, 1]
    blue = values[:, :, 2]
    y = (77 * red + 150 * green + 29 * blue + 128) >> 8
    cb = ((-43 * red - 85 * green + 128 * blue) >> 8) + 128
    cr = ((128 * red - 107 * green - 21 * blue) >> 8) + 128
    return np.stack(
        [np.clip(y, 0, 255), np.clip(cb, 0, 255), np.clip(cr, 0, 255)],
        axis=2,
    ).astype(np.uint8)


def cbcr_skin_mask(ycbcr: np.ndarray) -> np.ndarray:
    """Apply the inclusive frozen deployment thresholds."""

    source = np.asarray(ycbcr)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("expected a YCbCr image with three channels")
    cb = source[:, :, 1]
    cr = source[:, :, 2]
    return (
        (cb >= CB_MIN)
        & (cb <= CB_MAX)
        & (cr >= CR_MIN)
        & (cr <= CR_MAX)
    ).astype(np.uint8)


def block_vote_4x4(skin_mask: np.ndarray) -> np.ndarray:
    """Crop the deployed ROI and vote each 4x4 block with count >= 8."""

    source = (np.asarray(skin_mask) != 0).astype(np.uint8)
    if source.shape != (REFERENCE_HEIGHT, REFERENCE_WIDTH):
        raise ValueError(
            f"skin mask must be {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}"
        )
    crop = source[
        SOURCE_ROI_Y : SOURCE_ROI_Y + SOURCE_ROI_HEIGHT,
        SOURCE_ROI_X : SOURCE_ROI_X + SOURCE_ROI_WIDTH,
    ]
    counts = crop.reshape(
        ROI_HEIGHT, BLOCK_SIZE, ROI_WIDTH, BLOCK_SIZE
    ).sum(axis=(1, 3))
    return (counts >= BLOCK_SKIN_THRESHOLD).astype(np.uint8)


def majority_3x3_causal(block_mask: np.ndarray) -> np.ndarray:
    """Match the RTL causal 3x3 majority output at its emitted coordinates.

    At output (row, col), RTL uses raw rows row-2..row and columns col-2..col.
    Therefore the top two rows and left two columns are zero and the logical
    window centre is delayed by one sample in both axes.
    """

    source = (np.asarray(block_mask) != 0).astype(np.uint8)
    if source.shape != (ROI_HEIGHT, ROI_WIDTH):
        raise ValueError(f"block mask must have shape {(ROI_HEIGHT, ROI_WIDTH)}")
    integral = cv2.integral(source, sdepth=cv2.CV_32S)
    sums = (
        integral[3:, 3:]
        - integral[:-3, 3:]
        - integral[3:, :-3]
        + integral[:-3, :-3]
    )
    result = np.zeros_like(source)
    result[2:, 2:] = sums >= MAJORITY_THRESHOLD
    return result


def fit_rgb_to_reference(rgb: np.ndarray, *, mode: str = "letterbox") -> np.ndarray:
    """Place a public RGB image into the 1280x720 camera reference canvas.

    This operation is explicitly outside the FPGA: it defines how a public
    still image is presented to the otherwise bit-equivalent FPGA front end.
    Letterboxing is the default because it preserves hand geometry.
    """

    source = np.asarray(rgb)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("expected an RGB image with three channels")
    source = source[:, :, :3].astype(np.uint8, copy=False)
    if source.shape[:2] == (REFERENCE_HEIGHT, REFERENCE_WIDTH):
        return source.copy()
    if mode == "stretch":
        return cv2.resize(
            source,
            (REFERENCE_WIDTH, REFERENCE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
    if mode != "letterbox":
        raise ValueError("mode must be 'letterbox' or 'stretch'")
    scale = min(
        REFERENCE_WIDTH / source.shape[1],
        REFERENCE_HEIGHT / source.shape[0],
    )
    width = max(1, int(round(source.shape[1] * scale)))
    height = max(1, int(round(source.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(source, (width, height), interpolation=interpolation)
    canvas = np.zeros((REFERENCE_HEIGHT, REFERENCE_WIDTH, 3), dtype=np.uint8)
    x0 = (REFERENCE_WIDTH - width) // 2
    y0 = (REFERENCE_HEIGHT - height) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = resized
    return canvas


def preprocess_reference_rgb(rgb: np.ndarray) -> FpgaPreprocessResult:
    """Run the deployed chain from a 1280x720 RGB frame to a 32x32 mask."""

    source = np.asarray(rgb)
    if source.shape[:2] != (REFERENCE_HEIGHT, REFERENCE_WIDTH):
        raise ValueError(
            f"RGB frame must be {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}; "
            "call fit_rgb_to_reference explicitly for public still images"
        )
    ycbcr = rgb_to_ycbcr_fixed(source)
    skin = cbcr_skin_mask(ycbcr)
    blocks = block_vote_4x4(skin)
    majority = majority_3x3_causal(blocks)
    component = largest_component(majority, min_area=MIN_COMPONENT_AREA)
    if component is None:
        return FpgaPreprocessResult(
            ycbcr=ycbcr,
            skin_mask=skin,
            block_mask=blocks,
            majority_mask=majority,
            bbox=None,
            component_area=0,
            normalized_mask=None,
        )
    bbox, area = component
    normalized = normalize_bbox_roi(
        majority, bbox, output_size=32, margin=ROI_MARGIN
    )
    return FpgaPreprocessResult(
        ycbcr=ycbcr,
        skin_mask=skin,
        block_mask=blocks,
        majority_mask=majority,
        bbox=bbox,
        component_area=area,
        normalized_mask=normalized,
    )

