"""Convert the FPGA S5 majority-mask view into a normalized 32x32 hand ROI."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 720
SAMPLED_ROI_X = 32
SAMPLED_ROI_Y = 18
SAMPLED_ROI_WIDTH = 256
SAMPLED_ROI_HEIGHT = 144
SAMPLE_SCALE = 4


class S5FrameError(ValueError):
    """The input does not look like the FPGA S5 black/white debug view."""


class S5SyncHeaderError(S5FrameError):
    """The frame does not contain a valid V3 synchronized-S5 header."""


@dataclass(frozen=True)
class S5Diagnostics:
    grayscale_fraction: float
    black_white_fraction: float
    contrast: float


@dataclass(frozen=True)
class HandSample:
    image: np.ndarray
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    component_area: int
    fill_ratio: float
    diagnostics: S5Diagnostics


S5_SYNC_WORD = 0xA55A
S5_SYNC_HEADER_BITS = 32
S5_SYNC_CELL_SIZE = 4
S5_STATUS_BITS = 11
S5_STATUS_Y = 12


def decode_s5_sync_frame_id(
    frame: np.ndarray, *, threshold: float = 127.0
) -> int:
    """Decode the V3 capture header (0xA55A + 16-bit frame id, LSB first).

    The header occupies x=0..127 and y=0..3. Each bit is a 4x4 black/white
    cell, which is robust against normal HDMI capture-card compression. A
    missing sync word marks the warm-up frame or a non-V3 bitstream.
    """

    bgr = _reference_frame(frame)
    header = cv2.cvtColor(
        bgr[:S5_SYNC_CELL_SIZE, : S5_SYNC_HEADER_BITS * S5_SYNC_CELL_SIZE],
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)
    cells = header.reshape(
        S5_SYNC_CELL_SIZE,
        S5_SYNC_HEADER_BITS,
        S5_SYNC_CELL_SIZE,
    ).mean(axis=(0, 2))
    bits = cells >= threshold
    word = sum(int(bit) << index for index, bit in enumerate(bits))
    if (word & 0xFFFF) != S5_SYNC_WORD:
        raise S5SyncHeaderError(
            f"invalid synchronized-S5 header: 0x{word & 0xFFFF:04X}"
        )
    return (word >> 16) & 0xFFFF


def decode_s5_bnn_status(
    frame: np.ndarray, *, threshold: float = 127.0
) -> tuple[int, int]:
    """Decode the synchronized frame's FPGA gesture and confidence fields.

    The status occupies x=0..43 and y=12..15 as eleven 4x4 cells. Bits 0..2
    are gesture_id and bits 3..10 are confidence, all LSB first.
    """

    bgr = _reference_frame(frame)
    y0 = S5_STATUS_Y
    status = cv2.cvtColor(
        bgr[
            y0 : y0 + S5_SYNC_CELL_SIZE,
            : S5_STATUS_BITS * S5_SYNC_CELL_SIZE,
        ],
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)
    cells = status.reshape(
        S5_SYNC_CELL_SIZE,
        S5_STATUS_BITS,
        S5_SYNC_CELL_SIZE,
    ).mean(axis=(0, 2))
    bits = cells >= threshold
    word = sum(int(bit) << index for index, bit in enumerate(bits))
    return word & 0x7, (word >> 3) & 0xFF


def _reference_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        raise S5FrameError("expected a BGR video frame")
    bgr = frame[:, :, :3]
    if bgr.shape[:2] == (REFERENCE_HEIGHT, REFERENCE_WIDTH):
        return bgr
    return cv2.resize(
        bgr,
        (REFERENCE_WIDTH, REFERENCE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def decode_s5_mask(
    frame: np.ndarray,
    *,
    threshold: float = 127.0,
    validate: bool = True,
    min_grayscale_fraction: float = 0.94,
    min_black_white_fraction: float = 0.88,
    min_contrast: float = 70.0,
) -> tuple[np.ndarray, S5Diagnostics]:
    """Collapse the expanded 1280x720 S5 view back to its 256x144 mask.

    The FPGA expands each sampled mask bit to a 4x4 HDMI block.  Averaging the
    center of each block suppresses MJPG ringing while retaining the exact
    sampled geometry used by the hardware pipeline.
    """

    bgr = _reference_frame(frame)
    x0 = SAMPLED_ROI_X * SAMPLE_SCALE
    y0 = SAMPLED_ROI_Y * SAMPLE_SCALE
    x1 = x0 + SAMPLED_ROI_WIDTH * SAMPLE_SCALE
    y1 = y0 + SAMPLED_ROI_HEIGHT * SAMPLE_SCALE
    roi = bgr[y0:y1, x0:x1].astype(np.float32)

    channel_range = roi.max(axis=2) - roi.min(axis=2)
    grayscale_fraction = float(np.mean(channel_range <= 24.0))
    gray = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    black_white_fraction = float(np.mean((gray <= 64.0) | (gray >= 191.0)))
    contrast = float(np.percentile(gray, 99.5) - np.percentile(gray, 5.0))
    diagnostics = S5Diagnostics(
        grayscale_fraction=grayscale_fraction,
        black_white_fraction=black_white_fraction,
        contrast=contrast,
    )

    if validate:
        problems: list[str] = []
        if grayscale_fraction < min_grayscale_fraction:
            problems.append(f"grayscale={grayscale_fraction:.2f}")
        if black_white_fraction < min_black_white_fraction:
            problems.append(f"binary={black_white_fraction:.2f}")
        if contrast < min_contrast:
            problems.append(f"contrast={contrast:.0f}")
        if problems:
            raise S5FrameError(
                "frame is not a valid S5 mask (" + ", ".join(problems) + ")"
            )

    # Sample the center 2x2 pixels of every displayed 4x4 mask block.  The
    # outer pixels are more susceptible to capture-card scaling and ringing.
    blocks = gray.reshape(
        SAMPLED_ROI_HEIGHT,
        SAMPLE_SCALE,
        SAMPLED_ROI_WIDTH,
        SAMPLE_SCALE,
    )
    block_means = blocks[:, 1:3, :, 1:3].mean(axis=(1, 3))
    return (block_means >= threshold).astype(np.uint8), diagnostics


def largest_component(
    mask: np.ndarray, *, min_area: int = 80
) -> tuple[tuple[int, int, int, int], int] | None:
    """Return (bbox, area) for the largest 8-connected white component."""

    binary = (np.asarray(mask) != 0).astype(np.uint8)
    if binary.shape != (SAMPLED_ROI_HEIGHT, SAMPLED_ROI_WIDTH):
        raise ValueError(
            "mask must have shape "
            f"({SAMPLED_ROI_HEIGHT}, {SAMPLED_ROI_WIDTH})"
        )
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[index, cv2.CC_STAT_AREA])
    if area < min_area:
        return None
    bbox = tuple(int(value) for value in stats[index, :4])
    return bbox, area


def normalize_bbox_roi(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    output_size: int = 32,
    margin: float = 0.10,
) -> np.ndarray:
    """Crop a bbox, preserve aspect ratio, and center it on a square canvas."""

    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5)")
    binary = (np.asarray(mask) != 0).astype(np.uint8)
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive")

    pad_x = int(round(width * margin))
    pad_y = int(round(height * margin))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(binary.shape[1], x + width + pad_x)
    y1 = min(binary.shape[0], y + height + pad_y)
    crop = binary[y0:y1, x0:x1]

    side = max(crop.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    dst_y = (side - crop.shape[0]) // 2
    dst_x = (side - crop.shape[1]) // 2
    square[dst_y : dst_y + crop.shape[0], dst_x : dst_x + crop.shape[1]] = crop
    resized = cv2.resize(
        square,
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized != 0).astype(np.uint8)


def extract_hand_sample(
    frame: np.ndarray,
    *,
    threshold: float = 127.0,
    min_area: int = 80,
    margin: float = 0.10,
    validate_s5: bool = True,
) -> HandSample | None:
    mask, diagnostics = decode_s5_mask(
        frame, threshold=threshold, validate=validate_s5
    )
    component = largest_component(mask, min_area=min_area)
    if component is None:
        return None
    bbox, area = component
    image = normalize_bbox_roi(mask, bbox, output_size=32, margin=margin)
    _, _, width, height = bbox
    return HandSample(
        image=image,
        mask=mask,
        bbox=bbox,
        component_area=area,
        fill_ratio=float(area / (width * height)),
        diagnostics=diagnostics,
    )


def bbox_to_reference_pixels(
    bbox: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return (
        (SAMPLED_ROI_X + x) * SAMPLE_SCALE,
        (SAMPLED_ROI_Y + y) * SAMPLE_SCALE,
        width * SAMPLE_SCALE,
        height * SAMPLE_SCALE,
    )
