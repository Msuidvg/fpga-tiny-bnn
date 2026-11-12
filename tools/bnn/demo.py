"""Classify a binary 256x144 mask with the frozen Tiny-BNN V1 model."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .bitexact import BitExactModel, infer_bitexact, load_bitexact_model
from .preprocess import largest_component, normalize_bbox_roi


ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "testvectors" / "pipeline"


def read_vector(index: int) -> np.ndarray:
    words = (VECTORS / "raw_masks.mem").read_text(encoding="ascii").split()
    if not 0 <= index < len(words):
        raise ValueError(f"vector index must be in 0..{len(words) - 1}")
    packed = int(words[index], 16).to_bytes(144 * 256 // 8, "little")
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little").reshape(144, 256)


def read_mask(path: Path, shape: tuple[int, int] = (144, 256)) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    mask = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape != shape or mask.dtype != np.uint8:
        raise ValueError(f"input must be an 8-bit grayscale {shape[1]}x{shape[0]} binary mask")
    if not (np.isin(mask, (0, 1)).all() or np.isin(mask, (0, 255)).all()):
        raise ValueError("input mask pixels must be 0/1 or 0/255")
    return (mask != 0).astype(np.uint8)


def classify(mask: np.ndarray, model: BitExactModel) -> tuple[dict, np.ndarray | None]:
    component = largest_component(mask)
    if component is None:
        return {"status": "no_component", "bbox": None, "class_id": None, "scores": None}, None
    bbox, area = component
    roi = normalize_bbox_roi(mask, bbox)
    result = infer_bitexact(roi[np.newaxis], model)
    scores = result.scores[0].tolist()
    high, second = sorted(scores, reverse=True)[:2]
    return {
        "status": "classified",
        "bbox": bbox,
        "component_area": area,
        "class_id": int(result.predicted_labels[0]),
        "score_classes": model.class_labels.tolist(),
        "scores": scores,
        "score_margin": min(high - second, 255),
    }, roi


def write_preview(path: Path, mask: np.ndarray, result: dict, roi: np.ndarray | None) -> None:
    image = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    if result["bbox"] is not None:
        x, y, width, height = result["bbox"]
        cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), (216, 216, 46), 1)
    left = cv2.resize(image, (512, 288), interpolation=cv2.INTER_NEAREST)
    right = np.full((288, 160, 3), 28, dtype=np.uint8)
    if roi is not None:
        preview = cv2.resize(roi * 255, (144, 144), interpolation=cv2.INTER_NEAREST)
        right[42:186, 8:152] = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    caption = "No component" if roi is None else f"Class {result['class_id']}"
    cv2.putText(right, caption, (8, 223), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, payload = cv2.imencode(".png", np.hstack((left, right)))
    if not encoded:
        raise ValueError("could not encode PNG preview")
    path.write_bytes(payload.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--vector", type=int, help="index into the bundled parity set")
    inputs.add_argument("--mask", type=Path, help="grayscale binary mask, 256x144")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "v1" / "model.npz")
    parser.add_argument("--preview", type=Path, help="write a PNG with the box and normalized ROI")
    args = parser.parse_args()
    try:
        mask = read_vector(args.vector) if args.vector is not None else read_mask(args.mask)
        result, roi = classify(mask, load_bitexact_model(args.model))
        if args.preview:
            if args.preview.suffix.lower() != ".png":
                raise ValueError("preview path must end in .png")
            write_preview(args.preview, mask, result, roi)
    except (ValueError, OSError, cv2.error) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
