"""Load the portable curated binary-mask archive, without pickle."""
from pathlib import Path
import numpy as np

def load_archive(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        images = archive["images"]
        labels = archive["labels"]
        batches = archive["batches"]
        paths = archive["paths"]
    count = len(images)
    if count == 0 or images.shape != (count, 32, 32):
        raise ValueError("archive images must have nonempty shape (N,32,32)")
    if not np.all(np.isin(images, (0, 1))):
        raise ValueError("archive images must be binary 0/1")
    if labels.shape != (count,) or not np.all(np.isin(labels, (1,2,3,4,5))):
        raise ValueError("archive labels must be N values in 1..5")
    if batches.shape != (count,) or paths.shape != (count,):
        raise ValueError("archive batch/path counts must match images")
    if batches.dtype.kind not in "US" or paths.dtype.kind not in "US":
        raise ValueError("archive batch/path values must be strings")
    return dict(images=images.astype(np.uint8), labels=labels.astype(np.int64),
                batches=batches, paths=tuple(str(p) for p in paths))
