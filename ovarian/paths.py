"""Locations of data, weights and results, overridable with environment variables.

    OVARIAN_DATA     the Zenodo release, unpacked (layout in data/README.md)
    OVARIAN_WEIGHTS  the released model weights
    OVARIAN_RESULTS  where scripts write their outputs
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("OVARIAN_DATA", REPO / "data"))
WEIGHTS = Path(os.environ.get("OVARIAN_WEIGHTS", REPO / "weights"))
RESULTS = Path(os.environ.get("OVARIAN_RESULTS", REPO / "results"))

IMAGES = DATA / "images"
YOLO_LABELS = DATA / "yolo_labels"
CLINICAL = DATA / "clinical"
FEATURES = DATA / "features"
SPLITS = DATA / "splits"
SAM_MASKS = DATA / "sam_masks"
PREDICTIONS = DATA / "predictions"

SAM_CHECKPOINT = Path(os.environ.get("SAM_CHECKPOINT", WEIGHTS / "sam_vit_h_4b8939.pth"))


def results_dir(*parts: str) -> Path:
    path = RESULTS.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
