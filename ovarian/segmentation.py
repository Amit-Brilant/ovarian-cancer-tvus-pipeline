"""Lesion masks from SAM ViT-H prompted with the annotation boxes, followed by morphological refinement.

For each box SAM returns three candidate masks and the one with the highest predicted score is kept. The
refinement keeps the largest 8-connected component and applies an elliptical closing (9 px), preceded by
the hole-filling step of the original procedure (see `refine_mask`).
"""
from pathlib import Path

import cv2
import numpy as np

from ovarian import paths

CLOSE_KSIZE = 9


def label_path(stem: str, cls: str) -> Path:
    """YOLO annotation file of one pathological image."""
    return paths.YOLO_LABELS / f"{cls}_labels" / f"{stem}.txt"


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as an RGB uint8 array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_yolo_boxes(path: str | Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Pixel boxes (x1, y1, x2, y2) from a YOLO label file, truncated to integers and clipped to the image.

    Lines without exactly five fields and boxes that collapse after clipping are skipped.
    """
    boxes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        x1 = min(max(int((cx - bw / 2) * width), 0), width - 1)
        y1 = min(max(int((cy - bh / 2) * height), 0), height - 1)
        x2 = min(max(int((cx + bw / 2) * width), 0), width - 1)
        y2 = min(max(int((cy + bh / 2) * height), 0), height - 1)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def refine_mask(mask: np.ndarray, close_ksize: int = CLOSE_KSIZE) -> np.ndarray:
    """Hole-filling step, largest 8-connected component, elliptical closing; returns a 0/1 uint8 mask.

    The hole-filling step is kept exactly as in the procedure that produced the released masks: it floods
    the inverted mask from the top-left corner, so when that corner is background it leaves the mask
    unchanged and only holes narrower than the closing kernel are filled (by the closing).
    """
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    h, w = m.shape
    flood = cv2.bitwise_not(m)
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    filled = cv2.bitwise_or(m, cv2.bitwise_not(flood))

    n, labels, stats, _ = cv2.connectedComponentsWithStats((filled > 0).astype(np.uint8), connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        filled = np.where(labels == largest, 255, 0).astype(np.uint8)

    if close_ksize > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
    return (filled > 0).astype(np.uint8)


class BoxSegmenter:
    """Box-prompted SAM; the model is loaded on first use."""

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None, model_type: str = "vit_h"):
        self.checkpoint = Path(checkpoint) if checkpoint is not None else paths.SAM_CHECKPOINT
        self.device = device
        self.model_type = model_type
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            if not self.checkpoint.exists():
                raise FileNotFoundError(f"SAM checkpoint not found: {self.checkpoint}")
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint)).to(device)
            self._predictor = SamPredictor(sam)
        return self._predictor

    def raw_masks(self, image_rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[tuple[np.ndarray, float]]:
        """For each box, the highest-scoring of SAM's multimask outputs (0/1 uint8) and its predicted score."""
        import torch

        predictor = self.predictor
        out = []
        with torch.inference_mode():
            predictor.set_image(image_rgb)
            for box in boxes:
                box_t = torch.tensor([box], dtype=torch.float32, device=predictor.model.device)
                box_t = predictor.transform.apply_boxes_torch(box_t, image_rgb.shape[:2])
                masks, scores, _ = predictor.predict_torch(
                    point_coords=None, point_labels=None, boxes=box_t, multimask_output=True
                )
                scores = scores[0]
                best = 0 if scores.numel() == 1 else int(torch.argmax(scores).item())
                out.append((masks[0, best].cpu().numpy().astype(np.uint8), float(scores[best].item())))
        return out

    def segment(self, image_rgb: np.ndarray, boxes: list[tuple[int, int, int, int]],
                close_ksize: int = CLOSE_KSIZE) -> np.ndarray:
        """Union of the refined masks of all boxes, as a 0/1 uint8 array."""
        union = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        for mask, _ in self.raw_masks(image_rgb, boxes):
            union[refine_mask(mask, close_ksize) == 1] = 1
        return union

    def segment_file(self, image_path: str | Path, labels: str | Path, close_ksize: int = CLOSE_KSIZE) -> np.ndarray:
        """Mask of one image file prompted with the boxes of its YOLO label file."""
        image = read_rgb(image_path)
        h, w = image.shape[:2]
        return self.segment(image, read_yolo_boxes(labels, w, h), close_ksize)
