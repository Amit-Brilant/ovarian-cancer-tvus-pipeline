"""Generate the lesion masks of the pathological images with box-prompted SAM ViT-H.

Writes <out>/{benign,malignant}/<stem>.png as 0/255 PNGs, prompted with the YOLO annotation boxes.
"""
import argparse
from pathlib import Path

import cv2

from ovarian import paths
from ovarian.segmentation import CLOSE_KSIZE, BoxSegmenter, label_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="+", default=["benign", "malignant"], choices=["benign", "malignant"])
    parser.add_argument("--out", type=Path, default=None, help="output root (default: RESULTS/sam_masks)")
    parser.add_argument("--checkpoint", type=Path, default=paths.SAM_CHECKPOINT)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--close-ksize", type=int, default=CLOSE_KSIZE)
    parser.add_argument("--limit", type=int, default=None, help="first N images per class, in name order")
    parser.add_argument("--images", nargs="+", default=None, help="only these image names")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_root = args.out or paths.results_dir("sam_masks")
    segmenter = BoxSegmenter(args.checkpoint, device=args.device)
    for cls in args.classes:
        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted((paths.IMAGES / cls).glob("*.png"))
        if args.images:
            images = [p for p in images if p.name in set(args.images)]
        images = images[: args.limit]
        written = missing = 0
        for image_path in images:
            target = out_dir / image_path.name
            labels = label_path(image_path.stem, cls)
            if not labels.exists():
                missing += 1
                continue
            if target.exists() and not args.overwrite:
                continue
            mask = segmenter.segment_file(image_path, labels, args.close_ksize)
            cv2.imwrite(str(target), mask * 255)
            written += 1
        print(f"{cls}: {written} masks written, {missing} images without labels, to {out_dir}")


if __name__ == "__main__":
    main()
