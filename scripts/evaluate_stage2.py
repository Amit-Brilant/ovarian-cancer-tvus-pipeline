"""Evaluate Stage II detection weights on one split: mAP@0.5 and mAP@0.5:0.95.

The default image size is the one the released detector was trained at. Stage I and Stage III
instead run the detector at 640 when they crop (ovarian.stage3.DETECTOR_IMGSZ).

Writes RESULTS/stage2/eval_<split>.json. Run scripts/prepare_yolo_dataset.py first.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from ovarian import paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=paths.WEIGHTS / "stage2_yolov8m.pt")
    parser.add_argument("--data", type=Path, default=paths.RESULTS / "yolo_dataset" / "data.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found; run scripts/prepare_yolo_dataset.py first")
    out = paths.results_dir("stage2")
    metrics = YOLO(str(args.weights)).val(
        data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch, device=args.device,
        plots=False, project=str(out), name=f"val_{args.split}", exist_ok=True,
    )
    result = {
        "weights": args.weights.name,
        "split": args.split,
        "imgsz": args.imgsz,
        "n_images": sum(1 for _ in (args.data.parent / "images" / args.split).iterdir()),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
    }
    (out / f"eval_{args.split}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
