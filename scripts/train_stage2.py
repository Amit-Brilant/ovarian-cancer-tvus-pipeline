"""Train the Stage II lesion detector (YOLOv8m) with the settings of the released weights.

Image size 1280, 10 epochs, batch size 8, with the optimiser and learning rate chosen by Ultralytics
(optimizer='auto') and its built-in augmentation. These are the settings recorded in the released checkpoint, and evaluating that checkpoint at 1280 reproduces the mAP reported in the paper.
The two classes are the benign and malignant labels of the annotation; the rest of the pipeline uses only
the predicted box, never its class. Run scripts/prepare_yolo_dataset.py first.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from ovarian import paths, seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=paths.RESULTS / "yolo_dataset" / "data.yaml")
    parser.add_argument("--model", default="yolov8m.pt",
                        help="initial weights or a model yaml (yolov8m.yaml trains from scratch, without download)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="for example 0, cpu or mps (default: Ultralytics choice)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    parser.add_argument("--name", default="train")
    parser.add_argument("--no-plots", action="store_true", help="skip training plots")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found; run scripts/prepare_yolo_dataset.py first")
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="auto",
        seed=seeds.resolve(args.seed),
        deterministic=True,
        device=args.device,
        workers=args.workers,
        project=str(paths.results_dir("stage2")),
        name=args.name,
        exist_ok=True,
        plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
