"""Train the Stage I classifier (EfficientNet-B7 with the clinical CAM-loss) on the fixed patient-level split.

Detector boxes for the pathological images are computed once with the Stage II weights and cached in the output
folder. Defaults reproduce the released configuration; --limit and --epochs allow a quick smoke run.
"""
import argparse
import json
import time
from pathlib import Path

from ovarian import paths, seeds
from ovarian.stage1 import (
    Stage1Dataset,
    TrainConfig,
    default_device,
    detector_boxes,
    load_split,
    save_mask_previews,
    seed_everything,
    train,
)


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--limit", type=int, default=None, help="keep only the first N images per split and class")
    parser.add_argument("--detector", type=Path, default=paths.WEIGHTS / "stage2_yolov8m.pt")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--no-pretrained", action="store_true", help="start from random instead of ImageNet weights")
    parser.add_argument("--mask-previews", type=int, default=8, help="focus-mask overlays to save from the train split")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: results/stage1_train/run_<date>_<time>)")
    args = parser.parse_args()

    args.seed = seeds.resolve(args.seed)
    seed_everything(args.seed)
    cfg = TrainConfig(epochs=args.epochs, seed=args.seed, batch_size=args.batch_size, patience=args.patience,
                      num_workers=args.num_workers, pretrained=not args.no_pretrained)
    out_dir = args.out or paths.results_dir("stage1_train", time.strftime("run_%Y%m%d_%H%M%S"))
    frame = load_split(args.limit)

    start = time.time()
    boxes = detector_boxes(frame, args.detector, imgsz=cfg.detector_imgsz, conf=cfg.detector_conf,
                           iou=cfg.detector_iou, device=args.device, cache=out_dir / "detector_boxes.json")
    if args.mask_previews:
        train_frame = frame[frame.split == "train"].reset_index(drop=True)
        dataset = Stage1Dataset(train_frame, boxes, with_mask=True)
        count = min(args.mask_previews, len(dataset))
        save_mask_previews(dataset, out_dir / "mask_previews",
                           sorted({round(i * (len(dataset) - 1) / max(count - 1, 1)) for i in range(count)}))

    metrics = train(cfg, frame, boxes, out_dir, device=args.device)
    metrics["seconds"] = round(time.time() - start, 1)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
