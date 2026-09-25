"""Train a Stage III configuration with the protocol of the Methods.

Adam, plain cross-entropy, no augmentation and no early stopping; the checkpoint with the highest
image-level validation macro-F1 is kept. Configuration 2 uses a learning rate of 1e-4 with batch size 8,
Configuration 6 uses 1e-5 with batch size 4.

Configuration 6 needs the eleven descriptors of every training image. The data release carries the
descriptors of the automatic masks for validation and test; with --fallback-descriptors the training
images take the released annotation-box descriptors, which are correlated with the automatic ones but not
identical. Writes predictions and metrics for validation and test, as evaluate_stage3.py does.

The released checkpoints were trained with this protocol on the original cluster and reproduce the
reported numbers exactly; a rerun here reaches comparable but not bit-identical weights.
"""
import argparse
import json
from pathlib import Path

from ovarian import paths, seeds
from ovarian.stage1 import default_device
from ovarian.stage3 import (
    TrainConfig,
    detector_boxes,
    load_boxes,
    load_features,
    load_normalization,
    load_split,
    train,
)

SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=int, default=6, choices=[2, 6])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=None, help="default: the reported batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="default: the reported learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--features", type=Path, default=None,
                        help="folder with automatic_morphological_features_<split>.csv (default: DATA/features)")
    parser.add_argument("--fallback-descriptors", action="store_true",
                        help="use the released annotation-box descriptors where no automatic file covers a split")
    parser.add_argument("--normalization", type=Path, default=None,
                        help="default: WEIGHTS/stage3_config6_feature_normalization.json")
    parser.add_argument("--boxes", type=Path, default=None,
                        help="folder with detector_boxes_<split>.json (default: DATA/detector_boxes)")
    parser.add_argument("--detect", action="store_true", help="run the Stage II detector instead of the released boxes")
    parser.add_argument("--detector-weights", type=Path, default=paths.WEIGHTS / "stage2_yolov8m.pt")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: results/stage3_config<n>_train)")
    args = parser.parse_args()

    overrides = {"epochs": args.epochs, "weight_decay": args.weight_decay, "seed": seeds.resolve(args.seed),
                 "num_workers": args.num_workers}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    cfg = TrainConfig.for_configuration(args.configuration, **overrides)

    frame = load_split(list(SPLITS))
    features = mean = std = None
    if args.configuration != 2:
        names, mean, std = load_normalization(
            args.normalization or paths.WEIGHTS / "stage3_config6_feature_normalization.json")
        folder = args.features or paths.FEATURES
        features = load_features({s: folder / f"automatic_morphological_features_{s}.csv" for s in SPLITS},
                                 names, fallback=args.fallback_descriptors)

    boxes = {}
    for split in SPLITS:
        part = frame[frame.split == split].reset_index(drop=True)
        if args.detect:
            boxes |= detector_boxes(part, args.detector_weights, device=args.device)
        else:
            boxes |= load_boxes((args.boxes or paths.DATA / "detector_boxes") / f"detector_boxes_{split}.json")

    out_dir = args.out or paths.results_dir(f"stage3_config{args.configuration}_train")
    metrics = train(cfg, frame, boxes, out_dir, features, mean, std, args.device)
    print(json.dumps(metrics["splits"], indent=2))


if __name__ == "__main__":
    main()
