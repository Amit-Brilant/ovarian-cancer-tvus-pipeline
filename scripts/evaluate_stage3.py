"""Evaluate a released Stage III configuration on the fixed partition.

Writes per-image and per-patient predictions (the columns of the public prediction files), a metrics JSON
and, with --aggregation-rules, the alternative patient aggregation rules of Supplementary Table S5.

Configuration 6 needs the descriptors of the automatic cyst masks, which are released per split, and the
normalization statistics released with the weights. The detector boxes are read from the release; pass
--detect to run the Stage II detector instead.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ovarian import paths, seeds
from ovarian.stage1 import default_device, make_loader, seed_everything
from ovarian.stage3 import (
    STAGE3_WEIGHTS,
    Stage3Dataset,
    aggregation_rules,
    detector_boxes,
    evaluate,
    load_boxes,
    load_features,
    load_normalization,
    load_split,
    load_stage3_model,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=int, default=6, choices=[2, 6])
    parser.add_argument("--weights", type=Path, default=None, help="default: the released Configuration <n> weights")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--features", type=Path, default=None,
                        help="folder with automatic_morphological_features_<split>.csv (default: DATA/features)")
    parser.add_argument("--normalization", type=Path, default=None,
                        help="default: WEIGHTS/stage3_config6_feature_normalization.json")
    parser.add_argument("--boxes", type=Path, default=None,
                        help="folder with detector_boxes_<split>.json (default: DATA/detector_boxes)")
    parser.add_argument("--detect", action="store_true", help="run the Stage II detector instead of the released boxes")
    parser.add_argument("--detector-weights", type=Path, default=paths.WEIGHTS / "stage2_yolov8m.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    parser.add_argument("--aggregation-rules", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: results/stage3_config<n>)")
    args = parser.parse_args()

    seed_everything(seeds.resolve(args.seed))
    frame = load_split(args.splits)
    weights = args.weights or paths.WEIGHTS / STAGE3_WEIGHTS[args.configuration]
    model = load_stage3_model(args.configuration, weights, args.device)

    features = mean = std = None
    if args.configuration != 2:
        names, mean, std = load_normalization(
            args.normalization or paths.WEIGHTS / "stage3_config6_feature_normalization.json")
        folder = args.features or paths.FEATURES
        features = load_features({s: folder / f"automatic_morphological_features_{s}.csv" for s in args.splits}, names)

    predictions = []
    for split in args.splits:
        part = frame[frame.split == split].reset_index(drop=True)
        if args.detect:
            boxes = detector_boxes(part, args.detector_weights, device=args.device)
        else:
            boxes = load_boxes((args.boxes or paths.DATA / "detector_boxes") / f"detector_boxes_{split}.json")
        dataset = Stage3Dataset(part, boxes, features, mean, std)
        loader = make_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        predictions.append(evaluate(model, loader, part, args.device))

    predictions = pd.concat(predictions, ignore_index=True)
    out_dir = args.out or paths.results_dir(f"stage3_config{args.configuration}")
    extra = {"configuration": args.configuration, "weights": weights.name,
             "boxes": "detector" if args.detect else "released"}
    metrics = write_outputs(predictions, out_dir, extra)
    if args.aggregation_rules:
        rules = aggregation_rules(predictions)
        (Path(out_dir) / "stage3_aggregation_rules.json").write_text(json.dumps(rules, indent=2))
    print(json.dumps(metrics["splits"], indent=2))


if __name__ == "__main__":
    main()
