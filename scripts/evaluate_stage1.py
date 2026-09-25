"""Evaluate the released Stage I classifier on the fixed split.

Writes per-image and per-patient predictions (same columns as the public prediction files) and a metrics JSON.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ovarian import paths, seeds
from ovarian.stage1 import (
    Stage1Dataset,
    default_device,
    evaluate,
    load_split,
    load_stage1_model,
    make_loader,
    seed_everything,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=paths.WEIGHTS / "stage1_efficientnet_b7.pth")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: results/stage1)")
    args = parser.parse_args()

    seed_everything(seeds.resolve(args.seed))
    frame = load_split()
    model = load_stage1_model(args.weights, args.device)
    predictions = []
    for split in args.splits:
        part = frame[frame.split == split].reset_index(drop=True)
        loader = make_loader(Stage1Dataset(part), batch_size=args.batch_size, num_workers=args.num_workers)
        predictions.append(evaluate(model, loader, part, args.device)[0])

    out_dir = args.out or paths.results_dir("stage1")
    metrics = write_outputs(pd.concat(predictions, ignore_index=True), out_dir, {"weights": args.weights.name})
    print(json.dumps(metrics["splits"], indent=2))


if __name__ == "__main__":
    main()
