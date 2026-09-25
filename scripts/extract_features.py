"""Compute the eleven morphological descriptors of every pathological image from its mask.

Writes <out>/morphological_features_{benign,malignant}.csv with columns patient_id (the image stem) and the
descriptors, in the format of the data release.
"""
import argparse
from pathlib import Path

import pandas as pd

from ovarian import data, paths
from ovarian.features import descriptors, read_gray, read_mask


def class_table(cls: str, masks: Path) -> pd.DataFrame:
    """Descriptors of all images of one class that have a mask, in image name order."""
    rows = []
    for image_path in sorted((paths.IMAGES / cls).glob("*.png")):
        mask_path = masks / cls / image_path.name
        if not mask_path.exists():
            continue
        f = descriptors(read_gray(image_path), read_mask(mask_path))
        rows.append({"patient_id": image_path.stem, **f})
    df = pd.DataFrame(rows, columns=["patient_id", *data.DESCRIPTORS])
    df["area"] = df["area"].astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", type=Path, default=paths.SAM_MASKS, help="mask root with benign/ and malignant/")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: RESULTS/features)")
    args = parser.parse_args()

    out = args.out or paths.results_dir("features")
    out.mkdir(parents=True, exist_ok=True)
    for cls in ("benign", "malignant"):
        df = class_table(cls, args.masks)
        target = out / f"morphological_features_{cls}.csv"
        df.to_csv(target, index=False)
        print(f"{cls}: {len(df)} images, wrote {target}")


if __name__ == "__main__":
    main()
