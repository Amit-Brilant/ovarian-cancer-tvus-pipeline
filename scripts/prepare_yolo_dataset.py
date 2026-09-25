"""Build the Ultralytics detection dataset of Stage II from the Stage III partition and the YOLO annotations.

Creates <out>/images/{train,val,test}/<image> and <out>/labels/{train,val,test}/<stem>.txt as symbolic links
and <out>/data.yaml. Annotation class ids are 0 (benign) and 1 (malignant).
"""
import argparse
import json
from pathlib import Path

from ovarian import data, paths
from ovarian.segmentation import label_path

NAMES = {0: "benign", 1: "malignant"}
SPLITS = ("train", "val", "test")


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.absolute())


def label_classes(path: Path) -> set[int]:
    return {int(line.split()[0]) for line in path.read_text().splitlines() if line.strip()}


def build(out: Path, limit: int | None = None) -> dict[str, int]:
    """Link images and labels per split and write data.yaml; returns the number of images per split."""
    split = data.stage3_split().rename(columns={"class": "cls"})
    if limit is not None:
        split = split.sort_values("image").groupby(["split", "cls"], sort=False).head(limit)
    class_id = {name: k for k, name in NAMES.items()}
    counts = dict.fromkeys(SPLITS, 0)
    for part in SPLITS:
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
    for row in split.itertuples(index=False):
        image = data.image_path(row.image, row.cls)
        labels = label_path(Path(row.image).stem, row.cls)
        expected = class_id[row.cls]
        found = label_classes(labels)
        if found != {expected}:
            raise ValueError(f"{labels.name}: class ids {sorted(found)}, expected {expected}")
        link(image, out / "images" / row.split / row.image)
        link(labels, out / "labels" / row.split / labels.name)
        counts[row.split] += 1

    lines = [f"path: {json.dumps(str(out.resolve()))}"]
    lines += [f"{part}: images/{part}" for part in SPLITS]
    lines += ["names:"] + [f"  {k}: {v}" for k, v in NAMES.items()]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="dataset folder (default: RESULTS/yolo_dataset)")
    parser.add_argument("--limit", type=int, default=None, help="at most N images per split and class (smoke tests)")
    args = parser.parse_args()

    out = args.out or paths.results_dir("yolo_dataset")
    counts = build(out, args.limit)
    print(f"images per split: {counts}; wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
