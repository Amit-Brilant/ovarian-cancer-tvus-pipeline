"""Sensitivity of the eleven morphological descriptors to perturbation of the released SAM masks.

Step 1 recomputes the descriptors from the masks and compares them with the published descriptor tables.

Step 2 erodes and dilates every mask with disks of radius 2, 5 and 10 px and recomputes the descriptors.
Reported per perturbation:
  - per descriptor, Spearman correlation with the unperturbed value across images
  - per descriptor, median absolute relative change
  - a gradient boosting model on the standardized descriptors, fitted on the training images with
    unperturbed descriptors and evaluated on the test images with perturbed descriptors (image-level AUC,
    patient-level AUC and accuracy at 0.5)
  - five-fold patient-grouped cross-validated AUC over the training images when both the training and the
    held-out folds use perturbed descriptors

Writes RESULTS/mask_sensitivity.json.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from skimage.morphology import dilation, disk, erosion
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ovarian import data, paths, seeds
from ovarian.features import descriptors, entropy_map, read_gray, read_mask

FEATS = data.DESCRIPTORS
PERTURBATIONS = [("erode_2", -2), ("erode_5", -5), ("erode_10", -10),
                 ("dilate_2", 2), ("dilate_5", 5), ("dilate_10", 10)]


def load_table() -> pd.DataFrame:
    """Published descriptors with Stage III split, integer patient and label (malignant = 1)."""
    df = data.descriptors()
    split = data.stage3_split()
    df = df.merge(split[["image", "split"]], on="image", how="left")
    if df.split.isna().any():
        raise SystemExit("images without a Stage III split")
    df["patient"] = df.patient.astype(int)
    df["y"] = (df.cls == "malignant").astype(int)
    return df


def image_features(item: tuple[str, str]) -> dict[str, dict[str, float]]:
    """Descriptors of one image with its released mask and with each perturbed mask."""
    cls, image = item
    gray = read_gray(data.image_path(image, cls))
    mask = read_mask(paths.SAM_MASKS / cls / image)
    ent = entropy_map(gray)
    out = {"original": descriptors(gray, mask, ent)}
    for name, radius in PERTURBATIONS:
        op = erosion if radius < 0 else dilation
        out[name] = descriptors(gray, op(mask, disk(abs(radius))), ent)
    return out


def compute_all(table: pd.DataFrame, workers: int) -> dict[str, pd.DataFrame]:
    items = list(zip(table.cls, table.image, strict=True))
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(image_features, items, chunksize=4))
    else:
        results = [image_features(item) for item in items]
    names = ["original"] + [name for name, _ in PERTURBATIONS]
    return {name: pd.DataFrame([r[name] for r in results], index=table.stem) for name in names}


def tabular_eval(table: pd.DataFrame, train_feats: pd.DataFrame, test_feats: pd.DataFrame) -> dict:
    tr = (table.split == "train").values
    te = (table.split == "test").values
    scaler = StandardScaler().fit(train_feats[tr][FEATS].values)
    model = GradientBoostingClassifier(random_state=seeds.resolve()).fit(
        scaler.transform(train_feats[tr][FEATS].values), table.y[tr].values)
    p = model.predict_proba(scaler.transform(test_feats[te][FEATS].values))[:, 1]
    img_auc = roc_auc_score(table.y[te], p)
    pat = pd.DataFrame({"patient": table.patient[te].values, "y": table.y[te].values, "p": p}).groupby(
        "patient").agg(y=("y", "first"), p=("p", "mean"))
    correct = (pat.p >= 0.5).astype(int) == pat.y
    return {"image_auc": round(float(img_auc), 4), "patient_auc": round(float(roc_auc_score(pat.y, pat.p)), 4),
            "patient_accuracy": round(float(correct.mean()), 4),
            "patients_correct": int(correct.sum()), "n_patients": int(len(pat))}


def cv_auc(table: pd.DataFrame, feats: pd.DataFrame) -> float:
    tr = (table.split == "train").values
    X, y, g = feats[tr][FEATS].values, table.y[tr].values, table.patient[tr].values
    aucs = []
    for a, b in GroupKFold(n_splits=5).split(X, y, g):
        sc = StandardScaler().fit(X[a])
        m = GradientBoostingClassifier(random_state=seeds.resolve()).fit(sc.transform(X[a]), y[a])
        aucs.append(roc_auc_score(y[b], m.predict_proba(sc.transform(X[b]))[:, 1]))
    return round(float(np.mean(aucs)), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=1, help="parallel processes for the descriptor computation")
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    args = parser.parse_args()
    seeds.resolve(args.seed)

    table = load_table()
    print(f"{len(table)} images; computing descriptors and {len(PERTURBATIONS)} perturbations", flush=True)
    F = compute_all(table, args.workers)
    base = F["original"]

    pub = table.set_index("stem")[FEATS]
    rel = (base[FEATS].values - pub.values) / np.maximum(np.abs(pub.values), 1e-9)
    anchor = {f: {"median_abs_rel_diff": round(float(np.median(np.abs(rel[:, j]))), 5),
                  "share_within_1pct": round(float((np.abs(rel[:, j]) < 0.01).mean()), 4),
                  "spearman_vs_published": round(float(spearmanr(base[f], pub[f]).correlation), 4)}
              for j, f in enumerate(FEATS)}

    results = {"n_images": int(len(table)), "mask_source": str(paths.SAM_MASKS),
               "anchor_recomputed_vs_published": anchor,
               "baseline_tabular": {"test": tabular_eval(table, base, base), "cv_train_auc": cv_auc(table, base)},
               "perturbations": {}}
    for name, radius in PERTURBATIONS:
        pf = F[name]
        per_feat = {}
        for f in FEATS:
            rc = (pf[f].values - base[f].values) / np.maximum(np.abs(base[f].values), 1e-9)
            per_feat[f] = {"spearman": round(float(spearmanr(pf[f], base[f]).correlation), 4),
                           "median_abs_rel_change": round(float(np.median(np.abs(rc))), 4)}
        results["perturbations"][name] = {
            "radius_px": radius, "n_masks_emptied": int((pf.area == 0).sum()), "per_feature": per_feat,
            "test_model_trained_on_unperturbed": tabular_eval(table, base, pf),
            "cv_train_auc_perturbed_both_sides": cv_auc(table, pf)}
        print(name, results["perturbations"][name]["test_model_trained_on_unperturbed"],
              "cv", results["perturbations"][name]["cv_train_auc_perturbed_both_sides"], flush=True)

    out = paths.results_dir() / "mask_sensitivity.json"
    out.write_text(json.dumps(results, indent=1))
    print("baseline", results["baseline_tabular"])
    print("wrote", out)


if __name__ == "__main__":
    main()
