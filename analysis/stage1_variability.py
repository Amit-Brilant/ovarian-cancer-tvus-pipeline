"""Supplementary Table S3 (Stage I test metrics with bootstrap CIs) and Stage I variability.

Table S3 is computed from the released test predictions of the reported model. Image AUC uses the logit margin
(pathological minus normal); patients are aggregated by majority vote of the image predictions (first
image on a tie), and patient AUC uses the mean image probability. Confidence intervals: 1,000 bootstrap
resamples of images or patients, 2.5th and 97.5th percentiles.

The variability summary (mean, SD, min, max over the training runs) needs OVARIAN_PRIVATE/stage1_scores/metrics.csv.
Output: stage1_table_s3.json.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis import private_input, write_json
from ovarian import paths, seeds, stats

N_BOOT = 1000
KEYS = ["auc", "accuracy", "macro_f1", "sensitivity", "specificity", "ppv", "npv"]
SUMMARY = [("accuracy", "accuracy"), ("macro_f1", "macro_f1"), ("sensitivity_sick", "sensitivity"),
           ("specificity_healthy", "specificity"), ("roc_auc", "auc")]


def metrics(y, pred, score) -> dict:
    m = stats.binary_metrics(pred, y)
    out = {k: m[k] for k in ("tp", "tn", "fp", "fn", "accuracy", "macro_f1", "sensitivity", "specificity", "ppv", "npv")}
    out["auc"] = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
    return out


def level(y, pred, score, rng) -> dict:
    y, pred, score = (np.asarray(a) for a in (y, pred, score))
    draws = stats.bootstrap(lambda i: metrics(y[i], pred[i], score[i]), len(y), N_BOOT, rng, labels=y)
    return {"n": len(y), "point": metrics(y, pred, score),
            "ci95": {k: stats.percentile_ci([d[k] for d in draws]) for k in KEYS}}


def test_levels() -> tuple[pd.DataFrame, pd.DataFrame]:
    images = pd.read_csv(paths.PREDICTIONS / "stage1_image_predictions.csv", dtype={"patient": str})
    images = images[images.split == "test"].sort_values("evaluation_order").reset_index(drop=True)
    images["margin"] = images.logit_pathological - images.logit_normal
    patients = stats.patient_vote(images, pred="y_pred", prob="p_pathological")
    patients["y_true"] = images.groupby("patient", sort=False).y_true.first().to_numpy()

    released = pd.read_csv(paths.PREDICTIONS / "stage1_patient_predictions.csv", dtype={"patient": str})
    check = released[released.split == "test"].set_index("patient").loc[patients.patient]
    if not (check.y_pred_majority.to_numpy() == patients.y_pred_majority.to_numpy()).all():
        raise ValueError("patient majority votes differ from the released patient predictions")
    return images, patients


def run_variability() -> dict:
    m = pd.read_csv(private_input("stage1_scores", "metrics.csv"))
    test = m[m.split == "test"]
    out = {}
    for lvl in ("image", "patient"):
        for col, key in SUMMARY:
            v = test[test.level == lvl][col]
            out[f"{lvl}_{key}"] = {"mean": round(float(v.mean()), 3), "sd": round(float(v.std(ddof=1)), 3),
                                   "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    args = parser.parse_args(argv)
    rng = np.random.default_rng(seeds.resolve(args.seed))
    images, patients = test_levels()
    out = {"n_boot": N_BOOT,
           "levels": {"image": level(images.y_true, images.y_pred, images.margin, rng),
                      "patient": level(patients.y_true, patients.y_pred_majority, patients.prob_mean, rng)}}
    try:
        out["run_variability_test"] = run_variability()
    except FileNotFoundError as error:
        print(f"variability summary skipped: {error}")
    write_json(out, "stage1_table_s3.json")
    for name, lvl in out["levels"].items():
        p, ci = lvl["point"], lvl["ci95"]
        print(f"{name} n={lvl['n']} TN {p['tn']} FP {p['fp']} FN {p['fn']} TP {p['tp']}")
        for k in KEYS:
            print(f"  {k:12s} {p[k]:.4f} ({ci[k][0]:.4f} to {ci[k][1]:.4f})")


if __name__ == "__main__":
    main()
