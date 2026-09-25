"""Calibration of the Stage III patient probabilities, signal in the morphological descriptors, and split counts.

1. Calibration on the 21 test patients: Brier score and the slope and intercept of a logistic
   recalibration on the logit of the predicted probability. No binned reliability diagram is produced at
   this sample size.
2. Descriptor signal in the tabular branch alone (not importance inside the fusion model): univariate AUC
   per descriptor and permutation importance (AUC drop) of a gradient-boosting model on the eleven
   descriptors, with 5-fold patient-grouped cross-validation on the Stage III training images.
3. Image and patient counts per split and class for Stage I and Stage III.

Inputs: public release; calibration also needs OVARIAN_PRIVATE/test21_predictions.csv.
Output: calibration_and_importance.json.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from analysis import private_input, write_json
from ovarian import data, seeds



def calibration(prob, truth) -> dict:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    truth = np.asarray(truth, dtype=int)
    logit = np.log(prob / (1 - prob)).reshape(-1, 1)
    recal = LogisticRegression(penalty=None, max_iter=2000).fit(logit, truth)
    return {"brier": round(float(brier_score_loss(truth, prob)), 4),
            "calibration_slope": round(float(recal.coef_[0][0]), 4),
            "calibration_intercept": round(float(recal.intercept_[0]), 4),
            "mean_predicted": round(float(prob.mean()), 4),
            "observed_prevalence": round(float(truth.mean()), 4),
            "interpretation": ("slope below 1 indicates over-extreme probabilities; "
                               "slope above 1 indicates under-confident probabilities")}


def descriptor_table() -> pd.DataFrame:
    df = data.descriptors()
    df["truth"] = (df.cls == "malignant").astype(int)
    split = data.stage3_split().groupby("patient")["split"].first()
    df["split"] = df.patient.map(split)
    return df.dropna(subset=["split"])


def descriptor_signal(df: pd.DataFrame) -> dict:
    train = df[df.split == "train"]
    X = train[data.DESCRIPTORS].to_numpy(dtype=float)
    y = train.truth.to_numpy()
    groups = train.patient.astype(int).to_numpy()

    univariate = {}
    for i, name in enumerate(data.DESCRIPTORS):
        auc = roc_auc_score(y, X[:, i])
        univariate[name] = round(float(max(auc, 1 - auc)), 4)

    importances = {name: [] for name in data.DESCRIPTORS}
    fold_auc = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        model = GradientBoostingClassifier(random_state=seeds.resolve()).fit(scaler.transform(X[tr]), y[tr])
        fold_auc.append(roc_auc_score(y[te], model.predict_proba(scaler.transform(X[te]))[:, 1]))
        result = permutation_importance(model, scaler.transform(X[te]), y[te], n_repeats=30,
                                        random_state=seeds.resolve(), scoring="roc_auc")
        for name, value in zip(data.DESCRIPTORS, result.importances_mean, strict=True):
            importances[name].append(float(value))

    ranked = sorted(importances.items(), key=lambda kv: -np.mean(kv[1]))
    return {"scope": ("tabular branch only, eleven morphological descriptors; this is not "
                      "permutation importance inside the image-plus-tabular fusion model"),
            "n_images_train": len(train), "n_patients_train": int(train.patient.nunique()),
            "cross_validation": "5-fold, grouped by patient, on the training split only",
            "mean_cv_auc": round(float(np.mean(fold_auc)), 4),
            "univariate_auc": dict(sorted(univariate.items(), key=lambda kv: -kv[1])),
            "permutation_importance_auc_drop": {name: round(float(np.mean(v)), 4) for name, v in ranked}}


def split_counts(df: pd.DataFrame, group: pd.Series) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        part = df[df.split == split]
        out[split] = {g: {"patients": int(sub.patient.nunique()), "images": len(sub)}
                      for g, sub in part.groupby(group[part.index], sort=True)}
        out[split]["total"] = {"patients": int(part.patient.nunique()), "images": len(part)}
    return out


def counts() -> dict:
    stage1 = data.stage1_split()
    stage3 = data.stage3_split()
    return {"stage1": split_counts(stage1, stage1["class"].where(stage1["class"] == "normal", "pathological")),
            "stage3": split_counts(stage3, stage3["class"])}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    seeds.resolve(parser.parse_args(argv).seed)
    out = {}
    try:
        merged = pd.read_csv(private_input("test21_predictions.csv"))
        out["calibration"] = {"note": ("21 test patients. A binned reliability diagram is not interpretable at "
                                       "this size and is deliberately not produced."),
                              "config6": calibration(merged.c6s, merged.truth),
                              "config2": calibration(merged.c2s, merged.truth)}
    except FileNotFoundError as error:
        print(f"calibration skipped: {error}")
    out["morphological_feature_signal"] = descriptor_signal(descriptor_table())
    out["counts"] = counts()
    write_json(out, "calibration_and_importance.json")

    for cfg in ("config6", "config2"):
        if "calibration" in out:
            c = out["calibration"][cfg]
            print(f"{cfg}: Brier {c['brier']}, slope {c['calibration_slope']}, intercept {c['calibration_intercept']}")
    signal = out["morphological_feature_signal"]
    print(f"descriptors: mean grouped-CV AUC {signal['mean_cv_auc']}")
    for name, value in signal["permutation_importance_auc_drop"].items():
        print(f"  {name:16s} univariate AUC {signal['univariate_auc'][name]:<7} permutation {value}")


if __name__ == "__main__":
    main()
