"""Clinical baselines (CA-125, CA-125 with age, age), raw CA-125 cutoff sweep, and Supplementary Table S2.

Baselines mirror the pipeline design: logistic regression on standardised inputs fitted on the Stage III
training patients, decision threshold chosen on the validation patients (maximum accuracy over the
predicted probabilities), metrics on the test patients. CA-125 is log1p-transformed. CA-125 recorded as 0
is missing and replaced by the training median, for the fitted models and for the raw cutoff sweep alike.
The sweep calls a test patient malignant when CA-125 exceeds the cutoff (35 U/mL and every integer from
20 to 80 U/mL), without fitting.

Supplementary Table S2: Mann-Whitney U (continuous) and Fisher's exact test (binary), both two-sided, with
Benjamini-Hochberg adjustment over the ten tests; CA-125 uses the patients with a recorded value.

Inputs: public release; the comparison with the Stage III models and reader 1 also needs
OVARIAN_PRIVATE/test21_predictions.csv and reader_study/reader1.xlsx.
Outputs: clinical_baselines.json, clinical_baselines_validation.json, tableS2_fdr.json.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from analysis import private_input, readers, write_json
from ovarian import clinical, data, seeds, stats

N_BOOT = 2000
CONVENTIONAL_CUTOFF = 35
CUTOFFS = range(20, 81)
BASELINES = {"ca125_alone": ["log_ca125"], "ca125_plus_age": ["log_ca125", "Age"], "age_alone": ["Age"]}
CONFIGS = {"config6": "c6", "config2": "c2"}

TABLE_S2_CONTINUOUS = [("Age", "Age"), ("Years since menopause", "Years_since_menopause"),
                       ("Pregnancies", "Pregnancies_count"), ("Live births", "Actual_births_count"),
                       ("CA-125", "CA125_levels")]
TABLE_S2_BINARY = [("Current smoker", "Smoking"), ("Family history of cancer", "family_any"),
                   ("Personal history of cancer", "Background_diseases_Cancer"),
                   ("Diabetes", "Background_diseases_Diabetes"), ("Hypertension", "Background_diseases_Hypertension")]
FAMILY = ["Family_history_breast", "Family_history_ovaries", "Family_history_uterus", "Family_history_other"]


def cohort() -> tuple[pd.DataFrame, dict]:
    df = clinical.cohort()
    median = float(df.loc[df.split == "train", "CA125_levels"].median())
    missing = df.CA125_levels.isna()
    df["ca125_imputed"] = df.CA125_levels.fillna(median)
    df["log_ca125"] = np.log1p(df.ca125_imputed)
    info = {"n_missing": int(missing.sum()), "patients": df.loc[missing, "patient"].astype(int).tolist(),
            "imputed_with_training_median_u_per_ml": round(median, 2)}
    return df, info


def metrics(pred, truth, prob=None) -> dict:
    truth = np.asarray(truth)
    m = stats.binary_metrics(pred, truth)
    out = {"n": m["n"], "n_correct": m["n_correct"]}
    out.update({k: stats.rounded(m[k]) for k in ("accuracy", "sensitivity", "specificity", "ppv", "npv")})
    out["confusion"] = {k: m[k] for k in ("tp", "fp", "tn", "fn")}
    if prob is not None:
        prob = np.asarray(prob)
        out["auc"] = round(float(roc_auc_score(truth, prob)), 4)
        draws = stats.bootstrap(lambda i: roc_auc_score(truth[i], prob[i]), len(truth), N_BOOT,
                                np.random.default_rng(seeds.resolve()), labels=truth)
        if draws:
            out["auc_ci95"] = [round(v, 4) for v in stats.percentile_ci(draws)]
    return out


def best_threshold(prob: np.ndarray, truth: np.ndarray) -> float:
    """Accuracy-maximising threshold among the exact predicted probabilities (lowest on ties)."""
    best, best_acc = 0.5, -1.0
    for t in np.unique(prob):
        acc = ((prob >= t).astype(int) == truth).mean()
        if acc > best_acc:
            best, best_acc = float(t), float(acc)
    return best


def fit_baseline(df: pd.DataFrame, features: list[str]) -> tuple[dict, pd.Series]:
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    scaler = StandardScaler().fit(tr[features].to_numpy())
    model = LogisticRegression(max_iter=2000, random_state=seeds.resolve()).fit(scaler.transform(tr[features].to_numpy()),
                                                                     tr.label.to_numpy())
    p_val = model.predict_proba(scaler.transform(va[features].to_numpy()))[:, 1]
    threshold = best_threshold(p_val, va.label.to_numpy())
    p_test = model.predict_proba(scaler.transform(te[features].to_numpy()))[:, 1]
    pred = (p_test >= threshold).astype(int)

    result = metrics(pred, te.label.to_numpy(), p_test)
    result["validation"] = metrics((p_val >= threshold).astype(int), va.label.to_numpy(), p_val)
    result["validation"]["note"] = "threshold chosen on this split, so in-sample for the threshold"
    result.update({"features": features, "threshold_selected_on_validation": round(threshold, 4), "n_train": len(tr)})
    return result, pd.Series(pred, index=te.patient.to_numpy())


def cutoff_sweep(df: pd.DataFrame) -> dict:
    te = df[df.split == "test"]
    ca125, truth = te.ca125_imputed.to_numpy(), te.label.to_numpy()

    def at(cutoff: int) -> dict:
        m = metrics((ca125 > cutoff).astype(int), truth)
        return {k: m[k] for k in ("n_correct", "accuracy", "sensitivity", "specificity")}

    return {"rule": ("malignant if CA-125 > cutoff (U/mL); raw value (missing replaced by the training median), "
                     "no fitting, test patients only"),
            "n_test": len(te),
            "conventional_cutoff": {"cutoff_u_per_ml": CONVENTIONAL_CUTOFF, **at(CONVENTIONAL_CUTOFF)},
            "by_cutoff": {str(c): at(c) for c in CUTOFFS}}


def model_comparison(out: dict, predictions: dict[str, pd.Series]) -> None:
    """Stage III models and reader 1 on the same test patients, and exact McNemar tests against the baselines."""
    merged = pd.read_csv(private_input("test21_predictions.csv")).sort_values("original_patient_id")
    truth = merged.truth.to_numpy()
    for label, col in CONFIGS.items():
        out["baselines"][label] = metrics(merged[col].to_numpy(), truth, merged[f"{col}s"].to_numpy())
    reader1 = readers.load_reader("reader1").loc[merged.new_code.to_numpy()]
    raw = [None if pd.isna(v) else int(v) for v in reader1.call.to_numpy(dtype=object)]
    out["baselines"]["reader1_orads_ge4"] = metrics(readers.rule_a(raw, truth), truth)

    codes = merged.original_patient_id.astype(int).astype(str).str.zfill(3).to_numpy()
    out["model_vs_baseline"] = {}
    for label, col in CONFIGS.items():
        model_ok = merged[col].to_numpy() == truth
        for name, series in predictions.items():
            a_only, b_only, p = stats.mcnemar_exact(model_ok, series.loc[codes].to_numpy() == truth)
            out["model_vs_baseline"][f"{label}_vs_{name}"] = {
                "model_only_correct": a_only, "baseline_only_correct": b_only,
                "exact_mcnemar_p": round(p, 4), "significant_at_0.05": p < 0.05}


def table_s2(df: pd.DataFrame | None = None) -> dict:
    df = data.clinical() if df is None else df.copy()
    df["family_any"] = (df[FAMILY].fillna(0).sum(axis=1) > 0).astype(int)
    benign, malignant = df[df.label == 0], df[df.label == 1]
    names, raw = [], []
    for name, col in TABLE_S2_CONTINUOUS:
        names.append(name)
        raw.append(mannwhitneyu(benign[col].dropna(), malignant[col].dropna(), alternative="two-sided").pvalue)
    for name, col in TABLE_S2_BINARY:
        table = [[int((benign[col] == 1).sum()), int((benign[col] == 0).sum())],
                 [int((malignant[col] == 1).sum()), int((malignant[col] == 0).sum())]]
        names.append(name)
        raw.append(fisher_exact(table, alternative="two-sided")[1])
    fdr = stats.benjamini_hochberg(raw)
    return {n: {"raw": round(float(r), 4), "fdr": round(float(f), 4)} for n, r, f in zip(names, raw, fdr, strict=True)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    seeds.resolve(parser.parse_args(argv).seed)
    df, missing = cohort()
    counts = df.split.value_counts()
    out = {"design": (f"fit on {counts['train']} training patients, threshold chosen on {counts['val']} validation "
                      f"patients, reported on the {counts['test']} test patients"),
           "ca125_transform": "log1p", "ca125_missing": missing, "baselines": {}}
    predictions = {}
    for name, features in BASELINES.items():
        out["baselines"][name], predictions[name] = fit_baseline(df, features)
    try:
        model_comparison(out, predictions)
    except FileNotFoundError as error:
        print(f"comparison with the Stage III models skipped: {error}")
    out["ca125_cutoff_sweep"] = cutoff_sweep(df)
    out["not_reproducible"] = {
        "iota_simple_rules": ("requires prospectively recorded sonographic descriptors (locularity, papillary "
                              "projections, acoustic shadows, ascites) that were not collected in this retrospective cohort"),
        "adnex": ("same limitation, and additionally requires maximal lesion diameter in millimetres, which is "
                  "unavailable because pixel-to-millimetre calibration metadata were not retained"),
        "orads_as_independent_baseline": ("each reader's binary call is derived from their own O-RADS score by the "
                                          "O-RADS >= 4 threshold, so it is arithmetically identical to the reader "
                                          "column and is not an independent comparator")}
    write_json(out, "clinical_baselines.json")

    val = out["baselines"]["ca125_alone"]["validation"]
    write_json({"design": ("the CA-125 alone model fitted on the training patients, evaluated on the validation "
                           "patients at the threshold chosen on validation"),
                "ca125_alone_validation": {k: val[k] for k in ("n", "n_correct", "accuracy", "auc", "sensitivity",
                                                               "specificity")}},
               "clinical_baselines_validation.json")
    write_json(table_s2(df), "tableS2_fdr.json")

    for name, r in out["baselines"].items():
        print(f"{name:20s} {r['n_correct']}/{r['n']}  accuracy {r['accuracy']}  AUC {r.get('auc', 'n/a')}")
    sweep = out["ca125_cutoff_sweep"]
    best = max(sweep["by_cutoff"].items(), key=lambda kv: kv[1]["n_correct"])
    print(f"CA-125 > {CONVENTIONAL_CUTOFF} U/mL: {sweep['conventional_cutoff']['n_correct']}/{sweep['n_test']}; "
          f"best cutoff {best[0]} U/mL: {best[1]['n_correct']}/{sweep['n_test']}")


if __name__ == "__main__":
    main()
