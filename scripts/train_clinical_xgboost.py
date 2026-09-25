"""Train the clinical XGBoost pre-classifiers and write predictions, SHAP values and the PCA of the continuous variables.

    python scripts/train_clinical_xgboost.py                          # 15 and 5 features, 100 Optuna trials
    python scripts/train_clinical_xgboost.py --features 5 --model M   # score a saved model instead of training

Outputs under OVARIAN_RESULTS/clinical_xgboost/: one folder per feature set with the model, summary.json
(cross-validated macro-F1, parameters, metrics per split, SHAP shares), evaluation_results.csv,
patient_predictions.csv and shap_values.csv; pca_continuous.json and figures at the top level.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ovarian import clinical, paths  # noqa: E402


def shap_figure(share: pd.Series, path: Path, title: str) -> None:
    share = share.sort_values()
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(share) + 1.5))
    ax.barh(share.index, share.to_numpy(), color="#2563EB")
    for y, value in enumerate(share.to_numpy()):
        ax.text(value + 0.5, y, f"{value:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Share of mean |SHAP value| (%)")
    ax.set_title(title)
    ax.set_xlim(0, share.max() * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pca_figure(result: dict, path: Path) -> None:
    scores, ratio = result["scores"], result["explained_variance_ratio"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    for label, color, name in ((0, "#4878CF", "Benign"), (1, "#D65F5F", "Malignant")):
        part = scores[scores.label == label]
        left.scatter(part.PC1, part.PC2, c=color, label=name, alpha=0.7, edgecolors="white", linewidths=0.4, s=60)
    left.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}% variance)")
    left.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}% variance)")
    left.legend()
    loadings = pd.DataFrame(result["loadings"])
    loadings.plot.bar(ax=right, rot=35, color=["#5B7FA6", "#A65B5B"])
    right.axhline(0, color="black", linewidth=0.8)
    right.set_ylabel("Loading")
    fig.suptitle("PCA of the five continuous clinical variables")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(df: pd.DataFrame, n_features: int, args) -> dict:
    features = clinical.FEATURE_SETS[n_features]
    out = paths.results_dir("clinical_xgboost", f"{n_features}_features")
    X_train, y_train = clinical.arrays(df, features, "train")
    X_val, y_val = clinical.arrays(df, features, "val")

    if args.model:
        model = clinical.SavedModel(args.model)
        summary = {"source": "saved model", "iteration_range": list(model.iteration_range)}
    else:
        study = clinical.tune(X_train, y_train, n_trials=args.trials, seed=args.seed)
        model = clinical.fit(study.best_params, X_train, y_train, X_val, y_val, seed=args.seed)
        study.trials_dataframe().to_csv(out / "optuna_trials.csv", index=False)
        model.get_booster().save_model(out / "xgb_clinical_model.json")
        summary = {"source": "trained", "n_trials": args.trials, "cv_macro_f1": study.best_value,
                   "best_params": study.best_params, "best_iteration": int(model.best_iteration)}

    evaluation = [clinical.evaluate(model, *clinical.arrays(df, features, s), s) for s in clinical.SPLITS]
    pd.DataFrame(evaluation).drop(columns="confusion").to_csv(out / "evaluation_results.csv", index=False)
    clinical.patient_predictions(model, df, features).to_csv(out / "patient_predictions.csv", index=False)
    values = clinical.shap_values(model, df, features)
    values.to_csv(out / "shap_values.csv", index=False)
    share = clinical.shap_share(values, features)
    shap_figure(share, out / "shap_importance.png", f"Clinical XGBoost, {n_features} features")

    summary.update({"features": features, "evaluation": evaluation, "shap_share_percent": share.to_dict()})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=int, nargs="+", choices=sorted(clinical.FEATURE_SETS), default=[15, 5])
    parser.add_argument("--trials", type=int, default=clinical.N_TRIALS)
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    parser.add_argument("--model", type=Path, help="saved XGBoost model to score instead of training (one feature set)")
    args = parser.parse_args(argv)
    if args.model and len(args.features) != 1:
        parser.error("--model needs exactly one --features value")

    df = clinical.cohort()
    top = paths.results_dir("clinical_xgboost")
    pca = clinical.pca_continuous(df)
    pca_figure(pca, top / "pca_continuous.png")
    pca["scores"].to_csv(top / "pca_scores.csv", index=False)
    (top / "pca_continuous.json").write_text(json.dumps({k: v for k, v in pca.items() if k != "scores"}, indent=1))
    print("PCA explained variance:", [round(v, 4) for v in pca["explained_variance_ratio"]])

    for n in args.features:
        s = run(df, n, args)
        if "cv_macro_f1" in s:
            print(f"{n} features: CV macro-F1 {s['cv_macro_f1']:.4f}")
        for e in s["evaluation"]:
            print(f"  {e['split']:5s} n={e['n']:3d} accuracy {e['accuracy']:.4f} macro-F1 {e['macro_f1']:.4f}")
        print(f"  CA-125 share of mean |SHAP|: {s['shap_share_percent']['CA125_log1p']:.2f}%")


if __name__ == "__main__":
    main()
