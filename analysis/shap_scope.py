"""Share of CA-125 in the mean absolute SHAP value of the clinical XGBoost model, per attribution scope.

Reads the stored SHAP values of the model (one row per patient) rather than refitting it. By default the
values of the released model are read from OVARIAN_PRIVATE/xgb_output; --shap-dir can point instead to
the output folder of scripts/train_clinical_xgboost.py.

Output: shap_original_scope.json.
"""
import argparse
from pathlib import Path

import pandas as pd

from analysis import private_input, write_json
from ovarian import clinical, stats

META = {"Patient_code", "patient", "label", "Split", "split"}
SCOPES = {"all_patients": None, "train_only": ["train"], "validation_only": ["val"], "test_only": ["test"],
          "train_plus_validation": ["train", "val"]}


def scope_shares(values: pd.DataFrame) -> dict:
    split_col = "Split" if "Split" in values else "split"
    features = [c for c in values.columns if c not in META]
    binary = [f for f in features if f.startswith(("Background", "Family", "Smoking"))]
    out = {"n_features": len(features), "features": features, "scopes": {}}
    for scope, splits in SCOPES.items():
        sub = values if splits is None else values[values[split_col].isin(splits)]
        share = clinical.shap_share(sub, features)
        out["scopes"][scope] = {"n_patients": len(sub),
                                "ca125_share_percent": stats.rounded(share["CA125_log1p"], 2),
                                "other_features_share_percent": stats.rounded(100 - share["CA125_log1p"], 2),
                                "binary_features_share_percent": stats.rounded(share[binary].sum(), 2),
                                "ranking": {k: stats.rounded(v, 2) for k, v in share.items()}}
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shap-dir", type=Path, help="folder with shap_values.csv and evaluation_results.csv")
    args = parser.parse_args(argv)
    folder = args.shap_dir or private_input("xgb_output")
    out = scope_shares(pd.read_csv(folder / "shap_values.csv"))
    evaluation = folder / "evaluation_results.csv"
    if evaluation.exists():
        out["model_evaluation"] = pd.read_csv(evaluation).to_dict(orient="records")
    write_json(out, "shap_original_scope.json")
    for scope, v in out["scopes"].items():
        print(f"{scope:22s} n={v['n_patients']:3d}  CA-125 {v['ca125_share_percent']}%  "
              f"binary {v['binary_features_share_percent']}%")


if __name__ == "__main__":
    main()
