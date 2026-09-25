"""Clinical XGBoost pre-classifier (15 or 5 features): preparation, tuning, fitting, SHAP and PCA.

Hyperparameters are chosen with Optuna (TPE sampler, 100 trials) by 5-fold stratified cross-validation
on the training patients, maximising macro-F1; the final model is refitted on all training patients
with early stopping on the validation patients. CA-125 enters as log1p(CA-125), with values recorded as
0 left missing for XGBoost to route. XGBoost and Optuna are imported only where they are needed.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ovarian import data, seeds

N_TRIALS = 100
N_FOLDS = 5
EARLY_STOPPING = 20
SPLITS = ("train", "val", "test")

CONTINUOUS = ["Age", "Years_since_menopause", "Pregnancies_count", "Actual_births_count", "CA125_log1p"]
FEATURE_SETS = {15: CONTINUOUS + data.BINARY, 5: CONTINUOUS}


def patient_splits() -> pd.Series:
    """Stage III partition per patient."""
    return data.stage3_split().groupby("patient")["split"].first()


def cohort() -> pd.DataFrame:
    """Clinical table with the Stage III split and log1p(CA-125); CA-125 recorded as 0 is missing."""
    df = data.clinical()
    df["split"] = df.patient.map(patient_splits())
    if df.split.isna().any():
        raise ValueError(f"{int(df.split.isna().sum())} patients have no Stage III split")
    df["CA125_log1p"] = np.log1p(df.CA125_levels)
    return df


def arrays(df: pd.DataFrame, features: list[str], split: str) -> tuple[np.ndarray, np.ndarray]:
    part = df[df.split == split]
    return part[features].to_numpy(dtype=float), part.label.to_numpy()


def base_params(seed: int | None = None) -> dict:
    seed = seeds.resolve(seed)
    return {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
            "random_state": seed, "early_stopping_rounds": EARLY_STOPPING, "verbosity": 0}


def suggest(trial) -> dict:
    return {"n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0)}


def fit(params: dict, X: np.ndarray, y: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, seed: int | None = None):
    import xgboost as xgb

    model = xgb.XGBClassifier(**base_params(seed), **params)
    model.fit(X, y, eval_set=[(X_eval, y_eval)], verbose=False)
    return model


class SavedModel:
    """A saved binary XGBoost model with the classifier methods used here.

    Predictions use the trees up to the stored best iteration, as XGBClassifier does after early stopping.
    """

    def __init__(self, path):
        import xgboost as xgb

        self.booster = xgb.Booster(model_file=str(path))
        best = self.booster.attr("best_iteration")
        self.iteration_range = (0, int(best) + 1) if best is not None else (0, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.booster.inplace_predict(np.asarray(X, dtype=float), iteration_range=self.iteration_range, missing=np.nan)
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def get_booster(self):
        return self.booster


def tune(X: np.ndarray, y: np.ndarray, n_trials: int = N_TRIALS, seed: int | None = None):
    """Optuna study maximising mean cross-validated macro-F1 on the training patients."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = seeds.resolve(seed)
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(X, y))

    def objective(trial) -> float:
        params = suggest(trial)
        scores = [f1_score(y[va], fit(params, X[tr], y[tr], X[va], y[va], seed).predict(X[va]), average="macro")
                  for tr, va in folds]
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def evaluate(model, X: np.ndarray, y: np.ndarray, split: str) -> dict:
    pred = model.predict(X)
    out = {"split": split, "n": len(y), "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}
    for label, name in ((0, "benign"), (1, "malignant")):
        out[f"{name}_precision"] = precision_score(y, pred, pos_label=label, zero_division=0)
        out[f"{name}_recall"] = recall_score(y, pred, pos_label=label, zero_division=0)
        out[f"{name}_f1"] = f1_score(y, pred, pos_label=label, zero_division=0)
    out["confusion"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return out


def patient_predictions(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Probability of malignancy for every patient, train then validation then test."""
    frames = []
    for split in SPLITS:
        part = df[df.split == split][["patient", "label", "split"]].copy()
        part["xgb_prob_malignant"] = model.predict_proba(df.loc[part.index, features].to_numpy(dtype=float))[:, 1]
        part["xgb_pred"] = (part.xgb_prob_malignant >= 0.5).astype(int)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def shap_values(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Exact tree SHAP values (XGBoost pred_contribs, bias column dropped) for every patient."""
    import xgboost as xgb

    ordered = pd.concat([df[df.split == s] for s in SPLITS], ignore_index=True)
    matrix = xgb.DMatrix(ordered[features].to_numpy(dtype=float), feature_names=features, missing=np.nan)
    contribs = model.get_booster().predict(matrix, pred_contribs=True)[:, :-1]
    return pd.concat([ordered[["patient", "label", "split"]], pd.DataFrame(contribs, columns=features)], axis=1)


def shap_share(values: pd.DataFrame, features: list[str]) -> pd.Series:
    """Percentage of the mean absolute SHAP value per feature, largest first."""
    mean_abs = values[features].abs().mean()
    return (mean_abs / mean_abs.sum() * 100).sort_values(ascending=False)


def pca_continuous(df: pd.DataFrame | None = None) -> dict:
    """Two-component PCA of the five standardised continuous variables over all pathological patients.

    Missing values (CA-125 recorded as 0) are replaced by the column mean.
    """
    df = data.clinical() if df is None else df
    X = df[data.CONTINUOUS].astype(float)
    X = StandardScaler().fit_transform(X.fillna(X.mean()).to_numpy())
    pca = PCA(n_components=2).fit(X)
    scores = pca.transform(X)
    return {"n_patients": len(df), "features": list(data.CONTINUOUS),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "loadings": {f"PC{i + 1}": dict(zip(data.CONTINUOUS, pca.components_[i].tolist(), strict=True)) for i in range(2)},
            "scores": pd.DataFrame({"patient": df.patient.to_numpy(), "label": df.label.to_numpy(),
                                    "PC1": scores[:, 0], "PC2": scores[:, 1]})}
