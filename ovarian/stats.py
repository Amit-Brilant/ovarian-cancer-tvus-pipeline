"""Statistics shared by the analyses: aggregation, bootstrap, paired tests, multiplicity and agreement."""
import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def rounded(value, digits: int = 4):
    """Round for JSON output; undefined values become None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def binary_metrics(pred, truth) -> dict:
    """Confusion counts and the usual binary metrics (positive class = 1); undefined ratios are NaN."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    f1_pos, f1_neg = ratio(2 * tp, 2 * tp + fp + fn), ratio(2 * tn, 2 * tn + fp + fn)
    n = tp + fp + tn + fn
    return {"n": n, "n_correct": tp + tn, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": ratio(tp + tn, n), "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn), "macro_f1": (f1_pos + f1_neg) / 2}


def patient_vote(images: pd.DataFrame, patient: str = "patient", pred: str = "y_pred",
                 prob: str | None = None, tie_to_positive: bool = False) -> pd.DataFrame:
    """Patient-level majority vote over image predictions, patients in order of first appearance.

    A tied vote takes the prediction of the patient's first image (first-encounter rule), or the positive
    class with `tie_to_positive`, the rule of the Stage III evaluation. With `prob`, the mean image
    probability is returned as `prob_mean`.
    """
    rows = []
    for code, group in images.groupby(patient, sort=False):
        votes = group[pred].to_numpy().astype(int)
        n_pos = int(votes.sum())
        n_neg = len(votes) - n_pos
        row = {patient: code, "n_images": len(votes), "n_pred_0": n_neg, "n_pred_1": n_pos,
               "vote_tied": n_pos == n_neg,
               "y_pred_majority": (1 if tie_to_positive else int(votes[0])) if n_pos == n_neg
               else int(n_pos > n_neg)}
        if prob is not None:
            row["prob_mean"] = float(group[prob].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(statistic: Callable[[np.ndarray], object], n: int, n_boot: int, rng: np.random.Generator,
              labels: Sequence | None = None) -> list:
    """Nonparametric bootstrap: `statistic(indices)` over `n_boot` resamples of size n.

    With `labels`, resamples containing a single class are skipped (they still consume random draws).
    """
    labels = None if labels is None else np.asarray(labels)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if labels is not None and len(np.unique(labels[idx])) < 2:
            continue
        draws.append(statistic(idx))
    return draws


def percentile_ci(values: Sequence[float], level: float = 0.95) -> list[float]:
    """Percentile interval, ignoring NaN draws."""
    tail = (1 - level) / 2 * 100
    lo, hi = np.nanpercentile(np.asarray(values, dtype=float), [tail, 100 - tail])
    return [float(lo), float(hi)]


def mcnemar_exact(a_correct, b_correct) -> tuple[int, int, float]:
    """Two-sided exact McNemar test on paired correctness. Returns (a only correct, b only correct, p)."""
    a_correct, b_correct = np.asarray(a_correct, dtype=bool), np.asarray(b_correct, dtype=bool)
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    p = 1.0 if a_only + b_only == 0 else float(binomtest(a_only, a_only + b_only, 0.5).pvalue)
    return a_only, b_only, p


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, in input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p, kind="stable")):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values, in input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


def cohen_kappa(a, b) -> float:
    return float(cohen_kappa_score(a, b))


def fleiss_kappa(counts) -> float:
    """Fleiss' kappa from an items x categories matrix of rating counts (equal raters per item)."""
    counts = np.asarray(counts, dtype=float)
    n_items = counts.shape[0]
    n_raters = counts[0].sum()
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_e = (p_j ** 2).sum()
    return math.nan if p_e == 1 else float((p_i.mean() - p_e) / (1 - p_e))
