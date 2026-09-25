import math

import numpy as np
import pandas as pd
import pytest

from ovarian import stats


def test_binary_metrics_counts_and_ratios():
    m = stats.binary_metrics(pred=[1, 1, 0, 0, 1], truth=[1, 0, 0, 1, 1])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 1)
    assert m["n"] == 5 and m["n_correct"] == 3
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["ppv"] == pytest.approx(2 / 3)
    assert m["npv"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx((4 / 6 + 2 / 4) / 2)


def test_undefined_ratio_is_nan_and_rounds_to_none():
    m = stats.binary_metrics(pred=[1, 1], truth=[1, 0])
    assert math.isnan(m["npv"])
    assert stats.rounded(m["npv"]) is None
    assert stats.rounded(0.123456) == 0.1235


def test_patient_vote_majority_tie_and_order():
    images = pd.DataFrame({"patient": ["b", "a", "b", "a", "b", "c", "c"],
                           "y_pred": [0, 1, 1, 0, 1, 0, 0],
                           "p": [0.2, 0.9, 0.6, 0.4, 0.7, 0.1, 0.3]})
    out = stats.patient_vote(images, prob="p").set_index("patient")
    assert list(out.index) == ["b", "a", "c"]
    assert out.loc["b", "y_pred_majority"] == 1 and not out.loc["b", "vote_tied"]
    assert out.loc["a", "vote_tied"] and out.loc["a", "y_pred_majority"] == 1
    assert out.loc["c", "y_pred_majority"] == 0
    assert out.loc["b", "prob_mean"] == pytest.approx(0.5)


def test_bootstrap_skips_single_class_resamples():
    labels = np.array([0, 1])
    draws = stats.bootstrap(lambda i: labels[i].mean(), 2, 200, np.random.default_rng(0), labels=labels)
    assert 0 < len(draws) < 200
    assert all(d == 0.5 for d in draws)


def test_percentile_ci():
    assert stats.percentile_ci(np.arange(101)) == pytest.approx([2.5, 97.5])
    assert stats.percentile_ci([np.nan, *range(101)]) == pytest.approx([2.5, 97.5])


def test_mcnemar_exact():
    a = [True] * 6 + [False] * 2 + [True] * 3
    b = [False] * 6 + [True] * 2 + [True] * 3
    a_only, b_only, p = stats.mcnemar_exact(a, b)
    assert (a_only, b_only) == (6, 2)
    assert p == pytest.approx(2 * (1 + 8 + 28) / 256)
    assert stats.mcnemar_exact([True, False], [True, False])[2] == 1.0


def test_holm():
    np.testing.assert_allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(stats.holm([0.6, 0.5]), [1.0, 1.0])


def test_benjamini_hochberg():
    np.testing.assert_allclose(stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.2]), [0.04, 0.16 / 3, 0.16 / 3, 0.2])


def test_cohen_kappa():
    assert stats.cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)


def test_fleiss_kappa():
    assert stats.fleiss_kappa([[3, 0], [0, 3]]) == pytest.approx(1.0)
    assert stats.fleiss_kappa([[2, 1], [1, 2]]) == pytest.approx(-1 / 3)
    assert math.isnan(stats.fleiss_kappa([[3, 0], [3, 0]]))
