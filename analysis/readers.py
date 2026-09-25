"""Three-reader study on the 21 test patients (Table 3, agreement, McNemar) and Figure 3b.

Each reader's O-RADS score is mapped to a binary call, O-RADS >= 4 meaning malignant. A score recorded as
a range counts only when the whole range lies on one side of the threshold. Cases a reader could not
classify are counted as errors against that reader (rule A, reported in the manuscript) and, as an
alternative, excluded from that reader's metrics (rule B). Model-versus-reader comparisons use the exact
McNemar test, with Holm adjustment over the three readers separately for each model configuration.

Inputs (OVARIAN_PRIVATE): reader_study/reader{1,2,3}.xlsx and test21_predictions.csv.
Outputs: three_reader_results.json, three_reader_per_case.csv and figures/Figure_3b.{png,tif}.
"""
import argparse
import numbers
from itertools import combinations

import numpy as np
import pandas as pd

from analysis import output_dir, private_input, write_json
from ovarian import stats

THRESHOLD = 4
READERS = {"reader1": "co-author",
           "reader2": "co-author; also performed the de-identification of the dataset",
           "reader3": "not an author; the only fully independent reader"}
CONFIGS = {"config6": "c6", "config2": "c2"}


def parse_orads(value) -> tuple[int | None, int | None]:
    """(low, high) O-RADS score; (None, None) when the case was not classified."""
    if pd.isna(value):
        return None, None
    if isinstance(value, numbers.Number):
        return int(value), int(value)
    digits = "".join(c if c.isdigit() else " " for c in str(value)).split()
    if not digits:
        return None, None
    return int(digits[0]), int(digits[-1])


def binary_call(low: int | None, high: int | None) -> int | None:
    if low is None:
        return None
    if low >= THRESHOLD:
        return 1
    if high < THRESHOLD:
        return 0
    return None


def test_predictions() -> pd.DataFrame:
    """Per-patient truth and Stage III predictions of the 21 test patients, ordered by case number."""
    df = pd.read_csv(private_input("test21_predictions.csv"))
    return df.sort_values("new_code").reset_index(drop=True)


def load_reader(name: str) -> pd.DataFrame:
    df = pd.read_excel(private_input("reader_study", f"{name}.xlsx"))
    code_col = next(c for c in df.columns if "Patient" in str(c))
    orads_col = next(c for c in df.columns if "O-RADS" in str(c))
    rows = []
    for code, value in zip(df[code_col], df[orads_col], strict=True):
        low, high = parse_orads(value)
        rows.append({"code": int(code), "orads_raw": str(value).strip(), "call": binary_call(low, high)})
    return pd.DataFrame(rows).set_index("code")


def metrics(pred, truth) -> dict:
    m = stats.binary_metrics(pred, truth)
    out = {"n": m["n"], "n_correct": m["n_correct"]}
    out.update({k: stats.rounded(m[k]) for k in ("accuracy", "macro_f1", "sensitivity", "specificity", "ppv", "npv")})
    out["kappa_vs_truth"] = stats.rounded(stats.cohen_kappa(truth, pred))
    out["confusion"] = {k: m[k] for k in ("tp", "fp", "tn", "fn")}
    return out


def mcnemar(a, b, truth) -> dict:
    a_only, b_only, p = stats.mcnemar_exact(np.asarray(a) == truth, np.asarray(b) == truth)
    return {"a_only_correct": a_only, "b_only_correct": b_only, "exact_mcnemar_p": round(p, 4),
            "significant_at_0.05": p < 0.05, "_p": p}


def reader_calls(merged: pd.DataFrame) -> dict[str, dict]:
    """Per reader: raw calls in case order (None when unclassifiable) and the sheet rows."""
    out = {}
    for name in READERS:
        sheet = load_reader(name).loc[merged.new_code.to_numpy()]
        out[name] = {"raw": [None if pd.isna(v) else int(v) for v in sheet.call.to_numpy(dtype=object)],
                     "sheet": sheet}
    return out


def rule_a(raw: list, truth: np.ndarray) -> np.ndarray:
    """Unclassifiable cases counted as errors."""
    return np.array([1 - t if v is None else v for v, t in zip(raw, truth, strict=True)], dtype=int)


def analyse() -> tuple[dict, pd.DataFrame]:
    merged = test_predictions()
    truth = merged.truth.to_numpy()
    codes = merged.new_code.to_numpy()
    out = {"n_patients": len(merged), "author_status": READERS, "orads_malignant_threshold": THRESHOLD,
           "binary_rule": f"O-RADS >= {THRESHOLD} counted as malignant, applied identically to all readers",
           "readers": {}, "unclassifiable": {}, "models": {}, "model_vs_reader": {},
           "reader_agreement": {}, "panel_majority_vote": {}}

    calls = {}
    for name, entry in reader_calls(merged).items():
        raw = entry["raw"]
        missing = [int(c) for c, v in zip(codes, raw, strict=True) if v is None]
        keep = np.array([v is not None for v in raw])
        calls[name] = rule_a(raw, truth)
        result = {"is_author": name != "reader3", "n_unclassifiable": len(missing), "unclassifiable_codes": missing,
                  "rule_A_unreadable_counted_as_error": metrics(calls[name], truth)}
        if missing:
            result["rule_B_unreadable_excluded"] = metrics([v for v in raw if v is not None], truth[keep])
            rows = merged.set_index("new_code").loc[missing]
            out["unclassifiable"][name] = {"codes": missing,
                                           "study_codes": rows.original_patient_id.astype(int).tolist(),
                                           "true_class": rows.class_name.astype(str).tolist(),
                                           "raw_text": entry["sheet"].loc[missing, "orads_raw"].tolist()}
        out["readers"][name] = result

    for label, col in CONFIGS.items():
        out["models"][label] = metrics(merged[col].to_numpy(), truth)

    for label, col in CONFIGS.items():
        keys = [f"{label}_vs_{name}" for name in calls]
        for key, name in zip(keys, calls, strict=True):
            out["model_vs_reader"][key] = mcnemar(merged[col].to_numpy(), calls[name], truth)
        adjusted = stats.holm([out["model_vs_reader"][k].pop("_p") for k in keys])
        for key, p_adj in zip(keys, adjusted, strict=True):
            out["model_vs_reader"][key]["holm_adjusted_p"] = round(float(p_adj), 4)
            out["model_vs_reader"][key]["significant_at_0.05_holm"] = bool(p_adj < 0.05)
    out["model_vs_reader"]["holm_family"] = ("three reader comparisons per model configuration, "
                                             "adjusted separately for config6 and config2")

    for a, b in combinations(calls, 2):
        out["reader_agreement"][f"{a}__{b}"] = {"cohen_kappa": stats.rounded(stats.cohen_kappa(calls[a], calls[b])),
                                                "percent_agreement": round(float((calls[a] == calls[b]).mean()), 4)}
    stacked = np.vstack(list(calls.values()))
    positive = stacked.sum(axis=0)
    out["reader_agreement"]["fleiss_kappa_all_three"] = stats.rounded(
        stats.fleiss_kappa(np.column_stack([len(calls) - positive, positive])))

    panel = (positive >= 2).astype(int)
    out["panel_majority_vote"] = metrics(panel, truth)
    for label, col in CONFIGS.items():
        test = mcnemar(merged[col].to_numpy(), panel, truth)
        test.pop("_p")
        out["panel_majority_vote"][f"mcnemar_{label}_vs_panel"] = test

    per_case = merged[["new_code", "original_patient_id", "class_name", "truth", "c6", "c2"]].rename(
        columns={"original_patient_id": "study_code"})
    for name, pred in calls.items():
        per_case[name] = pred
    per_case["panel_majority"] = panel
    return out, per_case


def figure3b(results: dict) -> None:
    """Accuracy and macro-F1 of the three readers, their majority vote and the two model configurations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rule = "rule_A_unreadable_counted_as_error"
    rows = [("Reader 1\n(co-author)", results["readers"]["reader1"][rule]),
            ("Reader 2\n(co-author)", results["readers"]["reader2"][rule]),
            ("Reader 3\n(not an author)", results["readers"]["reader3"][rule]),
            ("Majority vote\nof 3 readers", results["panel_majority_vote"]),
            ("Configuration 2\n(ROI + CNN)", results["models"]["config2"]),
            ("Configuration 6\n(ROI + CNN + 11 morph)", results["models"]["config6"])]
    adjusted = ", ".join(f"{results['model_vs_reader'][f'config6_vs_reader{i}']['holm_adjusted_p']:.3f}" for i in (1, 2, 3))
    fleiss = results["reader_agreement"]["fleiss_kappa_all_three"]

    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=400)
    ax.set_facecolor("#F7F9FC")
    for offset, key, label, color in ((-width / 2, "accuracy", "Accuracy (%)", "#1F5FE0"),
                                      (width / 2, "macro_f1", "Macro-F1 (%)", "#9DC8F5")):
        bars = ax.bar(x + offset, [100 * r[1][key] for r in rows], width, label=label, color=color, edgecolor="#123C8C")
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=10.5)
    ax.set_ylim(0, 104)
    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_title("Stage III, unaided expert readers versus model (patient level, test set, n = 21)", fontsize=15, pad=34)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=2, frameon=False, fontsize=12)
    ax.yaxis.grid(True, color="white", linewidth=1.2)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.012,
             "No model-versus-reader difference reaches significance after Holm correction across the three "
             f"comparisons (adjusted p = {adjusted}).\nReader 3 recorded three cases as not classifiable from the "
             f"image; these are counted as errors against that reader. Inter-reader agreement: Fleiss' kappa = {fleiss:.3f}.",
             ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    png = output_dir("figures") / "Figure_3b.png"
    fig.savefig(png, dpi=400, facecolor="white")
    plt.close(fig)
    Image.open(png).convert("RGB").save(png.with_suffix(".tif"), compression="tiff_lzw", dpi=(400, 400))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-figure", action="store_true", help="skip Figure 3b")
    args = parser.parse_args(argv)
    results, per_case = analyse()
    write_json(results, "three_reader_results.json")
    per_case.to_csv(output_dir() / "three_reader_per_case.csv", index=False)
    if not args.no_figure:
        figure3b(results)
    for name, r in results["readers"].items():
        m = r["rule_A_unreadable_counted_as_error"]
        print(f"{name}: accuracy {m['accuracy']}, macro-F1 {m['macro_f1']}")
    for key, v in results["model_vs_reader"].items():
        if isinstance(v, dict):
            print(f"{key}: p {v['exact_mcnemar_p']}, Holm {v['holm_adjusted_p']}")
    print("Fleiss kappa:", results["reader_agreement"]["fleiss_kappa_all_three"])


if __name__ == "__main__":
    main()
