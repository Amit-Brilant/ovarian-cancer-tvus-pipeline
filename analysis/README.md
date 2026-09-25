# Analyses

Each analysis writes JSON (and, where noted, CSV or figures) to `$OVARIAN_RESULTS/analysis/`; the
manuscript numbers are taken from those files. Run from the repository root:

```bash
export OVARIAN_DATA=/path/to/zenodo_release
export OVARIAN_RESULTS=/path/to/results
export OVARIAN_PRIVATE=/path/to/private_inputs   # optional, see below
python -m analysis.run_all --seed <seed>          # or one module, e.g. python -m analysis.readers
```

| Manuscript object | Command | Output |
|---|---|---|
| Three-reader study: Table 3, Cohen and Fleiss kappa, exact McNemar with Holm adjustment | `python -m analysis.readers` | `three_reader_results.json`, `three_reader_per_case.csv` |
| Figure 3b (readers versus model; Holm-adjusted p-values and Fleiss kappa in the footnote) | `python -m analysis.readers` | `figures/Figure_3b.png`, `figures/Figure_3b.tif` |
| Clinical baselines: CA-125, CA-125 with age, age; raw CA-125 cutoff sweep, 20 to 80 U/mL | `python -m analysis.clinical_baselines` | `clinical_baselines.json`, `clinical_baselines_validation.json` |
| Supplementary Table S2, raw and false-discovery-rate adjusted p-values | `python -m analysis.clinical_baselines` | `tableS2_fdr.json` |
| Calibration of the Stage III probabilities; signal in the morphological descriptors; split counts | `python -m analysis.calibration_importance` | `calibration_and_importance.json` |
| Share of CA-125 in the mean absolute SHAP value, per attribution scope | `python -m analysis.shap_scope` | `shap_original_scope.json` |
| Supplementary Table S3 and Stage I variability across training runs | `python -m analysis.stage1_variability` | `stage1_table_s3.json` |
| Clinical XGBoost pre-classifiers (15 and 5 features), SHAP, PCA of the continuous variables | `python scripts/train_clinical_xgboost.py` | `$OVARIAN_RESULTS/clinical_xgboost/` |

Statistical helpers shared by all analyses are in `ovarian/stats.py`.

## Non-public inputs

Some analyses need inputs that are not part of the public data release: the reader spreadsheets, the
per-patient Stage III predictions on the test patients, the Stage I metrics of the training runs and the stored
SHAP values of the released clinical model. They are available from the corresponding author on
reasonable request. Place them under `$OVARIAN_PRIVATE` (default `private/` in the repository):

```
$OVARIAN_PRIVATE/
  reader_study/reader1.xlsx, reader2.xlsx, reader3.xlsx
      one row per test case: "Patient code" (case number 1 to 21) and "O-RADS score (0-5)"
  test21_predictions.csv
      one row per test patient: new_code (case number), original_patient_id (patient code),
      class_name, truth, c6, c6s, c2, c2s (binary prediction and probability of configurations 6 and 2)
  stage1_scores/metrics.csv
      one row per training run, split and level: split, level, accuracy, macro_f1,
      sensitivity_sick, specificity_healthy, roc_auc
  xgb_output/shap_values.csv, evaluation_results.csv
      SHAP values of the released clinical model, one row per patient (Patient_code, label, Split, features)
```

What runs without them:

- `clinical_baselines`: the three baselines, the cutoff sweep and Table S2 (the comparison with the
  Stage III models and reader 1 is skipped).
- `calibration_importance`: descriptor signal and counts (calibration is skipped).
- `stage1_variability`: Table S3, from the released predictions (the variability summary is skipped).
- `shap_scope --shap-dir $OVARIAN_RESULTS/clinical_xgboost/15_features`: the shares of a model trained
  with `scripts/train_clinical_xgboost.py`.
- `readers` needs the reader spreadsheets and the test predictions.
