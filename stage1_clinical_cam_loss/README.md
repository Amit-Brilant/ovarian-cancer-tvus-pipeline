# Stage I: normal-versus-pathological triage with the clinical CAM-loss

The Stage I model reported in the manuscript, published exactly as it was run. EfficientNet-B7 on the full
frame (224 x 224), trained with a clinical CAM-loss: class-weighted cross-entropy plus a term that penalises
class-activation-map (CAM) mass outside an automatically estimated region of interest,

    L = L_CE + lambda(epoch) * mean[ 1 - sum(A * M) / (sum(A) + eps) ],   A = softplus(true-class CAM)

with lambda = 0.25, ramped linearly over the first 5 epochs, and eps = 1e-8 (implemented as a floor on
sum(A), which is equivalent unless the CAM mass is itself below eps). The focus mask M is the
detector-derived region of interest (padded by 30 %, weight 1.0) with the rest of the ultrasound field at
weight 0.35 for pathological images, and the ultrasound field for normal images. The mask is used in the loss
only; at inference the network sees the image alone. Implementation: `utils/clinical_cam_loss.py`.

Augmentation: horizontal flip (p = 0.5), rotation up to 7 degrees, scaling 0.90 to 1.10, brightness 0.10 and
contrast 0.16 jitter; no translation, no vertical flip. AdamW (`torch.optim.AdamW`), learning rate 1e-4,
weight decay 1e-4, batch size 8, up to 50 epochs, early stopping on patient-level validation macro-F1 with
patience 12.

## Scope of this directory

- The reported model is the CAM-loss cell without artifact suppression (`cam_suppression_0`: attention
  weight 0.25, artifact suppression probability 0, zoom 0.90 to 1.10).
- The launcher `run_stage1_2x2_zoom_ablation.sh` also defines exploratory cells (cross-entropy only, and
  artifact suppression 0.5) that are not reported in the manuscript.
- The negative-control mask modes (`--attention-mask-mode shifted|random|permuted`) and the
  field-normalisation options (`--canonical-field-min-occupancy`, `--field-bbox-normalization`,
  `--suppress-artifacts-at-eval`) in the trainer are inert by default and were not reported.
- `stage1_effnet_ce_detector_gradcam.py` is included because the trainer imports helpers from it (dataset,
  split construction, detector boxes, metrics, Grad-CAM). Its own `main()` trains an earlier region-crop
  variant that is not the reported model.
- Retraining under further random seeds, reported in the manuscript as variability between training runs, is
  the same trainer invoked with a frozen patient-level split and a different `--seed`.

## Files

| File | Role |
|---|---|
| `stage1_effnet_ce_train_full_image_clinical_attention.py` | trainer |
| `utils/clinical_cam_loss.py` | the regulariser |
| `models/cam_features.py`, `models/networksVer3.py` | feature-map capture, EfficientNet wrapper |
| `stage1_effnet_ce_detector_gradcam.py`, `src/` | patient-level split, detector boxes, metrics, Grad-CAM |
| `run_stage1_2x2_zoom_ablation.sh` | the command that trained the reported model (cell `cam_suppression_0`) |

## Data layout

The loader (`src/data_loader.py`) expects one folder per class under `--data-root`, named `healthy`, `benign`
and `malignant`, and filenames of the form `PatientCode.ImageIndex.Date.png`, for example
`001.1.01-01-2020.png`:

```
/path/to/images/
  healthy/    001.1.01-01-2020.png  001.2.01-01-2020.png ...
  benign/     ...
  malignant/  ...
```

The public dataset uses `normal`, `benign` and `malignant` folders and its own filename scheme, described in
`../data/README.md`. Files must be renamed (and `normal` mapped to `healthy`), or the loader's
`parse_filename` function adapted, before running.

## Running

```bash
pip install -r requirements.txt
python stage1_effnet_ce_train_full_image_clinical_attention.py \
    --data-root /path/to/images --detector-checkpoint /path/to/stage2_yolov8m.pt \
    --output-root ./runs --attention-loss-weight 0.25 --attention-warmup-epochs 5 \
    --artifact-suppression-prob 0 --zoom-min 0.90 --zoom-max 1.10 \
    --seed <seed> --epochs 50 --patience 12 --batch-size 8
```

The files are published as they were run, with three exceptions that change no behaviour: the cluster paths
and the interpreter path are replaced by placeholders, no seed value is stored in the code (seeds are given
with `--seed` or the `SEED` environment variable), and the filename example in the docstring of
`src/parse_filename` uses a synthetic code and date instead of a real one. Pass the arguments above, or edit
the two variables at the top of the shell script (`PYTHON_BIN`, `PROJECT_ROOT`), before running elsewhere.
