# Data layout

The code expects the public data release, unpacked into one folder, with `OVARIAN_DATA` pointing at it.
Image files are named `<patient>.v<visit>.<image>.png`; the patient code is the first field.

```
$OVARIAN_DATA/
  images/{normal,benign,malignant}/        1,082 transvaginal ultrasound images, 1280 x 1280 PNG
  yolo_labels/{benign,malignant}_labels/   cyst bounding boxes, YOLO format, one file per pathological image
  clinical/processed_{benign,malignant}.xlsx
                                           15 clinical variables per pathological patient
  features/morphological_features_{benign,malignant}.csv
                                           the eleven morphological descriptors per pathological image,
                                           from the masks prompted with the annotation boxes
  features/automatic_morphological_features_{val,test}.csv
                                           the same descriptors from the masks prompted with the detector
                                           boxes; these are the inputs of Configuration 6
  detector_boxes/detector_boxes_{val,test}.json
                                           the Stage II box of each image, with the full-frame fallback
                                           where the detector found nothing
  splits/split_stage1_images.csv           per-image Stage I partition (train, val, test)
  splits/split_stage3_images.csv           per-image Stage III partition
  splits/split_analysis.csv                per-patient Stage III partition summary
  predictions/                             per-image and per-patient predictions of the reported models
  sam_masks/{benign,malignant}/            not distributed; regenerate with scripts/make_sam_masks.py
```

## Cohort and partitions

| Group | Patients | Images |
|---|:-:|:-:|
| Sonographically normal | 131 | 529 |
| Benign cyst | 68 | 291 |
| Malignant cyst | 61 | 262 |
| Total | 260 | 1,082 |

The two stages were split independently at patient level (about 70 / 15 / 15), each stratified over its own
denominator, so their test cohorts are different patients.

| Stage | Train | Validation | Test |
|---|:-:|:-:|:-:|
| Stage I (all patients), images | 753 | 143 | 186 |
| Stage III (pathological patients), patients / images | 89 / 353 | 19 / 104 | 21 / 96 |

## Clinical variables

Continuous: age, years since menopause, pregnancies, live births, CA-125 (U/mL). Binary: personal history of
cancer, diabetes, hypertension, ischaemic heart disease, dyslipidaemia, family history of breast, ovarian,
uterine or other cancer, and smoking. CA-125 was measured preoperatively in the pathological group only; a
value recorded as 0 (five benign patients) is treated as missing by `ovarian.data.clinical()`.

## Weights

`OVARIAN_WEIGHTS` points at the folder with the released model weights (Stage I, Stage II detector, Stage III)
and, for mask regeneration, the public SAM ViT-H checkpoint `sam_vit_h_4b8939.pth` from Meta AI.

```
$OVARIAN_WEIGHTS/
  stage1_efficientnet_b7.pth               Stage I triage classifier
  stage2_yolov8m.pt                        Stage II cyst detector
  stage3_config2_baseline.pth              Stage III, image only
  stage3_config6_morphological_fusion.pth  Stage III, image and the eleven descriptors
  stage3_config6_feature_normalization.json
                                           descriptor means and standard deviations, fitted on the
                                           training images only, applied before Configuration 6
  sam_vit_h_4b8939.pth                     public SAM ViT-H checkpoint (Meta AI), for mask regeneration
```

## Descriptors: two mask sources

The lesion masks come from SAM prompted with a box. The released `morphological_features_*` files use the
manual annotation box, and are the descriptors behind the clinical analyses. Configuration 6 instead uses
the box predicted by the Stage II detector, so that every input is produced by the pipeline itself; those
descriptors are released per split as `automatic_morphological_features_<split>.csv`. The two sets are
correlated but not interchangeable (Spearman 0.61 to 0.81 across the descriptors), so reproducing the
reported Stage III numbers requires the automatic files.
