"""Tables of the public data release: images, partitions, clinical variables and descriptors.

Image files are named <patient>.v<visit>.<image>.png; the patient is the first field.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ovarian import paths

CLASSES = ("normal", "benign", "malignant")
CONTINUOUS = ["Age", "Years_since_menopause", "Pregnancies_count", "Actual_births_count", "CA125_levels"]
BINARY = ["Background_diseases_Cancer", "Background_diseases_Diabetes", "Background_diseases_Hypertension",
          "Background_diseases_Ischemic_heart_disease", "Background_diseases_Dyslipidemia",
          "Family_history_breast", "Family_history_ovaries", "Family_history_uterus", "Family_history_other",
          "Smoking"]
DESCRIPTORS = ["area", "perimeter", "circularity", "eccentricity", "solidity", "extent", "aspect_ratio",
               "area_ratio", "intensity_mean", "intensity_std", "entropy_mean"]


def patient_of(image_name: str) -> str:
    return Path(image_name).name.split(".")[0]


def image_path(image_name: str, cls: str) -> Path:
    return paths.IMAGES / cls / image_name


def stage1_split() -> pd.DataFrame:
    """Per-image Stage I partition: image, class, patient, split (train, val, test)."""
    return pd.read_csv(paths.SPLITS / "split_stage1_images.csv", dtype={"patient": str})


def stage3_split() -> pd.DataFrame:
    """Per-image Stage III partition over the benign and malignant images."""
    return pd.read_csv(paths.SPLITS / "split_stage3_images.csv", dtype={"patient": str})


def clinical() -> pd.DataFrame:
    """One row per pathological patient with the 15 clinical variables and the label (malignant = 1).

    CA-125 recorded as 0 (five benign patients) is physiologically impossible and is returned as missing.
    """
    frames = []
    for label, cls in enumerate(("benign", "malignant")):
        df = pd.read_excel(paths.CLINICAL / f"processed_{cls}.xlsx")
        frames.append(df.assign(label=label))
    df = pd.concat(frames, ignore_index=True)
    df["patient"] = df.Patient_code.astype(int).astype(str).str.zfill(3)
    df["CA125_levels"] = df.CA125_levels.astype(float).where(df.CA125_levels > 0)
    return df


def descriptors() -> pd.DataFrame:
    """The eleven morphological descriptors per pathological image, with class and patient."""
    frames = []
    for cls in ("benign", "malignant"):
        df = pd.read_csv(paths.FEATURES / f"morphological_features_{cls}.csv")
        frames.append(df.assign(cls=cls))
    df = pd.concat(frames, ignore_index=True).rename(columns={"patient_id": "stem"})
    df["image"] = df.stem + ".png"
    df["patient"] = df.stem.map(patient_of)
    return df


def log1p_ca125(values: pd.Series) -> np.ndarray:
    return np.log1p(values.astype(float))
