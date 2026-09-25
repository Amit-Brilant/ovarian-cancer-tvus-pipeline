"""
Data loading and preprocessing pipeline for ovarian carcinoma detection.

CRITICAL:
This module performs splitting at PATIENT level (patient_code) to prevent leakage.
The same patient must not appear in different splits.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from src.config import DataConfig


def parse_filename(filepath: str) -> Tuple[str, int, str, int]:
    """
    Filename format: PatientCode.ImageNumber.VisitDate.ext
    Example: 001.1.01-01-2020.png
    """
    filename = Path(filepath).stem
    parts = filename.split(".")

    if len(parts) != 3:
        raise ValueError(
            f"Invalid filename format: {filename}. "
            f"Expected PatientCode.ImageNumber.VisitDate"
        )

    patient_code = parts[0]
    image_number = int(parts[1])
    visit_date = parts[2]

    parent_dir = Path(filepath).parent.name.lower()
    if parent_dir == "healthy":
        label = 0
    elif parent_dir in ["benign", "malignant"]:
        label = 1
    else:
        raise ValueError(
            f"Unknown directory: {parent_dir}. "
            f"Expected 'healthy', 'benign', or 'malignant'"
        )

    return patient_code, image_number, visit_date, label


def create_patient_visit_dataframe(data_root: str) -> pd.DataFrame:
    """
    Backward-compatible function name.
    Builds image-level dataframe with patient metadata and binary labels.
    """
    rows = []
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    for class_dir in ["healthy", "benign", "malignant"]:
        class_path = Path(data_root) / class_dir
        if not class_path.exists():
            print(f"Warning: Directory not found: {class_path}")
            continue

        for img_file in class_path.iterdir():
            if not img_file.is_file() or img_file.suffix.lower() not in valid_exts:
                continue

            try:
                patient_code, image_number, visit_date, label = parse_filename(str(img_file))
                rows.append(
                    {
                        "filepath": str(img_file),
                        "patient_code": patient_code,
                        "image_number": image_number,
                        "visit_date": visit_date,
                        "patient_visit_id": f"{patient_code}_{visit_date}",
                        "label": label,
                    }
                )
            except Exception as e:
                print(f"Error parsing {img_file.name}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No valid images found under: {data_root}")

    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"Total images: {len(df)}")
    print(f"Unique patient-visits: {df['patient_visit_id'].nunique()}")
    print(f"Unique patients: {df['patient_code'].nunique()}")
    print(f"\nLabel distribution:")
    print(f"  Healthy (0): {(df['label'] == 0).sum()} images")
    print(f"  Sick (1):    {(df['label'] == 1).sum()} images")
    print("=" * 70 + "\n")

    return df


def _safe_split(df: pd.DataFrame, test_size: float, random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Try stratified split by label; fallback to random split if not feasible.
    """
    try:
        return train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=random_seed,
        )
    except ValueError as e:
        print(f"Warning: stratified split failed ({e}). Falling back to random split.")
        return train_test_split(df, test_size=test_size, random_state=random_seed)


def stratified_patient_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by PATIENT (patient_code), stratified by patient-level label.

    patient-level label:
      if patient has any sick image => sick(1), else healthy(0)
    """
    if random_seed is None:
        raise ValueError("random_seed is required")
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total_ratio}"

    patient_summary = (
        df.groupby("patient_code")
        .agg(label=("label", "max"), n_images=("filepath", "count"))
        .reset_index()
    )

    print(f"Splitting {len(patient_summary)} patients...")

    train_val_pat, test_pat = _safe_split(
        patient_summary,
        test_size=test_ratio,
        random_seed=random_seed,
    )

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_pat, val_pat = _safe_split(
        train_val_pat,
        test_size=val_ratio_adjusted,
        random_seed=random_seed,
    )

    train_ids = set(train_pat["patient_code"])
    val_ids = set(val_pat["patient_code"])
    test_ids = set(test_pat["patient_code"])

    # Hard leakage checks (patient-level)
    assert len(train_ids & val_ids) == 0, "CRITICAL ERROR: Train-Val patient overlap detected!"
    assert len(train_ids & test_ids) == 0, "CRITICAL ERROR: Train-Test patient overlap detected!"
    assert len(val_ids & test_ids) == 0, "CRITICAL ERROR: Val-Test patient overlap detected!"

    train_df = df[df["patient_code"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["patient_code"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["patient_code"].isin(test_ids)].reset_index(drop=True)

    print("\n" + "=" * 70)
    print("Data Split Summary (PATIENT LEVEL)")
    print("=" * 70)
    print(f"Train: {len(train_ids)} patients, {len(train_df)} images")
    print(f"  - Healthy: {(train_df['label']==0).sum()}, Sick: {(train_df['label']==1).sum()}")
    print(f"\nValidation: {len(val_ids)} patients, {len(val_df)} images")
    print(f"  - Healthy: {(val_df['label']==0).sum()}, Sick: {(val_df['label']==1).sum()}")
    print(f"\nTest: {len(test_ids)} patients, {len(test_df)} images")
    print(f"  - Healthy: {(test_df['label']==0).sum()}, Sick: {(test_df['label']==1).sum()}")
    print("=" * 70 + "\n")

    print("✓ Split check passed - no patient overlap detected\n")
    return train_df, val_df, test_df


def stratified_patient_visit_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible function name.
    Internally uses patient-level split.
    """
    return stratified_patient_split(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )


def verify_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Explicit leakage checks at:
    1) patient level (patient_code)
    2) exam/visit level (patient_visit_id)
    3) file level (filepath)
    """
    checks = [
        ("patient_code", "Patient-level"),
        ("patient_visit_id", "Visit-level"),
        ("filepath", "Image-level"),
    ]

    print("=" * 70)
    print("Leakage Checks")
    print("=" * 70)

    for col, title in checks:
        tr = set(train_df[col].dropna().astype(str).tolist())
        va = set(val_df[col].dropna().astype(str).tolist())
        te = set(test_df[col].dropna().astype(str).tolist())

        tr_va = len(tr & va)
        tr_te = len(tr & te)
        va_te = len(va & te)

        print(f"{title} ({col}):")
        print(f"  train-val overlap: {tr_va}")
        print(f"  train-test overlap: {tr_te}")
        print(f"  val-test overlap: {va_te}")

        assert tr_va == 0, f"CRITICAL ERROR: {col} overlap train-val!"
        assert tr_te == 0, f"CRITICAL ERROR: {col} overlap train-test!"
        assert va_te == 0, f"CRITICAL ERROR: {col} overlap val-test!"

    print("✓ Split check passed - no overlap at patient/visit/image levels\n")


class OvarianDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        return_metadata: bool = False,
        metadata_key: str = "patient_code",
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_metadata = return_metadata
        self.metadata_key = metadata_key

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Keep float32 for compatibility with existing training code.
        label = torch.tensor(row["label"], dtype=torch.float32)

        if self.return_metadata:
            return image, label, row[self.metadata_key]
        return image, label


def get_transforms(config: DataConfig, is_training: bool = True) -> transforms.Compose:
    if is_training and config.use_augmentation:
        transform_list = [transforms.Resize(config.image_size)]

        if config.aug_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if config.aug_vertical_flip:
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        if config.aug_rotation:
            transform_list.append(transforms.RandomRotation(degrees=config.aug_rotation_degrees))
        if config.aug_color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=config.aug_color_jitter_brightness,
                    contrast=config.aug_color_jitter_contrast,
                    saturation=config.aug_color_jitter_saturation,
                    hue=config.aug_color_jitter_hue,
                )
            )
        if config.aug_gaussian_blur:
            transform_list.append(
                transforms.GaussianBlur(
                    kernel_size=config.aug_gaussian_blur_kernel,
                    sigma=(0.1, 2.0),
                )
            )

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if config.aug_random_erasing:
            transform_list.append(transforms.RandomErasing(p=config.aug_random_erasing_prob))
    else:
        transform_list = [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    return transforms.Compose(transform_list)


def create_dataloaders(
    data_config: DataConfig,
    flag_patient_visit_id: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main entry point.
    Uses PATIENT-level split (no patient appears in more than one split).

    If flag_patient_visit_id=True, metadata returned from val/test is patient_code.
    """
    np.random.seed(data_config.seed)
    torch.manual_seed(data_config.seed)

    df = create_patient_visit_dataframe(data_config.data_root)

    train_df, val_df, test_df = stratified_patient_split(
        df=df,
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        test_ratio=data_config.test_ratio,
        random_seed=data_config.seed,
    )
    verify_split_leakage(train_df, val_df, test_df)

    train_transform = get_transforms(data_config, is_training=True)
    val_transform = get_transforms(data_config, is_training=False)
    test_transform = get_transforms(data_config, is_training=False)

    metadata_key = "patient_code" if flag_patient_visit_id else "patient_visit_id"

    train_dataset = OvarianDataset(
        train_df,
        transform=train_transform,
        return_metadata=False,
        metadata_key=metadata_key,
    )
    val_dataset = OvarianDataset(
        val_df,
        transform=val_transform,
        return_metadata=flag_patient_visit_id,
        metadata_key=metadata_key,
    )
    test_dataset = OvarianDataset(
        test_df,
        transform=test_transform,
        return_metadata=flag_patient_visit_id,
        metadata_key=metadata_key,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=data_config.shuffle,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )

    print("DataLoaders created successfully:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}\n")

    return train_loader, val_loader, test_loader, train_df, val_df, test_df
