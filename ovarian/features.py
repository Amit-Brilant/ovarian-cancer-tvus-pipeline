"""The eleven morphological descriptors of a lesion mask on a grayscale ultrasound image."""
from pathlib import Path

import cv2
import numpy as np
from skimage import measure
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

from ovarian.data import DESCRIPTORS

ENTROPY_RADIUS = 3


def read_gray(path: str | Path) -> np.ndarray:
    """Read an image as 8-bit grayscale (OpenCV conversion, as used for the released descriptors)."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_mask(path: str | Path) -> np.ndarray:
    """Read a mask image as a boolean array (any nonzero pixel is foreground)."""
    return read_gray(path) > 0


def entropy_map(image: np.ndarray) -> np.ndarray:
    """Local Shannon entropy (bits) in a disk of radius 3; non-uint8 input is min-max scaled to uint8 first."""
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        low, high = image.min(), image.max()
        image = (image - low) / (high - low) if high > low else np.zeros_like(image)
        image = img_as_ubyte(image)
    return entropy(image, disk(ENTROPY_RADIUS))


def descriptors(image: np.ndarray, mask: np.ndarray, entropy_image: np.ndarray | None = None) -> dict[str, float]:
    """The eleven descriptors of the largest connected component of `mask` on grayscale `image`.

    `entropy_image` may be passed to reuse a precomputed `entropy_map(image)`. An empty mask gives zeros.
    """
    image = np.asarray(image)
    mask = np.asarray(mask).astype(bool)
    if image.ndim != 2 or mask.ndim != 2:
        raise ValueError("image and mask must be 2D")
    if image.shape != mask.shape:
        raise ValueError(f"shape mismatch: image {image.shape}, mask {mask.shape}")
    if not mask.any():
        return dict.fromkeys(DESCRIPTORS, 0.0)

    labeled = measure.label(mask)
    largest = max(measure.regionprops(labeled), key=lambda r: r.area)
    region = labeled == largest.label
    props = measure.regionprops(region.astype(np.uint8), intensity_image=image)[0]
    if entropy_image is None:
        entropy_image = entropy_map(image)

    h, w = image.shape
    area = float(props.area)
    perimeter = float(props.perimeter) if props.perimeter > 0 else 0.0
    minor, major = float(props.axis_minor_length), float(props.axis_major_length)
    pixels = image[region]
    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": 4.0 * np.pi * area / perimeter**2 if perimeter > 0 else 0.0,
        "eccentricity": float(props.eccentricity),
        "solidity": float(props.solidity),
        "extent": float(props.extent),
        "aspect_ratio": major / minor if minor > 0 else 0.0,
        "area_ratio": area / (h * w),
        "intensity_mean": float(np.mean(pixels)),
        "intensity_std": float(np.std(pixels)),
        "entropy_mean": float(np.mean(entropy_image[region])),
    }


def descriptors_from_files(image_path: str | Path, mask_path: str | Path) -> dict[str, float]:
    """Descriptors of one image file and its mask file."""
    return descriptors(read_gray(image_path), read_mask(mask_path))
