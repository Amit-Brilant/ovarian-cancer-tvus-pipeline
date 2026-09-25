import cv2
import numpy as np
import pytest

from ovarian.data import DESCRIPTORS
from ovarian.features import descriptors
from ovarian.segmentation import read_yolo_boxes, refine_mask

SIZE = 256


def disk_mask(cy: int, cx: int, radius: int, size: int = SIZE) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def test_circle_geometry():
    radius = 40
    mask = disk_mask(128, 128, radius)
    image = np.full((SIZE, SIZE), 100, dtype=np.uint8)
    f = descriptors(image, mask)

    assert list(f) == DESCRIPTORS
    assert f["area"] == mask.sum()
    assert f["area"] == pytest.approx(np.pi * radius**2, rel=0.01)
    assert f["perimeter"] == pytest.approx(2 * np.pi * radius, rel=0.05)
    assert f["circularity"] == pytest.approx(1.0, abs=0.1)
    assert f["eccentricity"] < 0.05
    assert f["aspect_ratio"] == pytest.approx(1.0, abs=0.01)
    assert f["solidity"] == pytest.approx(1.0, abs=0.02)
    assert f["extent"] == pytest.approx(mask.sum() / (2 * radius + 1) ** 2)
    assert f["area_ratio"] == pytest.approx(mask.sum() / SIZE**2)
    assert f["intensity_mean"] == 100.0
    assert f["intensity_std"] == 0.0
    assert f["entropy_mean"] == 0.0


def test_largest_component_and_ellipse():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[40:80, 30:230] = True
    mask |= disk_mask(200, 200, 10)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (SIZE, SIZE), dtype=np.uint8)
    f = descriptors(image, mask)

    assert f["area"] == 40 * 200
    assert f["extent"] == 1.0
    assert f["aspect_ratio"] == pytest.approx(5.0, rel=0.01)
    assert f["intensity_mean"] == pytest.approx(image[40:80, 30:230].mean())
    assert f["entropy_mean"] > 0


def test_empty_mask():
    f = descriptors(np.zeros((32, 32), np.uint8), np.zeros((32, 32), bool))
    assert all(v == 0.0 for v in f.values())


def test_refine_mask():
    ring = disk_mask(128, 128, 60) & ~disk_mask(128, 128, 20)
    blob = disk_mask(20, 20, 5)
    notch = np.zeros_like(ring)
    notch[126:130, 60:74] = True
    raw = (ring | blob) & ~notch
    assert raw[128, 128] == 0 and raw[128, 70] == 0 and raw[20, 20] == 1

    refined = refine_mask(raw.astype(np.uint8))
    assert refined.dtype == np.uint8
    assert set(np.unique(refined)) == {0, 1}
    assert refined[128, 128] == 0
    assert refined[20, 20] == 0
    assert refined[128, 70] == 1
    n_components = cv2.connectedComponents(refined, connectivity=8)[0] - 1
    assert n_components == 1
    np.testing.assert_array_equal(refine_mask(refined), refined)


def test_read_yolo_boxes(tmp_path):
    labels = tmp_path / "a.txt"
    labels.write_text("1 0.5 0.25 0.5 0.25\n0 0.99 0.5 0.1 0.2\nbad line\n0 0.5 0.5 0 0.1\n")
    boxes = read_yolo_boxes(labels, width=1280, height=640)
    assert boxes == [(320, 80, 960, 240), (1203, 256, 1279, 384)]
