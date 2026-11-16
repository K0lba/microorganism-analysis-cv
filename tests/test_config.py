from pathlib import Path

import cv2
import numpy as np
import pytest

from microtrack.io import Calibration, load_frames
from microtrack.preprocess import PreprocessConfig, denoise, preprocess
from microtrack.segmentation import SegmentationConfig, segment

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _load():
    yaml = pytest.importorskip("yaml") if hasattr(pytest, "importorskip") else __import__("yaml")
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_default_yaml_sections_are_accepted():
    cfg = _load()
    pre = PreprocessConfig.from_dict(cfg["preprocessing"])
    assert pre.enabled and pre.grayscale and pre.scale == 1.0
    assert pre.clahe is None and pre.denoise is None and pre.sharpness == 1.0   # enabled: false

    seg = SegmentationConfig.from_dict(cfg["segmentation"])
    assert seg.threshold.method == "otsu" and seg.threshold.objects == "auto"
    assert seg.morphology.open_ksize == 3 and seg.morphology.close_ksize == 3
    assert seg.filters.min_area == 3 and seg.filters.max_area is None
    assert not seg.split.enabled

    calib = Calibration.from_dict(cfg["calibration"])
    assert not calib.calibrated and calib.unit == "px"
    assert cfg["input"]["frame_stride"] == 1 and cfg["input"]["timestamp_source"] == "source_pts"


def test_enabled_flags_switch_operations_on():
    pre = PreprocessConfig.from_dict({
        "clahe": {"enabled": True, "clip_limit": 3.0, "tile_grid_size": 4},
        "denoise": {"enabled": True, "method": "median", "kernel_size": 5},
        "sharpen": {"enabled": True, "amount": 0.5},
    })
    assert pre.clahe.clip_limit == 3.0 and pre.clahe.tile_grid == 4
    assert pre.denoise.method == "median" and pre.denoise.kernel_size == 5
    assert pre.sharpness == pytest.approx(1.5)
    with pytest.raises(ValueError):
        PreprocessConfig.from_dict({"denoise": {"method": "median", "kernel_size": 4}})
    with pytest.raises(ValueError):
        PreprocessConfig.from_dict({"clahe": {"nope": 1}})
    with pytest.raises(ValueError):
        PreprocessConfig.from_dict({"sharpen": {"nope": 1}})


def test_preprocessing_disabled_returns_image_untouched():
    img = np.random.default_rng(0).integers(0, 255, (20, 20, 3), dtype=np.uint8)
    out, _ = preprocess(img, PreprocessConfig.from_dict({"enabled": False, "clahe": True}))
    assert np.array_equal(out, img)


def test_median_denoise_removes_impulse_noise():
    img = np.full((30, 30), 100, np.uint8)
    img[5, 5] = 255
    assert denoise(img, "median", kernel_size=3)[5, 5] == 100


def test_segmentation_aliases_from_project_yaml():
    seg = SegmentationConfig.from_dict({
        "method": "classical", "threshold_method": "adaptive",
        "morphology": {"enabled": False},
        "min_object_area_px": 10, "max_object_area_px": 500,
    })
    assert seg.threshold.method == "adaptive"
    assert seg.morphology.open_ksize == 0 and seg.morphology.close_ksize == 0
    assert not seg.morphology.fill_holes
    assert (seg.filters.min_area, seg.filters.max_area) == (10, 500)
    with pytest.raises(ValueError):
        SegmentationConfig.from_dict({"method": "unet"})


def test_um_filters_require_calibration():
    img = np.full((40, 40), 200, np.uint8)
    cv2.circle(img, (20, 20), 8, 60, -1)
    cfg = SegmentationConfig.from_dict({"filters": {"unit": "um", "min_area": 10}})
    with pytest.raises(ValueError):
        segment(img, cfg, Calibration.from_dict({"pixel_size_um": None}))
    assert len(segment(img, cfg, Calibration.from_dict({"pixel_size_um": 0.5}))) == 1


def test_calibration_from_dict_and_rescale_keeps_flag():
    c = Calibration.from_dict({"pixel_size_um": 0.25, "unit": "um"})
    assert c.calibrated and c.pixel_size_um == 0.25
    assert not Calibration.from_dict(None).rescaled(2.0).calibrated
    with pytest.raises(ValueError):
        Calibration.from_dict({"pixels": 1})


def test_timestamp_source_source_pts(tmp_path):
    path = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (32, 32))
    for i in range(16):
        writer.write(np.full((32, 32, 3), 5 * i, np.uint8))
    writer.release()
    pts = load_frames(path, step=4, timestamp_source="source_pts")
    fps = load_frames(path, step=4, timestamp_source="fps")
    assert [f.index for f in pts] == [0, 4, 8, 12]
    assert [f.time_s for f in pts] == pytest.approx([0.0, 0.5, 1.0, 1.5])
    assert [f.time_s for f in pts] == pytest.approx([f.time_s for f in fps])
    # явный fps приоритетнее отметок контейнера
    over = load_frames(path, step=4, timestamp_source="source_pts", fps=4.0)
    assert [f.time_s for f in over] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        load_frames(path, timestamp_source="wall")
