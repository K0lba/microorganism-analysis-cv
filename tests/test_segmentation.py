import cv2
import numpy as np
import pytest

from microtrack.io import Calibration, Frame
from microtrack.preprocess import PreprocessConfig
from microtrack.segmentation import (
    FilterConfig,
    SegmentationConfig,
    SplitConfig,
    ThresholdConfig,
    segment,
    segment_frame,
    split_touching,
)


def make_scene(shapes, size=(160, 200), dark=True, noise=4.0, gradient=0.0, seed=0):
    """shapes: [(cx, cy, a, b, angle)] — эллипсы (полуоси a, b)."""
    h, w = size
    rng = np.random.default_rng(seed)
    bg, fg = (200.0, 80.0) if dark else (60.0, 180.0)
    img = np.full((h, w), bg, np.float32)
    if gradient:
        img += np.linspace(-gradient, gradient, w, dtype=np.float32)[None, :]
    m = np.zeros((h, w), np.uint8)
    for cx, cy, a, b, ang in shapes:
        cv2.ellipse(m, (cx, cy), (a, b), ang, 0, 360, 255, -1)
    img[m > 0] = fg + (img[m > 0] - bg) * 0  # объекты однотонные
    if gradient:
        img[m > 0] += np.linspace(-gradient, gradient, w, dtype=np.float32)[None, :].repeat(h, 0)[m > 0]
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


THREE = [(40, 40, 12, 8, 0), (110, 60, 14, 6, 30), (60, 115, 10, 10, 0)]


def test_otsu_dark_objects_count_and_masks():
    res = segment(make_scene(THREE))
    assert len(res) == 3
    assert res.mask.dtype == np.uint8 and set(np.unique(res.mask)) <= {0, 255}
    assert res.labels.dtype == np.int32 and res.labels.max() == 3
    assert res.mask.shape == res.gray.shape
    # площадь близка к площади эллипса
    areas = sorted(o.area_px for o in res.objects)
    expected = sorted(np.pi * a * b for _, _, a, b, _ in THREE)
    assert areas == pytest.approx(expected, rel=0.12)


def test_auto_polarity_bright_objects():
    res = segment(make_scene(THREE, dark=False))
    assert len(res) == 3


def test_explicit_polarity_wrong_gives_background_blob():
    cfg = SegmentationConfig(threshold=ThresholdConfig(objects="bright"))
    res = segment(make_scene(THREE, dark=True), cfg)
    assert len(res) != 3  # инвертированная маска не даёт 3 объекта


def test_adaptive_handles_uneven_illumination():
    img = make_scene(THREE, gradient=50.0)
    otsu = segment(img)
    cfg = SegmentationConfig(threshold=ThresholdConfig(method="adaptive", block_size=61, offset=10))
    adapt = segment(img, cfg)
    assert len(adapt) == 3
    assert len(otsu) != 3 or sum(o.area_px for o in otsu.objects) > 1.3 * sum(o.area_px for o in adapt.objects)


def test_fixed_threshold():
    cfg = SegmentationConfig(threshold=ThresholdConfig(method="fixed", fixed_value=140, objects="dark"))
    assert len(segment(make_scene(THREE), cfg)) == 3


def test_color_input_is_converted():
    gray = make_scene(THREE)
    assert len(segment(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))) == 3


def test_area_filter_removes_specks_and_reports_reason():
    shapes = THREE + [(150, 140, 3, 3, 0)]            # пятно ~28 px
    res = segment(make_scene(shapes), SegmentationConfig(filters=FilterConfig(min_area=60)))
    assert len(res) == 3 and res.rejected.get("min_area", 0) == 1
    res = segment(make_scene(shapes), SegmentationConfig(filters=FilterConfig(min_area=10)))
    assert len(res) == 4
    res = segment(make_scene(THREE), SegmentationConfig(filters=FilterConfig(max_area=300)))
    assert all(o.area_px <= 300 for o in res.objects) and res.rejected.get("max_area", 0) >= 1


def test_geometry_filters():
    shapes = [(40, 40, 10, 10, 0), (120, 60, 30, 5, 0)]   # круг и палочка
    img = make_scene(shapes)
    circles = segment(img, SegmentationConfig(filters=FilterConfig(min_circularity=0.8)))
    assert len(circles) == 1 and circles.objects[0].aspect_ratio < 1.3
    rods = segment(img, SegmentationConfig(filters=FilterConfig(min_length=40)))
    assert len(rods) == 1 and rods.objects[0].aspect_ratio > 4
    thin = segment(img, SegmentationConfig(filters=FilterConfig(max_aspect_ratio=2.0)))
    assert len(thin) == 1
    wide = segment(img, SegmentationConfig(filters=FilterConfig(min_width=15)))
    assert len(wide) == 1
    for o in segment(img).objects:
        assert 0 < o.circularity <= 1 and 0 < o.solidity <= 1


def test_exclude_border():
    shapes = THREE + [(2, 80, 10, 10, 0)]
    assert len(segment(make_scene(shapes))) == 4
    res = segment(make_scene(shapes), SegmentationConfig(filters=FilterConfig(exclude_border=True)))
    assert len(res) == 3 and res.rejected.get("border") == 1


def test_units_um_use_calibration():
    img = make_scene(THREE)
    calib = Calibration(pixel_size_um=0.5)             # 1 px² = 0.25 um²
    cfg = SegmentationConfig(filters=FilterConfig(unit="um", min_area=100.0))  # = 400 px
    res = segment(img, cfg, calib)
    assert all(o.area_px >= 400 for o in res.objects) and len(res) < 3
    o = segment(img, calibration=calib).objects[0]
    assert o.area_um2 == pytest.approx(o.area_px * 0.25)


def test_morphology_fills_holes_and_removes_noise():
    img = make_scene([(60, 60, 25, 25, 0)], noise=0.0)
    cv2.circle(img, (60, 60), 6, 200, -1)              # светлая дырка внутри
    img[10, 10] = 0                                    # одиночный тёмный пиксель
    res = segment(img)
    assert len(res) == 1
    assert res.mask[60, 60] == 255                     # дырка заполнена
    no_fill = segment(img, SegmentationConfig.from_dict({"morphology": {"fill_holes": False}}))
    assert no_fill.mask[60, 60] == 0


def test_object_info_consistency():
    res = segment(make_scene(THREE))
    for o in res.objects:
        assert (res.labels == o.label).sum() == o.area_px
        x, y, w, h = o.bbox
        assert x <= o.centroid[0] <= x + w and y <= o.centroid[1] <= y + h
        assert o.contour.ndim == 2 and o.contour.shape[1] == 2
        assert o.length_px >= o.width_px > 0


def test_empty_image():
    res = segment(np.full((50, 60), 120, np.uint8))
    assert len(res) == 0 and res.mask.sum() == 0 and res.labels.max() == 0


TOUCHING = [(70, 60, 16, 16, 0), (98, 60, 16, 16, 0)]   # два круга с перекрытием


def test_split_touching_experimental():
    img = make_scene(TOUCHING)
    assert len(segment(img)) == 1                        # без разделения — один объект
    cfg = SegmentationConfig(split=SplitConfig(enabled=True))
    res = segment(img, cfg)
    assert len(res) == 2
    a, b = sorted(o.area_px for o in res.objects)
    assert b / a < 1.3                                   # объекты примерно равны


def test_split_does_not_cut_single_objects():
    img = make_scene([(40, 40, 14, 14, 0), (110, 60, 28, 7, 20)])  # круг и палочка
    res = segment(img, SegmentationConfig(split=SplitConfig(enabled=True)))
    assert len(res) == 2


def test_split_touching_function_on_mask():
    mask = np.zeros((100, 140), np.uint8)
    cv2.circle(mask, (50, 50), 18, 255, -1)
    cv2.circle(mask, (80, 50), 18, 255, -1)
    out = split_touching(mask)
    n, _ = cv2.connectedComponents((out > 0).astype(np.uint8), connectivity=4)
    assert n - 1 == 2
    assert out.sum() < mask.sum()                        # линия разреза убрала пиксели


def test_config_from_dict_and_validation():
    cfg = SegmentationConfig.from_dict({
        "threshold": {"method": "adaptive", "block_size": 41},
        "filters": {"min_area": 50, "min_circularity": 0.5},
        "split": True,
    })
    assert cfg.threshold.method == "adaptive" and cfg.split.enabled
    for bad in ({"threshold": {"method": "x"}}, {"threshold": {"block_size": 10}},
                {"filters": {"unit": "mm"}}, {"filters": {"min_area": 10, "max_area": 5}},
                {"nope": {}}, {"threshold": {"nope": 1}}, {"split": {"peak_fraction": 0}}):
        with pytest.raises(ValueError):
            SegmentationConfig.from_dict(bad)


def test_segment_frame_with_preprocess_rescales_calibration():
    img = make_scene(THREE)
    frame = Frame(img, index=3, time_s=0.3, calibration=Calibration(0.4))
    res = segment_frame(frame, preprocess_cfg=PreprocessConfig(scale=0.5, median_ksize=3))
    assert res.gray.shape == (80, 100)
    assert res.calibration.pixel_size_um == pytest.approx(0.8)
    assert len(res) == 3
