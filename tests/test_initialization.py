import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import init_target as CLI  # noqa: E402

from microtrack.initialization import (  # noqa: E402
    InitConfig, InitializationError, InitSession, Selection, initialize_target,
    load_target, render_overlay, save_target,
)
from microtrack.io import Calibration  # noqa: E402
from microtrack.preprocess import PreprocessConfig  # noqa: E402
from microtrack.segmentation import describe_mask  # noqa: E402
from microtrack.segmentation.classical import SplitConfig  # noqa: E402
from microtrack.segmentation.metrics import iou  # noqa: E402
from microtrack.segmentation.pretrained import Prompts, Segmenter, prompts_from_labels  # noqa: E402
from microtrack.synthetic import MODEL_SCENARIOS, generate_scene  # noqa: E402

SPECS = {s.name: s for s in MODEL_SCENARIOS}


def scene(name="clean", seed=1):
    img, gt = generate_scene(SPECS[name], seed)
    return img, gt, prompts_from_labels(gt)


def best(img, sel, **kw):
    return initialize_target(img, sel, kw.pop("config", None), **kw)


# --------------------------------------------------------------------------- #
# Selection и конфигурация
# --------------------------------------------------------------------------- #
def test_selection_box_is_normalized_and_clipped():
    s = Selection.from_box(50, 40, 10, 20)
    assert s.box == (10.0, 20.0, 50.0, 40.0)
    v = Selection.from_box(-10, -5, 500, 90).validated((100, 200))
    assert v.box == (0.0, 0.0, 200.0, 90.0)


def test_selection_validation_errors():
    with pytest.raises(ValueError):
        Selection.from_point(300, 10).validated((100, 200))
    with pytest.raises(ValueError):
        Selection.from_box(10, 10, 11, 12).validated((100, 100))
    with pytest.raises(ValueError):
        Selection.from_roi([(1, 1), (5, 5)])
    with pytest.raises(ValueError):
        Selection.from_roi([(10, 10), (20, 20), (30, 30)]).validated((100, 100))   # вырожденный
    with pytest.raises(ValueError):
        Selection("lasso").validated((10, 10))


def test_selection_roundtrip_and_regions():
    for s in (Selection.from_point(3, 4), Selection.from_box(1, 2, 30, 40),
              Selection.from_roi([(5, 5), (50, 8), (30, 40)])):
        assert Selection.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    assert Selection.from_box(10, 10, 30, 25).region_mask((50, 50)).sum() == 20 * 15
    tri = Selection.from_roi([(10, 10), (40, 10), (10, 40)])
    assert 400 < tri.region_mask((60, 60)).sum() < 500
    assert tri.bounding_box() == (10.0, 10.0, 40.0, 40.0)
    with pytest.raises(ValueError):
        Selection.from_point(1, 1).region_mask((5, 5))


def test_init_config_from_dict_and_validation():
    cfg = InitConfig.from_dict({"method": "grabcut", "threshold": {"objects": "dark"},
                                "split": {"enabled": True, "peak_fraction": 0.8}, "min_area": 30,
                                "preprocess": {"median_ksize": 3}})
    assert cfg.method == "grabcut" and cfg.threshold.objects == "dark" and cfg.split.enabled
    assert cfg.preprocess.median_ksize == 3
    for bad in ({"method": "magic"}, {"nope": 1}, {"threshold": {"nope": 1}}, {"max_area_frac": 0},
                {"threshold": {"objects": "grey"}}, {"smooth_sigma": -1}, {"window_start": 2}):
        with pytest.raises(ValueError):
            InitConfig.from_dict(bad)


def test_default_yaml_initialization_section_is_accepted():
    yaml = pytest.importorskip("yaml") if hasattr(pytest, "importorskip") else __import__("yaml")
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = InitConfig.from_dict(yaml.safe_load(path.read_text(encoding="utf-8"))["initialization"])
    assert cfg.method == "auto" and cfg.threshold.method == "otsu" and not cfg.split.enabled


def test_describe_mask_takes_largest_component():
    m = np.zeros((40, 40), np.uint8)
    m[2:6, 2:6] = 255
    m[15:35, 10:30] = 255
    info = describe_mask(m, Calibration(0.5))
    assert info.area_px == 400 and info.area_um2 == pytest.approx(100)
    assert info.centroid == pytest.approx((19.5, 24.5))
    with pytest.raises(ValueError):
        describe_mask(np.zeros((5, 5)))


# --------------------------------------------------------------------------- #
# Классический алгоритм: точка
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["clean", "noisy", "uneven_light", "bright_on_dark", "low_contrast",
                                  "small_objects", "rods", "dense"])
def test_point_selects_clicked_object(name):
    img, gt, pr = scene(name)
    for k in (0, 1):
        t = best(img, Selection.from_point(*pr.points[k]))
        g = gt == np.unique(gt[gt > 0])[k]
        assert iou(t.mask > 0, g) > 0.8
        assert abs(t.area_px - g.sum()) < 0.25 * g.sum()
        ys, xs = np.nonzero(g)
        assert np.hypot(t.center[0] - xs.mean(), t.center[1] - ys.mean()) < 2.0


def test_result_fields_are_consistent():
    img, gt, pr = scene()
    t = best(img, Selection.from_point(*pr.points[0]), calibration=Calibration(0.5), frame_index=7, time_s=0.7)
    assert t.mask.dtype == np.uint8 and set(np.unique(t.mask)) == {0, 255} and t.mask.shape == img.shape
    assert t.area_px == int((t.mask > 0).sum()) and t.area_um2 == pytest.approx(t.area_px * 0.25)
    assert t.contour.ndim == 2 and t.contour.shape[1] == 2 and len(t.contour) > 8
    assert t.frame_index == 7 and t.time_s == 0.7 and t.method == "threshold"
    assert "um^2" in t.summary()
    assert best(img, Selection.from_point(*pr.points[0])).area_um2 is None        # без калибровки


def test_click_beside_object_snaps_with_warning():
    img, gt, pr = scene()
    g = gt == 1
    ys, xs = np.nonzero(g)
    t = best(img, Selection.from_point(xs.max() + 4, pr.points[0][1]))
    assert iou(t.mask > 0, g) > 0.8
    assert any("ближайший" in w for w in t.warnings)


def test_click_on_empty_background_fails_cleanly():
    img, gt, pr = scene("clean")
    ys, xs = np.nonzero(gt == 0)
    far = None
    for y, x in zip(ys[::500], xs[::500]):                 # фон, далеко от любого объекта
        d = cv2.distanceTransform((gt == 0).astype(np.uint8), cv2.DIST_L2, 3)[y, x]
        if d > 40:
            far = (x, y)
            break
    assert far is not None
    with pytest.raises(InitializationError):
        best(img, Selection.from_point(*far))


def test_noise_speck_does_not_attract_snapping():
    img = np.full((120, 120), 200, np.uint8)
    img[50:53, 50:53] = 60                                # пятно 3x3 = 9 px < min_area
    with pytest.raises(InitializationError):
        best(img, Selection.from_point(56, 51))


def test_touching_objects_need_split_option():
    img, gt, pr = scene("touching", 4)
    ids = np.unique(gt[gt > 0])
    k = 0                                                 # метки 1 и 2 — слипшаяся пара
    merged = best(img, Selection.from_point(*pr.points[k]))
    assert merged.area_px > 1.5 * (gt == ids[k]).sum()
    cfg = InitConfig(split=SplitConfig(enabled=True, peak_fraction=0.8))
    split = best(img, Selection.from_point(*pr.points[k]), config=cfg)
    assert iou(split.mask > 0, gt == ids[k]) > 0.85


def test_forced_polarity_and_max_area_guard():
    img, gt, pr = scene()
    wrong = InitConfig.from_dict({"threshold": {"objects": "bright"}})
    t = best(img, Selection.from_point(*pr.points[0]), config=wrong)    # порог выбрал бы фон -> запасной путь
    assert any("наращивание" in w for w in t.warnings) and iou(t.mask > 0, gt == 1) > 0.8
    ok = InitConfig.from_dict({"threshold": {"objects": "dark"}})
    assert best(img, Selection.from_point(*pr.points[0]), config=ok).area_px > 50


def test_min_area_guard():
    img, gt, pr = scene()
    with pytest.raises(InitializationError):
        best(img, Selection.from_point(*pr.points[0]), config=InitConfig(min_area=100000))


def test_preprocess_cannot_change_geometry():
    img, gt, pr = scene()
    cfg = InitConfig(preprocess=PreprocessConfig(scale=2.0, median_ksize=3))
    t = best(img, Selection.from_point(*pr.points[0]), config=cfg)
    assert t.mask.shape == img.shape and iou(t.mask > 0, gt == 1) > 0.8


# --------------------------------------------------------------------------- #
# Рамка, ROI, GrabCut
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["clean", "noisy", "uneven_light", "bright_on_dark", "low_contrast", "rods"])
def test_box_tight_and_loose(name):
    img, gt, pr = scene(name)
    g = gt == 1
    x0, y0, x1, y1 = pr.boxes[0]
    for pad in (0, 6):
        t = best(img, Selection.from_box(x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        assert iou(t.mask > 0, g) > 0.8


def test_roi_polygon():
    img, gt, pr = scene("clean")
    x0, y0, x1, y1 = pr.boxes[0]
    roi = Selection.from_roi([(x0 - 5, y0 - 3), (x1 + 4, y0 - 6), (x1 + 6, y1 + 5), (x0 - 4, y1 + 4)])
    t = best(img, roi)
    assert iou(t.mask > 0, gt == 1) > 0.9 and t.selection.kind == "roi"


def test_box_clips_mask_to_region_by_default():
    img, gt, pr = scene("large_objects")
    x0, y0, x1, y1 = pr.boxes[0]
    half = Selection.from_box(x0, y0, (x0 + x1) / 2, y1)        # рамка покрывает половину объекта
    clipped = best(img, half)
    free = best(img, half, config=InitConfig(clip_to_region=False))
    assert clipped.area_px < 0.6 * (gt == 1).sum() < 0.9 * free.area_px
    assert iou(free.mask > 0, gt == 1) > 0.9


def test_box_on_empty_region_fails():
    img = np.full((100, 100), 180, np.uint8)
    with pytest.raises(InitializationError):
        best(img, Selection.from_box(20, 20, 60, 60))
    noise = np.clip(np.random.default_rng(0).normal(150, 4, (100, 100)), 0, 255).astype(np.uint8)
    with pytest.raises(InitializationError):
        best(noise, Selection.from_box(20, 20, 60, 60))                  # чистый шум — не объект
    with pytest.raises(InitializationError):
        best(noise, Selection.from_point(50, 50))


def test_grabcut_box_and_roi_but_not_point():
    img, gt, pr = scene("clean")
    cfg = InitConfig(method="grabcut")
    t = best(img, Selection.from_box(*pr.boxes[0]), config=cfg)
    assert t.method == "grabcut" and iou(t.mask > 0, gt == 1) > 0.75
    x0, y0, x1, y1 = pr.boxes[0]
    t = best(img, Selection.from_roi([(x0 - 4, y0 - 4), (x1 + 4, y0 - 4), (x1 + 4, y1 + 4), (x0 - 4, y1 + 4)]),
             config=cfg)
    assert iou(t.mask > 0, gt == 1) > 0.75
    with pytest.raises(ValueError):
        best(img, Selection.from_point(*pr.points[0]), config=cfg)


# --------------------------------------------------------------------------- #
# Готовая модель с подсказкой (заглушки)
# --------------------------------------------------------------------------- #
class FakePoints(Segmenter):
    mode, name = "points", "fake-points"

    def __call__(self, image, prompts=None):
        n, lab = cv2.connectedComponents((image < 140).astype(np.uint8), connectivity=4)
        x, y = np.round(prompts.points[0]).astype(int)
        return (lab == lab[y, x]).astype(np.int32) if lab[y, x] > 0 else np.zeros(image.shape[:2], np.int32)


class FakeBoxes(Segmenter):
    mode, name = "boxes", "fake-boxes"

    def __call__(self, image, prompts=None):
        n, lab = cv2.connectedComponents((image < 140).astype(np.uint8), connectivity=4)
        x0, y0, x1, y1 = np.round(prompts.boxes[0]).astype(int)
        ids, counts = np.unique(lab[y0:y1, x0:x1][lab[y0:y1, x0:x1] > 0], return_counts=True)
        return (lab == ids[np.argmax(counts)]).astype(np.int32) if len(ids) else np.zeros(image.shape[:2], np.int32)


class AutoOnly(Segmenter):
    mode = "auto"


def test_model_points_and_boxes_with_native_prompts():
    img, gt, pr = scene("clean")
    t = best(img, Selection.from_point(*pr.points[0]), segmenter=FakePoints())
    assert t.method == "model:fake-points" and iou(t.mask > 0, gt == 1) > 0.9 and not t.warnings
    t = best(img, Selection.from_box(*pr.boxes[0]), segmenter=FakeBoxes())
    assert t.method == "model:fake-boxes" and iou(t.mask > 0, gt == 1) > 0.9 and not t.warnings


def test_model_derives_missing_prompt_from_classical():
    img, gt, pr = scene("clean")
    t = best(img, Selection.from_box(*pr.boxes[0]), segmenter=FakePoints())
    assert iou(t.mask > 0, gt == 1) > 0.9 and any("точка для модели" in w for w in t.warnings)
    t = best(img, Selection.from_point(*pr.points[0]), segmenter=FakeBoxes())
    assert iou(t.mask > 0, gt == 1) > 0.9 and any("рамка для модели" in w for w in t.warnings)


def test_model_roi_is_clipped_to_polygon():
    img, gt, pr = scene("large_objects")
    x0, y0, x1, y1 = pr.boxes[0]
    roi = Selection.from_roi([(x0, y0), ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1), (x0, y1)])
    t = best(img, roi, segmenter=FakeBoxes())
    assert t.area_px < 0.6 * (gt == 1).sum()


def test_model_errors():
    img, gt, pr = scene("clean")
    with pytest.raises(ValueError):
        best(img, Selection.from_point(*pr.points[0]), segmenter=AutoOnly())
    with pytest.raises(ValueError):
        best(img, Selection.from_point(*pr.points[0]), config=InitConfig(method="model"))
    class Nothing(FakePoints):
        def __call__(self, image, prompts=None):
            return np.zeros(image.shape[:2], np.int32)
    with pytest.raises(InitializationError):
        best(img, Selection.from_point(*pr.points[0]), segmenter=Nothing())


# --------------------------------------------------------------------------- #
# Сохранение, загрузка, визуальная проверка
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip(tmp_path):
    img, gt, pr = scene()
    t = best(img, Selection.from_box(*pr.boxes[0]), calibration=Calibration(0.4), frame_index=3, time_s=0.3)
    paths = save_target(t, tmp_path / "init", img)
    assert {"mask", "json", "overlay"} == set(paths) and all(p.exists() for p in paths.values())
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and data["area_px"] == t.area_px
    assert data["center"]["x"] == pytest.approx(t.center[0]) and len(data["contour"]) == len(t.contour)
    back = load_target(tmp_path / "init")
    assert np.array_equal(back.mask, t.mask) and back.area_px == t.area_px
    assert back.center == pytest.approx(t.center) and np.array_equal(back.contour, t.contour)
    assert back.selection == t.selection and back.method == t.method
    assert back.frame_index == 3 and back.time_s == 0.3
    assert back.calibration.pixel_size_um == 0.4 and back.area_um2 == pytest.approx(t.area_um2)
    assert back.image_shape == img.shape


def test_save_without_image_skips_overlay_and_uncalibrated_roundtrip(tmp_path):
    img, gt, pr = scene()
    t = best(img, Selection.from_point(*pr.points[0]))
    paths = save_target(t, tmp_path, None)
    assert "overlay" not in paths
    back = load_target(tmp_path)
    assert not back.calibration.calibrated and back.area_um2 is None
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["area_um2"] is None


def test_load_rejects_unknown_schema(tmp_path):
    img, gt, pr = scene()
    save_target(best(img, Selection.from_point(*pr.points[0])), tmp_path)
    p = tmp_path / "init.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["schema_version"] = 99
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError):
        load_target(tmp_path)


def test_overlay_marks_contour_center_and_selection():
    img, gt, pr = scene()
    t = best(img, Selection.from_point(*pr.points[0]))
    before = img.copy()
    out = render_overlay(img, t)
    assert out.shape == img.shape + (3,) and out.dtype == np.uint8 and np.array_equal(img, before)
    cx, cy = (int(round(v)) for v in t.center)
    assert tuple(out[cy, cx]) == (0, 0, 255)                          # красный маркер центра
    x, y = t.contour[0]
    assert out[y, x, 1] > out[y, x, 0] + 80                           # зелёный контур
    inside = t.mask > 0
    assert (out[inside][:, 1].astype(int) - out[inside][:, 0]).mean() > 20     # зелёный оттенок маски
    plain = render_overlay(img, None, Selection.from_box(10, 10, 60, 50), pending={"corner": (5, 5)})
    assert plain.shape == out.shape and not np.array_equal(plain[..., 0], plain[..., 1])


# --------------------------------------------------------------------------- #
# Диалог выбора и повторный выбор
# --------------------------------------------------------------------------- #
def test_session_select_undo_reselect():
    img, gt, pr = scene("clean")
    s = InitSession(img)
    s.click(*pr.points[0])
    first = s.result
    assert first is not None and len(s.history) == 1 and not s.accepted
    s.click(*pr.points[1])
    assert s.result is not first and len(s.history) == 2
    assert s.result.center != first.center
    s.key("u")
    assert s.result is first and len(s.history) == 1
    s.key("c")                                                          # выбрать заново
    assert s.result is None and not s.history
    s.click(*pr.points[2])
    assert s.result is not None


def test_session_failed_click_keeps_previous_result():
    img, gt, pr = scene("clean")
    s = InitSession(img)
    s.click(*pr.points[0])
    prev = s.result
    s.click(img.shape[1] + 50, 10)                                      # вне кадра
    assert s.message.startswith("error") and s.result is prev and len(s.history) == 1
    dist = cv2.distanceTransform((gt == 0).astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)                # самая «фоновая» точка
    s.click(x, y)                                                       # клик по пустому фону не роняет диалог
    assert s.message.startswith("error") and s.result is prev


def test_session_box_modes_and_drag():
    img, gt, pr = scene("clean")
    s = InitSession(img)
    s.key("b")
    x0, y0, x1, y1 = pr.boxes[0]
    s.click(x0 - 3, y0 - 3)
    assert s.result is None and "opposite" in s.message
    s.click(x1 + 3, y1 + 3)
    assert s.result is not None and s.result.selection.kind == "box"
    x0, y0, x1, y1 = pr.boxes[1]
    s.drag((x1 + 3, y1 + 3), (x0 - 3, y0 - 3))                          # протягивание в любом направлении
    assert s.result.selection.kind == "box" and len(s.history) == 2
    s.key("p")
    s.drag((0, 0), (10, 10))                                            # в режиме точки перетаскивание игнорируется
    assert len(s.history) == 2


def test_session_roi_close_undo_vertex_and_accept():
    img, gt, pr = scene("clean")
    s = InitSession(img)
    s.key("r")
    x0, y0, x1, y1 = pr.boxes[0]
    for p in [(x0 - 4, y0 - 4), (x1 + 4, y0 - 4), (x1 + 4, y1 + 4), (3, 3)]:
        s.click(*p)
    s.key("u")                                                          # убрать лишнюю вершину
    s.click(x0 - 4, y1 + 4)
    assert s.result is None
    s.key("enter")                                                      # замкнуть полигон
    assert s.result is not None and s.result.selection.kind == "roi" and not s.accepted
    s.key("enter")                                                      # принять
    assert s.accepted and "accepted" in s.message
    n = len(s.history)
    s.click(*pr.points[1])
    assert len(s.history) == n and "already accepted" in s.message      # после принятия клики игнорируются
    s.key("c")
    assert not s.accepted and s.result is None
    s.key("p")
    s.click(*pr.points[1])
    assert s.result is not None and s.result.selection.kind == "point"


def test_session_keys_actions_and_save(tmp_path):
    img, gt, pr = scene("clean")
    s = InitSession(img, calibration=Calibration(0.5), frame_index=2, time_s=0.2)
    assert s.key("s") == "save" and s.key("q") == "quit" and s.key("esc") == "quit" and s.key("x") == ""
    s.key("enter")
    assert not s.accepted and "nothing to accept" in s.message
    with pytest.raises(InitializationError):
        s.save(tmp_path)
    s.click(*pr.points[0])
    s.key("enter")
    paths = s.save(tmp_path / "out")
    assert paths["overlay"].exists() and load_target(tmp_path / "out").frame_index == 2
    with pytest.raises(ValueError):
        s.set_mode("lasso")
    frame = s.render()
    assert frame.shape == img.shape + (3,)


# --------------------------------------------------------------------------- #
# Командная строка
# --------------------------------------------------------------------------- #
def test_cli_point_box_roi_and_failure(tmp_path):
    img, gt, pr = scene("clean")
    src = tmp_path / "frame.png"
    cv2.imwrite(str(src), img)
    x, y = pr.points[0]
    out = tmp_path / "p"
    assert CLI.main(["--source", str(src), "--point", str(x), str(y), "--out", str(out),
                     "--pixel-size-um", "0.5"]) == 0
    t = load_target(out)
    assert iou(t.mask > 0, gt == 1) > 0.9 and t.area_um2 == pytest.approx(t.area_px * 0.25)
    x0, y0, x1, y1 = pr.boxes[1]
    assert CLI.main(["--source", str(src), "--box", str(x0), str(y0), str(x1), str(y1),
                     "--out", str(tmp_path / "b")]) == 0
    assert CLI.main(["--source", str(src), "--roi", str(x0 - 4), str(y0 - 4), str(x1 + 4), str(y0 - 4),
                     str(x1 + 4), str(y1 + 4), str(x0 - 4), str(y1 + 4), "--out", str(tmp_path / "r")]) == 0
    assert CLI.main(["--source", str(src), "--point", "-5", "5", "--out", str(tmp_path / "bad")]) == 1
    assert not (tmp_path / "bad" / "init.json").exists()
    with pytest.raises(SystemExit):
        CLI.main(["--source", str(src), "--roi", "1", "2", "3", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        CLI.main(["--source", str(src), "--frame", "5", "--point", "1", "1", "--out", str(tmp_path / "x")])


def test_cli_reads_frame_from_video_with_time(tmp_path):
    img, gt, pr = scene("clean")
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (img.shape[1], img.shape[0]))
    for _ in range(6):
        w.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    w.release()
    x, y = pr.points[0]
    out = tmp_path / "o"
    assert CLI.main(["--source", str(path), "--frame", "3", "--point", str(x), str(y), "--out", str(out)]) == 0
    t = load_target(out)
    assert t.frame_index == 3 and t.time_s == pytest.approx(0.6)


def test_cli_missing_model_falls_back_to_classical(tmp_path, monkeypatch):
    from microtrack.segmentation.pretrained import ModelUnavailable, Sam2PromptSegmenter

    def _raise_unavailable(self):
        raise ModelUnavailable("тест: модель недоступна")

    monkeypatch.setattr(Sam2PromptSegmenter, "load", _raise_unavailable)
    img, gt, pr = scene("clean")
    src = tmp_path / "f.png"
    cv2.imwrite(str(src), img)
    x0, y0, x1, y1 = pr.boxes[0]
    rc = CLI.main(["--source", str(src), "--box", str(x0), str(y0), str(x1), str(y1),
                   "--model", "sam2-boxes", "--out", str(tmp_path / "o")])
    assert rc == 0 and load_target(tmp_path / "o").method == "threshold"


def test_cli_without_selection_and_without_display_fails_cleanly(tmp_path):
    import os
    img, gt, pr = scene("clean")
    src = tmp_path / "f.png"
    cv2.imwrite(str(src), img)
    old = {k: os.environ.pop(k, None) for k in ("DISPLAY", "WAYLAND_DISPLAY")}
    try:
        if sys.platform.startswith("linux"):
            assert not CLI.gui_available()
            assert CLI.main(["--source", str(src), "--out", str(tmp_path / "o")]) == 2
    finally:
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v
