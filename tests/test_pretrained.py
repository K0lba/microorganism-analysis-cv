import sys
import types
from contextlib import contextmanager
from importlib.machinery import ModuleSpec

import cv2
import numpy as np
import pytest

from microtrack.segmentation import pretrained as P
from microtrack.segmentation.metrics import evaluate, match_objects
from microtrack.synthetic import SceneSpec, generate_scene

SPEC = SceneSpec("t", n_objects=6, elongation=(1.0, 1.6), noise=0.0, blur=0.0)


def scene(seed=3):
    return generate_scene(SPEC, seed)


# --------------------------------------------------------------------------- #
# Заглушки внешних библиотек (проверяют проводку адаптеров, а не сами модели)
# --------------------------------------------------------------------------- #
def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = ModuleSpec(name, None)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


@contextmanager
def fake_modules(mapping):
    old = {k: sys.modules.get(k) for k in mapping}
    sys.modules.update(mapping)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _dark_components(gray):
    n, lab = cv2.connectedComponents((gray < 140).astype(np.uint8))
    return lab


def fake_cellpose(calls):
    class CellposeModel:
        def __init__(self, gpu=False, **kw):
            calls.append(("init", gpu, kw))

        def eval(self, img, **kw):
            calls.append(("eval", kw))
            return _dark_components(img).astype(np.uint16), None, None

    models = _mod("cellpose.models", CellposeModel=CellposeModel)
    core = _mod("cellpose.core", use_gpu=lambda: False)
    cp = _mod("cellpose", models=models, core=core)
    return {"cellpose": cp, "cellpose.models": models, "cellpose.core": core}


def fake_sam2():
    class Predictor:
        def __init__(self, model):
            self.model = model

        def set_image(self, rgb):
            self.lab = _dark_components(rgb[..., 0])

        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
            if point_coords is not None:
                x, y = point_coords[0]
                l = self.lab[int(round(y)), int(round(x))]
                good = (self.lab == l) if l > 0 else np.zeros_like(self.lab, bool)
                return np.stack([np.zeros_like(good), good]), np.array([0.1, 0.9]), None
            x0, y0, x1, y1 = np.round(box).astype(int)
            ids, counts = np.unique(self.lab[y0:y1, x0:x1][self.lab[y0:y1, x0:x1] > 0], return_counts=True)
            m = (self.lab == ids[np.argmax(counts)]) if len(ids) else np.zeros_like(self.lab, bool)
            return m[None], np.array([0.95]), None

    class AutoGen:
        def __init__(self, model, **kw):
            self.kw = kw

        def generate(self, rgb):
            lab = _dark_components(rgb[..., 0])
            out = [{"segmentation": lab == i, "predicted_iou": 0.9, "area": int((lab == i).sum())}
                   for i in range(1, lab.max() + 1)]
            out.append({"segmentation": lab == 0, "predicted_iou": 0.99, "area": int((lab == 0).sum())})
            return out                                   # последняя — «весь фон», как у настоящего SAM

    build = _mod("sam2.build_sam", build_sam2=lambda cfg, ckpt, device=None: object(),
                 build_sam2_hf=lambda mid, device=None: object())
    amg = _mod("sam2.automatic_mask_generator", SAM2AutomaticMaskGenerator=AutoGen)
    pred = _mod("sam2.sam2_image_predictor", SAM2ImagePredictor=Predictor)
    root = _mod("sam2")
    return {"sam2": root, "sam2.build_sam": build, "sam2.automatic_mask_generator": amg,
            "sam2.sam2_image_predictor": pred}


def fake_stardist(fail=False):
    class StarDist2D:
        @classmethod
        def from_pretrained(cls, name):
            if fail:
                raise RuntimeError("network unreachable")
            return cls()

        def predict_instances(self, x, prob_thresh=None, nms_thresh=None):
            n, lab = cv2.connectedComponents((x > 0.5).astype(np.uint8))     # светлые объекты
            return lab, {}

    def normalize(x, pmin, pmax):
        lo, hi = np.percentile(x, pmin), np.percentile(x, pmax)
        return (x - lo) / (hi - lo + 1e-8)

    models = _mod("stardist.models", StarDist2D=StarDist2D)
    return {"stardist": _mod("stardist"), "stardist.models": models,
            "csbdeep": _mod("csbdeep"), "csbdeep.utils": _mod("csbdeep.utils", normalize=normalize)}


# --------------------------------------------------------------------------- #
# Подсказки и маски
# --------------------------------------------------------------------------- #
def test_prompts_points_inside_and_boxes_tight():
    _, gt = scene()
    pr = P.prompts_from_labels(gt)
    assert len(pr) == 6 and pr.points.shape == (6, 2) and pr.boxes.shape == (6, 4)
    for k, i in enumerate(np.unique(gt[gt > 0])):
        x, y = pr.points[k].astype(int)
        assert gt[y, x] == i
        ys, xs = np.nonzero(gt == i)
        assert pr.boxes[k].tolist() == [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


def test_prompt_point_is_inside_concave_object():
    lab = np.zeros((40, 40), np.int32)
    lab[5:35, 5:12] = 1
    lab[28:35, 5:35] = 1                                 # Г-образная фигура: центроид вне объекта
    pr = P.prompts_from_labels(lab)
    x, y = pr.points[0].astype(int)
    assert lab[y, x] == 1


def test_prompt_jitter_is_reproducible_and_clipped():
    _, gt = scene()
    a = P.prompts_from_labels(gt, 3.0, 3.0, np.random.default_rng(1))
    b = P.prompts_from_labels(gt, 3.0, 3.0, np.random.default_rng(1))
    c = P.prompts_from_labels(gt)
    assert np.array_equal(a.points, b.points) and not np.array_equal(a.points, c.points)
    h, w = gt.shape
    assert (a.points >= 0).all() and (a.points[:, 0] <= w - 1).all() and (a.points[:, 1] <= h - 1).all()
    assert (a.boxes[:, 2] > a.boxes[:, 0]).all() and (a.boxes[:, 3] > a.boxes[:, 1]).all()


def test_empty_labels_give_empty_prompts():
    pr = P.prompts_from_labels(np.zeros((10, 10), np.int32))
    assert len(pr) == 0 and pr.boxes.shape == (0, 4)


def test_labels_from_masks_score_order_and_filters():
    a = np.zeros((20, 20), bool); a[2:10, 2:10] = True
    b = np.zeros((20, 20), bool); b[6:14, 6:14] = True            # перекрывает a
    tiny = np.zeros((20, 20), bool); tiny[0, 0] = True
    huge = np.ones((20, 20), bool)
    lab = P.labels_from_masks([a, b, tiny, huge], [0.5, 0.9, 0.9, 0.99], min_area=4, max_area_frac=0.5)
    assert lab.max() == 2 and lab[7, 7] == lab[10, 10]            # перекрытие отдано маске b (оценка выше)
    assert lab[3, 3] != lab[10, 10] and lab[3, 3] > 0
    assert lab[0, 0] == 0
    assert np.array_equal(np.unique(lab), [0, 1, 2])


def test_labels_from_masks_drops_fully_covered_and_handles_empty():
    small = np.zeros((10, 10), bool); small[3:5, 3:5] = True
    big = np.zeros((10, 10), bool); big[1:9, 1:9] = True
    lab = P.labels_from_masks([small, big], [0.1, 0.9])
    assert lab.max() == 1                                         # small полностью перекрыта
    assert P.labels_from_masks([], shape=(4, 5)).shape == (4, 5)


# --------------------------------------------------------------------------- #
# Адаптеры на заглушках
# --------------------------------------------------------------------------- #
def test_cellpose_adapter_calls_and_output():
    img, gt = scene()
    calls = []
    with fake_modules(fake_cellpose(calls)):
        seg = P.CellposeSegmenter("cpsam")
        seg.load()
        lab = seg(img)
        seg3 = P.CellposeSegmenter("cyto3")
        seg3.load()
        seg3(img)
    assert lab.dtype == np.int32
    assert evaluate(lab, gt)["f1"] == 1.0
    init_kw = [c for c in calls if c[0] == "init"]
    eval_kw = [c[1] for c in calls if c[0] == "eval"]
    assert "model_type" not in init_kw[0][2] and init_kw[1][2]["model_type"] == "cyto3"
    assert "channels" not in eval_kw[0] and eval_kw[1]["channels"] == [0, 0]


def test_cellpose_load_failure_becomes_model_unavailable():
    mapping = fake_cellpose([])
    def boom(self, gpu=False, **kw):
        raise OSError("weights download failed")
    mapping["cellpose.models"].CellposeModel.__init__ = boom
    with fake_modules(mapping):
        with pytest.raises(P.ModelUnavailable) as e:
            P.CellposeSegmenter("cpsam").load()
    assert "weights download failed" in str(e.value)


def test_stardist_adapter_inverts_bright_background():
    img, gt = scene()                                  # тёмные объекты на светлом фоне
    with fake_modules(fake_stardist()):
        seg = P.StarDistSegmenter()
        seg.load()
        assert evaluate(seg(img), gt)["f1"] == 1.0
        raw = P.StarDistSegmenter(auto_invert=False)
        raw.load()
        assert evaluate(raw(img), gt)["f1"] < 1.0


def test_stardist_weights_failure():
    with fake_modules(fake_stardist(fail=True)):
        with pytest.raises(P.ModelUnavailable) as e:
            P.StarDistSegmenter().load()
    assert "network unreachable" in str(e.value)


def test_sam2_auto_filters_background_mask():
    img, gt = scene()
    with fake_modules(fake_sam2()):
        P._SAM2_CACHE.clear()
        seg = P.Sam2AutoSegmenter()
        seg.load()
        lab = seg(img)
    assert lab.max() == 6 and evaluate(lab, gt)["f1"] == 1.0
    assert lab[0, 0] == 0                              # «весь фон» не попал в разметку


@pytest.mark.parametrize("mode", ["points", "boxes"])
def test_sam2_prompt_modes(mode):
    img, gt = scene()
    with fake_modules(fake_sam2()):
        P._SAM2_CACHE.clear()
        seg = P.Sam2PromptSegmenter(mode)
        seg.load()
        lab = seg(img, P.prompts_from_labels(gt))
        empty = seg(img, None)
    assert match_objects(lab, gt).f1 == 1.0
    assert empty.shape == gt.shape and empty.max() == 0


def test_sam2_prompt_mode_validation_and_model_cache():
    with pytest.raises(ValueError):
        P.Sam2PromptSegmenter("lasso")
    with fake_modules(fake_sam2()):
        P._SAM2_CACHE.clear()
        a = P.Sam2PromptSegmenter("boxes"); a.load()
        b = P.Sam2AutoSegmenter(); b.load()
        assert len(P._SAM2_CACHE) == 1                 # модель загружена один раз на оба режима
    P._SAM2_CACHE.clear()


# --------------------------------------------------------------------------- #
# Реестр
# --------------------------------------------------------------------------- #
def test_registry_is_consistent():
    assert {"cellpose_cpsam", "stardist_fluo", "omnipose_bact_phase", "sam2_auto",
            "sam2_boxes_classical", "sam2_points_oracle"} <= set(P.MODEL_REGISTRY)
    for name, s in P.MODEL_REGISTRY.items():
        assert s.name == name and s.mode in ("auto", "points", "boxes")
        assert (s.mode == "auto") == (s.prompt_source is None)
        assert s.prompt_source in (None, "oracle", "classical") and s.requires


def test_load_model_reports_missing_packages():
    spec = P.ModelSpec("x", "x", "auto", None, lambda: P.Segmenter(), ("definitely_not_installed_pkg",))
    with pytest.raises(P.ModelUnavailable) as e:
        P.load_model(spec)
    assert "definitely_not_installed_pkg" in str(e.value)
    ok = P.ModelSpec("y", "y", "auto", None, lambda: P.Segmenter(), ("numpy",))
    assert isinstance(P.load_model(ok), P.Segmenter)
