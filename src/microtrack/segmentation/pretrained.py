"""Адаптеры готовых моделей сегментации (без дообучения).

Единый интерфейс: ``segmenter(image, prompts=None) -> labels`` (int32, 0 — фон).
Тяжёлые зависимости (torch, cellpose, stardist, sam2 ...) импортируются лениво:
если пакета или весов нет, ``load_model`` бросает :class:`ModelUnavailable`
с понятной причиной, а скрипт сравнения просто пропускает модель.

Режимы:
  auto    — модель сама находит объекты (Cellpose, StarDist, Omnipose, SAM2 auto);
  points  — по одной точке на объект (SAM2 image predictor);
  boxes   — по одной рамке на объект (SAM2 image predictor).
Подсказки берутся из эталона (``oracle`` — верхняя граница, «идеальный пользователь»)
или из classical-детектора (``classical`` — реальный гибрид без разметки).

ВНИМАНИЕ: код адаптеров написан по документации (Cellpose 4.x, StarDist 0.9,
SAM 2.1, Omnipose) и проверен только на заглушках — на реальных весах его нужно
прогнать там, где есть сеть/GPU.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..preprocess import ensure_uint8, to_grayscale


class ModelUnavailable(RuntimeError):
    """Модель нельзя запустить: нет пакета, весов, сети или несовместима версия."""


# --------------------------------------------------------------------------- #
# Подсказки (точки и рамки)
# --------------------------------------------------------------------------- #
@dataclass
class Prompts:
    points: np.ndarray      # (N, 2) x, y
    boxes: np.ndarray       # (N, 4) x0, y0, x1, y1 (x1, y1 — исключая)

    def __len__(self) -> int:
        return len(self.points)


def prompts_from_labels(labels: np.ndarray, point_jitter: float = 0.0,
                        box_jitter: float = 0.0, rng: Optional[np.random.Generator] = None,
                        ) -> Prompts:
    """Точка внутри каждого объекта и его охватывающая рамка.

    Точка — самый «глубокий» пиксель объекта (максимум distance transform), поэтому
    всегда лежит внутри даже у несвыпуклых форм. ``*_jitter`` (px, sigma) имитирует
    неточный ввод пользователя или детектора.
    """
    rng = rng or np.random.default_rng(0)
    h, w = labels.shape
    pts, boxes = [], []
    for i in np.unique(labels[labels > 0]):
        ys, xs = np.nonzero(labels == i)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        crop = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        dist = cv2.distanceTransform(np.pad(crop, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        p = np.array([x0 + cx, y0 + cy], float)
        b = np.array([x0, y0, x1, y1], float)
        if point_jitter:
            p += rng.normal(0, point_jitter, 2)
        if box_jitter:
            b += rng.normal(0, box_jitter, 4)
        p = np.clip(p, 0, [w - 1, h - 1])
        b = np.clip(b, 0, [w, h, w, h])
        b[2] = max(b[2], b[0] + 2)
        b[3] = max(b[3], b[1] + 2)
        pts.append(p)
        boxes.append(b)
    return Prompts(np.array(pts, float).reshape(-1, 2), np.array(boxes, float).reshape(-1, 4))


def labels_from_masks(masks: Sequence[np.ndarray], scores: Optional[Sequence[float]] = None,
                      shape: Optional[Tuple[int, int]] = None, min_area: int = 0,
                      max_area_frac: float = 1.0) -> np.ndarray:
    """Набор бинарных масок -> экземплярные метки.

    Маски с большей оценкой рисуются поверх (перекрытия достаются им); слишком
    малые и слишком большие (обычно «весь фон» у SAM) отбрасываются.
    """
    if len(masks) == 0:
        return np.zeros(shape or (0, 0), np.int32)
    h, w = np.asarray(masks[0]).shape
    order = np.argsort(scores, kind="stable") if scores is not None else range(len(masks))
    out = np.zeros((h, w), np.int32)
    k = 0
    for idx in order:
        m = np.asarray(masks[idx]) > 0
        area = int(m.sum())
        if area < max(min_area, 1) or area > max_area_frac * h * w:
            continue
        k += 1
        out[m] = k
    ids = np.unique(out[out > 0])                 # метки, полностью перекрытые другими, исчезают
    remap = np.zeros(k + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[out]


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #
def _gray_u8(image: np.ndarray) -> np.ndarray:
    return to_grayscale(ensure_uint8(image))


def _rgb_u8(image: np.ndarray) -> np.ndarray:
    g = _gray_u8(image) if image.ndim == 2 or image.shape[2] != 3 else ensure_uint8(image)
    return np.stack([g] * 3, axis=-1) if g.ndim == 2 else g


def _torch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


def _import(module: str, hint: str):
    try:
        return importlib.import_module(module)
    except Exception as e:                       # ImportError и ошибки инициализации (CUDA, TF)
        raise ModelUnavailable(f"{hint}: {type(e).__name__}: {e}") from e


@contextlib.contextmanager
def _torch_load_on_cpu():
    """Без CUDA веса, сохранённые на GPU, грузятся на CPU (torch.load без map_location падает)."""
    try:
        import torch
    except ImportError:
        yield
        return
    if torch.cuda.is_available():
        yield
        return
    orig = torch.load

    def patched(*args, **kwargs):
        # принудительно: библиотеки часто передают map_location=cuda явно (cellpose_omni)
        kwargs["map_location"] = torch.device("cpu")
        return orig(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = orig


class Segmenter:
    mode = "auto"

    def load(self) -> None:
        """Загрузка модели и весов (тяжёлая, может бросить ModelUnavailable)."""

    def __call__(self, image: np.ndarray, prompts: Optional[Prompts] = None) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Cellpose / Omnipose
# --------------------------------------------------------------------------- #
class CellposeSegmenter(Segmenter):
    """Cellpose 4 (``cpsam``) или Cellpose 3 (``cyto3`` и др. через model_type)."""

    def __init__(self, model: str = "cpsam", diameter: Optional[float] = None,
                 flow_threshold: float = 0.4, cellprob_threshold: float = 0.0,
                 gpu: Optional[bool] = None):
        self.model, self.diameter = model, diameter
        self.flow_threshold, self.cellprob_threshold, self.gpu = flow_threshold, cellprob_threshold, gpu

    def load(self) -> None:
        cp = _import("cellpose", "cellpose не установлен")
        # cellpose.models — подмодуль, который `import cellpose` сам не подгружает
        models = _import("cellpose.models", "cellpose.models не загружается")
        core = getattr(cp, "core", None)
        use_gpu = self.gpu if self.gpu is not None else bool(core and core.use_gpu())
        try:
            with _torch_load_on_cpu():
                if self.model == "cpsam":
                    self._m = models.CellposeModel(gpu=use_gpu)
                else:
                    self._m = models.CellposeModel(gpu=use_gpu, model_type=self.model)
        except Exception as e:
            raise ModelUnavailable(f"cellpose[{self.model}]: не удалось загрузить веса: {e}") from e

    def __call__(self, image, prompts=None):
        kw = dict(diameter=self.diameter, flow_threshold=self.flow_threshold,
                  cellprob_threshold=self.cellprob_threshold)
        if self.model != "cpsam":
            kw["channels"] = [0, 0]
        out = self._m.eval(_gray_u8(image), **kw)
        return np.asarray(out[0]).astype(np.int32)


class OmniposeSegmenter(Segmenter):
    """Omnipose (форк cellpose_omni) с бактериальной моделью, напр. bact_phase_omni."""

    def __init__(self, model_type: str = "bact_phase_omni", gpu: Optional[bool] = None):
        self.model_type, self.gpu = model_type, gpu

    def load(self) -> None:
        mod = _import("cellpose_omni", "omnipose (cellpose_omni) не установлен")
        core = getattr(mod, "core", None)
        use_gpu = self.gpu if self.gpu is not None else bool(core and core.use_gpu())
        models = _import("cellpose_omni.models", "cellpose_omni.models не загружается")
        try:
            with _torch_load_on_cpu():
                self._m = models.CellposeModel(gpu=use_gpu, model_type=self.model_type)
        except Exception as e:
            raise ModelUnavailable(f"omnipose[{self.model_type}]: не удалось загрузить веса: {e}") from e

    def __call__(self, image, prompts=None):
        out = self._m.eval(_gray_u8(image), channels=[0, 0], omni=True)
        return np.asarray(out[0]).astype(np.int32)


# --------------------------------------------------------------------------- #
# StarDist
# --------------------------------------------------------------------------- #
class StarDistSegmenter(Segmenter):
    """StarDist 2D, предобученные модели для ядер (fluo, dsb2018) и H&E.

    Модели ожидают светлые объекты на тёмном фоне, поэтому при ``auto_invert``
    светлый фон (медиана > 127) инвертируется.
    """

    def __init__(self, name: str = "2D_versatile_fluo", prob_thresh: Optional[float] = None,
                 nms_thresh: Optional[float] = None, auto_invert: bool = True):
        self.name, self.prob_thresh, self.nms_thresh, self.auto_invert = name, prob_thresh, nms_thresh, auto_invert

    def load(self) -> None:
        models = _import("stardist.models", "stardist/tensorflow не установлены")
        self._normalize = _import("csbdeep.utils", "csbdeep не установлен").normalize
        try:
            self._m = models.StarDist2D.from_pretrained(self.name)
        except Exception as e:
            raise ModelUnavailable(f"stardist[{self.name}]: не удалось загрузить веса: {e}") from e
        if self._m is None:
            raise ModelUnavailable(f"stardist: модель '{self.name}' не найдена")

    def __call__(self, image, prompts=None):
        g = _gray_u8(image)
        if self.auto_invert and np.median(g) > 127:
            g = 255 - g
        x = self._normalize(g.astype(np.float32), 1, 99.8)
        labels, _ = self._m.predict_instances(x, prob_thresh=self.prob_thresh,
                                              nms_thresh=self.nms_thresh)
        return np.asarray(labels).astype(np.int32)


# --------------------------------------------------------------------------- #
# SAM 2
# --------------------------------------------------------------------------- #
_SAM2_CACHE: Dict[tuple, object] = {}


def _load_sam2(hf_id: str, checkpoint: Optional[str], config: str, device: str):
    key = (hf_id, checkpoint, config, device)
    if key in _SAM2_CACHE:
        return _SAM2_CACHE[key]
    build = _import("sam2.build_sam", "sam2 не установлен")
    try:
        if checkpoint:
            model = build.build_sam2(config, checkpoint, device=device)
        else:
            model = build.build_sam2_hf(hf_id, device=device)
    except Exception as e:
        raise ModelUnavailable(f"sam2: не удалось загрузить веса ({hf_id or checkpoint}): {e}") from e
    _SAM2_CACHE[key] = model
    return model


@dataclass
class Sam2Options:
    hf_id: str = field(default_factory=lambda: os.environ.get("SAM2_HF_ID", "facebook/sam2.1-hiera-small"))
    checkpoint: Optional[str] = field(default_factory=lambda: os.environ.get("SAM2_CHECKPOINT"))
    config: str = field(default_factory=lambda: os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml"))
    device: Optional[str] = None
    min_area: int = 20
    max_area_frac: float = 0.2          # отбрасывает маску «весь фон»


class Sam2AutoSegmenter(Segmenter):
    mode = "auto"

    def __init__(self, points_per_side: int = 32, pred_iou_thresh: float = 0.8,
                 stability_score_thresh: float = 0.9, options: Optional[Sam2Options] = None):
        self.o = options or Sam2Options()
        self.pps, self.iou_t, self.stab_t = points_per_side, pred_iou_thresh, stability_score_thresh

    def load(self) -> None:
        model = _load_sam2(self.o.hf_id, self.o.checkpoint, self.o.config, self.o.device or _torch_device())
        amg = _import("sam2.automatic_mask_generator", "sam2 не установлен")
        self._gen = amg.SAM2AutomaticMaskGenerator(
            model, points_per_side=self.pps, pred_iou_thresh=self.iou_t,
            stability_score_thresh=self.stab_t, min_mask_region_area=self.o.min_area)

    def __call__(self, image, prompts=None):
        res = self._gen.generate(_rgb_u8(image))
        return labels_from_masks([r["segmentation"] for r in res],
                                 [r["predicted_iou"] for r in res],
                                 shape=image.shape[:2], min_area=self.o.min_area,
                                 max_area_frac=self.o.max_area_frac)


class Sam2PromptSegmenter(Segmenter):
    def __init__(self, mode: str = "boxes", options: Optional[Sam2Options] = None):
        if mode not in ("points", "boxes"):
            raise ValueError("mode: points | boxes")
        self.mode, self.o = mode, options or Sam2Options()

    def load(self) -> None:
        model = _load_sam2(self.o.hf_id, self.o.checkpoint, self.o.config, self.o.device or _torch_device())
        pred = _import("sam2.sam2_image_predictor", "sam2 не установлен")
        self._pred = pred.SAM2ImagePredictor(model)

    def __call__(self, image, prompts=None):
        h, w = image.shape[:2]
        if prompts is None or len(prompts) == 0:
            return np.zeros((h, w), np.int32)
        self._pred.set_image(_rgb_u8(image))
        masks, scores = [], []
        for k in range(len(prompts)):
            if self.mode == "points":
                m, s, _ = self._pred.predict(point_coords=prompts.points[k:k + 1],
                                             point_labels=np.array([1]), multimask_output=True)
            else:
                m, s, _ = self._pred.predict(box=prompts.boxes[k], multimask_output=False)
            j = int(np.argmax(s))
            masks.append(np.asarray(m[j]) > 0)
            scores.append(float(s[j]))
        return labels_from_masks(masks, scores, shape=(h, w), min_area=self.o.min_area,
                                 max_area_frac=self.o.max_area_frac)


# --------------------------------------------------------------------------- #
# Реестр
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    mode: str                                   # auto | points | boxes
    prompt_source: Optional[str]                # None | oracle | classical
    factory: Callable[[], Segmenter]
    requires: Tuple[str, ...]
    description: str = ""


def _spec(name, family, mode, source, factory, requires, desc=""):
    return ModelSpec(name, family, mode, source, factory, tuple(requires), desc)


_SAM = ("torch", "sam2")
MODEL_REGISTRY: Dict[str, ModelSpec] = {s.name: s for s in [
    _spec("cellpose_cpsam", "cellpose", "auto", None, lambda: CellposeSegmenter("cpsam"),
          ("torch", "cellpose"), "Cellpose-SAM (v4), универсальная модель"),
    _spec("cellpose_cyto3", "cellpose", "auto", None, lambda: CellposeSegmenter("cyto3"),
          ("torch", "cellpose"), "Cellpose 3, cyto3 (только cellpose<4)"),
    _spec("omnipose_bact_phase", "omnipose", "auto", None, lambda: OmniposeSegmenter("bact_phase_omni"),
          ("torch", "cellpose_omni"), "Omnipose, бактерии, фазовый контраст"),
    _spec("stardist_fluo", "stardist", "auto", None, lambda: StarDistSegmenter("2D_versatile_fluo"),
          ("tensorflow", "stardist", "csbdeep"), "StarDist, ядра (флуоресценция)"),
    _spec("sam2_auto", "sam2", "auto", None, lambda: Sam2AutoSegmenter(), _SAM,
          "SAM 2.1, автоматическая генерация масок по сетке точек"),
    _spec("sam2_points_oracle", "sam2", "points", "oracle", lambda: Sam2PromptSegmenter("points"), _SAM,
          "SAM 2.1, точки из эталона (верхняя граница)"),
    _spec("sam2_boxes_oracle", "sam2", "boxes", "oracle", lambda: Sam2PromptSegmenter("boxes"), _SAM,
          "SAM 2.1, рамки из эталона (верхняя граница)"),
    _spec("sam2_points_classical", "sam2", "points", "classical", lambda: Sam2PromptSegmenter("points"), _SAM,
          "SAM 2.1, точки из classical-детектора"),
    _spec("sam2_boxes_classical", "sam2", "boxes", "classical", lambda: Sam2PromptSegmenter("boxes"), _SAM,
          "SAM 2.1, рамки из classical-детектора"),
]}


def missing_requirements(spec: ModelSpec) -> List[str]:
    out = []
    for pkg in spec.requires:
        try:
            if importlib.util.find_spec(pkg) is None:
                out.append(pkg)
        except (ImportError, ValueError):
            out.append(pkg)
    return out


def load_model(spec: "ModelSpec | str", registry: Optional[Dict[str, ModelSpec]] = None) -> Segmenter:
    """Создаёт и загружает модель; при проблемах бросает ModelUnavailable."""
    if isinstance(spec, str):
        spec = (registry or MODEL_REGISTRY)[spec]
    missing = missing_requirements(spec)
    if missing:
        raise ModelUnavailable("не установлены пакеты: " + ", ".join(missing))
    seg = spec.factory()
    seg.load()
    return seg
