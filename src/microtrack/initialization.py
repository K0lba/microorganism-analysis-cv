"""Ручная инициализация целевого объекта на первом кадре.

Пользователь указывает объект точкой, рамкой или областью интереса (ROI-полигоном).
Маска строится классическим алгоритмом или готовой моделью с подсказкой (SAM 2 и др.).
Результат (маска, центр, площадь, внешний контур) сохраняется в папку и затем
используется трекингом как начальное состояние.

Состав:
  Selection / InitConfig   — что выбрал пользователь и как строить маску;
  initialize_target        — выбор -> TargetInit (без графики, легко тестировать);
  save_target / load_target — init_mask.png, init.json, init_overlay.png;
  render_overlay           — визуальная проверка (контур, центр, выбранная подсказка);
  InitSession              — «состояние диалога»: клики, повторный выбор, отмена, принятие.
Окно OpenCV, которое кормит InitSession событиями мыши и клавиатуры, живёт в scripts/init_target.py.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .io import Calibration
from .preprocess import PreprocessConfig, ensure_uint8, preprocess, to_grayscale
from .segmentation.classical import (
    MorphologyConfig, ObjectInfo, SplitConfig, ThresholdConfig,
    binarize, clean_mask, describe_mask, split_touching,
)
from .segmentation.pretrained import Prompts, Segmenter
from .visualization import put_label

SCHEMA_VERSION = 1


class InitializationError(RuntimeError):
    """Объект по выбору получить не удалось (пользователь может выбрать заново)."""


# --------------------------------------------------------------------------- #
# Выбор пользователя
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Selection:
    kind: str                                                   # point | box | roi
    point: Optional[Tuple[float, float]] = None                 # x, y
    box: Optional[Tuple[float, float, float, float]] = None     # x0, y0, x1, y1
    polygon: Optional[Tuple[Tuple[float, float], ...]] = None   # вершины ROI

    @classmethod
    def from_point(cls, x: float, y: float) -> "Selection":
        return cls("point", point=(float(x), float(y)))

    @classmethod
    def from_box(cls, x0: float, y0: float, x1: float, y1: float) -> "Selection":
        return cls("box", box=(float(min(x0, x1)), float(min(y0, y1)),
                               float(max(x0, x1)), float(max(y0, y1))))

    @classmethod
    def from_roi(cls, vertices: Sequence[Sequence[float]]) -> "Selection":
        poly = tuple((float(x), float(y)) for x, y in vertices)
        if len(poly) < 3:
            raise ValueError("ROI: нужно минимум 3 вершины")
        return cls("roi", polygon=poly)

    def validated(self, shape: Tuple[int, int]) -> "Selection":
        """Проверка и обрезка по границам кадра (shape = (h, w))."""
        h, w = shape
        if self.kind == "point":
            x, y = self.point
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(f"Точка ({x:.0f}, {y:.0f}) вне кадра {w}x{h}")
            return self
        if self.kind == "box":
            x0, y0, x1, y1 = self.box
            x0, x1 = max(0.0, x0), min(float(w), x1)
            y0, y1 = max(0.0, y0), min(float(h), y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                raise ValueError("Рамка слишком мала или вне кадра (минимум 3x3 px)")
            return Selection("box", box=(x0, y0, x1, y1))
        if self.kind == "roi":
            poly = tuple((min(max(x, 0.0), w - 1.0), min(max(y, 0.0), h - 1.0)) for x, y in self.polygon)
            if len(poly) < 3 or cv2.contourArea(np.array(poly, np.float32)) < 9:
                raise ValueError("ROI слишком мала или вырождена")
            return Selection("roi", polygon=poly)
        raise ValueError(f"Неизвестный тип выбора: {self.kind}")

    def bounding_box(self) -> Tuple[float, float, float, float]:
        if self.kind == "box":
            return self.box
        if self.kind == "roi":
            xs, ys = zip(*self.polygon)
            return (min(xs), min(ys), max(xs), max(ys))
        x, y = self.point
        return (x, y, x, y)

    def region_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Бинарная область выбора (для рамки и ROI)."""
        m = np.zeros(shape, np.uint8)
        if self.kind == "box":
            x0, y0, x1, y1 = (int(round(v)) for v in self.box)
            m[y0:y1, x0:x1] = 1
        elif self.kind == "roi":
            cv2.fillPoly(m, [np.round(np.array(self.polygon)).astype(np.int32)], 1)
        else:
            raise ValueError("У точки нет области")
        return m.astype(bool)

    def to_dict(self) -> dict:
        return {"kind": self.kind,
                "point": list(self.point) if self.point else None,
                "box": list(self.box) if self.box else None,
                "polygon": [list(p) for p in self.polygon] if self.polygon else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Selection":
        if d["kind"] == "point":
            return cls.from_point(*d["point"])
        if d["kind"] == "box":
            return cls.from_box(*d["box"])
        return cls.from_roi(d["polygon"])


# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #
@dataclass
class InitConfig:
    method: str = "auto"                 # auto | threshold | grabcut | model
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    snap_radius: float = 15.0            # точка на фоне: искать ближайший объект в радиусе, px
    grow_tolerance: float = 12.0         # допуск яркости наращивания области (запасной путь для точки)
    box_margin: float = 0.3              # ширина кольца фона вокруг рамки/ROI, доля размера
    clip_to_region: bool = True          # обрезать маску рамкой/ROI
    min_area: int = 20
    max_area_frac: float = 0.5           # больше этой доли кадра — скорее всего выбран фон
    grabcut_iterations: int = 5
    smooth_sigma: float = 0.5            # гауссово сглаживание перед порогом (0 — выключено)
    min_contrast: float = 15.0           # минимальная разница яркости объект/фон в окне (уровни 0-255)
    window_start: int = 16               # точка: стартовый радиус локального окна, px (растёт в 1.5 раза)
    split: Optional[SplitConfig] = None  # разделять слипшиеся (только для точки), экспериментально
    preprocess: Optional[PreprocessConfig] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "InitConfig":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Неизвестные параметры инициализации: {sorted(unknown)}")
        for name, klass in (("threshold", ThresholdConfig), ("morphology", MorphologyConfig),
                            ("split", SplitConfig)):
            v = d.get(name)
            if isinstance(v, dict):
                bad = set(v) - {f.name for f in fields(klass)}
                if bad:
                    raise ValueError(f"Неизвестные параметры в '{name}': {sorted(bad)}")
                d[name] = klass(**v)
            elif v is None:
                d.pop(name, None)
        if isinstance(d.get("preprocess"), dict):
            d["preprocess"] = PreprocessConfig.from_dict(d["preprocess"])
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.method not in ("auto", "threshold", "grabcut", "model"):
            raise ValueError("method: auto | threshold | grabcut | model")
        if self.threshold.objects not in ("dark", "bright", "auto"):
            raise ValueError("threshold.objects: dark | bright | auto")
        if self.threshold.method not in ("otsu", "adaptive", "fixed"):
            raise ValueError("threshold.method: otsu | adaptive | fixed")
        if self.snap_radius < 0 or self.grow_tolerance <= 0 or self.box_margin < 0:
            raise ValueError("snap_radius/box_margin >= 0, grow_tolerance > 0")
        if not 0 < self.max_area_frac <= 1:
            raise ValueError("max_area_frac должен быть в (0, 1]")
        if self.min_area < 1 or self.grabcut_iterations < 1 or self.window_start < 4:
            raise ValueError("min_area, grabcut_iterations >= 1; window_start >= 4")
        if self.smooth_sigma < 0 or self.min_contrast < 0:
            raise ValueError("smooth_sigma и min_contrast не могут быть отрицательными")


# --------------------------------------------------------------------------- #
# Результат
# --------------------------------------------------------------------------- #
@dataclass
class TargetInit:
    mask: np.ndarray                 # uint8 0/255, размер кадра
    info: ObjectInfo                 # центр, площадь, контур, bbox, показатели формы
    selection: Selection
    method: str
    warnings: List[str]
    frame_index: int
    time_s: float
    calibration: Calibration
    image_shape: Tuple[int, int]

    @property
    def center(self) -> Tuple[float, float]:
        return self.info.centroid

    @property
    def area_px(self) -> int:
        return self.info.area_px

    @property
    def area_um2(self) -> Optional[float]:
        return self.info.area_um2 if self.calibration.calibrated else None

    @property
    def contour(self) -> np.ndarray:
        return self.info.contour

    def summary(self) -> str:
        cx, cy = self.center
        area = (f"{self.area_um2:.1f} um^2 ({self.area_px} px)" if self.area_um2 is not None
                else f"{self.area_px} px")
        return f"center=({cx:.1f}, {cy:.1f}), area={area}, method={self.method}"


# --------------------------------------------------------------------------- #
# Классический алгоритм
# --------------------------------------------------------------------------- #
def _prepare_gray(image: np.ndarray, cfg: InitConfig) -> np.ndarray:
    if cfg.preprocess is not None:
        g, _ = preprocess(image, replace(cfg.preprocess, grayscale=True, scale=1.0))
        g = to_grayscale(g)
    else:
        g = to_grayscale(ensure_uint8(image))
    if cfg.smooth_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), cfg.smooth_sigma)
    return g


def _largest_component(mask: np.ndarray) -> Optional[np.ndarray]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=4)
    if n < 2:
        return None
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _best_component(binary: np.ndarray, region: np.ndarray) -> Optional[np.ndarray]:
    """Компонента с наибольшим пересечением с областью выбора."""
    n, lab = cv2.connectedComponents((binary > 0).astype(np.uint8), connectivity=4)
    if n < 2:
        return None
    overlap = np.bincount(lab[region], minlength=n)
    overlap[0] = 0
    if overlap.max() == 0:
        return None
    return lab == int(np.argmax(overlap))


def _class_contrast(g: np.ndarray) -> float:
    """Разница средних яркостей двух классов Otsu в окне; у пустого фона она мала."""
    if g.size < 10 or int(g.min()) == int(g.max()):
        return 0.0
    t, _ = cv2.threshold(g.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = g[g <= t], g[g > t]
    return float(hi.mean() - lo.mean()) if lo.size and hi.size else 0.0


def _dark_from_point(gray: np.ndarray, x: int, y: int) -> bool:
    """Полярность по самому клику: яркость под точкой сравнивается с порогом Otsu."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    patch = gray[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
    return float(np.median(patch)) <= t


def _dark_from_region(g: np.ndarray, reg: np.ndarray, ring: np.ndarray, fallback: bool) -> bool:
    """Объект = класс внутри области, который сильнее отличается от кольца фона вокруг неё."""
    vals = g[reg]
    if vals.size < 10 or ring.sum() < 20:
        return fallback
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = vals[vals <= t], vals[vals > t]
    if lo.size == 0 or hi.size == 0:
        return fallback
    ring_med = float(np.median(g[ring]))
    return abs(lo.mean() - ring_med) > abs(hi.mean() - ring_med)


def _region_grow(gray: np.ndarray, x: int, y: int, tol: float, morph: MorphologyConfig) -> np.ndarray:
    h, w = gray.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    t = (tol,) * 4
    cv2.floodFill(gray.copy(), mask, (x, y), 255, t, t, 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    grown = clean_mask(mask[1:-1, 1:-1].copy(), morph)
    return grown > 0


def _point_component(g: np.ndarray, lx: int, ly: int, dark: bool, cfg: InitConfig, free: Tuple[bool, ...]):
    """Компонента под точкой в окне g при заданной полярности; None, если объект «не замкнут»."""
    if _class_contrast(g) < cfg.min_contrast:
        return None
    tcfg = replace(cfg.threshold, objects="dark" if dark else "bright")
    binary = clean_mask(binarize(g, tcfg), cfg.morphology)
    if cfg.split is not None and cfg.split.enabled:
        binary = split_touching(binary, cfg.split.peak_fraction, cfg.split.smooth_sigma,
                                cfg.split.min_component_area)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=4)
    warnings: List[str] = []
    cand = int(lab[ly, lx])
    if cand == 0 and cfg.snap_radius > 0 and n > 1:
        r = int(math.ceil(cfg.snap_radius))
        y0, y1, x0, x1 = max(0, ly - r), min(g.shape[0], ly + r + 1), max(0, lx - r), min(g.shape[1], lx + r + 1)
        big = stats[:, cv2.CC_STAT_AREA] >= cfg.min_area          # мелкие пятна шума не «притягивают» клик
        big[0] = False
        ys, xs = np.nonzero(big[lab[y0:y1, x0:x1]])
        if ys.size:
            d = np.hypot(xs + x0 - lx, ys + y0 - ly)
            k = int(np.argmin(d))
            if d[k] <= cfg.snap_radius:
                cand = int(lab[ys[k] + y0, xs[k] + x0])
                warnings.append(f"точка вне объекта: выбран ближайший на расстоянии {d[k]:.1f} px")
    if cand == 0:
        return None
    x, y, w, h, area = (int(v) for v in stats[cand])
    hh, ww = g.shape
    touches = ((x == 0 and not free[0]) or (y == 0 and not free[1]) or
               (x + w >= ww and not free[2]) or (y + h >= hh and not free[3]))
    if touches or area > cfg.max_area_frac * hh * ww:
        return None
    return lab == cand, warnings


def _segment_point(gray: np.ndarray, sel: Selection, cfg: InitConfig):
    """Точка: порог считается в растущем окне вокруг клика, чтобы перепад освещения и соседи не мешали.

    Берётся первое окно, где объект под точкой замкнут (не упирается в край окна) и меньше
    половины окна. Сначала пробуется полярность по яркости под кликом, затем обратная
    (клик мог попасть на фон рядом с объектом). Запасной путь — наращивание области.
    """
    h, w = gray.shape
    x = min(max(int(round(sel.point[0])), 0), w - 1)
    y = min(max(int(round(sel.point[1])), 0), h - 1)
    if cfg.threshold.objects == "auto":
        first = _dark_from_point(gray, x, y)
        polarities = [first, not first]
    else:
        polarities = [cfg.threshold.objects == "dark"]

    radii, r = [], cfg.window_start
    while r < max(h, w):
        radii.append(r)
        r = int(r * 1.5)
    radii.append(max(h, w))
    for r in radii:
        x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
        g = gray[y0:y1, x0:x1]
        free = (x0 == 0, y0 == 0, x1 == w, y1 == h)           # стороны окна, совпадающие с краем кадра
        for dark in polarities:
            res = _point_component(g, x - x0, y - y0, dark, cfg, free)
            if res is not None:
                comp, warnings = res
                out = np.zeros((h, w), bool)
                out[y0:y1, x0:x1] = comp
                return out, warnings

    grown = _region_grow(gray, x, y, cfg.grow_tolerance, cfg.morphology)
    if grown.sum() == 0:
        raise InitializationError("объект не найден: попробуйте рамку или другую точку")
    warnings = ["порог не выделил замкнутый объект, использовано наращивание области от точки"]
    # наращивание останавливается на склоне края и занижает маску: уточняем порогом в окне вокруг неё
    ys, xs = np.nonzero(grown)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = max(3, int(round(0.3 * max(x1 - x0, y1 - y0))))
    region = np.zeros((h, w), bool)
    region[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)] = True
    try:
        auto = replace(cfg, threshold=replace(cfg.threshold, objects="auto"))   # полярность по фону вокруг
        refined, _ = _segment_region(gray, region, auto)
    except InitializationError:
        return grown, warnings
    if refined[y, x] and 0.8 * grown.sum() <= refined.sum() <= 4 * grown.sum():
        return refined, warnings
    return grown, warnings


def _segment_region(gray: np.ndarray, region: np.ndarray, cfg: InitConfig):
    """Рамка/ROI: порог по локальному окну (область + кольцо фона), объект — внутри области."""
    h, w = gray.shape
    ys, xs = np.nonzero(region)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    m = max(3, int(round(cfg.box_margin * max(x1 - x0, y1 - y0))))
    cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
    g = gray[cy0:cy1, cx0:cx1]
    reg = region[cy0:cy1, cx0:cx1]
    if _class_contrast(g) < cfg.min_contrast:
        raise InitializationError("в выбранной области нет контраста: объект не виден")
    ring = cv2.dilate(reg.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * m + 1, 2 * m + 1))).astype(bool) & ~reg

    objects = cfg.threshold.objects
    if objects == "auto":
        fallback = float(np.median(g[reg])) < float(np.median(g))
        dark = _dark_from_region(g, reg, ring, fallback)
    else:
        dark = objects == "dark"
    tcfg = replace(cfg.threshold, objects="dark" if dark else "bright")
    binary = binarize(g, tcfg)
    if cfg.clip_to_region:
        binary[~reg] = 0
    binary = clean_mask(binary, cfg.morphology)
    comp = _best_component(binary, reg)
    if comp is None:
        raise InitializationError("в выбранной области объект не найден")
    out = np.zeros((h, w), bool)
    out[cy0:cy1, cx0:cx1] = comp
    return out, []


def _segment_grabcut(gray: np.ndarray, sel: Selection, cfg: InitConfig):
    if sel.kind == "point":
        raise ValueError("grabcut требует рамку или ROI, а не точку")
    h, w = gray.shape
    x0, y0, x1, y1 = (int(round(v)) for v in sel.bounding_box())
    x1, y1 = min(w, max(x1, x0 + 3)), min(h, max(y1, y0 + 3))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)      # GrabCut чувствителен к шуму: сглаживаем сильнее
    try:
        cv2.grabCut(cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR), mask, (x0, y0, x1 - x0, y1 - y0),
                    bgd, fgd, cfg.grabcut_iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        raise InitializationError(f"GrabCut не сработал: {e}") from e
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    fg = clean_mask(fg, cfg.morphology) > 0
    if sel.kind == "roi":
        fg &= sel.region_mask((h, w))
    comp = _best_component(fg, sel.region_mask((h, w)))
    if comp is None:
        raise InitializationError("GrabCut не выделил объект")
    return comp, []


# --------------------------------------------------------------------------- #
# Готовая модель с подсказкой
# --------------------------------------------------------------------------- #
def _segment_model(image: np.ndarray, gray: np.ndarray, sel: Selection, cfg: InitConfig,
                   segmenter: Segmenter):
    h, w = gray.shape
    mode = getattr(segmenter, "mode", "auto")
    if mode not in ("points", "boxes"):
        raise ValueError("Нужна модель с подсказками (mode = points | boxes), а не автоматическая")
    warnings: List[str] = []

    def classical_mask() -> np.ndarray:
        if sel.kind == "point":
            return _segment_point(gray, sel, cfg)[0]
        return _segment_region(gray, sel.region_mask((h, w)), cfg)[0]

    if mode == "points":
        if sel.point is not None:
            px, py = sel.point
        else:                                   # точка не задана: самый «глубокий» пиксель classical-маски
            base = classical_mask().astype(np.uint8)
            dist = cv2.distanceTransform(base, cv2.DIST_L2, 3)
            py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
            warnings.append("точка для модели получена из classical-маски выбранной области")
        prompts = Prompts(np.array([[px, py]], float), np.array([[px - 1, py - 1, px + 1, py + 1]], float))
    else:
        if sel.kind != "point":
            x0, y0, x1, y1 = sel.bounding_box()
        else:                                   # рамка не задана: bbox classical-маски вокруг точки
            ys, xs = np.nonzero(classical_mask())
            x0, x1, y0, y1 = xs.min() - 2, xs.max() + 3, ys.min() - 2, ys.max() + 3
            warnings.append("рамка для модели получена из classical-маски вокруг точки")
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        prompts = Prompts(np.array([[cx, cy]], float), np.array([[x0, y0, x1, y1]], float))

    labels = np.asarray(segmenter(image, prompts))
    if labels.shape[:2] != (h, w) or labels.max() == 0:
        raise InitializationError("модель не вернула маску для выбранной подсказки")
    mask = labels > 0
    if sel.kind == "roi":
        mask &= sel.region_mask((h, w))
    return mask, warnings


# --------------------------------------------------------------------------- #
# Публичный интерфейс
# --------------------------------------------------------------------------- #
def initialize_target(
    image: np.ndarray,
    selection: Selection,
    config: Optional[InitConfig] = None,
    segmenter: Optional[Segmenter] = None,
    calibration: Optional[Calibration] = None,
    frame_index: int = 0,
    time_s: float = 0.0,
) -> TargetInit:
    """Выбор пользователя -> начальная маска, центр, площадь, контур.

    method="auto": с ``segmenter`` — модель с подсказкой, без него — классический алгоритм.
    Бросает ValueError (некорректный выбор) или InitializationError (объект не найден).
    """
    cfg = config or InitConfig()
    cfg.validate()
    image = np.asarray(image)
    h, w = image.shape[:2]
    sel = selection.validated((h, w))
    calib = calibration or Calibration.from_dict(None)      # без калибровки площадь только в px
    method = cfg.method
    if method == "auto":
        method = "model" if segmenter is not None else "threshold"
    if method == "model" and segmenter is None:
        raise ValueError("method='model' требует передать segmenter")

    gray = _prepare_gray(image, cfg)
    if method == "model":
        mask, warns = _segment_model(image, gray, sel, cfg, segmenter)
        method_name = f"model:{getattr(segmenter, 'name', type(segmenter).__name__)}"
    elif method == "grabcut":
        mask, warns = _segment_grabcut(gray, sel, cfg)
        method_name = "grabcut"
    elif sel.kind == "point":
        mask, warns = _segment_point(gray, sel, cfg)
        method_name = "threshold"
    else:
        mask, warns = _segment_region(gray, sel.region_mask((h, w)), cfg)
        method_name = "threshold"

    comp = _largest_component(mask)
    if comp is None or comp.sum() < cfg.min_area:
        raise InitializationError(f"объект слишком мал (меньше {cfg.min_area} px)")
    if comp.sum() > cfg.max_area_frac * h * w:
        raise InitializationError(
            f"выделено более {cfg.max_area_frac:.0%} кадра: вероятно, выбран фон")
    mask_u8 = comp.astype(np.uint8) * 255
    return TargetInit(mask_u8, describe_mask(mask_u8, calib), sel, method_name, warns,
                      int(frame_index), float(time_s), calib, (h, w))


# --------------------------------------------------------------------------- #
# Сохранение и загрузка
# --------------------------------------------------------------------------- #
MASK_FILE, JSON_FILE, OVERLAY_FILE = "init_mask.png", "init.json", "init_overlay.png"


def save_target(target: TargetInit, directory, image: Optional[np.ndarray] = None) -> Dict[str, Path]:
    """Пишет init_mask.png, init.json и (если передан кадр) init_overlay.png."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    i = target.info
    data = {
        "schema_version": SCHEMA_VERSION,
        "frame_index": target.frame_index, "time_s": target.time_s,
        "image_shape": list(target.image_shape),
        "calibration": {"pixel_size_um": target.calibration.pixel_size_um,
                        "unit": target.calibration.unit, "calibrated": target.calibration.calibrated},
        "selection": target.selection.to_dict(), "method": target.method, "warnings": target.warnings,
        "center": {"x": i.centroid[0], "y": i.centroid[1]},
        "area_px": int(i.area_px), "area_um2": target.area_um2,
        "bbox": list(i.bbox), "perimeter_px": i.perimeter_px,
        "length_px": i.length_px, "width_px": i.width_px, "aspect_ratio": i.aspect_ratio,
        "circularity": i.circularity, "solidity": i.solidity,
        "contour": i.contour.astype(int).tolist(),
    }
    paths = {"mask": out / MASK_FILE, "json": out / JSON_FILE}
    cv2.imwrite(str(paths["mask"]), target.mask)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    if image is not None:
        paths["overlay"] = out / OVERLAY_FILE
        cv2.imwrite(str(paths["overlay"]), render_overlay(image, target))
    return paths


def load_target(directory) -> TargetInit:
    """Читает результат save_target (для трекера: начальное состояние объекта)."""
    d = Path(directory)
    data = json.loads((d / JSON_FILE).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Неподдерживаемая версия схемы: {data.get('schema_version')}")
    mask = cv2.imread(str(d / MASK_FILE), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(d / MASK_FILE)
    c = data["calibration"]
    calib = Calibration(c["pixel_size_um"], c["unit"], calibrated=c["calibrated"])
    info = ObjectInfo(
        label=1, area_px=data["area_px"], area_um2=calib.area_px_to_um2(data["area_px"]),
        centroid=(data["center"]["x"], data["center"]["y"]), bbox=tuple(data["bbox"]),
        contour=np.array(data["contour"], int).reshape(-1, 2), perimeter_px=data["perimeter_px"],
        length_px=data["length_px"], width_px=data["width_px"], aspect_ratio=data["aspect_ratio"],
        circularity=data["circularity"], solidity=data["solidity"])
    return TargetInit(mask, info, Selection.from_dict(data["selection"]), data["method"],
                      list(data["warnings"]), data["frame_index"], data["time_s"], calib,
                      tuple(data["image_shape"]))


# --------------------------------------------------------------------------- #
# Визуальная проверка
# --------------------------------------------------------------------------- #
_GREEN, _RED, _YELLOW, _CYAN = (0, 220, 0), (0, 0, 255), (0, 220, 255), (255, 200, 0)


def _put_text(img, text, org):
    put_label(img, text, org, (255, 255, 255), 0.5)


def render_overlay(image: np.ndarray, target: Optional[TargetInit] = None,
                   selection: Optional[Selection] = None, pending: Optional[dict] = None,
                   lines: Sequence[str] = (), alpha: float = 0.35) -> np.ndarray:
    """BGR-кадр с маской, контуром, центром и выбранной подсказкой (подписи — ASCII)."""
    g = ensure_uint8(np.asarray(image))
    out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g.ndim == 2 else g[:, :, :3].copy()
    if target is not None:
        tint = out.copy()
        tint[target.mask > 0] = _GREEN
        out = cv2.addWeighted(tint, alpha, out, 1 - alpha, 0)
        cv2.polylines(out, [target.contour.reshape(-1, 1, 2).astype(np.int32)], True, _GREEN, 1, cv2.LINE_AA)
    sel = selection or (target.selection if target is not None else None)
    if sel is not None:
        if sel.kind == "point":
            p = tuple(int(round(v)) for v in sel.point)
            cv2.circle(out, p, 5, _YELLOW, 1, cv2.LINE_AA)
            cv2.circle(out, p, 1, _YELLOW, -1)
        elif sel.kind == "box":
            x0, y0, x1, y1 = (int(round(v)) for v in sel.box)
            cv2.rectangle(out, (x0, y0), (x1, y1), _YELLOW, 1, cv2.LINE_AA)
        else:
            cv2.polylines(out, [np.round(np.array(sel.polygon)).astype(np.int32).reshape(-1, 1, 2)],
                          True, _YELLOW, 1, cv2.LINE_AA)
    if target is not None:
        cx, cy = (int(round(v)) for v in target.center)
        cv2.drawMarker(out, (cx, cy), _RED, cv2.MARKER_CROSS, 12, 2)
    if pending:
        corner, pts = pending.get("corner"), pending.get("polygon") or []
        if corner is not None:
            cv2.drawMarker(out, tuple(int(round(v)) for v in corner), _CYAN, cv2.MARKER_TILTED_CROSS, 10, 1)
        if pts:
            arr = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [arr], False, _CYAN, 1, cv2.LINE_AA)
            for p in arr.reshape(-1, 2):
                cv2.circle(out, tuple(int(v) for v in p), 2, _CYAN, -1)
    y = 18
    for line in list(lines) + ([] if target is None else _target_lines(target)):
        _put_text(out, line, (8, y))
        y += 18
    return out


def _target_lines(t: TargetInit) -> List[str]:
    cx, cy = t.center
    area = f"{t.area_um2:.1f} um2" if t.area_um2 is not None else f"{t.area_px} px"
    return [f"area: {area}", f"center: ({cx:.1f}, {cy:.1f})", f"method: {t.method}"]


# --------------------------------------------------------------------------- #
# Диалог выбора: клики, повторный выбор, отмена, принятие
# --------------------------------------------------------------------------- #
HELP = "p point | b box | r roi | Enter accept (or close roi) | u undo | c re-select | s save | q quit"


class InitSession:
    """Состояние диалога без графики: окно только передаёт сюда клики и клавиши.

    Клавиши (``key``): p, b, r — режим точка/рамка/ROI; enter — принять (в режиме ROI —
    замкнуть полигон); u — отменить последний шаг; c — выбрать заново; s — сохранить; q/esc — выход.
    Ошибка выбора не ломает диалог: сообщение попадает в ``message``, прошлый результат остаётся.
    """

    def __init__(self, image: np.ndarray, config: Optional[InitConfig] = None,
                 segmenter: Optional[Segmenter] = None, calibration: Optional[Calibration] = None,
                 frame_index: int = 0, time_s: float = 0.0):
        self.image = np.asarray(image)
        self.config, self.segmenter = config or InitConfig(), segmenter
        self.calibration, self.frame_index, self.time_s = calibration or Calibration.from_dict(None), frame_index, time_s
        self.mode = "point"
        self.result: Optional[TargetInit] = None
        self.history: List[TargetInit] = []
        self.accepted = False
        self.message = HELP
        self._corner: Optional[Tuple[float, float]] = None
        self._poly: List[Tuple[float, float]] = []

    # -- выбор ------------------------------------------------------------- #
    def set_mode(self, mode: str) -> None:
        if mode not in ("point", "box", "roi"):
            raise ValueError("mode: point | box | roi")
        self.mode, self._corner, self._poly = mode, None, []
        self.message = f"mode: {mode}"

    def click(self, x: float, y: float) -> None:
        if self.accepted:
            self.message = "already accepted: press c to re-select"
        elif self.mode == "point":
            self._run(Selection.from_point(x, y))
        elif self.mode == "box":
            if self._corner is None:
                self._corner = (float(x), float(y))
                self.message = "box: click the opposite corner"
            else:
                (x0, y0), self._corner = self._corner, None
                self._run(Selection.from_box(x0, y0, x, y))
        else:
            self._poly.append((float(x), float(y)))
            self.message = f"roi: {len(self._poly)} vertices, Enter to close"

    def drag(self, p0, p1) -> None:
        """Перетаскивание мышью задаёт рамку (только в режиме box)."""
        if self.accepted or self.mode != "box":
            return
        self._corner = None
        self._run(Selection.from_box(p0[0], p0[1], p1[0], p1[1]))

    def _run(self, sel: Selection) -> None:
        try:
            res = initialize_target(self.image, sel, self.config, self.segmenter, self.calibration,
                                    self.frame_index, self.time_s)
        except (InitializationError, ValueError) as e:
            self.message = f"error: {e} — choose again"
            return
        self.history.append(res)
        self.result = res
        self.message = res.summary() + (" | " + "; ".join(res.warnings) if res.warnings else "")

    # -- действия ---------------------------------------------------------- #
    def undo(self) -> None:
        if self._poly:
            self._poly.pop()
        elif self._corner is not None:
            self._corner = None
        elif self.history:
            self.accepted = False
            self.history.pop()
            self.result = self.history[-1] if self.history else None
        self.message = "undo"

    def reset(self) -> None:
        """Повторный выбор объекта с нуля."""
        self.result, self.accepted, self._corner, self._poly = None, False, None, []
        self.history.clear()
        self.message = "cleared: select the object again"

    def accept(self) -> bool:
        if self.result is None:
            self.message = "nothing to accept: select an object first"
            return False
        self.accepted = True
        self.message = "accepted: " + self.result.summary()
        return True

    def save(self, directory) -> Dict[str, Path]:
        if self.result is None:
            raise InitializationError("нечего сохранять: объект не выбран")
        return save_target(self.result, directory, self.image)

    def key(self, name: str) -> str:
        """Обработка клавиши; возвращает действие для окна: '', 'save' или 'quit'."""
        name = name.lower()
        if name in ("p", "b", "r"):
            self.set_mode({"p": "point", "b": "box", "r": "roi"}[name])
        elif name in ("enter", "a"):
            if self.mode == "roi" and len(self._poly) >= 3:
                poly, self._poly = self._poly, []
                self._run(Selection.from_roi(poly))
            else:
                self.accept()
        elif name == "u":
            self.undo()
        elif name == "c":
            self.reset()
        elif name == "s":
            return "save"
        elif name in ("q", "esc"):
            return "quit"
        return ""

    def render(self) -> np.ndarray:
        pending = {"corner": self._corner, "polygon": self._poly}
        lines = [f"[{self.mode}] " + ("ACCEPTED" if self.accepted else "select the object")]
        return render_overlay(self.image, self.result, None, pending, lines)
