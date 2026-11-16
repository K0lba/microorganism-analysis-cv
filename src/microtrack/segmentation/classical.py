"""Классический baseline сегментации (без обучения нейросетей).

Этапы: оттенки серого → бинаризация (Otsu / адаптивная / фиксированная) →
морфология → (опционально) разделение слипшихся объектов → поиск компонент и
контуров → фильтрация по площади, размерам и форме.

Все параметры собраны в :class:`SegmentationConfig` и читаются из словаря/YAML.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..io import Calibration, Frame
from ..preprocess import PreprocessConfig, ensure_uint8, preprocess, to_grayscale


# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #
def _normalize_dict(d: Optional[dict]) -> dict:
    """Приводит секцию ``segmentation`` из YAML проекта к вложенной схеме конфига.

    Принимаются плоские ключи проекта: method, threshold_method, morphology.enabled /
    kernel_size, min_object_area_px, max_object_area_px.
    """
    d = dict(d or {})
    if d.pop("method", "classical") != "classical":
        raise ValueError("segmentation.method: для classical.py допустим только 'classical'")

    thr = dict(d.get("threshold") or {})
    if "threshold_method" in d:
        thr.setdefault("method", d.pop("threshold_method"))
    if thr:
        d["threshold"] = thr

    if "morphology" in d:
        morph = dict(d["morphology"] or {})
        enabled = morph.pop("enabled", True)
        ks = morph.pop("kernel_size", None)
        if ks is not None:
            morph.setdefault("open_ksize", ks)
            morph.setdefault("close_ksize", ks)
        if not enabled:
            morph.update(open_ksize=0, close_ksize=0, fill_holes=False)
        d["morphology"] = morph

    flt = dict(d.get("filters") or {})
    for old, new in (("min_object_area_px", "min_area"), ("max_object_area_px", "max_area")):
        if old in d:
            val = d.pop(old)
            if val is not None:
                flt.setdefault(new, val)
    if flt:
        d["filters"] = flt
    return d


@dataclass
class ThresholdConfig:
    method: str = "otsu"            # otsu | adaptive | fixed
    objects: str = "auto"           # dark | bright | auto (объекты — меньшая часть кадра)
    block_size: int = 51            # adaptive: окно (нечётное >= 3)
    offset: float = 5.0             # adaptive: насколько пиксель должен отличаться от локального среднего
    fixed_value: int = 128          # fixed: порог


@dataclass
class MorphologyConfig:
    open_ksize: int = 3             # 0 — выключено
    close_ksize: int = 3            # 0 — выключено
    kernel: str = "ellipse"         # ellipse | rect
    fill_holes: bool = True


@dataclass
class FilterConfig:
    unit: str = "px"                # px | um — единицы порогов ниже (площадь в unit², размеры в unit)
    min_area: float = 20.0
    max_area: Optional[float] = None
    min_length: Optional[float] = None      # длинная сторона минимального прямоугольника
    max_length: Optional[float] = None
    min_width: Optional[float] = None       # короткая сторона минимального прямоугольника
    max_width: Optional[float] = None
    min_circularity: Optional[float] = None  # 4πA/P², 1 — круг
    min_solidity: Optional[float] = None     # площадь / площадь выпуклой оболочки
    max_aspect_ratio: Optional[float] = None  # длина / ширина
    exclude_border: bool = False            # убрать объекты, касающиеся границы кадра


@dataclass
class SplitConfig:
    """Экспериментальное разделение слипшихся объектов (distance transform + затопление от маркеров)."""
    enabled: bool = False
    peak_fraction: float = 0.6      # маркер = область, где расстояние >= доля * максимум в компоненте
    smooth_sigma: float = 1.0       # сглаживание карты расстояний
    min_component_area: int = 40    # меньшие компоненты не делятся


@dataclass
class SegmentationConfig:
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "SegmentationConfig":
        d = _normalize_dict(d)
        parts = {"threshold": ThresholdConfig, "morphology": MorphologyConfig,
                 "filters": FilterConfig, "split": SplitConfig}
        unknown = set(d) - set(parts)
        if unknown:
            raise ValueError(f"Неизвестные секции сегментации: {sorted(unknown)}")
        kwargs = {}
        for name, klass in parts.items():
            sub = d.get(name)
            if sub is None:
                continue
            if name == "split" and sub is True:
                sub = {"enabled": True}
            bad = set(sub) - {f.name for f in fields(klass)}
            if bad:
                raise ValueError(f"Неизвестные параметры в '{name}': {sorted(bad)}")
            kwargs[name] = klass(**sub)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        t, m, f, s = self.threshold, self.morphology, self.filters, self.split
        if t.method not in ("otsu", "adaptive", "fixed"):
            raise ValueError(f"Неизвестный метод бинаризации: {t.method}")
        if t.objects not in ("dark", "bright", "auto"):
            raise ValueError("threshold.objects: dark | bright | auto")
        if t.block_size < 3 or t.block_size % 2 == 0:
            raise ValueError("threshold.block_size должен быть нечётным и >= 3")
        if m.kernel not in ("ellipse", "rect"):
            raise ValueError("morphology.kernel: ellipse | rect")
        for name in ("open_ksize", "close_ksize"):
            if getattr(m, name) < 0:
                raise ValueError(f"morphology.{name} не может быть отрицательным")
        if f.unit not in ("px", "um"):
            raise ValueError("filters.unit: px | um")
        if f.max_area is not None and f.max_area < f.min_area:
            raise ValueError("filters.max_area меньше min_area")
        if not 0 < s.peak_fraction <= 1:
            raise ValueError("split.peak_fraction должен быть в (0, 1]")


# --------------------------------------------------------------------------- #
# Результат
# --------------------------------------------------------------------------- #
@dataclass
class ObjectInfo:
    label: int
    area_px: int
    area_um2: float
    centroid: Tuple[float, float]           # (x, y), пиксели
    bbox: Tuple[int, int, int, int]         # x, y, w, h
    contour: np.ndarray                     # (N, 2) int, внешний контур (x, y)
    perimeter_px: float
    length_px: float
    width_px: float
    aspect_ratio: float
    circularity: float
    solidity: float


@dataclass
class SegmentationResult:
    gray: np.ndarray                        # изображение, по которому шла сегментация
    raw_mask: np.ndarray                    # uint8 0/255 после морфологии, до фильтрации
    mask: np.ndarray                        # uint8 0/255, только принятые объекты
    labels: np.ndarray                      # int32, 0 — фон, 1..N — объекты
    objects: List[ObjectInfo]
    rejected: Dict[str, int]                # причина → сколько компонент отброшено
    calibration: Calibration

    def __len__(self) -> int:
        return len(self.objects)


# --------------------------------------------------------------------------- #
# Шаги
# --------------------------------------------------------------------------- #
def _objects_are_dark(gray: np.ndarray, setting: str) -> bool:
    if setting == "dark":
        return True
    if setting == "bright":
        return False
    # auto: объекты — меньшая часть кадра по Otsu
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (b > 0).mean() > 0.5


def binarize(gray: np.ndarray, cfg: ThresholdConfig) -> np.ndarray:
    """Возвращает uint8 0/255, где 255 — объекты."""
    dark = _objects_are_dark(gray, cfg.objects)
    if cfg.method == "otsu":
        mode = cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY
        _, b = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    elif cfg.method == "fixed":
        mode = cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY
        _, b = cv2.threshold(gray, cfg.fixed_value, 255, mode)
    else:  # adaptive
        mode = cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY
        c = cfg.offset if dark else -cfg.offset
        b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode,
                                  cfg.block_size, c)
    return b


def _kernel(shape: str, k: int) -> np.ndarray:
    kind = cv2.MORPH_ELLIPSE if shape == "ellipse" else cv2.MORPH_RECT
    return cv2.getStructuringElement(kind, (k, k))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def clean_mask(mask: np.ndarray, cfg: MorphologyConfig) -> np.ndarray:
    out = mask
    if cfg.open_ksize > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, _kernel(cfg.kernel, cfg.open_ksize))
    if cfg.close_ksize > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _kernel(cfg.kernel, cfg.close_ksize))
    if cfg.fill_holes:
        out = _fill_holes(out)
    return out


def _flood_from_markers(markers: np.ndarray, dist: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Затопление от маркеров внутри ``inside`` по убыванию расстояния до фона.

    Простая замена cv2.watershed: тот ориентируется на перепады цвета, а не на
    высоту рельефа, и «протекает» наружу по плоской карте расстояний.
    """
    import heapq

    lab = markers.copy()
    lab[~inside] = 0
    h, w = lab.shape
    heap = []
    ys, xs = np.nonzero(lab)
    for n, (y, x) in enumerate(zip(ys.tolist(), xs.tolist())):
        heap.append((-float(dist[y, x]), n, y, x))
    heapq.heapify(heap)
    counter = len(heap)
    while heap:
        _, _, y, x = heapq.heappop(heap)
        l = lab[y, x]
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and inside[ny, nx] and lab[ny, nx] == 0:
                lab[ny, nx] = l
                counter += 1
                heapq.heappush(heap, (-float(dist[ny, nx]), counter, ny, nx))
    return lab


def split_touching(mask: np.ndarray, peak_fraction: float = 0.6,
                   smooth_sigma: float = 1.0, min_component_area: int = 40) -> np.ndarray:
    """ЭКСПЕРИМЕНТАЛЬНО: делит слипшиеся объекты линиями разреза шириной 1 px.

    Для каждой компоненты строится карта расстояний до фона; маркеры — области
    с расстоянием >= peak_fraction * максимум. Если маркеров >= 2, компонента
    затапливается от маркеров, а линия раздела между объектами обнуляется
    в маске (чистый Python, на больших кадрах заметно медленнее). Результат
    нужно размечать с 4-связностью (так делает :func:`segment`).
    """
    binary = (mask > 0).astype(np.uint8)
    out = binary.copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    pad = 3
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_component_area:
            continue
        crop = np.pad((lab[y:y + h, x:x + w] == i).astype(np.uint8), pad)
        dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
        if smooth_sigma > 0:
            dist = cv2.GaussianBlur(dist, (0, 0), smooth_sigma)
        peaks = (dist >= peak_fraction * dist.max()) & (crop > 0)
        k, markers = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
        k -= 1                                  # число маркеров без фона
        if k < 2:
            continue
        lab_c = _flood_from_markers(markers.astype(np.int32), dist, crop > 0)

        # линия разреза: у каждой границы между метками обнуляем пиксель с большей меткой
        cut = np.zeros(lab_c.shape, bool)
        for dy, dx in ((0, 1), (1, 0)):
            a_ = lab_c[: lab_c.shape[0] - dy, : lab_c.shape[1] - dx]
            b_ = lab_c[dy:, dx:]
            diff = (a_ > 0) & (b_ > 0) & (a_ != b_)
            cut[: lab_c.shape[0] - dy, : lab_c.shape[1] - dx] |= diff & (a_ > b_)
            cut[dy:, dx:] |= diff & (b_ > a_)
        region = out[y:y + h, x:x + w]
        region[cut[pad:pad + h, pad:pad + w]] = 0
    return (out * 255).astype(np.uint8)


def _contour_of(label_crop: np.ndarray, x0: int, y0: int) -> np.ndarray:
    padded = np.pad(label_crop.astype(np.uint8), 1)
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea) if len(contours) > 1 else contours[0]
    return c.reshape(-1, 2) + np.array([x0 - 1, y0 - 1])


def _measure(label: int, crop: np.ndarray, stats_row, centroid, calib: Calibration) -> ObjectInfo:
    x, y, w, h, area = (int(v) for v in stats_row)
    contour = _contour_of(crop, x, y)
    cnt = contour.reshape(-1, 1, 2).astype(np.int32)
    perimeter = float(cv2.arcLength(cnt, True))
    circ = min(1.0, 4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    solidity = min(1.0, area / hull_area) if hull_area > 0 else 1.0
    if len(contour) >= 3:
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    else:
        rw, rh = float(w), float(h)
    length, width = max(rw, rh), min(rw, rh)
    # у очень тонких/маленьких объектов minAreaRect даёт 0 по ширине
    width = max(width, 1.0)
    length = max(length, width)
    return ObjectInfo(
        label=label, area_px=area, area_um2=calib.area_px_to_um2(area),
        centroid=(float(centroid[0]), float(centroid[1])), bbox=(x, y, w, h),
        contour=contour, perimeter_px=perimeter, length_px=length, width_px=width,
        aspect_ratio=length / width, circularity=circ, solidity=solidity,
    )


def describe_mask(mask: np.ndarray, calibration: Optional[Calibration] = None,
                  label: int = 1) -> ObjectInfo:
    """Характеристики одиночного объекта по бинарной маске (берётся наибольшая 4-связная компонента).

    Возвращает центр, площадь (px и в единицах калибровки), bbox, внешний контур и показатели формы.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n < 2:
        raise ValueError("Маска пуста")
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[i, :4])
    crop = lab[y:y + h, x:x + w] == i
    return _measure(label, crop, stats[i], cents[i], calibration or Calibration())


def _reject_reason(o: ObjectInfo, f: FilterConfig, calib: Calibration,
                   shape: Tuple[int, int]) -> Optional[str]:
    # пороги -> пиксели
    if f.unit == "um":
        s = calib.pixel_size_um
        k_len, k_area = 1.0 / s, 1.0 / (s * s)
    else:
        k_len = k_area = 1.0
    scale = lambda v, k: None if v is None else v * k
    min_a, max_a = f.min_area * k_area, scale(f.max_area, k_area)

    if o.area_px < min_a:
        return "min_area"
    if max_a is not None and o.area_px > max_a:
        return "max_area"
    if f.min_length is not None and o.length_px < f.min_length * k_len:
        return "min_length"
    if f.max_length is not None and o.length_px > f.max_length * k_len:
        return "max_length"
    if f.min_width is not None and o.width_px < f.min_width * k_len:
        return "min_width"
    if f.max_width is not None and o.width_px > f.max_width * k_len:
        return "max_width"
    if f.min_circularity is not None and o.circularity < f.min_circularity:
        return "min_circularity"
    if f.min_solidity is not None and o.solidity < f.min_solidity:
        return "min_solidity"
    if f.max_aspect_ratio is not None and o.aspect_ratio > f.max_aspect_ratio:
        return "max_aspect_ratio"
    if f.exclude_border:
        x, y, w, h = o.bbox
        if x == 0 or y == 0 or x + w >= shape[1] or y + h >= shape[0]:
            return "border"
    return None


# --------------------------------------------------------------------------- #
# Публичный интерфейс
# --------------------------------------------------------------------------- #
def segment(
    image: np.ndarray,
    cfg: Optional[SegmentationConfig] = None,
    calibration: Optional[Calibration] = None,
    preprocess_cfg: Optional[PreprocessConfig] = None,
) -> SegmentationResult:
    """Полный classical baseline: изображение → маски и список объектов.

    Если задан ``preprocess_cfg``, сначала применяется предобработка (масштаб
    кадра при этом пересчитывает калибровку).
    """
    cfg = cfg or SegmentationConfig()
    cfg.validate()
    calib = calibration or Calibration()
    if cfg.filters.unit == "um" and not calib.calibrated:
        raise ValueError("filters.unit = 'um', но калибровка не задана (pixel_size_um: null)")

    if preprocess_cfg is not None:
        gray, calib = preprocess(image, replace(preprocess_cfg, grayscale=True), calib)
        gray = to_grayscale(gray)
    else:
        gray = to_grayscale(ensure_uint8(image))

    raw = clean_mask(binarize(gray, cfg.threshold), cfg.morphology)
    work = raw
    if cfg.split.enabled:
        work = split_touching(raw, cfg.split.peak_fraction, cfg.split.smooth_sigma,
                              cfg.split.min_component_area)

    n, lab, stats, cents = cv2.connectedComponentsWithStats(
        (work > 0).astype(np.uint8), connectivity=4)
    labels = np.zeros(gray.shape, np.int32)
    objects: List[ObjectInfo] = []
    rejected: Dict[str, int] = {}
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        crop = lab[y:y + h, x:x + w] == i
        info = _measure(len(objects) + 1, crop, stats[i], cents[i], calib)
        reason = _reject_reason(info, cfg.filters, calib, gray.shape)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        labels[y:y + h, x:x + w][crop] = info.label
        objects.append(info)

    return SegmentationResult(
        gray=gray, raw_mask=raw, mask=((labels > 0) * 255).astype(np.uint8),
        labels=labels, objects=objects, rejected=rejected, calibration=calib,
    )


def segment_frame(frame: Frame, cfg: Optional[SegmentationConfig] = None,
                  preprocess_cfg: Optional[PreprocessConfig] = None) -> SegmentationResult:
    return segment(frame.image, cfg, frame.calibration, preprocess_cfg)
