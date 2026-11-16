"""Настраиваемая предобработка кадров.

Каждая операция включается отдельно и может быть отключена. Порядок задаётся
``PreprocessConfig.order``. Изменение масштаба кадра пересчитывает калибровку.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional, Tuple

import cv2
import numpy as np

from .io import Calibration, Frame

DEFAULT_ORDER = ("grayscale", "rescale", "denoise", "median", "clahe", "sharpness")


# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #
@dataclass
class ClaheConfig:
    clip_limit: float = 2.0
    tile_grid: int = 8


@dataclass
class DenoiseConfig:
    method: str = "gaussian"      # gaussian | bilateral | nlm | median
    strength: float = 1.0         # sigma (gaussian) / h (nlm) / sigmaColor (bilateral)
    kernel_size: int = 3          # только для median (нечётное >= 3)


@dataclass
class PreprocessConfig:
    enabled: bool = True                     # False — кадр отдаётся как есть (только uint8)
    grayscale: bool = True
    scale: float = 1.0                       # 1.0 — без изменения размера
    denoise: Optional[DenoiseConfig] = None
    median_ksize: int = 0                    # 0 — выключено, иначе нечётное >= 3
    clahe: Optional[ClaheConfig] = None
    sharpness: float = 1.0                   # 1 — без изменений, >1 резче, <1 мягче
    order: Tuple[str, ...] = DEFAULT_ORDER

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PreprocessConfig":
        """Создаёт конфиг из словаря (например, из секции YAML)."""
        d = dict(d or {})

        # секция sharpen {enabled, amount} -> sharpness = 1 + amount
        if "sharpen" in d:
            sh = d.pop("sharpen")
            if isinstance(sh, dict):
                bad = set(sh) - {"enabled", "amount"}
                if bad:
                    raise ValueError(f"Неизвестные параметры sharpen: {sorted(bad)}")
                if sh.get("enabled", True):
                    d.setdefault("sharpness", 1.0 + float(sh.get("amount", 1.0)))
            elif sh:
                d.setdefault("sharpness", 2.0)

        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Неизвестные параметры предобработки: {sorted(unknown)}")

        def _section(name, klass, aliases=None):
            v = d.get(name)
            if isinstance(v, dict):
                v = dict(v)
                if not v.pop("enabled", True):
                    d[name] = None
                    return
                for old, new in (aliases or {}).items():
                    if old in v:
                        v[new] = v.pop(old)
                bad = set(v) - {f.name for f in fields(klass)}
                if bad:
                    raise ValueError(f"Неизвестные параметры в '{name}': {sorted(bad)}")
                d[name] = klass(**v)
            elif v is True:
                d[name] = klass()
            elif v is False:
                d[name] = None
            elif isinstance(v, str) and klass is DenoiseConfig:
                d[name] = DenoiseConfig(method=v)

        _section("clahe", ClaheConfig, {"tile_grid_size": "tile_grid"})
        _section("denoise", DenoiseConfig)
        if "order" in d:
            d["order"] = tuple(d["order"])
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale должен быть положительным")
        if self.median_ksize and (self.median_ksize < 3 or self.median_ksize % 2 == 0):
            raise ValueError("median_ksize должен быть нечётным и >= 3 (или 0)")
        if self.sharpness < 0:
            raise ValueError("sharpness не может быть отрицательным")
        if self.denoise and self.denoise.method not in ("gaussian", "bilateral", "nlm", "median"):
            raise ValueError(f"Неизвестный метод шумоподавления: {self.denoise.method}")
        if self.denoise and self.denoise.method == "median" and (
                self.denoise.kernel_size < 3 or self.denoise.kernel_size % 2 == 0):
            raise ValueError("denoise.kernel_size должен быть нечётным и >= 3")
        if self.clahe and (self.clahe.clip_limit <= 0 or self.clahe.tile_grid < 1):
            raise ValueError("Некорректные параметры CLAHE")
        bad = set(self.order) - set(DEFAULT_ORDER)
        if bad:
            raise ValueError(f"Неизвестные операции в order: {sorted(bad)}")


# --------------------------------------------------------------------------- #
# Отдельные операции
# --------------------------------------------------------------------------- #
def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Приводит изображение к uint8 (uint16 → /257, float → min-max)."""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return np.round(img / 257.0).astype(np.uint8)
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, np.uint8)
    return np.round((img - lo) / (hi - lo) * 255).astype(np.uint8)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """CLAHE; для цветных кадров применяется к яркости (канал L в Lab)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    if img.ndim == 2:
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_median(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError("ksize должен быть нечётным и >= 3")
    return cv2.medianBlur(img, ksize)


def adjust_sharpness(img: np.ndarray, factor: float = 1.0, sigma: float = 1.5) -> np.ndarray:
    """Unsharp mask: factor=1 — без изменений, >1 повышает резкость, <1 смягчает."""
    if factor == 1.0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, factor, blurred, 1.0 - factor, 0)


def denoise(img: np.ndarray, method: str = "gaussian", strength: float = 1.0,
            kernel_size: int = 3) -> np.ndarray:
    """Подавление шума: gaussian | bilateral | nlm (non-local means) | median."""
    if method == "median":
        return apply_median(img, kernel_size)
    if strength <= 0:
        return img
    if method == "gaussian":
        return cv2.GaussianBlur(img, (0, 0), strength)
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=0, sigmaColor=strength * 25, sigmaSpace=strength * 3)
    if method == "nlm":
        if img.ndim == 2:
            return cv2.fastNlMeansDenoising(img, None, h=strength * 5)
        return cv2.fastNlMeansDenoisingColored(img, None, strength * 5, strength * 5)
    raise ValueError(f"Неизвестный метод шумоподавления: {method}")


def rescale(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)


# --------------------------------------------------------------------------- #
# Пайплайн
# --------------------------------------------------------------------------- #
def preprocess(
    img: np.ndarray,
    cfg: Optional[PreprocessConfig] = None,
    calibration: Optional[Calibration] = None,
) -> Tuple[np.ndarray, Calibration]:
    """Применяет включённые операции. Возвращает (изображение uint8, калибровка)."""
    cfg = cfg or PreprocessConfig()
    cfg.validate()
    calib = calibration or Calibration()
    out = ensure_uint8(img)
    if not cfg.enabled:
        return out, calib

    for op in cfg.order:
        if op == "grayscale" and cfg.grayscale:
            out = to_grayscale(out)
        elif op == "rescale" and cfg.scale != 1.0:
            out = rescale(out, cfg.scale)
            calib = calib.rescaled(cfg.scale)
        elif op == "denoise" and cfg.denoise is not None:
            out = denoise(out, cfg.denoise.method, cfg.denoise.strength, cfg.denoise.kernel_size)
        elif op == "median" and cfg.median_ksize:
            out = apply_median(out, cfg.median_ksize)
        elif op == "clahe" and cfg.clahe is not None:
            out = apply_clahe(out, cfg.clahe.clip_limit, cfg.clahe.tile_grid)
        elif op == "sharpness" and cfg.sharpness != 1.0:
            out = adjust_sharpness(out, cfg.sharpness)
    return out, calib


def preprocess_frame(frame: Frame, cfg: Optional[PreprocessConfig] = None) -> Frame:
    """Предобработка кадра с сохранением индекса и исходного времени."""
    img, calib = preprocess(frame.image, cfg, frame.calibration)
    return replace(frame, image=img, calibration=calib)
