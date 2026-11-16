"""Единый интерфейс загрузки изображений и видео.

Любой источник (одиночное изображение, TIFF-стек, папка с кадрами, видеофайл)
отдаётся как поток объектов :class:`Frame` с исходным индексом и временем кадра.
Прореживание (``step``) не меняет время: кадр N остаётся кадром N.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"}

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Калибровка
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Calibration:
    """Масштаб микроскопа.

    pixel_size_um — размер одного пикселя в микрометрах (мкм/пикс).
    Можно задать напрямую или вычислить через :meth:`from_reference`.
    """

    pixel_size_um: float = 1.0
    unit: str = "um"
    description: str = ""
    calibrated: bool = True         # False — масштаб неизвестен, размеры остаются в пикселях

    def __post_init__(self) -> None:
        if not np.isfinite(self.pixel_size_um) or self.pixel_size_um <= 0:
            raise ValueError("pixel_size_um должен быть положительным числом")

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "Calibration":
        """Секция ``calibration`` конфига; ``pixel_size_um: null`` — не откалибровано."""
        d = dict(d or {})
        bad = set(d) - {"pixel_size_um", "unit", "description"}
        if bad:
            raise ValueError(f"Неизвестные параметры калибровки: {sorted(bad)}")
        if d.get("pixel_size_um") is None:
            return cls(1.0, "px", "uncalibrated", calibrated=False)
        return cls(float(d["pixel_size_um"]), d.get("unit", "um"), d.get("description", ""))

    @classmethod
    def from_reference(cls, length_um: float, length_px: float, **kw) -> "Calibration":
        """Калибровка по эталону: известная длина в мкм и её длина в пикселях."""
        if length_px <= 0 or length_um <= 0:
            raise ValueError("length_um и length_px должны быть положительными")
        return cls(pixel_size_um=length_um / length_px, **kw)

    def px_to_um(self, value_px: float) -> float:
        return value_px * self.pixel_size_um

    def um_to_px(self, value_um: float) -> float:
        return value_um / self.pixel_size_um

    def area_px_to_um2(self, area_px: float) -> float:
        return area_px * self.pixel_size_um ** 2

    def rescaled(self, scale: float) -> "Calibration":
        """Калибровка после изменения размера кадра в ``scale`` раз."""
        if scale <= 0:
            raise ValueError("scale должен быть положительным")
        return Calibration(self.pixel_size_um / scale, self.unit, self.description, self.calibrated)


# --------------------------------------------------------------------------- #
# Кадр
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    image: np.ndarray
    index: int                      # исходный номер кадра в источнике
    time_s: float                   # исходное время, сек
    source: str = ""
    calibration: Calibration = field(default_factory=Calibration)

    @property
    def shape(self) -> tuple:
        return self.image.shape


# --------------------------------------------------------------------------- #
# Чтение
# --------------------------------------------------------------------------- #
def is_video(path: PathLike) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image(path: PathLike) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _finalize(img: np.ndarray, grayscale: bool) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return _to_gray(img) if grayscale else img


def read_image(path: PathLike, grayscale: bool = False) -> np.ndarray:
    """Читает одиночное изображение (PNG/JPG/TIFF/BMP), сохраняя разрядность."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not is_image(path):
        raise ValueError(f"Неподдерживаемый формат изображения: {path.suffix}")
    # imdecode, а не imread: корректно работает с не-ASCII путями
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return _finalize(img, grayscale)


def _check_step(step: int) -> None:
    if not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("step должен быть целым числом >= 1")


def _selected(index: int, step: int, start: int, end: Optional[int]) -> bool:
    return index >= start and (end is None or index < end) and (index - start) % step == 0


def _iter_video(path: Path, step: int, start: int, end: Optional[int],
                fps: Optional[float], timestamp_source: str = "fps") -> Iterator[tuple]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    try:
        file_fps = cap.get(cv2.CAP_PROP_FPS)
        use_fps = fps if fps else file_fps
        if not use_fps or use_fps <= 0 or not np.isfinite(use_fps):
            raise ValueError("FPS видео не определён, передайте fps явно")
        index = 0
        while True:
            # grab() не декодирует кадр — пропуск лишних кадров дешёвый
            if not cap.grab():
                break
            if end is not None and index >= end:
                break
            if _selected(index, step, start, end):
                t = index / use_fps
                if timestamp_source == "source_pts" and not fps:
                    # отметка кадра из контейнера; если её нет — index / fps
                    pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if np.isfinite(pos) and (pos > 0 or index == 0):
                        t = pos / 1000.0
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield index, frame, t
            index += 1
    finally:
        cap.release()


def iter_frames(
    source: PathLike,
    step: int = 1,
    start: int = 0,
    end: Optional[int] = None,
    fps: Optional[float] = None,
    grayscale: bool = False,
    calibration: Optional[Calibration] = None,
    timestamp_source: str = "fps",
) -> Iterator[Frame]:
    """Единый итератор кадров.

    source: видеофайл, изображение, многостраничный TIFF или папка с изображениями.
    step:   брать каждый N-й кадр; индекс и время остаются исходными.
    start/end: диапазон исходных индексов [start, end).
    fps:    частота кадров (для видео — переопределяет значение из файла;
            для папки/стека задаёт время кадра, по умолчанию 1 кадр = 1 с).
    timestamp_source: ``fps`` — время = index / fps; ``source_pts`` — отметки
            времени из контейнера видео (если их нет, то index / fps).
            Явно заданный ``fps`` всегда имеет приоритет.
    """
    if timestamp_source not in ("fps", "source_pts"):
        raise ValueError("timestamp_source: fps | source_pts")
    _check_step(step)
    if start < 0 or (end is not None and end < start):
        raise ValueError("Некорректный диапазон start/end")
    if fps is not None and fps <= 0:
        raise ValueError("fps должен быть положительным")
    calib = calibration or Calibration()
    source = Path(source)
    name = str(source)

    if source.is_dir():
        files = sorted(p for p in source.iterdir() if is_image(p))
        if not files:
            raise ValueError(f"В папке нет изображений: {source}")
        dt = 1.0 / fps if fps else 1.0
        for i, p in enumerate(files):
            if _selected(i, step, start, end):
                yield Frame(read_image(p, grayscale), i, i * dt, str(p), calib)
        return

    if not source.exists():
        raise FileNotFoundError(source)

    if is_video(source):
        for i, frame, t in _iter_video(source, step, start, end, fps, timestamp_source):
            yield Frame(_finalize(frame, grayscale), i, t, name, calib)
        return

    if is_image(source):
        if source.suffix.lower() in {".tif", ".tiff"}:
            ok, pages = cv2.imreadmulti(str(source), flags=cv2.IMREAD_UNCHANGED)
            if ok and len(pages) > 1:
                dt = 1.0 / fps if fps else 1.0
                for i, page in enumerate(pages):
                    if _selected(i, step, start, end):
                        yield Frame(_finalize(page, grayscale), i, i * dt, name, calib)
                return
        if _selected(0, step, start, end):
            yield Frame(read_image(source, grayscale), 0, 0.0, name, calib)
        return

    raise ValueError(f"Неподдерживаемый формат источника: {source.suffix}")


def load_frames(source: PathLike, **kwargs) -> list:
    """То же, что :func:`iter_frames`, но возвращает список."""
    return list(iter_frames(source, **kwargs))


def video_info(path: PathLike) -> dict:
    """Метаданные видео: fps, число кадров, размер."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    try:
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()
