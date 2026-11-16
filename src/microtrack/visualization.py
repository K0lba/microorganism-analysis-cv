"""Визуализация трекинга: маски, контуры, ID и траектории поверх кадра.

Обозначения на кадре:
  заливка и контур цвета трека   — объект найден и сопоставлен;
  красный контур                 — сомнительная сегментация (seg_flag != ok);
  ``ID?`` жёлтым                 — неоднозначное сопоставление (ambiguous);
  серый крестик ``ID lost``      — объект пропал, показано предсказанное положение;
  линия цвета трека              — последние наблюдения траектории.

Видео пишется во временный каталог с ASCII-путём и затем переносится:
``cv2.VideoWriter`` на Windows не открывает пути с кириллицей.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .preprocess import ensure_uint8

BAD_SEG_COLOR = (0, 0, 255)
AMBIGUOUS_COLOR = (0, 255, 255)
LOST_COLOR = (170, 170, 170)


def to_bgr(image: np.ndarray) -> np.ndarray:
    img = ensure_uint8(np.asarray(image))
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def track_color(track_id: int) -> Tuple[int, int, int]:
    """Устойчивый цвет трека (BGR); соседние ID получают далёкие оттенки."""
    hue = int((track_id * 0.618033988749895 % 1.0) * 180)
    hsv = np.uint8([[[hue, 210, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def label_edges(labels: np.ndarray) -> np.ndarray:
    """Границы между разными метками (включая границу с фоном)."""
    lab = np.asarray(labels)
    edge = np.zeros(lab.shape, bool)
    diff_x = lab[:, 1:] != lab[:, :-1]
    diff_y = lab[1:, :] != lab[:-1, :]
    edge[:, 1:] |= diff_x
    edge[:, :-1] |= diff_x
    edge[1:, :] |= diff_y
    edge[:-1, :] |= diff_y
    return edge & (lab > 0)


def _fit_scale(text: str, scale: float, max_width: int) -> float:
    """Уменьшить масштаб шрифта, чтобы строка не выходила за max_width px."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    while scale > 0.25 and cv2.getTextSize(text, font, scale, 1)[0][0] > max_width:
        scale -= 0.02
    return scale


def put_label(img, text, org, color=(255, 255, 255), scale=0.4):
    """Подпись с чёрной обводкой.

    Толстый ``putText`` (thickness=3) в OpenCV рисуется не по центру тонкого
    глифа, а сдвинутой копией. Обводка — тот же шрифт толщиной 1 вокруг точки.
    Масштаб уменьшается, если строка не помещается в ширину кадра.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = _fit_scale(text, scale, img.shape[1] - int(org[0]) - 4)
    x, y = int(org[0]), int(org[1])
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            cv2.putText(img, text, (x + dx, y + dy), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def draw_tracking_frame(image: np.ndarray, masks: Sequence[np.ndarray],
                        assigned: Dict[int, int], tracker, frame: int,
                        time_s: Optional[float] = None, trail: int = 20,
                        alpha: float = 0.4, show_ids: bool = True) -> np.ndarray:
    """Кадр с наложением состояния ``Tracker`` сразу после ``tracker.update(frame, masks)``.

    ``assigned`` — результат ``update``: {индекс маски: id трека}.
    """
    vis = to_bgr(image)
    h, w = vis.shape[:2]
    tl = np.zeros((h, w), np.int32)
    for i, m in enumerate(masks):
        tid = assigned.get(i)
        if tid is not None:
            tl[np.asarray(m, bool)] = tid

    current = {tid: tr.observations[-1] for tid, tr in tracker.tracks.items()
               if tr.observations and tr.observations[-1].frame == frame}

    fg = tl > 0
    if fg.any():
        lut = np.zeros((int(tl.max()) + 1, 3), np.uint8)
        for tid in np.unique(tl[fg]):
            lut[tid] = track_color(int(tid))
        color = lut[tl]
        vis[fg] = (vis[fg] * (1.0 - alpha) + color[fg] * alpha).astype(np.uint8)
        edge = label_edges(tl)
        bad_ids = [tid for tid, o in current.items()
                   if o.observed and str(o.extra.get("seg_flag", "ok")) != "ok"]
        bad = np.isin(tl, bad_ids) if bad_ids else np.zeros_like(fg)
        vis[edge & ~bad] = color[edge & ~bad]
        vis[edge & bad] = BAD_SEG_COLOR

    n_obs = n_lost = n_amb = 0
    for tid, o in current.items():
        tr = tracker.tracks[tid]
        col = track_color(tid)
        pts = np.array([[round(p.features.center[0]), round(p.features.center[1])]
                        for p in tr.observations[-trail:]], np.int32)
        if len(pts) > 1:
            cv2.polylines(vis, [pts], False, col, 1, cv2.LINE_AA)
        cx, cy = int(pts[-1][0]), int(pts[-1][1])
        if o.observed:
            n_obs += 1
            n_amb += int(o.ambiguous)
            if show_ids:
                put_label(vis, f"{tid}?" if o.ambiguous else str(tid), (cx + 4, cy - 4),
                      AMBIGUOUS_COLOR if o.ambiguous else col)
        else:
            n_lost += 1
            cv2.drawMarker(vis, (cx, cy), LOST_COLOR, cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)
            if show_ids:
                put_label(vis, f"{tid} lost", (cx + 4, cy - 4), LOST_COLOR)

    head = f"frame {frame}"
    if time_s is not None:
        head += f"  t={time_s:.2f}s"
    stats = f"objects {n_obs}  lost {n_lost}  ambiguous {n_amb}"
    put_label(vis, head, (6, 14), (255, 255, 255), 0.45)
    put_label(vis, stats, (6, 28), (255, 255, 255), 0.45)
    put_label(vis, "red: bad seg  ?: ambiguous  x: lost", (6, h - 8), (255, 255, 255), 0.4)
    return vis


def write_image(path, image: np.ndarray) -> str:
    """PNG/JPG по любому пути (cv2.imwrite не открывает пути с кириллицей)."""
    p = Path(path)
    ok, buf = cv2.imencode(p.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Не удалось закодировать изображение: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(p))
    return str(p)


class OverlayWriter:
    """Собирает кадры с наложением: MP4, PNG каждого кадра и превью последнего."""

    def __init__(self, out_dir, fps: float = 5.0, save_video: bool = True,
                 save_frames: bool = False, video_name: str = "overlay.mp4"):
        self.out_dir = Path(out_dir)
        self.fps = float(fps) if fps and fps > 0 else 5.0
        self.save_video, self.save_frames = save_video, save_frames
        self.video_path = self.out_dir / video_name
        self._writer = None
        self._tmp: Optional[str] = None
        self._last: Optional[np.ndarray] = None
        self.files: Dict[str, str] = {}

    def add(self, image: np.ndarray, frame: int) -> None:
        self._last = image
        if self.save_frames:
            write_image(self.out_dir / "frames" / f"frame_{frame:06d}.png", image)
            self.files["frames"] = str(self.out_dir / "frames")
        if not self.save_video:
            return
        if self._writer is None:
            fd, self._tmp = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            h, w = image.shape[:2]
            self._writer = cv2.VideoWriter(self._tmp, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self._writer.isOpened():
                warnings.warn("Не удалось открыть VideoWriter (mp4v); видео с наложением не сохраняется")
                self.save_video = False
                self._writer = None
                return
        self._writer.write(image)

    def close(self) -> Dict[str, str]:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self.out_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(self._tmp, self.video_path)
            self.files["video"] = str(self.video_path)
        elif self._tmp and os.path.exists(self._tmp):
            os.remove(self._tmp)
        if self._last is not None:
            self.files["preview"] = write_image(self.out_dir / "preview.png", self._last)
        return dict(self.files)
