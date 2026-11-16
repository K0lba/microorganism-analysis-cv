"""Динамика во времени: смещения, скорости, физические размеры, относительные изменения.

Главный принцип: время берётся из **временных отметок кадров**, а не из номера
обработанного кадра. Поэтому скорость не зависит от того, обрабатывали ли вы
каждый кадр, пропускали ли кадры или были ли интервалы между кадрами неравномерными.

Вход — строки наблюдений (``Tracker.to_rows()``): как минимум ``track_id``,
``frame``, ``x``, ``y`` (px); необязательно ``observed`` (0 — предсказанное
положение, по умолчанию пропускается) и любые поля морфометрии
(``area``, ``length``, ``width``, ``perimeter`` ...).

Выход — строки на каждое наблюдение со столбцами:

  t_s, elapsed_s        время кадра и время от первого наблюдения трека, с
  dt_s                  интервал до предыдущего наблюдения того же ID, с
  frames_since_prev     сколько кадров прошло с предыдущего наблюдения
  x_px, y_px            положение; dx_px, dy_px, displacement_px — смещение
  vx_px_s, vy_px_s,     компоненты и модуль скорости между соседними
  speed_px_s            наблюдениями (смещение / dt)
  direction_deg         направление движения, градусы; система координат
                        изображения: ось y вниз, поэтому +90° — вниз
  path_length_px,       пройденный путь и смещение от первого наблюдения
  net_displacement_px
  <поле>_rel_change     (значение - начальное) / начальное для площадей и
                        линейных размеров (0.5 = +50 %)

При наличии калибровки (µm на px) добавляются столбцы ``*_um`` / ``*_um_s`` /
``*_um2`` для положения, смещений, скоростей и размеров.
Первое наблюдение трека не имеет dt, смещения и скорости (NaN).
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .morphometry import AREA_FIELDS, LINEAR_FIELDS, scale_features

NAN = float("nan")

_FRAME_COLS = ("frame", "frame_idx", "frame_index", "frame_id", "index", "idx")
_TIME_COLS = ("time_s", "timestamp_s", "time_sec", "time", "timestamp", "t_s", "t",
              "seconds", "time_ms", "timestamp_ms", "t_ms")
_SEQ_COLS = ("sequence", "seq", "sequence_id", "seq_id", "video", "video_id")
_UNIT_TO_S = {"s": 1.0, "ms": 1e-3, "us": 1e-6}


# --------------------------------------------------------------------------
# Время
# --------------------------------------------------------------------------

class Timeline:
    """Соответствие «номер кадра -> время, с».

    Создаётся из списка/словаря отметок, из CSV (``frames.csv``) или из
    постоянной частоты кадров (запасной вариант, когда отметок нет).
    """

    def __init__(self, times: Optional[Mapping[int, float]] = None,
                 fps: Optional[float] = None, t0: float = 0.0):
        if times is None and fps is None:
            raise ValueError("Нужны либо отметки времени, либо fps")
        if fps is not None and fps <= 0:
            raise ValueError("fps должен быть положительным")
        self._times = {int(k): float(v) for k, v in (times or {}).items()}
        self._fps, self._t0 = fps, float(t0)

    @classmethod
    def from_fps(cls, fps: float, t0: float = 0.0) -> "Timeline":
        return cls(fps=fps, t0=t0)

    @classmethod
    def from_sequence(cls, times: Sequence[float], first_frame: int = 0) -> "Timeline":
        return cls({first_frame + i: t for i, t in enumerate(times)})

    @classmethod
    def from_csv(cls, path, frame_col: Optional[str] = None,
                 time_col: Optional[str] = None, unit: Optional[str] = None,
                 sequence: Optional[object] = None,
                 sequence_col: Optional[str] = None) -> "Timeline":
        """Прочитать отметки из CSV.

        Названия столбцов определяются автоматически (см. ``_FRAME_COLS``,
        ``_TIME_COLS``) или задаются явно. ``unit`` — "s", "ms" или "us"; если не
        задан, берётся из суффикса имени столбца (``..._ms``), иначе секунды.
        Если в файле несколько последовательностей, нужно указать ``sequence``.
        """
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            rows = list(reader)
        low = {c.lower(): c for c in cols}

        def pick(explicit, candidates, what):
            if explicit:
                if explicit not in cols:
                    raise ValueError(f"Столбца {explicit!r} нет в {path}; есть: {cols}")
                return explicit
            for c in candidates:
                if c in low:
                    return low[c]
            raise ValueError(f"Не найден столбец {what} в {path}; есть: {cols}. "
                             f"Укажите его явно.")

        fcol = pick(frame_col, _FRAME_COLS, "номера кадра")
        tcol = pick(time_col, _TIME_COLS, "времени")
        if unit is None:
            unit = "ms" if tcol.lower().endswith("_ms") else \
                   "us" if tcol.lower().endswith("_us") else "s"
        if unit not in _UNIT_TO_S:
            raise ValueError(f"unit должен быть одним из {list(_UNIT_TO_S)}")

        scol = sequence_col or next((low[c] for c in _SEQ_COLS if c in low), None)
        if scol is not None:
            values = sorted({r[scol] for r in rows})
            if sequence is None:
                if len(values) > 1:
                    raise ValueError(f"В файле несколько последовательностей {values}; "
                                     f"укажите sequence=...")
            else:
                rows = [r for r in rows if _same(r[scol], sequence)]
                if not rows:
                    raise ValueError(f"Последовательность {sequence!r} не найдена среди {values}")
        elif sequence is not None:
            raise ValueError("В файле нет столбца последовательности")

        times: Dict[int, float] = {}
        for r in rows:
            fr = int(float(r[fcol]))
            if fr in times:
                raise ValueError(f"Кадр {fr} встречается в {path} несколько раз")
            times[fr] = float(r[tcol]) * _UNIT_TO_S[unit]
        return cls(times)

    def time(self, frame: int) -> float:
        frame = int(frame)
        if frame in self._times:
            return self._times[frame]
        if self._fps is not None:
            return self._t0 + frame / self._fps
        raise KeyError(f"Нет временной отметки для кадра {frame}")

    def __contains__(self, frame: int) -> bool:
        return int(frame) in self._times or self._fps is not None


def _same(a, b) -> bool:
    if str(a) == str(b):
        return True
    try:
        return int(float(a)) == int(float(b))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# Калибровка
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """Масштаб изображения: сколько микрометров в одном пикселе."""

    um_per_px: Optional[float] = None

    def __post_init__(self):
        if self.um_per_px is not None and not self.um_per_px > 0:
            raise ValueError("um_per_px должен быть положительным числом")

    @property
    def calibrated(self) -> bool:
        return self.um_per_px is not None

    @classmethod
    def from_reference(cls, length_um: float, length_px: float) -> "Calibration":
        """По известному отрезку (шкала, сетка): length_um µm занимают length_px px."""
        if length_um <= 0 or length_px <= 0:
            raise ValueError("Длины должны быть положительными")
        return cls(length_um / length_px)

    def length(self, px: float) -> float:
        return px * self.um_per_px if self.calibrated else NAN

    def area(self, px2: float) -> float:
        return px2 * self.um_per_px ** 2 if self.calibrated else NAN


# --------------------------------------------------------------------------
# Расчёт
# --------------------------------------------------------------------------

def _is_observed(row: Mapping) -> bool:
    return bool(int(row.get("observed", 1)))


def _relative_fields(first: Mapping) -> List[str]:
    return [k for k in first
            if (k in AREA_FIELDS or k in LINEAR_FIELDS)
            and isinstance(first[k], (int, float, np.integer, np.floating))]


def compute_temporal(rows: Iterable[Mapping], timeline: Timeline,
                     calibration: Optional[Calibration] = None,
                     include_predicted: bool = False) -> List[dict]:
    """Временные признаки для всех наблюдений всех треков."""
    cal = calibration or Calibration()
    by_track: Dict[int, List[Mapping]] = {}
    for r in rows:
        if include_predicted or _is_observed(r):
            by_track.setdefault(int(r["track_id"]), []).append(r)

    out: List[dict] = []
    for tid in sorted(by_track):
        obs = sorted(by_track[tid], key=lambda r: int(r["frame"]))
        rel_fields = _relative_fields(obs[0])
        first = obs[0]
        t0 = timeline.time(first["frame"])
        path = 0.0
        prev = None
        for r in obs:
            frame = int(r["frame"])
            t = timeline.time(frame)
            x, y = float(r["x"]), float(r["y"])
            row = {"track_id": tid, "frame": frame, "t_s": t, "elapsed_s": t - t0,
                   "dt_s": NAN, "frames_since_prev": None,
                   "x_px": x, "y_px": y, "dx_px": NAN, "dy_px": NAN,
                   "displacement_px": NAN, "vx_px_s": NAN, "vy_px_s": NAN,
                   "speed_px_s": NAN, "direction_deg": NAN}
            if prev is not None:
                dt = t - prev["t"]
                dx, dy = x - prev["x"], y - prev["y"]
                disp = math.hypot(dx, dy)
                row.update(dt_s=dt, frames_since_prev=frame - prev["frame"],
                           dx_px=dx, dy_px=dy, displacement_px=disp)
                if dt > 0:
                    row.update(vx_px_s=dx / dt, vy_px_s=dy / dt, speed_px_s=disp / dt)
                else:
                    warnings.warn(f"Трек {tid}: отметки времени кадров {prev['frame']} и "
                                  f"{frame} не возрастают (dt={dt:g} с); скорость не считается")
                if disp > 0:
                    row["direction_deg"] = math.degrees(math.atan2(dy, dx))
                path += disp
            row["path_length_px"] = path
            row["net_displacement_px"] = math.hypot(x - float(first["x"]),
                                                    y - float(first["y"]))

            if cal.calibrated:
                s = cal.um_per_px
                for src, dst in (("x_px", "x_um"), ("y_px", "y_um"), ("dx_px", "dx_um"),
                                 ("dy_px", "dy_um"), ("displacement_px", "displacement_um"),
                                 ("vx_px_s", "vx_um_s"), ("vy_px_s", "vy_um_s"),
                                 ("speed_px_s", "speed_um_s"),
                                 ("path_length_px", "path_length_um"),
                                 ("net_displacement_px", "net_displacement_um")):
                    row[dst] = row[src] * s

            # морфометрические поля: исходные (px), калиброванные и относительные
            metric = {k: r[k] for k in r
                      if (k in AREA_FIELDS or k in LINEAR_FIELDS)
                      and isinstance(r[k], (int, float, np.integer, np.floating))}
            row.update(metric)
            if cal.calibrated:
                row.update(scale_features(metric, cal.um_per_px))
            for k in rel_fields:
                v0 = float(first[k])
                row[f"{k}_rel_change"] = (float(r[k]) - v0) / v0 if v0 != 0 and k in r else NAN
            out.append(row)
            prev = {"t": t, "x": x, "y": y, "frame": frame}
    return out


def summarize_tracks(temporal_rows: Sequence[Mapping],
                     calibration: Optional[Calibration] = None) -> List[dict]:
    """Итоги по каждому треку: длительность, путь, средняя/макс. скорость, прямолинейность."""
    cal = calibration or Calibration()
    by_track: Dict[int, List[Mapping]] = {}
    for r in temporal_rows:
        by_track.setdefault(int(r["track_id"]), []).append(r)
    out = []
    for tid in sorted(by_track):
        rs = by_track[tid]
        first, last = rs[0], rs[-1]
        duration = last["t_s"] - first["t_s"]
        path = last["path_length_px"]
        speeds = np.array([r["speed_px_s"] for r in rs if np.isfinite(r["speed_px_s"])])
        item = {
            "track_id": tid, "n_observations": len(rs),
            "frame_start": first["frame"], "frame_end": last["frame"],
            "t_start_s": first["t_s"], "t_end_s": last["t_s"], "duration_s": duration,
            "path_length_px": path, "net_displacement_px": last["net_displacement_px"],
            "straightness": last["net_displacement_px"] / path if path > 0 else NAN,
            "mean_speed_px_s": path / duration if duration > 0 else NAN,
            "median_speed_px_s": float(np.median(speeds)) if speeds.size else NAN,
            "max_speed_px_s": float(speeds.max()) if speeds.size else NAN,
        }
        if cal.calibrated:
            for k in ("path_length", "net_displacement"):
                item[f"{k}_um"] = item[f"{k}_px"] * cal.um_per_px
            for k in ("mean_speed", "median_speed", "max_speed"):
                item[f"{k}_um_s"] = item[f"{k}_px_s"] * cal.um_per_px
        for k in ("area_rel_change", "length_rel_change", "width_rel_change"):
            if k in last:
                item[f"{k}_final"] = last[k]
        out.append(item)
    return out


def analyze_tracker(tracker, timeline: Timeline,
                    calibration: Optional[Calibration] = None):
    """Удобная связка: (строки по наблюдениям, итоги по трекам) из ``Tracker``."""
    rows = compute_temporal(tracker.to_rows(include_predicted=True), timeline, calibration)
    return rows, summarize_tracks(rows, calibration)


def save_rows_csv(rows: Sequence[Mapping], path) -> str:
    """Записать строки в CSV; NaN и None — пустые ячейки."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or ["track_id"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None or (isinstance(v, float) and math.isnan(v))
                            else v) for k, v in r.items()})
    return str(p)
