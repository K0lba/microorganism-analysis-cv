"""Экспорт таблицы наблюдений для Модуля 1.

Одна строка = одно наблюдение одного объекта (track_id) в одном кадре.
Таблица самодокументирована: ``COLUMNS`` задаёт имя, тип, единицу измерения,
допустимость пустых значений и описание каждого столбца; по нему же строится
``docs/export_schema.md`` (``schema_markdown``) и выполняется проверка
``validate_table``.

Единицы: длины в px и µm, площади в px² и µm², время в секундах, углы в градусах,
скорости в px/с и µm/с. Столбцы с µm пусты (NA), если нет калибровки.
Дополнительные условия эксперимента добавляются столбцами ``cond_<имя>``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .temporal import Calibration, Timeline, compute_temporal

SCHEMA_VERSION = "1.1"
KEY = ["experiment_id", "sequence_id", "track_id", "frame"]


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str          # str | int | float | bool | Int (nullable int)
    unit: str           # "px", "µm", "px²", "µm²", "s", "deg", "px/s", "µm/s", "1", "—"
    nullable: bool
    description: str
    group: str


def _c(name, dtype, unit, nullable, group, description):
    return Column(name, dtype, unit, nullable, description, group)


COLUMNS: List[Column] = [
    # --- идентификаторы ---
    _c("experiment_id", "str", "—", False, "ключи", "Идентификатор эксперимента (из конфигурации)"),
    _c("sequence_id", "str", "—", False, "ключи", "Идентификатор видео/последовательности кадров"),
    _c("track_id", "int", "—", False, "ключи", "Устойчивый идентификатор объекта в пределах последовательности (с 1)"),
    _c("frame", "int", "—", False, "ключи", "Номер кадра в исходной последовательности (не в обработанной)"),
    _c("timestamp_s", "float", "s", False, "ключи", "Время кадра от начала записи, с"),
    # --- положение ---
    _c("x_px", "float", "px", False, "положение", "Координата центроида X (вправо), px"),
    _c("y_px", "float", "px", False, "положение", "Координата центроида Y (вниз), px"),
    _c("x_um", "float", "µm", True, "положение", "X в микрометрах; NA без калибровки"),
    _c("y_um", "float", "µm", True, "положение", "Y в микрометрах; NA без калибровки"),
    # --- движение ---
    _c("dt_s", "float", "s", True, "движение", "Интервал до предыдущего наблюдения трека, с; NA у первого наблюдения"),
    _c("frames_since_prev", "Int", "—", True, "движение", "Число кадров с предыдущего наблюдения (1 — соседний); NA у первого"),
    _c("displacement_px", "float", "px", True, "движение", "Смещение центроида от предыдущего наблюдения; NA у первого"),
    _c("displacement_um", "float", "µm", True, "движение", "То же в µm; NA без калибровки или у первого"),
    _c("speed_px_s", "float", "px/s", True, "движение", "Скорость = смещение / dt; NA у первого или при dt <= 0"),
    _c("speed_um_s", "float", "µm/s", True, "движение", "Скорость в µm/с; NA без калибровки"),
    _c("direction_deg", "float", "deg", True, "движение", "Направление движения, ось Y вниз (+90° — вниз); NA при нулевом смещении"),
    _c("path_length_px", "float", "px", False, "движение", "Пройденный путь от первого наблюдения трека"),
    _c("path_length_um", "float", "µm", True, "движение", "То же в µm"),
    _c("net_displacement_px", "float", "px", False, "движение", "Расстояние от первого наблюдения трека"),
    _c("net_displacement_um", "float", "µm", True, "движение", "То же в µm"),
    # --- морфология ---
    _c("area_px2", "float", "px²", False, "морфология", "Площадь маски"),
    _c("area_um2", "float", "µm²", True, "морфология", "Площадь в µm²"),
    _c("perimeter_px", "float", "px", False, "морфология", "Периметр внешнего контура"),
    _c("perimeter_um", "float", "µm", True, "морфология", "Периметр в µm"),
    _c("length_px", "float", "px", False, "морфология", "Длина: большая сторона минимального описанного повёрнутого прямоугольника"),
    _c("length_um", "float", "µm", True, "морфология", "Длина в µm"),
    _c("width_px", "float", "px", False, "морфология", "Ширина: меньшая сторона того же прямоугольника"),
    _c("width_um", "float", "µm", True, "морфология", "Ширина в µm"),
    _c("major_axis_px", "float", "px", False, "морфология", "Большая ось эквивалентного эллипса (по моментам)"),
    _c("minor_axis_px", "float", "px", False, "морфология", "Малая ось эквивалентного эллипса"),
    _c("aspect_ratio", "float", "1", False, "морфология", "length / width, >= 1"),
    _c("circularity", "float", "1", False, "морфология", "4*pi*S / P^2, 0..1"),
    _c("roundness", "float", "1", False, "морфология", "4*S / (pi * major^2), 0..1"),
    _c("solidity", "float", "1", False, "морфология", "S / площадь выпуклой оболочки, 0..1"),
    _c("eccentricity", "float", "1", False, "морфология", "Эксцентриситет эквивалентного эллипса, 0..1"),
    _c("orientation_deg", "float", "deg", False, "морфология", "Угол большой оси к оси X, -90..90"),
    _c("radial_mean_px", "float", "px", False, "морфология", "Среднее расстояние от центроида до контура"),
    _c("radial_mean_um", "float", "µm", True, "морфология", "То же в µm"),
    _c("radial_cv", "float", "1", False, "морфология", "Коэффициент вариации радиальных расстояний"),
    _c("radial_max_min_ratio", "float", "1", False, "морфология", "Максимум / минимум радиального расстояния"),
    _c("extent", "float", "1", False, "морфология", "Площадь / (длина × ширина) минимального прямоугольника"),
    _c("convexity", "float", "1", False, "морфология", "Периметр выпуклой оболочки / периметр, 0..1"),
    _c("roughness", "float", "1", False, "морфология", "Среднее отклонение r(θ) от сглаженного профиля / средний радиус"),
    _c("bending_energy", "float", "1", False, "морфология", "Безразмерная энергия изгиба контура; у круга ≈ 1"),
    _c("radial_entropy", "float", "1", False, "морфология", "Нормированная энтропия распределения r(θ), 0..1"),
    _c("fourier_2", "float", "1", False, "морфология", "Вторая гармоника r(θ): вытянутость, |F2|/|F0|"),
    _c("fourier_3", "float", "1", False, "морфология", "Третья гармоника r(θ): трёхлепестковая асимметрия"),
    _c("center_shift_circ", "float", "1", False, "морфология", "Сдвиг центроида от центра описанной окружности / её радиус"),
    # --- динамика размеров ---
    _c("area_rel_change", "float", "1", False, "динамика", "(S - S0) / S0 относительно первого наблюдения трека (0.5 = +50 %)"),
    _c("length_rel_change", "float", "1", False, "динамика", "Относительное изменение длины"),
    _c("width_rel_change", "float", "1", False, "динамика", "Относительное изменение ширины"),
    # --- качество трекинга ---
    _c("observed", "bool", "—", False, "качество трекинга", "True — объект найден в кадре; False — положение предсказано (пропуск)"),
    _c("match_confidence", "float", "1", False, "качество трекинга", "Уверенность сопоставления с предыдущим кадром, 0..1 (1 у первого наблюдения)"),
    _c("match_iou", "float", "1", False, "качество трекинга", "IoU с предыдущей маской, сдвинутой по предсказанию"),
    _c("ambiguous", "bool", "—", False, "качество трекинга", "True — был почти равноценный альтернативный кандидат"),
    _c("track_n_obs", "int", "—", False, "качество трекинга", "Число наблюдений трека в таблице"),
    _c("track_mean_confidence", "float", "1", False, "качество трекинга", "Средняя уверенность сопоставления по треку"),
    _c("track_status", "str", "—", False, "качество трекинга", "active | lost | finished на конец обработки"),
    # --- качество сегментации ---
    _c("seg_contrast", "float", "gray", False, "качество сегментации", "Разница средней яркости объекта и фона вокруг, уровни 0..255"),
    _c("seg_cnr", "float", "1", False, "качество сегментации", "Контраст к шуму фона"),
    _c("touches_border", "bool", "—", False, "качество сегментации", "Маска касается края кадра"),
    _c("seg_flag", "str", "—", False, "качество сегментации", "ok или причины через запятую: low_contrast, border, irregular, tiny, area_jump"),
    # --- калибровка и условия ---
    _c("um_per_px", "float", "µm/px", True, "калибровка", "Масштаб; NA — калибровки нет, µm-столбцы пусты"),
    _c("calibration_source", "str", "—", False, "калибровка", "Источник масштаба: config | reference | none"),
    _c("timestamp_source", "str", "—", False, "калибровка", "Источник времени: csv | fps | video"),
    _c("fps_nominal", "float", "1/s", True, "калибровка", "Номинальная частота кадров, если известна"),
    _c("frame_step", "int", "—", False, "калибровка", "Шаг субдискретизации: обрабатывался каждый N-й кадр"),
    _c("pipeline_version", "str", "—", False, "калибровка", "Версия конвейера"),
    _c("schema_version", "str", "—", False, "калибровка", "Версия схемы таблицы"),
]
COLUMN_BY_NAME: Dict[str, Column] = {c.name: c for c in COLUMNS}
COND_PREFIX = "cond_"

_PD_DTYPE = {"str": "string", "int": "int64", "float": "float64", "bool": "boolean", "Int": "Int64"}


@dataclass
class ExportMeta:
    experiment_id: str
    sequence_id: str = "seq"
    calibration: Calibration = field(default_factory=Calibration)
    calibration_source: str = "none"
    timestamp_source: str = "fps"
    fps_nominal: Optional[float] = None
    frame_step: int = 1
    conditions: Mapping[str, object] = field(default_factory=dict)
    pipeline_version: str = "0.1.0"


# --------------------------------------------------------------------------
# Построение таблицы
# --------------------------------------------------------------------------

def _f(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def build_table(tracker, timeline: Timeline, meta: ExportMeta,
                include_predicted: bool = False) -> pd.DataFrame:
    """Собрать таблицу наблюдений из ``Tracker`` (треки + морфометрия + качество)."""
    rows_r = tracker.to_rows(include_predicted=include_predicted)
    temporal = compute_temporal(rows_r, timeline, meta.calibration,
                                include_predicted=include_predicted)
    by_key = {(int(r["track_id"]), int(r["frame"])): r for r in rows_r}
    cal = meta.calibration
    um = cal.um_per_px if cal.calibrated else float("nan")

    n_obs: Dict[int, int] = {}
    conf_sum: Dict[int, List[float]] = {}
    for r in rows_r:
        if int(r["observed"]):
            n_obs[int(r["track_id"])] = n_obs.get(int(r["track_id"]), 0) + 1
            conf_sum.setdefault(int(r["track_id"]), []).append(float(r["confidence"]))

    out = []
    for t in temporal:
        tid, fr = int(t["track_id"]), int(t["frame"])
        r = by_key[(tid, fr)]
        rec = {
            "experiment_id": meta.experiment_id, "sequence_id": meta.sequence_id,
            "track_id": tid, "frame": fr, "timestamp_s": t["t_s"],
            "x_px": t["x_px"], "y_px": t["y_px"],
            "x_um": t.get("x_um", np.nan), "y_um": t.get("y_um", np.nan),
            "dt_s": t["dt_s"], "frames_since_prev": t["frames_since_prev"],
            "displacement_px": t["displacement_px"], "displacement_um": t.get("displacement_um", np.nan),
            "speed_px_s": t["speed_px_s"], "speed_um_s": t.get("speed_um_s", np.nan),
            "direction_deg": t["direction_deg"],
            "path_length_px": t["path_length_px"], "path_length_um": t.get("path_length_um", np.nan),
            "net_displacement_px": t["net_displacement_px"],
            "net_displacement_um": t.get("net_displacement_um", np.nan),
            "area_px2": _f(t.get("area")), "area_um2": _f(t.get("area_um2")),
            "perimeter_px": _f(t.get("perimeter")), "perimeter_um": _f(t.get("perimeter_um")),
            "length_px": _f(t.get("length")), "length_um": _f(t.get("length_um")),
            "width_px": _f(t.get("width")), "width_um": _f(t.get("width_um")),
            "major_axis_px": _f(t.get("major_axis")), "minor_axis_px": _f(t.get("minor_axis")),
            "aspect_ratio": _f(r.get("aspect_ratio")), "circularity": _f(r.get("circularity")),
            "roundness": _f(r.get("roundness")), "solidity": _f(r.get("solidity")),
            "eccentricity": _f(r.get("eccentricity")),
            "orientation_deg": _f(r.get("orientation_deg")),
            "radial_mean_px": _f(t.get("radial_mean")), "radial_mean_um": _f(t.get("radial_mean_um")),
            "radial_cv": _f(r.get("radial_cv")),
            "radial_max_min_ratio": _f(r.get("radial_max_min_ratio")),
            "extent": _f(r.get("extent")), "convexity": _f(r.get("convexity")),
            "roughness": _f(r.get("roughness")), "bending_energy": _f(r.get("bending_energy")),
            "radial_entropy": _f(r.get("radial_entropy")),
            "fourier_2": _f(r.get("fourier_2")), "fourier_3": _f(r.get("fourier_3")),
            "center_shift_circ": _f(r.get("center_shift_circ")),
            "area_rel_change": _f(t.get("area_rel_change")),
            "length_rel_change": _f(t.get("length_rel_change")),
            "width_rel_change": _f(t.get("width_rel_change")),
            "observed": bool(int(r["observed"])), "match_confidence": float(r["confidence"]),
            "match_iou": float(r["iou"]), "ambiguous": bool(int(r["ambiguous"])),
            "track_n_obs": n_obs.get(tid, 0),
            "track_mean_confidence": float(np.mean(conf_sum[tid])) if tid in conf_sum else float("nan"),
            "track_status": tracker.tracks[tid].status,
            "seg_contrast": _f(r.get("seg_contrast")), "seg_cnr": _f(r.get("seg_cnr")),
            "touches_border": bool(r.get("touches_border", False)),
            "seg_flag": str(r.get("seg_flag", "ok")),
            "um_per_px": um, "calibration_source": meta.calibration_source,
            "timestamp_source": meta.timestamp_source,
            "fps_nominal": meta.fps_nominal if meta.fps_nominal is not None else float("nan"),
            "frame_step": int(meta.frame_step), "pipeline_version": meta.pipeline_version,
            "schema_version": SCHEMA_VERSION,
        }
        for k, v in meta.conditions.items():
            rec[f"{COND_PREFIX}{k}"] = v
        out.append(rec)

    cond_cols = [f"{COND_PREFIX}{k}" for k in meta.conditions]
    df = pd.DataFrame(out, columns=[c.name for c in COLUMNS] + cond_cols)
    df = _flag_area_jumps(df)
    return coerce_types(df)


def _flag_area_jumps(df: pd.DataFrame, max_ratio: float = 0.5) -> pd.DataFrame:
    """Добавить area_jump в seg_flag, если площадь резко изменилась между соседними наблюдениями."""
    if df.empty:
        return df
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    prev = df.groupby("track_id")["area_px2"].shift(1)
    jump = ((df["area_px2"] / prev - 1.0).abs() > max_ratio) & prev.notna()
    flags = df["seg_flag"].astype(str)
    df["seg_flag"] = np.where(jump, np.where(flags == "ok", "area_jump", flags + ",area_jump"), flags)
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Привести столбцы к типам схемы (лишние cond_* не трогаем)."""
    df = df.copy()
    for c in COLUMNS:
        if c.name in df.columns:
            df[c.name] = df[c.name].astype(_PD_DTYPE[c.dtype])
    return df


# --------------------------------------------------------------------------
# Проверка таблицы
# --------------------------------------------------------------------------

@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self):
        if self.errors:
            raise ValueError("Таблица не прошла проверку:\n  " + "\n  ".join(self.errors))

    def __str__(self) -> str:
        if self.ok and not self.warnings:
            return "OK"
        return "\n".join([f"ERROR: {e}" for e in self.errors] + [f"WARN: {w}" for w in self.warnings])


_RANGES = {  # name: (min, max)
    "match_confidence": (0.0, 1.0), "track_mean_confidence": (0.0, 1.0), "match_iou": (0.0, 1.0),
    "circularity": (0.0, 1.0), "roundness": (0.0, 1.0), "solidity": (0.0, 1.0),
    "eccentricity": (0.0, 1.0), "aspect_ratio": (1.0, math.inf),
    "area_px2": (0.0, math.inf), "length_px": (0.0, math.inf), "width_px": (0.0, math.inf),
    "orientation_deg": (-90.0, 90.0), "dt_s": (0.0, math.inf), "speed_px_s": (0.0, math.inf),
    "direction_deg": (-180.0, 180.0), "radial_cv": (0.0, math.inf),
    "extent": (0.0, 1.0), "convexity": (0.0, 1.0), "roughness": (0.0, math.inf),
    "bending_energy": (0.0, math.inf), "radial_entropy": (0.0, 1.0),
    "fourier_2": (0.0, math.inf), "fourier_3": (0.0, math.inf),
    "center_shift_circ": (0.0, math.inf),
}
_UNIT_PAIRS = [  # (px-столбец, µm-столбец, степень масштаба)
    ("x_px", "x_um", 1), ("y_px", "y_um", 1), ("displacement_px", "displacement_um", 1),
    ("speed_px_s", "speed_um_s", 1), ("path_length_px", "path_length_um", 1),
    ("net_displacement_px", "net_displacement_um", 1), ("area_px2", "area_um2", 2),
    ("perimeter_px", "perimeter_um", 1), ("length_px", "length_um", 1),
    ("width_px", "width_um", 1), ("radial_mean_px", "radial_mean_um", 1),
]


def _kind_ok(series: pd.Series, dtype: str) -> bool:
    t = series.dtype
    if dtype == "str":
        return pd.api.types.is_string_dtype(t) or t == object
    if dtype in ("int", "Int"):
        return pd.api.types.is_integer_dtype(t)
    if dtype == "float":
        return pd.api.types.is_float_dtype(t)
    if dtype == "bool":
        return pd.api.types.is_bool_dtype(t)
    return False


def validate_table(df: pd.DataFrame, rtol: float = 1e-6) -> ValidationReport:
    """Проверка: столбцы, типы, пустые значения, диапазоны, единицы, уникальность, время."""
    rep = ValidationReport()
    names = [c.name for c in COLUMNS]

    # 1. столбцы
    missing = [n for n in names if n not in df.columns]
    if missing:
        rep.errors.append(f"нет столбцов: {missing}")
    extra = [c for c in df.columns if c not in COLUMN_BY_NAME and not c.startswith(COND_PREFIX)]
    if extra:
        rep.warnings.append(f"столбцы вне схемы: {extra}")
    present = [c for c in COLUMNS if c.name in df.columns]

    # 2. типы и пустые значения
    for c in present:
        s = df[c.name]
        if not _kind_ok(s, c.dtype):
            rep.errors.append(f"{c.name}: тип {s.dtype}, ожидается {c.dtype}")
            continue
        n_null = int(s.isna().sum())
        if n_null and not c.nullable:
            rep.errors.append(f"{c.name}: {n_null} пустых значений, столбец не допускает NA")
        if c.dtype == "float" and np.isinf(s.dropna().astype(float)).any():
            rep.errors.append(f"{c.name}: содержит inf")

    if df.empty:
        rep.warnings.append("таблица пуста")
        return rep
    if missing:
        return rep

    # 3. уникальность
    dup = df.duplicated(KEY, keep=False)
    if dup.any():
        rep.errors.append(f"{int(dup.sum())} строк с повторяющимся ключом {KEY}")

    # 4. диапазоны
    tol = 1e-6
    for name, (lo, hi) in _RANGES.items():
        s = df[name].dropna().astype(float)
        bad = int(((s < lo - tol) | (s > hi + tol)).sum())
        if bad:
            rep.errors.append(f"{name}: {bad} значений вне диапазона [{lo}, {hi}]")
    if (df["frame_step"] < 1).any():
        rep.errors.append("frame_step < 1")
    um_vals = df["um_per_px"].dropna()
    if (um_vals <= 0).any():
        rep.errors.append("um_per_px <= 0")

    # 5. время и порядок внутри трека
    for tid, g in df.sort_values("frame").groupby("track_id"):
        if (np.diff(g["frame"].to_numpy()) <= 0).any():
            rep.errors.append(f"track {tid}: номера кадров не возрастают")
        if (np.diff(g["timestamp_s"].to_numpy().astype(float)) <= 0).any():
            rep.errors.append(f"track {tid}: timestamp_s не возрастает строго")
        if g["track_n_obs"].nunique() > 1:
            rep.errors.append(f"track {tid}: track_n_obs не постоянен")
        n_first_null = int(g["speed_px_s"].isna().sum())
        if n_first_null > 1:
            rep.warnings.append(f"track {tid}: {n_first_null} наблюдений без скорости (ожидается 1)")
    obs_counts = df[df["observed"]].groupby("track_id").size()
    for tid, n in obs_counts.items():
        declared = int(df.loc[df["track_id"] == tid, "track_n_obs"].iloc[0])
        if declared != n:
            rep.errors.append(f"track {tid}: track_n_obs={declared}, а наблюдений {n}")

    # 6. единицы измерения: калибровка согласована с µm-столбцами
    calibrated = df["um_per_px"].notna()
    if (df["um_per_px"].nunique(dropna=True) > 1):
        rep.errors.append("um_per_px различается между строками одной таблицы")
    for px, um_name, power in _UNIT_PAIRS:
        s_px, s_um = df[px].astype(float), df[um_name].astype(float)
        if (~calibrated & s_um.notna()).any():
            rep.errors.append(f"{um_name}: значения есть, а калибровки нет")
        if (calibrated & s_um.isna() & s_px.notna()).any():
            rep.errors.append(f"{um_name}: пусто при заданной калибровке")
        both = calibrated & s_um.notna() & s_px.notna()
        if both.any():
            expect = s_px[both] * df.loc[both, "um_per_px"].astype(float) ** power
            if not np.allclose(s_um[both], expect, rtol=max(rtol, 1e-6), atol=1e-9):
                rep.errors.append(f"{um_name}: не согласовано с {px} * um_per_px^{power}")
    if df["calibration_source"].eq("none").any() and calibrated.any():
        rep.errors.append("calibration_source='none', но um_per_px задан")
    return rep


# --------------------------------------------------------------------------
# Запись и чтение
# --------------------------------------------------------------------------

def export_csv(df: pd.DataFrame, path) -> str:
    """CSV (UTF-8, запятая, NA = пустая ячейка, без индекса)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, na_rep="", encoding="utf-8")
    return str(p)


def export_parquet(df: pd.DataFrame, path) -> str:
    """Parquet (нужен pyarrow или fastparquet): типы схемы сохраняются без потерь."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, index=False)
    except ImportError as e:
        raise ImportError("Для экспорта в Parquet установите pyarrow: pip install pyarrow") from e
    return str(p)


def to_dataframe(tracker, timeline: Timeline, meta: ExportMeta, validate: bool = True,
                 include_predicted: bool = False) -> pd.DataFrame:
    """Таблица как pandas DataFrame (с проверкой по умолчанию)."""
    df = build_table(tracker, timeline, meta, include_predicted)
    if validate:
        validate_table(df).raise_if_errors()
    return df


def read_table(path) -> pd.DataFrame:
    """Прочитать CSV или Parquet обратно с типами схемы."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return coerce_types(pd.read_parquet(p))
    dtypes = {c.name: _PD_DTYPE[c.dtype] for c in COLUMNS}
    df = pd.read_csv(p, dtype=dtypes, keep_default_na=True, na_values=[""])
    return coerce_types(df)


def export_all(df: pd.DataFrame, out_dir, basename: str = "observations",
               formats: Sequence[str] = ("csv",), strict: bool = True) -> Dict[str, str]:
    """Проверить и записать таблицу во все запрошенные форматы."""
    rep = validate_table(df)
    if strict:
        rep.raise_if_errors()
    out: Dict[str, str] = {}
    for fmt in formats:
        fmt = fmt.lower()
        path = Path(out_dir) / f"{basename}.{fmt}"
        if fmt == "csv":
            out["csv"] = export_csv(df, path)
        elif fmt == "parquet":
            out["parquet"] = export_parquet(df, path)
        else:
            raise ValueError(f"Неизвестный формат экспорта: {fmt}")
    return out


def schema_markdown() -> str:
    """Спецификация столбцов в Markdown."""
    lines = [f"# Спецификация таблицы наблюдений (схема {SCHEMA_VERSION})", "",
             "Одна строка = одно наблюдение одного объекта в одном кадре. "
             f"Уникальный ключ: `{', '.join(KEY)}`.", "",
             "Дополнительные условия эксперимента добавляются столбцами `cond_<имя>` "
             "(значения берутся из конфигурации, одинаковы для всех строк).", "",
             "Типы: `str` — строка, `int` — целое без пропусков, `Int` — целое с возможными NA, "
             "`float` — вещественное, `bool` — логическое. "
             "NA в CSV — пустая ячейка.", ""]
    group = None
    for c in COLUMNS:
        if c.group != group:
            group = c.group
            lines += ["", f"## {group}", "", "| Столбец | Тип | Единица | NA | Описание |",
                      "|---|---|---|---|---|"]
        lines.append(f"| `{c.name}` | {c.dtype} | {c.unit} | {'да' if c.nullable else 'нет'} | {c.description} |")
    lines += ["", "## Проверки при экспорте (`validate_table`)", "",
              "1. Присутствуют все столбцы схемы; типы совпадают с таблицей выше.",
              "2. Пустые значения только в столбцах с NA = да; нет `inf`.",
              "3. Ключ уникален; в треке `frame` и `timestamp_s` строго возрастают.",
              "4. Диапазоны: доли и уверенности в [0, 1], `aspect_ratio` >= 1, площади и длины > 0.",
              "5. Единицы: µm-столбцы пусты без калибровки и равны px-столбцам × `um_per_px` "
              "(в степени 2 для площади) при калибровке.",
              "6. `track_n_obs` совпадает с числом наблюдений трека в таблице."]
    return "\n".join(lines) + "\n"
