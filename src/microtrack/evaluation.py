"""Сопоставление измерений с эталоном и оценка ошибок признаков.

Общие функции для scripts/feature_errors.py и scripts/plot_sizes.py:
сопоставление track_id → object_id, дополнение эталона площадью/периметром
и расчёт относительной ошибки каждого признака.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .tracking_metrics import evaluate_tracking

# Соответствие «эталонный признак → измеренный столбец таблицы».
# Эталон: length_px, width_px (из objects.csv). Площадь и периметр считаем из них,
# если генератор ещё не записал измерения истинной маски.
FEATURE_MAP: Dict[str, str] = {
    "length_px": "length_px",
    "width_px": "width_px",
    "area_px2": "area_px2",
    "perimeter_px": "perimeter_px",
    "aspect_ratio": "aspect_ratio",
}
# Сравниваются только когда есть и в эталоне, и в таблице измерений.
OPTIONAL_FEATURES = (
    "circularity", "solidity", "extent", "roughness", "bending_energy", "radial_cv",
)


def load_gt(path) -> pd.DataFrame:
    """objects.csv синтетики → DataFrame с колонками gt_id, frame, x_px, y_px, length_px, width_px."""
    return pd.read_csv(path).rename(columns={"frame_index": "frame", "object_id": "gt_id"})


def match_tracks(table: pd.DataFrame, gt: pd.DataFrame, max_distance: float = 30.0) -> pd.DataFrame:
    """Сопоставить track_id → gt_id по близости центроидов; вернуть таблицу с колонкой gt_id."""
    pred_rows = [{"track_id": int(r["track_id"]), "frame": int(r["frame"]),
                  "x_px": float(r["x_px"]), "y_px": float(r["y_px"])}
                 for _, r in table.iterrows()]
    gt_rows = [{"object_id": int(r["gt_id"]), "frame": int(r["frame"]),
                "x_px": float(r["x_px"]), "y_px": float(r["y_px"])}
               for _, r in gt.iterrows()]
    tr = evaluate_tracking(pred_rows, gt_rows, max_distance=max_distance)
    pid_to_gid = {m[0]: m[1] for m in tr.matches}
    out = table.copy()
    out["gt_id"] = out["track_id"].map(pid_to_gid)
    return out.dropna(subset=["gt_id"])


def gt_area_perimeter(gt: pd.DataFrame) -> pd.DataFrame:
    """Дополнить эталон площадью и периметром.

    Если objects.csv уже содержит измерения истинной маски, они сохраняются.
    Иначе площадь и периметр восстанавливаются как у эллипса с осями length/width.
    """
    g = gt.copy()
    a, b = g["length_px"] / 2.0, g["width_px"] / 2.0
    if "area_px2" not in g.columns:
        g["area_px2"] = np.pi * a * b
    if "perimeter_px" not in g.columns:
        g["perimeter_px"] = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    if "aspect_ratio" not in g.columns:
        g["aspect_ratio"] = g["length_px"] / g["width_px"].clip(lower=1e-6)
    return g


def compute_errors(table: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Для каждого признака: средняя/медианная относительная ошибка, bias, RMSE."""
    gt = gt_area_perimeter(gt)
    merged = table.merge(gt, left_on=["gt_id", "frame"], right_on=["gt_id", "frame"],
                          suffixes=("_pred", "_gt"))
    rows = []
    names = dict(FEATURE_MAP)
    for name in OPTIONAL_FEATURES:
        if name in table.columns and name in gt.columns:
            names[name] = name
    for gt_name, pred_name in names.items():
        pred_col = pred_name if pred_name in merged.columns else f"{pred_name}_pred"
        gt_col = gt_name if gt_name in merged.columns else f"{gt_name}_gt"
        if pred_col not in merged.columns or gt_col not in merged.columns:
            continue
        p = merged[pred_col].astype(float)
        g = merged[gt_col].astype(float).clip(lower=1e-6)
        rel = (p - g) / g
        abs_err = (p - g).abs()
        rows.append({
            "feature": pred_name,
            "n": int(len(p)),
            "mean_rel_error": float(rel.mean()),
            "median_rel_error": float(rel.median()),
            "mean_abs_rel_error": float(abs_err.div(g).mean()),
            "bias": float((p - g).mean()),
            "rmse": float(np.sqrt(((p - g) ** 2).mean())),
            "gt_mean": float(g.mean()),
            "pred_mean": float(p.mean()),
        })
    return pd.DataFrame(rows)
