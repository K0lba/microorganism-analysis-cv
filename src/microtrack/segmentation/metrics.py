"""Метрики качества сегментации: маски (IoU, Dice) и обнаружение объектов."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Бинарные маски
# --------------------------------------------------------------------------- #
def _bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union; две пустые маски дают 1.0."""
    a, b = _bool(a), _bool(b)
    if a.shape != b.shape:
        raise ValueError(f"Размеры масок различаются: {a.shape} и {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Коэффициент Дайса; две пустые маски дают 1.0."""
    a, b = _bool(a), _bool(b)
    if a.shape != b.shape:
        raise ValueError(f"Размеры масок различаются: {a.shape} и {b.shape}")
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / total)


def mask_coverage(mask: np.ndarray) -> float:
    """Доля кадра под маской."""
    return float(_bool(mask).mean())


# --------------------------------------------------------------------------- #
# Объекты
# --------------------------------------------------------------------------- #
def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IoU между всеми парами (предсказанный объект, эталонный объект).

    Возвращает (id предсказанных, id эталонных, матрица [n_pred, n_gt]).
    """
    pred = np.asarray(pred).astype(np.int64)
    gt = np.asarray(gt).astype(np.int64)
    if pred.shape != gt.shape:
        raise ValueError(f"Размеры разметок различаются: {pred.shape} и {gt.shape}")
    p_ids = np.unique(pred[pred > 0])
    g_ids = np.unique(gt[gt > 0])
    mat = np.zeros((len(p_ids), len(g_ids)), np.float64)
    if len(p_ids) == 0 or len(g_ids) == 0:
        return p_ids, g_ids, mat

    p_area = np.bincount(pred.ravel())[p_ids]
    g_area = np.bincount(gt.ravel())[g_ids]
    both = (pred > 0) & (gt > 0)
    pi = np.searchsorted(p_ids, pred[both])
    gi = np.searchsorted(g_ids, gt[both])
    inter = np.zeros_like(mat)
    np.add.at(inter, (pi, gi), 1)
    union = p_area[:, None] + g_area[None, :] - inter
    return p_ids, g_ids, inter / union


@dataclass
class DetectionResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    matches: List[Tuple[int, int, float]] = field(default_factory=list)  # (pred_id, gt_id, iou)


def match_objects(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.5) -> DetectionResult:
    """Сопоставление один-к-одному по IoU (жадно, по убыванию IoU).

    Пара считается совпадением при IoU >= iou_threshold. Пустые pred и gt — идеальный результат.
    """
    p_ids, g_ids, mat = iou_matrix(pred, gt)
    n_p, n_g = len(p_ids), len(g_ids)
    matches: List[Tuple[int, int, float]] = []
    if n_p and n_g:
        order = np.dstack(np.unravel_index(np.argsort(-mat, axis=None, kind="stable"), mat.shape))[0]
        used_p, used_g = set(), set()
        for i, j in order:
            v = mat[i, j]
            if v < iou_threshold or v <= 0:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            matches.append((int(p_ids[i]), int(g_ids[j]), float(v)))
    tp = len(matches)
    fp, fn = n_p - tp, n_g - tp
    if n_p == 0 and n_g == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / n_p if n_p else 0.0
        recall = tp / n_g if n_g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0
    return DetectionResult(tp, fp, fn, precision, recall, f1, mean_iou, matches)


# --------------------------------------------------------------------------- #
# Сводка по кадру
# --------------------------------------------------------------------------- #
def evaluate(pred_labels: np.ndarray, gt_labels: np.ndarray, iou_threshold: float = 0.5,
             elapsed_s: Optional[float] = None) -> Dict[str, float]:
    """Все метрики для одного кадра: маски, обнаружение, счётчики, покрытие, время."""
    det = match_objects(pred_labels, gt_labels, iou_threshold)
    n_pred = int(len(np.unique(pred_labels[pred_labels > 0])))
    n_gt = int(len(np.unique(gt_labels[gt_labels > 0])))
    out = {
        "iou": iou(pred_labels, gt_labels),
        "dice": dice(pred_labels, gt_labels),
        "precision": det.precision,
        "recall": det.recall,
        "f1": det.f1,
        "mean_matched_iou": det.mean_matched_iou,
        "tp": det.tp, "fp": det.fp, "fn": det.fn,
        "n_pred": n_pred, "n_gt": n_gt, "count_error": n_pred - n_gt,
        "coverage_pred": mask_coverage(pred_labels),
        "coverage_gt": mask_coverage(gt_labels),
    }
    if elapsed_s is not None:
        out["time_ms"] = elapsed_s * 1000.0
    return out
