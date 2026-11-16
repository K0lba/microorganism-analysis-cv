"""Метрики трекинга против эталона: IDF1, переключения ID, точность/полнота ID.

Сопоставление предсказанных треков с эталонными основано на совпадении
объектов по кадрам: для каждого кадра предсказанные детекции сопоставляются
эталонным по близости центроидов (или IoU масок, если есть), а затем для каждой
пары (предсказанный трек, эталонный объект) считается число кадров, где их
детекции совпали. Оптимальное назначение треков — венгерским алгоритмом.

Метрики:
  IDTP / IDFP / IDFN   совпадения по идентичности (как в IDF1)
  IDF1                2*IDTP / (2*IDTP + IDFP + IDFN), 0..1
  id_precision        IDTP / (IDTP + IDFP)
  id_recall           IDTP / (IDTP + IDFN)
  n_switches          сколько раз эталонный объект сменил предсказанный ID
  switch_rate         n_switches / (число кадров с эталонным объектом - число треков)
  mostly_tracked      доля эталонных объектов, прослеженных >= 80 % кадров
  mostly_lost         доля эталонных объектов, прослеженных < 20 % кадров
  fragments           число раз, когда эталонный объект «теряется» и возвращается
  mota                MOTA: 1 - (misses + false_pos + switches) / total_gt
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class FrameMatch:
    """Сопоставление детекций в одном кадре: {pred_id: gt_id} и обратно."""
    pred_to_gt: Dict[int, int]
    gt_to_pred: Dict[int, int]


def match_by_centroids(pred_rows: Sequence[dict], gt_rows: Sequence[dict],
                       max_distance: float = 30.0) -> FrameMatch:
    """Сопоставить детекции одного кадра по близости центроидов (жадно, по убыванию расстояния)."""
    pairs = []
    for p in pred_rows:
        for g in gt_rows:
            d = math.hypot(float(p["x_px"]) - float(g["x_px"]),
                           float(p["y_px"]) - float(g["y_px"]))
            if d <= max_distance:
                pairs.append((d, int(p["track_id"]), int(g["object_id"])))
    pairs.sort()
    used_p, used_g = set(), set()
    p2g, g2p = {}, {}
    for _, pid, gid in pairs:
        if pid in used_p or gid in used_g:
            continue
        used_p.add(pid)
        used_g.add(gid)
        p2g[pid] = gid
        g2p[gid] = pid
    return FrameMatch(p2g, g2p)


def match_by_iou(pred_labels: np.ndarray, gt_labels: np.ndarray,
                 iou_threshold: float = 0.5) -> FrameMatch:
    """Сопоставить детекции кадра по IoU масок (нужны предсказанные и эталонные метки)."""
    from .segmentation.metrics import iou_matrix
    p_ids, g_ids, mat = iou_matrix(pred_labels, gt_labels)
    p2g, g2p = {}, {}
    if len(p_ids) and len(g_ids):
        order = np.dstack(np.unravel_index(np.argsort(-mat, axis=None, kind="stable"), mat.shape))[0]
        used_p, used_g = set(), set()
        for i, j in order:
            if mat[i, j] < iou_threshold:
                break
            if int(p_ids[i]) in used_p or int(g_ids[j]) in used_g:
                continue
            used_p.add(int(p_ids[i]))
            used_g.add(int(g_ids[j]))
            p2g[int(p_ids[i])] = int(g_ids[j])
            g2p[int(g_ids[j])] = int(p_ids[i])
    return FrameMatch(p2g, g2p)


@dataclass
class TrackingReport:
    idtp: int
    idfp: int
    idfn: int
    idf1: float
    id_precision: float
    id_recall: float
    n_switches: int
    switch_rate: float
    n_gt_tracks: int
    n_pred_tracks: int
    mostly_tracked: float
    mostly_lost: float
    fragments: int
    mota: float
    matches: List[Tuple[int, int, int]] = field(default_factory=list)  # (pred_id, gt_id, shared_frames)
    per_gt: Dict[int, dict] = field(default_factory=dict)


def evaluate_tracking(pred_table, gt_table, max_distance: float = 30.0) -> TrackingReport:
    """Оценить трекинг: pred_table и gt_table — строки с track_id/object_id, frame, x_px, y_px.

    pred_table: iterable of dict с ключами track_id, frame, x_px, y_px.
    gt_table:   iterable of dict с ключами object_id, frame, x_px, y_px.
    """
    by_frame: Dict[int, Tuple[List[dict], List[dict]]] = {}
    for r in pred_table:
        by_frame.setdefault(int(r["frame"]), ([], []))[0].append(r)
    for r in gt_table:
        by_frame.setdefault(int(r["frame"]), ([], []))[1].append(r)

    # Совпадения по кадрам: для каждого gt_id — последовательность pred_id (или None)
    gt_frames: Dict[int, List[Tuple[int, Optional[int]]]] = {}
    overlap: Dict[Tuple[int, int], int] = {}  # (pred_id, gt_id) -> shared frames
    for frame in sorted(by_frame):
        preds, gts = by_frame[frame]
        m = match_by_centroids(preds, gts, max_distance)
        for gid in {int(g["object_id"]) for g in gts}:
            pid = m.gt_to_pred.get(gid)
            gt_frames.setdefault(gid, []).append((frame, pid))
            if pid is not None:
                overlap[(pid, gid)] = overlap.get((pid, gid), 0) + 1

    gt_ids = sorted(gt_frames)
    pred_ids = sorted({pid for pid, _ in overlap})
    n_gt = len(gt_ids)
    n_pred = len(pred_ids)
    total_frames = len(by_frame)

    # Венгерский алгоритм: максимизируем совпадения -> минимизируем -overlap
    cost = np.zeros((max(n_pred, 1), max(n_gt, 1)))
    for (pid, gid), n in overlap.items():
        i = pred_ids.index(pid)
        j = gt_ids.index(gid)
        cost[i, j] = -n
    rows, cols = linear_sum_assignment(cost)
    matches: List[Tuple[int, int, int]] = []
    pred_to_gt: Dict[int, int] = {}
    for r, c in zip(rows, cols):
        if r < n_pred and c < n_gt and cost[r, c] < 0:
            pid, gid = pred_ids[r], gt_ids[c]
            shared = overlap[(pid, gid)]
            matches.append((pid, gid, shared))
            pred_to_gt[pid] = gid

    idtp = sum(m[2] for m in matches)
    total_pred = sum(len(p) for p, _ in by_frame.values()) if by_frame else 0
    # total_pred — число предсказанных наблюдений; но для IDF1 нужны только «уникальные» детекции
    total_pred_obs = sum(1 for frame, (preds, _) in by_frame.items() for _ in preds)
    total_gt_obs = sum(1 for frame, (_, gts) in by_frame.items() for _ in gts)
    idfp = total_pred_obs - idtp
    idfn = total_gt_obs - idtp
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 1.0
    id_prec = idtp / (idtp + idfp) if (idtp + idfp) else 1.0
    id_rec = idtp / (idtp + idfn) if (idtp + idfn) else 1.0

    # Переключения ID и фрагменты
    n_switches = 0
    fragments = 0
    per_gt: Dict[int, dict] = {}
    mostly_tracked = mostly_lost = 0
    for gid in gt_ids:
        seq = [pid for _, pid in gt_frames[gid]]
        observed_frames = len(seq)
        # переключения: сколько раз сменился pred_id (не None) при ненулевом предыдущем (не None)
        switches = 0
        prev = None
        for pid in seq:
            if pid is not None and prev is not None and pid != prev:
                switches += 1
            prev = pid
        # фрагменты: число «разрывов» (None между наблюдениями)
        frags = 0
        seen = False
        for pid in seq:
            if pid is None and seen:
                frags += 1
                seen = False
            elif pid is not None:
                seen = True
        n_switches += switches
        fragments += frags
        track_ratio = observed_frames / total_frames if total_frames else 0.0
        if track_ratio >= 0.8:
            mostly_tracked += 1
        elif track_ratio < 0.2:
            mostly_lost += 1
        matched_pid = next((pid for pid in seq if pid is not None), None)
        per_gt[gid] = {"switches": switches, "fragments": frags, "tracked_ratio": track_ratio,
                       "n_frames": observed_frames, "matched_pred": matched_pid}

    switch_rate = n_switches / max(n_gt, 1)
    mota = 1.0 - (idfn + idfp + n_switches) / max(total_gt_obs, 1)

    return TrackingReport(
        idtp=idtp, idfp=idfp, idfn=idfn, idf1=idf1,
        id_precision=id_prec, id_recall=id_rec,
        n_switches=n_switches, switch_rate=switch_rate,
        n_gt_tracks=n_gt, n_pred_tracks=n_pred,
        mostly_tracked=mostly_tracked / max(n_gt, 1),
        mostly_lost=mostly_lost / max(n_gt, 1),
        fragments=fragments, mota=mota,
        matches=matches, per_gt=per_gt,
    )


def report_to_dict(r: TrackingReport) -> dict:
    """Плоский словарь метрик для CSV/JSON."""
    return {
        "idtp": r.idtp, "idfp": r.idfp, "idfn": r.idfn,
        "idf1": round(r.idf1, 4), "id_precision": round(r.id_precision, 4),
        "id_recall": round(r.id_recall, 4),
        "n_switches": r.n_switches, "switch_rate": round(r.switch_rate, 4),
        "n_gt_tracks": r.n_gt_tracks, "n_pred_tracks": r.n_pred_tracks,
        "mostly_tracked": round(r.mostly_tracked, 4),
        "mostly_lost": round(r.mostly_lost, 4),
        "fragments": r.fragments, "mota": round(r.mota, 4),
    }
