"""Тесты метрик трекинга против эталона: IDF1, переключения ID, MOTA."""
import numpy as np
import pandas as pd
import pytest

from microtrack.tracking_metrics import (
    evaluate_tracking,
    match_by_centroids,
    report_to_dict,
)


def _rows(pairs):
    """pairs: [(track_id, object_id, frame, x, y), ...] → pred и gt строки."""
    pred = [{"track_id": t, "frame": f, "x_px": x, "y_px": y} for t, o, f, x, y in pairs]
    gt = [{"object_id": o, "frame": f, "x_px": x, "y_px": y} for t, o, f, x, y in pairs]
    return pred, gt


def test_perfect_tracking():
    pred, gt = _rows([(1, 10, 0, 10, 20), (1, 10, 1, 14, 22),
                      (2, 20, 0, 50, 60), (2, 20, 1, 54, 62)])
    r = evaluate_tracking(pred, gt, max_distance=30)
    assert r.idf1 == pytest.approx(1.0)
    assert r.n_switches == 0 and r.mota == pytest.approx(1.0)
    assert r.n_gt_tracks == 2 and r.n_pred_tracks == 2


def test_id_switch():
    # эталон 10: кадр 0 → pred 1, кадр 1 → pred 2 (переключение)
    pred, gt = _rows([(1, 10, 0, 10, 20), (2, 10, 1, 14, 22),
                      (3, 20, 0, 50, 60), (3, 20, 1, 54, 62)])
    r = evaluate_tracking(pred, gt, max_distance=30)
    assert r.n_switches == 1
    assert r.idf1 < 1.0
    assert r.switch_rate == pytest.approx(0.5)


def test_missed_object():
    # эталон 20 виден в кадрах 0 и 1, но предсказан только в кадре 0
    pred, gt = _rows([(1, 10, 0, 10, 20), (1, 10, 1, 14, 22),
                      (2, 20, 0, 50, 60),
                      (None, 20, 1, 54, 62)])  # None — нет предсказания
    pred = [p for p in pred if p["track_id"] is not None]
    gt = [g for g in gt]
    r = evaluate_tracking(pred, gt, max_distance=30)
    assert r.idfn > 0
    assert r.idf1 < 1.0
    assert r.fragments >= 1


def test_false_positive():
    # лишний предсказанный трек 3 (нет эталона 30)
    pred, gt = _rows([(1, 10, 0, 10, 20), (1, 10, 1, 14, 22),
                      (2, 20, 0, 50, 60), (2, 20, 1, 54, 62),
                      (3, 30, 0, 90, 90), (3, 30, 1, 92, 92)])
    # убираем эталон 30 — трек 3 становится ложной положительной детекцией
    gt = [g for g in gt if g["object_id"] != 30]
    r = evaluate_tracking(pred, gt, max_distance=30)
    assert r.idfp > 0
    assert r.idf1 < 1.0


def test_match_by_centroids_greedy():
    preds = [{"track_id": 1, "x_px": 10.0, "y_px": 20.0},
             {"track_id": 2, "x_px": 50.0, "y_px": 60.0}]
    gts = [{"object_id": 10, "x_px": 11.0, "y_px": 21.0},
           {"object_id": 20, "x_px": 80.0, "y_px": 80.0}]
    m = match_by_centroids(preds, gts, max_distance=30)
    assert m.pred_to_gt == {1: 10}
    assert m.gt_to_pred == {10: 1}


def test_report_to_dict_keys():
    pred, gt = _rows([(1, 10, 0, 10, 20), (1, 10, 1, 14, 22)])
    r = evaluate_tracking(pred, gt)
    d = report_to_dict(r)
    assert {"idf1", "n_switches", "mota", "id_precision", "id_recall"} <= set(d)
    assert 0 <= d["idf1"] <= 1


def test_mostly_tracked_lost():
    # 6 кадров; эталон 10 — все 6, 20 — 1 из 6 (< 20 %), 30 — 3 из 6
    pairs = [(1, 10, f, 10 + 4 * f, 20 + 2 * f) for f in range(6)]
    pairs += [(2, 20, 0, 50, 60)]
    pairs += [(3, 30, f, 90 + 2 * f, 90 + 2 * f) for f in range(3)]
    pred, gt = _rows(pairs)
    r = evaluate_tracking(pred, gt, max_distance=30)
    assert r.mostly_tracked > 0   # объект 10: 6/6
    assert r.mostly_lost > 0       # объект 20: 1/6 < 0.2
