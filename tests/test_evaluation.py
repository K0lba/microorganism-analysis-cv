"""Тесты модуля evaluation: сопоставление треков и ошибки признаков."""
import numpy as np
import pandas as pd
import pytest

from microtrack.evaluation import compute_errors, gt_area_perimeter, load_gt, match_tracks


def _table():
    return pd.DataFrame([
        {"track_id": 1, "frame": 0, "x_px": 10.0, "y_px": 20.0, "area_px2": 300.0,
         "length_px": 25.0, "width_px": 14.0, "perimeter_px": 62.0, "aspect_ratio": 1.8},
        {"track_id": 1, "frame": 1, "x_px": 14.0, "y_px": 22.0, "area_px2": 310.0,
         "length_px": 26.0, "width_px": 14.5, "perimeter_px": 63.0, "aspect_ratio": 1.8},
    ])


def _gt():
    return pd.DataFrame([
        {"gt_id": 10, "frame": 0, "x_px": 11.0, "y_px": 21.0, "length_px": 24.0, "width_px": 13.0},
        {"gt_id": 10, "frame": 1, "x_px": 15.0, "y_px": 23.0, "length_px": 25.0, "width_px": 13.5},
    ])


def test_match_tracks_adds_gt_id():
    m = match_tracks(_table(), _gt(), max_distance=30)
    assert "gt_id" in m.columns
    assert len(m) == 2 and (m["gt_id"] == 10).all()


def test_gt_area_perimeter():
    g = gt_area_perimeter(_gt())
    assert "area_px2" in g and "perimeter_px" in g and "aspect_ratio" in g
    a, b = 24 / 2, 13 / 2
    assert g["area_px2"].iloc[0] == pytest.approx(np.pi * a * b)


def test_compute_errors_columns():
    errs = compute_errors(match_tracks(_table(), _gt()), _gt())
    assert len(errs) == 5
    assert set(errs["feature"]) == {"length_px", "width_px", "area_px2", "perimeter_px", "aspect_ratio"}
    assert (errs["n"] == 2).all()
    assert errs["mean_abs_rel_error"].between(0, 2).all()


def test_compute_errors_zero_when_perfect():
    table = _table()
    gt_df = _gt()
    g = gt_area_perimeter(gt_df)
    table["length_px"] = g["length_px"].values
    table["width_px"] = g["width_px"].values
    table["area_px2"] = g["area_px2"].values
    table["perimeter_px"] = g["perimeter_px"].values
    table["aspect_ratio"] = g["aspect_ratio"].values
    errs = compute_errors(match_tracks(table, gt_df), gt_df)
    assert errs["mean_rel_error"].abs().max() < 1e-6
    assert errs["bias"].abs().max() < 1e-6


def test_load_gt_renames_columns(tmp_path):
    p = tmp_path / "objects.csv"
    p.write_text("object_id,frame_index,x_px,y_px,length_px,width_px\n10,0,5.0,6.0,20.0,10.0\n",
                 encoding="utf-8")
    gt = load_gt(p)
    assert "gt_id" in gt.columns and "frame" in gt.columns
    assert int(gt["gt_id"].iloc[0]) == 10
