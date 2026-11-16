"""Тесты экспорта: схема, типы, пустые значения, единицы, уникальность, форматы."""
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from microtrack import export as ex
from microtrack.export import (COLUMNS, ExportMeta, build_table, export_all, export_csv,
                               export_parquet, read_table, schema_markdown, validate_table)
from microtrack.morphometry import morphometry_features
from microtrack.quality import segmentation_quality
from microtrack.temporal import Calibration, Timeline
from microtrack.tracker import Tracker

H, W = 120, 200
TIMES = [0.0, 0.1, 0.25, 0.30, 0.55, 0.60, 0.80, 0.90]


def make_tracker(n_frames=8, hidden=None, scale_object=None):
    """Два объекта; кадр-фон серый, объекты тёмные. hidden: {frame: {object_idx}}."""
    hidden = hidden or {}
    state = {"gray": None}

    def feature_fn(mask):
        d = morphometry_features(mask)
        d.update(segmentation_quality(state["gray"], mask, d))
        return d

    tr = Tracker(feature_fn=feature_fn)
    for f in range(n_frames):
        img = np.full((H, W), 200, np.uint8)
        masks = []
        for i, (x0, y0, vx, vy, a, b) in enumerate([(30, 30, 4, 2, 9, 6), (150, 80, -4, -1, 8, 8)]):
            if i in hidden.get(f, ()):
                continue
            m = np.zeros((H, W), np.uint8)
            aa = a * (scale_object(f) if scale_object and i == 0 else 1)
            cv2.ellipse(m, (x0 + vx * f, y0 + vy * f), (int(aa), b), 0, 0, 360, 1, -1)
            img[m > 0] = 60
            masks.append(m.astype(bool))
        state["gray"] = img
        tr.update(f, masks)
    tr.finalize()
    return tr


def meta(**kw):
    base = dict(experiment_id="exp1", sequence_id="seq_000",
                calibration=Calibration(0.5), calibration_source="config",
                timestamp_source="csv", fps_nominal=10.0, frame_step=1,
                conditions={"medium": "PBS", "temperature_c": 25})
    base.update(kw)
    return ExportMeta(**base)


@pytest.fixture
def table():
    return build_table(make_tracker(), Timeline.from_sequence(TIMES), meta())


# ---------------------------------------------------------------- схема и типы

def test_schema_has_unique_names_and_valid_dtypes():
    names = [c.name for c in COLUMNS]
    assert len(names) == len(set(names))
    assert {c.dtype for c in COLUMNS} <= {"str", "int", "float", "bool", "Int"}
    assert all(c.unit and c.description for c in COLUMNS)
    for k in ("experiment_id", "track_id", "timestamp_s"):
        assert k in names


def test_table_columns_types_and_validation(table):
    rep = validate_table(table)
    assert rep.ok, str(rep)
    assert list(table.columns[:len(COLUMNS)]) == [c.name for c in COLUMNS]
    assert "cond_medium" in table.columns and "cond_temperature_c" in table.columns
    assert table["track_id"].dtype == "int64" and table["timestamp_s"].dtype == "float64"
    assert str(table["observed"].dtype) == "boolean" and str(table["frames_since_prev"].dtype) == "Int64"
    assert len(table) == 2 * 8
    assert not table.duplicated(ex.KEY).any()
    assert (table["experiment_id"] == "exp1").all() and (table["schema_version"] == ex.SCHEMA_VERSION).all()


def test_first_observation_has_na_motion_only(table):
    for _, g in table.groupby("track_id"):
        g = g.sort_values("frame")
        first, rest = g.iloc[0], g.iloc[1:]
        assert pd.isna(first["dt_s"]) and pd.isna(first["speed_px_s"]) and pd.isna(first["frames_since_prev"])
        assert rest["speed_px_s"].notna().all() and rest["dt_s"].notna().all()
        assert first["area_rel_change"] == 0.0


def test_timestamps_come_from_timeline(table):
    g = table[table["track_id"] == table["track_id"].min()].sort_values("frame")
    assert list(g["timestamp_s"]) == pytest.approx(TIMES)
    assert list(g["dt_s"].iloc[1:]) == pytest.approx(list(np.diff(TIMES)))


def test_units_consistent_with_calibration(table):
    s = 0.5
    assert table["um_per_px"].eq(s).all()
    assert (table["area_um2"] == pytest.approx(table["area_px2"] * s ** 2)) if False else True
    np.testing.assert_allclose(table["area_um2"], table["area_px2"] * s ** 2)
    np.testing.assert_allclose(table["length_um"], table["length_px"] * s)
    np.testing.assert_allclose(table["x_um"], table["x_px"] * s)
    ok = table["speed_px_s"].notna()
    np.testing.assert_allclose(table.loc[ok, "speed_um_s"], table.loc[ok, "speed_px_s"] * s)


def test_uncalibrated_table_has_na_micrometers():
    t = build_table(make_tracker(), Timeline.from_sequence(TIMES),
                    meta(calibration=Calibration(), calibration_source="none"))
    rep = validate_table(t)
    assert rep.ok, str(rep)
    assert t["um_per_px"].isna().all()
    for c in ("x_um", "area_um2", "speed_um_s", "length_um", "radial_mean_um"):
        assert t[c].isna().all(), c
    assert t["area_px2"].notna().all()


def test_quality_columns(table):
    assert table["match_confidence"].between(0, 1).all()
    assert (table["seg_contrast"] > 100).all()               # тёмные объекты на светлом фоне
    assert (table["seg_flag"] == "ok").all()
    assert table["track_n_obs"].eq(8).all()
    assert table["track_status"].eq("finished").all()


def test_conditions_columns():
    t = build_table(make_tracker(n_frames=3), Timeline.from_fps(10), meta(conditions={"pH": 7.4}))
    assert (t["cond_pH"] == 7.4).all()


# ---------------------------------------------------------------- пропуски и флаги

def test_predicted_rows_excluded_then_included():
    tr = make_tracker(hidden={3: {0}, 4: {0}})
    tl = Timeline.from_sequence(TIMES)
    t = build_table(tr, tl, meta())
    assert validate_table(t).ok
    g = t[t["track_id"] == 1]
    assert len(g) == 6 and g["observed"].all()
    gap = g.sort_values("frame").iloc[3]                      # первое наблюдение после пропуска
    assert int(gap["frames_since_prev"]) == 3 and gap["dt_s"] == pytest.approx(TIMES[5] - TIMES[2])
    full = build_table(tr, tl, meta(), include_predicted=True)
    assert validate_table(full).ok, str(validate_table(full))
    g = full[full["track_id"] == 1]
    assert len(g) == 8 and (~g["observed"]).sum() == 2


def test_area_jump_flag():
    tr = make_tracker(scale_object=lambda f: 1.0 if f < 4 else 1.8)
    t = build_table(tr, Timeline.from_sequence(TIMES), meta())
    flagged = t[t["seg_flag"].str.contains("area_jump")]
    assert list(flagged["frame"]) == [4] and flagged["track_id"].iloc[0] == 1


# ---------------------------------------------------------------- проверки ловят ошибки

def _errors(df):
    return " | ".join(validate_table(df).errors)


def test_validation_detects_duplicate_key(table):
    bad = pd.concat([table, table.iloc[[3]]], ignore_index=True)
    assert "повторяющимся ключом" in _errors(bad)


def test_validation_detects_nulls_in_required(table):
    bad = table.copy()
    bad.loc[2, "area_px2"] = np.nan
    assert "area_px2" in _errors(bad)


def test_validation_detects_wrong_unit_conversion(table):
    bad = table.copy()
    bad.loc[1, "area_um2"] = bad.loc[1, "area_px2"] * 0.5      # забыли возвести масштаб в квадрат
    assert "area_um2" in _errors(bad)


def test_validation_detects_um_without_calibration(table):
    bad = table.copy()
    bad["um_per_px"] = pd.array([np.nan] * len(bad), dtype="float64")
    bad["calibration_source"] = "none"
    assert "калибровки нет" in _errors(bad)


def test_validation_detects_time_problems(table):
    bad = table.copy()
    idx = bad[(bad["track_id"] == 1) & (bad["frame"] == 3)].index[0]
    bad.loc[idx, "timestamp_s"] = 0.0
    assert "не возрастает" in _errors(bad)


def test_validation_detects_range_and_type_errors(table):
    bad = table.copy()
    bad.loc[0, "match_confidence"] = 1.5
    bad.loc[1, "circularity"] = -0.1
    assert "match_confidence" in _errors(bad) and "circularity" in _errors(bad)
    bad = table.copy()
    bad["track_id"] = bad["track_id"].astype("float64")
    assert "track_id: тип" in _errors(bad)
    bad = table.drop(columns=["solidity"])
    assert "нет столбцов" in _errors(bad)


def test_validation_detects_inconsistent_track_n_obs(table):
    bad = table.copy()
    bad.loc[bad["track_id"] == 1, "track_n_obs"] = 99
    assert "track_n_obs" in _errors(bad)


def test_unknown_columns_are_warnings_only(table):
    t = table.copy()
    t["foo"] = 1
    rep = validate_table(t)
    assert rep.ok and any("foo" in w for w in rep.warnings)


# ---------------------------------------------------------------- форматы

def test_csv_roundtrip_preserves_values_types_and_na(table, tmp_path):
    path = export_csv(table, tmp_path / "obs.csv")
    back = read_table(path)
    assert validate_table(back).ok
    base = [c.name for c in COLUMNS]
    pd.testing.assert_frame_equal(table[base].reset_index(drop=True), back[base], check_dtype=True)
    assert back["frames_since_prev"].isna().sum() == table["frames_since_prev"].isna().sum()
    text = Path(path).read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("experiment_id,sequence_id,track_id,frame,timestamp_s")


def test_export_all_validates_and_writes(table, tmp_path):
    out = export_all(table, tmp_path, formats=("csv",))
    assert Path(out["csv"]).exists()
    bad = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        export_all(bad, tmp_path / "x", formats=("csv",))
    assert "csv" in export_all(bad, tmp_path / "y", formats=("csv",), strict=False)
    with pytest.raises(ValueError):
        export_all(table, tmp_path, formats=("xlsx",))


def test_parquet_roundtrip(table, tmp_path):
    pytest.importorskip("pyarrow")
    path = export_parquet(table, tmp_path / "obs.parquet")
    back = read_table(path)
    base = [c.name for c in COLUMNS]
    pd.testing.assert_frame_equal(table[base].reset_index(drop=True), back[base].reset_index(drop=True))


def test_parquet_without_engine_gives_clear_error(table, tmp_path, monkeypatch):
    def boom(self, *a, **k):
        raise ImportError("no engine")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(ImportError):
        export_parquet(table, tmp_path / "x.parquet")


def test_to_dataframe_validates():
    df = ex.to_dataframe(make_tracker(n_frames=4), Timeline.from_fps(10), meta())
    assert isinstance(df, pd.DataFrame) and len(df) == 8


def test_schema_markdown_lists_all_columns_and_matches_repo_doc():
    md = schema_markdown()
    for c in COLUMNS:
        assert f"`{c.name}`" in md
    doc = Path(__file__).resolve().parents[1] / "docs" / "export_schema.md"
    if doc.exists():
        assert doc.read_text(encoding="utf-8") == md, "docs/export_schema.md устарел: python scripts/gen_schema_doc.py"
