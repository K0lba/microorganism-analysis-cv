"""Сквозной конвейер на синтетических кадрах с известными траекториями."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from microtrack.export import validate_table
from microtrack.io import Calibration
from microtrack.pipeline import PipelineConfig, relabel_min_area, run_pipeline, tracker_config_from_dict
from microtrack.visualization import write_image

H, W = 120, 160
# (x0, y0, vx, vy, a, b) — px и px за кадр
OBJECTS = [(25, 30, 4, 2, 9, 6), (130, 40, -4, 2, 7, 7), (60, 95, 3, -2, 10, 5)]


def write_frames(folder: Path, n: int = 8) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for f in range(n):
        img = np.full((H, W), 200, np.uint8)
        for x0, y0, vx, vy, a, b in OBJECTS:
            cv2.ellipse(img, (x0 + vx * f, y0 + vy * f), (a, b), 0, 0, 360, 60, -1)
        write_image(folder / f"frame_{f:03d}.png", img)
    return folder


def config(**kw) -> PipelineConfig:
    cfg = PipelineConfig.from_dict({
        "input": {"fps": 4.0},
        "segmentation": {"min_object_area_px": 20},
        "output": {"save_parquet": False, "save_visualization": True},
    })
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_frames_to_table(tmp_path):
    src = write_frames(tmp_path / "кадры")          # кириллица в пути — как у пользователя
    out = tmp_path / "результат"
    res = run_pipeline(src, config(), out)
    assert res.n_frames == 8 and len(res.tracker.tracks) == 3
    assert res.report.ok, str(res.report)
    t = res.table
    assert len(t) == 3 * 8 and t["observed"].all()
    assert sorted(t["track_id"].unique()) == [1, 2, 3]
    g = t[t["track_id"] == 1].sort_values("frame")
    assert list(g["timestamp_s"]) == pytest.approx([f / 4.0 for f in range(8)])
    np.testing.assert_allclose(g["speed_px_s"].iloc[1:], np.hypot(4, 2) * 4.0, rtol=0.05)
    assert t["um_per_px"].isna().all() and (t["calibration_source"] == "none").all()

    back = pd.read_csv(out / "observations.csv")
    assert len(back) == len(t)
    summary = pd.read_csv(out / "tracks_summary.csv")
    assert len(summary) == 3 and (summary["n_observations"] == 8).all()
    assert (out / "preview.png").exists()
    info = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert info["n_tracks"] == 3 and info["validation"]["ok"]


def test_stride_keeps_source_time_and_calibration(tmp_path):
    src = write_frames(tmp_path / "frames")
    cfg = config(frame_stride=2, calibration=Calibration(0.5))
    cfg.output.save_visualization = False
    res = run_pipeline(src, cfg, tmp_path / "out")
    t = res.table
    assert res.report.ok, str(res.report)
    assert sorted(t["frame"].unique()) == [0, 2, 4, 6]
    rest = t[t["frames_since_prev"].notna()]
    assert (rest["frames_since_prev"] == 2).all()
    assert rest["dt_s"].to_numpy() == pytest.approx(0.5)
    np.testing.assert_allclose(t["x_um"], t["x_px"] * 0.5)
    assert (t["calibration_source"] == "config").all()


def test_conditions_and_experiment_id(tmp_path):
    src = write_frames(tmp_path / "frames", n=3)
    cfg = config(experiment_id="exp42", conditions={"medium": "PBS", "temperature_c": 37})
    cfg.output.save_visualization = False
    t = run_pipeline(src, cfg, tmp_path / "out").table
    assert (t["experiment_id"] == "exp42").all()
    assert (t["cond_medium"] == "PBS").all() and (t["cond_temperature_c"] == 37).all()


def test_default_yaml_builds_pipeline_config():
    cfg = PipelineConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert cfg.segmentation_method == "classical"
    assert cfg.tracking.min_gate == 30.0 and cfg.tracking.max_missed == 2
    assert not cfg.calibration.calibrated
    assert cfg.frame_stride == 1 and cfg.timestamp_source == "source_pts"


def test_tracking_aliases_and_errors():
    cfg = tracker_config_from_dict({"max_displacement_px": 12, "max_missed_frames": 1,
                                    "use_shape_similarity": False})
    assert (cfg.min_gate, cfg.max_missed, cfg.w_shape) == (12, 1, 0.0)
    with pytest.raises(ValueError):
        tracker_config_from_dict({"nope": 1})
    with pytest.raises(ValueError):
        PipelineConfig.from_dict({"output": {"nope": True}})


def test_unknown_model_is_rejected(tmp_path):
    src = write_frames(tmp_path / "frames", n=2)
    with pytest.raises(ValueError):
        run_pipeline(src, config(segmentation_method="unet"), tmp_path / "out")


def test_relabel_min_area():
    lab = np.zeros((10, 10), np.int32)
    lab[0:2, 0:2] = 5          # 4 px
    lab[5:9, 5:9] = 7          # 16 px
    out = relabel_min_area(lab, 10)
    assert sorted(np.unique(out)) == [0, 1] and int((out == 1).sum()) == 16


def test_empty_source_and_no_objects(tmp_path):
    empty = tmp_path / "blank"
    empty.mkdir()
    write_image(empty / "f0.png", np.full((40, 40), 200, np.uint8))
    cfg = config()
    cfg.output.save_visualization = False
    res = run_pipeline(empty, cfg, tmp_path / "out")
    assert len(res.table) == 0 and res.report.ok
    assert validate_table(res.table).warnings
