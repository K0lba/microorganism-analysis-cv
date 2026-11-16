"""Тесты временных признаков: интервалы берутся из отметок времени, а не из номеров кадров."""
import math
import warnings

import cv2
import numpy as np
import pytest

from microtrack.morphometry import morphometry_features
from microtrack.temporal import (Calibration, Timeline, analyze_tracker,
                                 compute_temporal, save_rows_csv, summarize_tracks)
from microtrack.tracker import Tracker


def line_rows(frames, times, v=(12.0, -5.0), p0=(100.0, 80.0), tid=1, **extra):
    """Объект движется равномерно ПО ВРЕМЕНИ: p(t) = p0 + v*t (px/с)."""
    rows = []
    for f, t in zip(frames, times):
        r = {"track_id": tid, "frame": f, "x": p0[0] + v[0] * t, "y": p0[1] + v[1] * t,
             "observed": 1}
        r.update({k: fn(t) if callable(fn) else fn for k, fn in extra.items()})
        rows.append(r)
    return rows


SPEED = math.hypot(12.0, -5.0)  # 13 px/с


def speeds(rows):
    return np.array([r["speed_px_s"] for r in rows])


# ---------------------------------------------------------------- интервалы

def test_uniform_fps():
    frames = list(range(10))
    tl = Timeline.from_fps(10.0)
    res = compute_temporal(line_rows(frames, [f / 10 for f in frames]), tl)
    assert math.isnan(res[0]["dt_s"]) and math.isnan(res[0]["speed_px_s"])
    assert all(r["dt_s"] == pytest.approx(0.1) for r in res[1:])
    assert np.allclose(speeds(res)[1:], SPEED)
    assert res[-1]["path_length_px"] == pytest.approx(SPEED * 0.9)
    assert res[-1]["elapsed_s"] == pytest.approx(0.9)


@pytest.mark.parametrize("fps", [2.0, 10.0, 25.0, 60.0])
def test_speed_independent_of_frame_rate(fps):
    frames = list(range(12))
    res = compute_temporal(line_rows(frames, [f / fps for f in frames]),
                           Timeline.from_fps(fps))
    assert np.allclose(speeds(res)[1:], SPEED)
    assert res[1]["dt_s"] == pytest.approx(1 / fps)


def test_irregular_timestamps_use_time_not_frame_index():
    times = [0.0, 0.10, 0.35, 0.40, 1.00, 1.05, 2.00]
    frames = list(range(len(times)))
    rows = line_rows(frames, times)
    res = compute_temporal(rows, Timeline.from_sequence(times))
    assert np.allclose(speeds(res)[1:], SPEED)                    # скорость постоянна
    disp = [r["displacement_px"] for r in res[1:]]
    assert max(disp) / min(disp) > 5                              # смещения на кадр — разные
    assert [r["dt_s"] for r in res[1:]] == pytest.approx(list(np.diff(times)))
    # ошибочный расчёт «по номеру кадра» дал бы другой результат:
    naive = np.array(disp)                                        # dt = 1 кадр
    assert not np.allclose(naive, SPEED)


def test_skipped_frames_between_observations():
    """Обработаны только кадры 0, 3, 4, 9 из видео 20 fps."""
    fps = 20.0
    frames = [0, 3, 4, 9]
    res = compute_temporal(line_rows(frames, [f / fps for f in frames]),
                           Timeline.from_fps(fps))
    assert [r["frames_since_prev"] for r in res[1:]] == [3, 1, 5]
    assert [r["dt_s"] for r in res[1:]] == pytest.approx([0.15, 0.05, 0.25])
    assert np.allclose(speeds(res)[1:], SPEED)


def test_same_physics_different_sampling():
    """Одно движение, снятое с разной частотой и с прореживанием, даёт ту же скорость."""
    dense_t = [i / 50 for i in range(101)]
    dense = compute_temporal(line_rows(range(101), dense_t), Timeline.from_fps(50.0))
    thin_idx = list(range(0, 101, 5))   # включает последний кадр 100
    thin = compute_temporal(line_rows(thin_idx, [dense_t[i] for i in thin_idx]),
                            Timeline.from_fps(50.0))
    assert np.nanmean(speeds(dense)) == pytest.approx(np.nanmean(speeds(thin)))
    assert dense[-1]["path_length_px"] == pytest.approx(thin[-1]["path_length_px"], rel=0.01)


def test_non_increasing_timestamps_give_nan_and_warning():
    times = [0.0, 0.1, 0.1, 0.3]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = compute_temporal(line_rows(range(4), times), Timeline.from_sequence(times))
    assert any("не возрастают" in str(x.message) for x in w)
    assert math.isnan(res[2]["speed_px_s"]) and math.isfinite(res[3]["speed_px_s"])


def test_missing_timestamp_raises():
    with pytest.raises(KeyError):
        compute_temporal(line_rows([0, 1, 2], [0, 0.1, 0.2]), Timeline({0: 0.0, 1: 0.1}))


def test_predicted_rows_excluded_by_default():
    rows = line_rows(range(5), [i / 10 for i in range(5)])
    rows[2]["observed"] = 0
    tl = Timeline.from_fps(10.0)
    res = compute_temporal(rows, tl)
    assert [r["frame"] for r in res] == [0, 1, 3, 4]
    assert res[2]["dt_s"] == pytest.approx(0.2)
    assert np.allclose(speeds(res)[1:], SPEED)
    assert len(compute_temporal(rows, tl, include_predicted=True)) == 5


def test_direction_in_image_coordinates():
    rows = [{"track_id": 1, "frame": i, "x": 10.0, "y": 10.0 + 4 * i} for i in range(3)]
    res = compute_temporal(rows, Timeline.from_fps(1.0))
    assert res[1]["direction_deg"] == pytest.approx(90.0)        # вниз
    rows = [{"track_id": 1, "frame": i, "x": 10.0 + 4 * i, "y": 10.0} for i in range(3)]
    assert compute_temporal(rows, Timeline.from_fps(1.0))[1]["direction_deg"] == pytest.approx(0.0)
    still = [{"track_id": 1, "frame": i, "x": 5.0, "y": 5.0} for i in range(3)]
    r = compute_temporal(still, Timeline.from_fps(1.0))[1]
    assert r["speed_px_s"] == 0.0 and math.isnan(r["direction_deg"])


# ---------------------------------------------------------------- калибровка

def test_calibration_scales_everything_consistently():
    s = 0.5
    frames = list(range(6))
    times = [0.0, 0.2, 0.3, 0.7, 0.8, 1.0]
    rows = line_rows(frames, times, area=lambda t: 100.0, length=40.0, perimeter=90.0,
                     circularity=0.8)
    res = compute_temporal(rows, Timeline.from_sequence(times), Calibration(s))
    r = res[3]
    assert r["speed_um_s"] == pytest.approx(r["speed_px_s"] * s)
    assert r["displacement_um"] == pytest.approx(r["displacement_px"] * s)
    assert r["x_um"] == pytest.approx(r["x_px"] * s)
    assert r["area_um2"] == pytest.approx(100.0 * s ** 2)
    assert r["length_um"] == pytest.approx(40.0 * s)
    assert r["perimeter_um"] == pytest.approx(45.0)
    assert "circularity_um" not in r                      # безразмерные не масштабируются
    assert res[-1]["path_length_um"] == pytest.approx(res[-1]["path_length_px"] * s)
    # без калибровки µm-столбцов нет
    assert "speed_um_s" not in compute_temporal(rows, Timeline.from_sequence(times))[1]


def test_calibration_validation_and_reference():
    with pytest.raises(ValueError):
        Calibration(0.0)
    with pytest.raises(ValueError):
        Calibration(-1.0)
    cal = Calibration.from_reference(length_um=100.0, length_px=250.0)
    assert cal.um_per_px == pytest.approx(0.4)
    assert cal.length(10) == pytest.approx(4.0) and cal.area(10) == pytest.approx(1.6)
    assert math.isnan(Calibration().length(10))


# ---------------------------------------------------------------- относительные изменения

def test_relative_changes_vs_first_observation():
    times = [0.0, 0.5, 1.0, 2.5]
    area = {0.0: 100.0, 0.5: 150.0, 1.0: 100.0, 2.5: 50.0}
    length = {0.0: 20.0, 0.5: 30.0, 1.0: 20.0, 2.5: 10.0}
    rows = line_rows(range(4), times)
    for r, t in zip(rows, times):
        r["area"], r["length"] = area[t], length[t]
    res = compute_temporal(rows, Timeline.from_sequence(times))
    assert [r["area_rel_change"] for r in res] == pytest.approx([0.0, 0.5, 0.0, -0.5])
    assert [r["length_rel_change"] for r in res] == pytest.approx([0.0, 0.5, 0.0, -0.5])


def test_relative_change_zero_initial_is_nan():
    rows = line_rows(range(3), [0, 1, 2], area=lambda t: 0.0 if t == 0 else 5.0)
    res = compute_temporal(rows, Timeline.from_fps(1.0))
    assert all(math.isnan(r["area_rel_change"]) for r in res)


def test_relative_change_reference_is_first_observed_not_first_frame():
    rows = line_rows(range(4), [0, 1, 2, 3], area=lambda t: 100.0 * (1 + t))
    rows[0]["observed"] = 0                                # первый кадр — предсказание
    res = compute_temporal(rows, Timeline.from_fps(1.0))
    assert res[0]["frame"] == 1
    assert [r["area_rel_change"] for r in res] == pytest.approx([0.0, 0.5, 1.0])


# ---------------------------------------------------------------- несколько треков и итоги

def test_multiple_tracks_independent_and_summary():
    tl = Timeline.from_fps(10.0)
    a = line_rows(range(11), [i / 10 for i in range(11)], v=(30.0, 0.0), tid=1)
    # трек 2: движение по окружности радиуса 20 px, 1 оборот за 1 с
    b = [{"track_id": 2, "frame": i, "observed": 1,
          "x": 100 + 20 * math.cos(2 * math.pi * i / 10),
          "y": 100 + 20 * math.sin(2 * math.pi * i / 10)} for i in range(11)]
    c = [{"track_id": 3, "frame": i, "x": 5.0, "y": 5.0, "observed": 1} for i in range(5)]
    res = compute_temporal(b + c + a, tl)                        # порядок на входе — любой
    assert [r["track_id"] for r in res] == sorted(r["track_id"] for r in res)
    summ = {s["track_id"]: s for s in summarize_tracks(res)}
    assert summ[1]["straightness"] == pytest.approx(1.0)
    assert summ[1]["mean_speed_px_s"] == pytest.approx(30.0)
    assert summ[1]["max_speed_px_s"] == pytest.approx(30.0)
    assert summ[1]["duration_s"] == pytest.approx(1.0)
    assert summ[2]["net_displacement_px"] == pytest.approx(0.0, abs=1e-9)
    assert summ[2]["straightness"] == pytest.approx(0.0, abs=1e-9)
    chord = 2 * 20 * math.sin(math.pi / 10)                       # длина хорды за 0.1 с
    assert summ[2]["mean_speed_px_s"] == pytest.approx(chord / 0.1)
    assert summ[3]["path_length_px"] == 0.0 and math.isnan(summ[3]["straightness"])
    cal = summarize_tracks(compute_temporal(a, tl, Calibration(0.25)), Calibration(0.25))[0]
    assert cal["mean_speed_um_s"] == pytest.approx(7.5)


# ---------------------------------------------------------------- время из CSV

def test_timeline_from_csv_ms_and_sequences(tmp_path):
    p = tmp_path / "frames.csv"
    p.write_text("sequence,frame_index,timestamp_ms\n"
                 "seq_000,0,0\nseq_000,1,40\nseq_000,2,100\n"
                 "seq_001,0,5\nseq_001,1,25\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Timeline.from_csv(p)                                  # неоднозначно: две последовательности
    tl = Timeline.from_csv(p, sequence="seq_000")
    assert [tl.time(i) for i in range(3)] == pytest.approx([0.0, 0.04, 0.10])
    assert Timeline.from_csv(p, sequence="seq_001").time(1) == pytest.approx(0.025)
    with pytest.raises(ValueError):
        Timeline.from_csv(p, sequence="nope")
    with pytest.raises(KeyError):
        tl.time(7)


def test_timeline_from_csv_explicit_columns_and_errors(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("n,sec\n0,0.0\n1,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Timeline.from_csv(p)                                  # столбцы не распознаны
    tl = Timeline.from_csv(p, frame_col="n", time_col="sec")
    assert tl.time(1) == pytest.approx(0.5)
    q = tmp_path / "dup.csv"
    q.write_text("frame,time_s\n0,0\n0,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Timeline.from_csv(q)


def test_timeline_requires_source():
    with pytest.raises(ValueError):
        Timeline()
    with pytest.raises(ValueError):
        Timeline.from_fps(0)


# ---------------------------------------------------------------- связка с трекером

def test_integration_tracker_morphometry_temporal(tmp_path):
    """Растущий круг, движущийся по прямой; кадры сняты с неравномерным интервалом."""
    times = [0.0, 0.4, 0.6, 1.2, 1.4, 2.0]                  # неравные интервалы
    v = (25.0, 10.0)                                         # px/с; при этих t координаты целые
    tracker = Tracker(feature_fn=morphometry_features)
    for f, t in enumerate(times):
        m = np.zeros((200, 300), np.uint8)
        r = int(round(10 + 5 * t))
        cv2.circle(m, (int(round(40 + v[0] * t)), int(round(40 + v[1] * t))), r, 1, -1)
        tracker.update(f, [m.astype(bool)])
    tracker.finalize()
    assert len(tracker.tracks) == 1

    cal = Calibration(0.2)
    rows, summary = analyze_tracker(tracker, Timeline.from_sequence(times), cal)
    sp = np.array([r["speed_px_s"] for r in rows[1:]])
    assert np.allclose(sp, math.hypot(*v), rtol=0.01)
    assert rows[-1]["area_rel_change"] == pytest.approx((20 ** 2 - 10 ** 2) / 10 ** 2, rel=0.1)
    assert rows[-1]["length_rel_change"] == pytest.approx(1.0, rel=0.1)      # диаметр 20 -> 40
    assert rows[-1]["length_um"] == pytest.approx(rows[-1]["length"] * 0.2)
    assert summary[0]["mean_speed_um_s"] == pytest.approx(summary[0]["mean_speed_px_s"] * 0.2)

    out = save_rows_csv(rows, tmp_path / "temporal.csv")
    lines = open(out, encoding="utf-8").read().splitlines()
    assert lines[0].startswith("track_id,frame,t_s") and len(lines) == 1 + len(times)
    assert lines[1].split(",")[4] == ""                      # dt первого наблюдения — пусто
