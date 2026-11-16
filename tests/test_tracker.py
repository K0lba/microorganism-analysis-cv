"""Тесты трекера на синтетических траекториях с известными ID."""
import cv2
import numpy as np
import pytest

from microtrack.tracker import (Tracker, TrackerConfig, compute_features,
                                masks_from_label_image, track_sequence)

H, W = 120, 160


def ellipse_mask(cx, cy, a, b, angle=0.0):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (a, b), angle, 0, 360, 1, -1)
    return m.astype(bool)


# Эталон: gt_id -> (x0, y0, vx, vy, a, b, angle)
OBJECTS = {
    10: (20, 20, 2.0, 1.0, 8, 5, 0),
    20: (130, 30, -2.0, 1.5, 6, 6, 0),
    30: (30, 95, 1.5, -1.0, 10, 4, 30),
}


def make_sequence(objects, n_frames=30, hidden=None, seed=0):
    """Вернуть (frames_masks, gt) где gt[t] = {gt_id: det_index}.

    hidden: {gt_id: set(frames)} — кадры, где объект не виден (пропуск детекции).
    Порядок детекций в кадре перемешивается: трекер не должен на него опираться.
    """
    rng = np.random.default_rng(seed)
    hidden = hidden or {}
    frames, gt = [], []
    for t in range(n_frames):
        items = [(gid, p) for gid, p in objects.items() if t not in hidden.get(gid, ())]
        rng.shuffle(items)
        masks, mapping = [], {}
        for i, (gid, (x0, y0, vx, vy, a, b, ang)) in enumerate(items):
            masks.append(ellipse_mask(x0 + vx * t, y0 + vy * t, a, b, ang))
            mapping[gid] = i
        frames.append(masks)
        gt.append(mapping)
    return frames, gt


def collect_track_ids(tracker_fn, frames):
    """Прогнать трекер; вернуть список {det_index: track_id} по кадрам."""
    tr = Tracker()
    per_frame = [tr.update(t, m) for t, m in enumerate(frames)]
    tr.finalize()
    return tr, per_frame


def id_map(per_frame, gt):
    """gt_id -> множество track_id, которые ему назначались."""
    out = {}
    for assigned, mapping in zip(per_frame, gt):
        for gid, di in mapping.items():
            out.setdefault(gid, set()).add(assigned.get(di))
    return out


def test_features_ellipse():
    f = compute_features(ellipse_mask(60, 50, 14, 8))
    assert abs(f.center[0] - 60) < 0.5 and abs(f.center[1] - 50) < 0.5
    assert abs(f.major - 28) < 2 and abs(f.minor - 16) < 2
    assert 0.8 < f.circularity <= 1.0
    assert compute_features(np.zeros((10, 10), bool)) is None


def test_stable_ids_multiple_objects():
    frames, gt = make_sequence(OBJECTS)
    tr, per_frame = collect_track_ids(None, frames)
    ids = id_map(per_frame, gt)
    assert all(len(v) == 1 and None not in v for v in ids.values()), ids
    assert len({next(iter(v)) for v in ids.values()}) == 3  # разные ID у разных объектов
    assert len(tr.tracks) == 3
    for t in tr.tracks.values():
        assert len(t.observations) == len(frames) and t.mean_confidence > 0.8


def test_trajectories_match_ground_truth():
    frames, gt = make_sequence(OBJECTS)
    tr, per_frame = collect_track_ids(None, frames)
    for gid, (x0, y0, vx, vy, *_) in OBJECTS.items():
        track = tr.tracks[next(iter(id_map(per_frame, gt)[gid]))]
        traj = track.trajectory()
        for fr, x, y in traj:
            assert abs(x - (x0 + vx * fr)) < 1.0 and abs(y - (y0 + vy * fr)) < 1.0


def test_crossing_objects_keep_ids():
    """Два одинаковых объекта проходят друг сквозь друга по диагонали."""
    objs = {1: (30, 30, 3.0, 2.0, 7, 7, 0), 2: (110, 30, -3.0, 2.0, 7, 7, 0)}
    frames, gt = make_sequence(objs, n_frames=25)
    tr, per_frame = collect_track_ids(None, frames)
    ids = id_map(per_frame, gt)
    assert all(len(v) == 1 for v in ids.values()), ids
    assert len(tr.tracks) == 2


def test_short_gap_keeps_id():
    frames, gt = make_sequence(OBJECTS, hidden={10: {10, 11, 12}})
    tr, per_frame = collect_track_ids(None, frames)
    ids = id_map(per_frame, gt)
    assert len(ids[10] - {None}) == 1
    track = tr.tracks[next(iter(ids[10] - {None}))]
    pred = [o for o in track.observations if not o.observed]
    assert [o.frame for o in pred] == [10, 11, 12]
    assert len(tr.tracks) == 3  # новых треков не появилось
    x0, y0, vx, vy, *_ = OBJECTS[10]  # предсказание близко к истине
    for o in pred:
        assert abs(o.features.center[0] - (x0 + vx * o.frame)) < 2.0


def test_long_gap_finishes_track_and_new_id():
    cfg = TrackerConfig(max_missed=3)
    frames, gt = make_sequence(OBJECTS, hidden={10: set(range(8, 16))})
    tr = track_sequence(frames, cfg)
    finished = [t for t in tr.tracks.values() if t.end_frame == 7]
    assert len(finished) == 1 and finished[0].status == "finished"
    assert all(o.observed for o in finished[0].observations)  # хвост предсказаний срезан
    assert len(tr.tracks) == 4  # после возвращения объект получил новый ID


def test_new_object_spawns_track():
    frames, gt = make_sequence({10: OBJECTS[10]}, n_frames=10)
    extra = ellipse_mask(100, 90, 6, 6)
    for t in range(5, 10):
        frames[t].append(extra)
    tr = track_sequence(frames)
    assert len(tr.tracks) == 2
    assert sorted(len(t.observations) for t in tr.tracks.values()) == [5, 10]


def test_single_target_mode_ignores_others():
    from types import SimpleNamespace
    frames, gt = make_sequence(OBJECTS, n_frames=20)
    init_mask = frames[0][gt[0][20]]  # выбираем объект 20
    tr = Tracker()
    track = tr.init_from_target(SimpleNamespace(mask=init_mask), frame=0)
    for t in range(1, 20):
        assigned = tr.update(t, frames[t])
        assert list(assigned.values()) == [track.id]
        assert assigned and gt[t][20] in assigned
    assert len(tr.tracks) == 1 and len(track.observations) == 20


def test_label_image_input():
    frames, gt = make_sequence(OBJECTS, n_frames=5)
    tr = Tracker()
    for t, masks in enumerate(frames):
        labels = np.zeros((H, W), np.int32)
        for i, m in enumerate(masks, start=1):
            labels[m] = i
        tr.update(t, labels)
    assert len(tr.tracks) == 3
    assert len(masks_from_label_image(labels)) == 3


def test_confidence_drops_for_odd_match():
    """Объект резко меняет размер: сопоставление допустимо, но уверенность ниже."""
    good = [ellipse_mask(40 + 2 * t, 50, 8, 8) for t in range(6)]
    bad = good[:3] + [ellipse_mask(46 + 2, 50, 12, 12)] + good[4:]
    t_good, t_bad = track_sequence(good), track_sequence(bad)
    c_good = t_good.tracks[1].observations[3].confidence
    c_bad = t_bad.tracks[1].observations[3].confidence
    assert c_good > 0.9 and c_bad < c_good


def test_frame_order_enforced():
    tr = Tracker()
    tr.update(3, [ellipse_mask(30, 30, 5, 5)])
    with pytest.raises(ValueError):
        tr.update(3, [])


def test_save_csv(tmp_path):
    frames, _ = make_sequence(OBJECTS, n_frames=6)
    tr = track_sequence(frames)
    path = tr.save_csv(tmp_path / "tracks.csv")
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    assert lines[0].startswith("track_id,frame,x,y,area")
    assert len(lines) == 1 + 3 * 6
