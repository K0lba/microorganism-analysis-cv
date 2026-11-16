import numpy as np

from src.generate_synthetic import build_scene, draw_cell_mask, make_tracks


def test_cell_shapes_stay_fixed_and_are_not_bare_ellipses():
    rng = np.random.default_rng(3)
    tracks = make_tracks(rng, 180, 140, 12, "shapes")
    rods = [t for t in tracks if t["cell_type"] == "rod"]
    diplos = [t for t in tracks if t["cell_type"] == "diplo"]
    assert rods and diplos
    rod = rods[0]
    first = draw_cell_mask((140, 180), 90, 70, rod["length"], rod["width"], rod["angle"],
                           rod["cell_type"], rod["harmonics"])
    second = draw_cell_mask((140, 180), 90, 70, rod["length"], rod["width"], rod["angle"],
                            rod["cell_type"], rod["harmonics"])
    assert np.array_equal(first, second) and int(first.sum()) > 20
    diplo = diplos[0]
    peanut = draw_cell_mask((140, 180), 90, 70, diplo["length"], diplo["width"], 0.0,
                            "diplo", diplo["harmonics"])
    assert int(peanut.sum()) > int(np.pi * (diplo["width"] / 2) ** 2)


def test_ids_and_positions_persist_between_frames():
    rng = np.random.default_rng(7)
    config = {
        "background_mean": 35, "background_noise": 5.0, "background_texture": 4.0,
        "vignette": 0.08, "cell_intensity_min": 35, "cell_intensity_max": 115,
        "contrast_min": 0.65, "contrast_max": 1.35, "brightness_shift": 18,
        "blur_probability": 0.0, "sensor_noise": 0.0,
    }
    tracks = make_tracks(rng, 128, 96, 6, "test")
    a = build_scene(rng, 128, 96, tracks, 0, 5.0, "test", config)
    b = build_scene(rng, 128, 96, tracks, 1, 5.0, "test", config)
    assert a[0].shape == (96, 128)
    assert a[1].shape == (96, 128)
    assert a[2].ndim == 3 and a[2].shape[1:] == (96, 128)
    assert len(a[3]) == len(b[3]) == 6
    assert {r["object_id"] for r in a[3]} == {r["object_id"] for r in b[3]}
    assert all(r["timestamp_sec"] == 0.2 for r in b[3])
    assert all(r["area_px2"] > 0 and r["perimeter_px"] > 0 for r in a[3])
    assert {r["cell_type"] for r in a[3]} <= {"coccus", "oval", "rod", "diplo"}
