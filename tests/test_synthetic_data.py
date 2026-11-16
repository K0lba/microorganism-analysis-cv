import numpy as np
import pytest

from microtrack.synthetic import DEFAULT_SCENARIOS, SceneSpec, generate_dataset, generate_scene


def _adjacent_pairs(labels):
    pairs = set()
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (a > 0) & (b > 0) & (a != b)
        pairs |= {tuple(sorted(p)) for p in zip(a[m].tolist(), b[m].tolist())}
    return pairs


@pytest.mark.parametrize("spec", DEFAULT_SCENARIOS)
def test_scene_shapes_and_object_count(spec):
    img, gt = generate_scene(spec, seed=3)
    assert img.shape == gt.shape == spec.size
    assert img.dtype == np.uint8 and gt.dtype == np.int32
    ids = np.unique(gt[gt > 0])
    assert ids.tolist() == list(range(1, spec.n_objects + 1))


def test_deterministic_by_seed():
    spec = DEFAULT_SCENARIOS[0]
    a1, g1 = generate_scene(spec, 5)
    a2, g2 = generate_scene(spec, 5)
    a3, _ = generate_scene(spec, 6)
    assert np.array_equal(a1, a2) and np.array_equal(g1, g2)
    assert not np.array_equal(a1, a3)


def test_polarity_and_contrast():
    dark, gt = generate_scene(SceneSpec("d", dark=True, noise=0, blur=0), 1)
    bright, gt2 = generate_scene(SceneSpec("b", dark=False, background=45, noise=0, blur=0), 1)
    assert dark[gt > 0].mean() < dark[gt == 0].mean()
    assert bright[gt2 > 0].mean() > bright[gt2 == 0].mean()
    assert dark[gt == 0].mean() - dark[gt > 0].mean() == pytest.approx(110)


def test_objects_stay_inside_margin():
    spec = SceneSpec("m", n_objects=15, margin=5)
    _, gt = generate_scene(spec, 2)
    assert (gt[:5] == 0).all() and (gt[-5:] == 0).all()
    assert (gt[:, :5] == 0).all() and (gt[:, -5:] == 0).all()


def test_touching_pairs_are_adjacent_and_others_are_not():
    spec = next(s for s in DEFAULT_SCENARIOS if s.name == "touching")
    _, gt = generate_scene(spec, 4)
    pairs = _adjacent_pairs(gt)
    assert len(pairs) == spec.touching_pairs
    # слипшиеся пары создаются первыми: метки (1,2), (3,4), ...
    assert pairs == {(2 * i + 1, 2 * i + 2) for i in range(spec.touching_pairs)}
    _, clean = generate_scene(DEFAULT_SCENARIOS[0], 4)
    assert _adjacent_pairs(clean) == set()


def test_illumination_gradient_changes_background():
    img, gt = generate_scene(SceneSpec("g", gradient=50, noise=0, blur=0, n_objects=3), 0)
    left, right = img[:, :20][gt[:, :20] == 0].mean(), img[:, -20:][gt[:, -20:] == 0].mean()
    assert right - left > 60


def test_irregular_contours_are_not_plain_ellipses():
    from microtrack.morphometry import compute_morphometry
    from microtrack.synthetic import EXTRA_SCENARIOS

    spec = next(s for s in EXTRA_SCENARIOS if s.name == "irregular")
    img, gt = generate_scene(spec, seed=2)
    assert img.dtype == np.uint8 and set(np.unique(gt)) >= set(range(1, spec.n_objects + 1))
    rough, plain = [], []
    clean = generate_scene(DEFAULT_SCENARIOS[0], seed=2)[1]
    for labels, bucket in ((gt, rough), (clean, plain)):
        for lab in np.unique(labels):
            if lab == 0:
                continue
            bucket.append(compute_morphometry(labels == lab).roughness)
    assert np.mean(rough) > np.mean(plain)


def test_phase_contrast_has_a_bright_rim():
    from microtrack.synthetic import EXTRA_SCENARIOS

    spec = next(s for s in EXTRA_SCENARIOS if s.name == "phase_contrast")
    img, gt = generate_scene(spec, seed=1)
    body = img[gt > 0].mean()
    kernel = np.ones((5, 5), np.uint8)
    import cv2
    ring = cv2.dilate((gt > 0).astype(np.uint8), kernel) - (gt > 0).astype(np.uint8)
    assert img[ring > 0].mean() > body


def test_generate_dataset_counts_and_names():
    items = list(generate_dataset(DEFAULT_SCENARIOS[:2], images_per_scenario=3, seed=1))
    assert len(items) == 6
    assert [n for n, *_ in items] == ["clean"] * 3 + ["noisy"] * 3
    assert not np.array_equal(items[0][2], items[1][2])
