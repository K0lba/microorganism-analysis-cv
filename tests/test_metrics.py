import numpy as np
import pytest

from microtrack.segmentation.metrics import (
    dice, evaluate, iou, iou_matrix, mask_coverage, match_objects,
)


def square(shape, y, x, size, value=1, out=None):
    m = np.zeros(shape, np.int32) if out is None else out
    m[y:y + size, x:x + size] = value
    return m


def test_iou_dice_known_overlap():
    a = square((30, 30), 0, 0, 10)
    b = square((30, 30), 0, 5, 10)            # пересечение 5x10 = 50, объединение 150
    assert iou(a, b) == pytest.approx(50 / 150)
    assert dice(a, b) == pytest.approx(2 * 50 / 200)


def test_iou_dice_identical_disjoint_empty():
    a = square((20, 20), 2, 2, 8)
    assert iou(a, a) == 1.0 and dice(a, a) == 1.0
    far = square((20, 20), 12, 12, 6)
    assert iou(a, far) == 0.0 and dice(a, far) == 0.0
    z = np.zeros((20, 20), np.int32)
    assert iou(z, z) == 1.0 and dice(z, z) == 1.0
    assert iou(a, z) == 0.0 and dice(a, z) == 0.0
    with pytest.raises(ValueError):
        iou(a, np.zeros((5, 5)))


def test_iou_ignores_label_values():
    a = square((20, 20), 2, 2, 8, value=7)
    b = square((20, 20), 2, 2, 8, value=1)
    assert iou(a, b) == 1.0


def test_mask_coverage():
    assert mask_coverage(square((20, 20), 0, 0, 10)) == pytest.approx(0.25)
    assert mask_coverage(np.zeros((5, 5))) == 0.0


def test_iou_matrix_values():
    gt = square((40, 40), 0, 0, 10, 1)
    square((40, 40), 20, 20, 10, 2, out=gt)
    pred = square((40, 40), 0, 5, 10, 5)               # перекрывает первый эталон наполовину
    p_ids, g_ids, m = iou_matrix(pred, gt)
    assert p_ids.tolist() == [5] and g_ids.tolist() == [1, 2]
    assert m[0, 0] == pytest.approx(50 / 150) and m[0, 1] == 0.0


def test_detection_perfect():
    gt = square((50, 50), 0, 0, 10, 1)
    square((50, 50), 20, 20, 10, 2, out=gt)
    det = match_objects(gt.copy(), gt)
    assert (det.tp, det.fp, det.fn) == (2, 0, 0)
    assert det.precision == det.recall == det.f1 == 1.0 and det.mean_matched_iou == 1.0


def test_detection_tp_fp_fn():
    gt = square((60, 60), 0, 0, 10, 1)
    square((60, 60), 20, 20, 10, 2, out=gt)
    square((60, 60), 40, 40, 10, 3, out=gt)
    pred = square((60, 60), 0, 1, 10, 1)               # IoU = 90/110 ≈ 0.82 -> TP
    square((60, 60), 20, 20, 10, 2, out=pred)          # идеальный TP
    square((60, 60), 50, 5, 6, 3, out=pred)            # ложный объект
    det = match_objects(pred, gt)                      # третий эталон не найден
    assert (det.tp, det.fp, det.fn) == (2, 1, 1)
    assert det.precision == pytest.approx(2 / 3) and det.recall == pytest.approx(2 / 3)
    assert det.f1 == pytest.approx(2 / 3)
    assert det.mean_matched_iou == pytest.approx((90 / 110 + 1.0) / 2)


def test_iou_threshold_boundary():
    gt = square((30, 30), 0, 0, 10)
    pred = square((30, 30), 0, 5, 10)                  # IoU = 1/3
    assert match_objects(pred, gt, 0.5).tp == 0
    assert match_objects(pred, gt, 0.3).tp == 1


def test_merged_and_split_objects():
    # один предсказанный объект на два эталона: IoU с каждым = 0.5, засчитывается один
    gt = square((40, 60), 5, 5, 10, 1)
    square((40, 60), 5, 15, 10, 2, out=gt)
    merged = np.zeros_like(gt)
    merged[5:15, 5:25] = 1
    det = match_objects(merged, gt)
    assert (det.tp, det.fp, det.fn) == (1, 0, 1)
    # один эталон, разрезанный на две части: засчитывается только лучшее совпадение
    gt1 = square((30, 30), 0, 0, 10, 1)
    halves = np.zeros_like(gt1)
    halves[0:10, 0:7] = 1
    halves[0:10, 7:10] = 2
    det = match_objects(halves, gt1)
    assert (det.tp, det.fp, det.fn) == (1, 1, 0)


def test_empty_cases():
    z = np.zeros((20, 20), np.int32)
    gt = square((20, 20), 2, 2, 5)
    both = match_objects(z, z)
    assert both.f1 == 1.0 and both.tp == 0
    none_found = match_objects(z, gt)
    assert (none_found.tp, none_found.fn) == (0, 1) and none_found.f1 == 0.0
    only_fp = match_objects(gt, z)
    assert (only_fp.fp, only_fp.recall) == (1, 0.0) and only_fp.f1 == 0.0


def test_evaluate_summary_keys():
    gt = square((40, 40), 0, 0, 10, 1)
    square((40, 40), 20, 20, 10, 2, out=gt)
    pred = square((40, 40), 0, 0, 10, 1)
    out = evaluate(pred, gt, elapsed_s=0.012)
    assert out["n_pred"] == 1 and out["n_gt"] == 2 and out["count_error"] == -1
    assert out["coverage_pred"] == pytest.approx(100 / 1600)
    assert out["coverage_gt"] == pytest.approx(200 / 1600)
    assert out["time_ms"] == pytest.approx(12.0)
    assert out["iou"] == pytest.approx(0.5) and out["dice"] == pytest.approx(2 * 100 / 300)
    assert out["recall"] == 0.5 and out["precision"] == 1.0
