import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_models as C  # noqa: E402

from microtrack.segmentation import pretrained as P  # noqa: E402
from microtrack.synthetic import SceneSpec  # noqa: E402

CIRCLES = SceneSpec("circles", n_objects=6, elongation=(1.0, 1.0), noise=0.0, blur=0.0)


class BoxPainter(P.Segmenter):
    """Рисует закрашенный прямоугольник по каждой рамке — проверка гибридного режима."""
    mode = "boxes"

    def __call__(self, image, prompts=None):
        out = np.zeros(image.shape[:2], np.int32)
        for k, (x0, y0, x1, y1) in enumerate(np.round(prompts.boxes).astype(int), start=1):
            out[y0:y1, x0:x1] = k
        return out


class Broken(P.Segmenter):
    def load(self):
        raise P.ModelUnavailable("нет весов")


class Crashing(P.Segmenter):
    def __call__(self, image, prompts=None):
        raise ValueError("inference failed")


def spec(name, factory, mode="auto", source=None, requires=("numpy",)):
    return P.ModelSpec(name, name, mode, source, factory, requires)


REGISTRY = {
    "boxes_oracle": spec("boxes_oracle", BoxPainter, "boxes", "oracle"),
    "boxes_classical": spec("boxes_classical", BoxPainter, "boxes", "classical"),
    "missing": spec("missing", P.Segmenter, requires=("definitely_not_installed_pkg",)),
    "no_weights": spec("no_weights", Broken),
    "crash": spec("crash", Crashing),
}


def test_run_comparison_statuses_and_rows():
    status, rows = C.run_comparison(list(REGISTRY), [CIRCLES], 2, 0, include_references=True,
                                    registry=REGISTRY)
    st = {s["model"]: s for s in status}
    assert st["boxes_oracle"]["status"] == "ok" and st["ref:otsu"]["status"] == "ok"
    assert st["missing"]["status"] == "skipped" and "definitely_not_installed_pkg" in st["missing"]["reason"]
    assert st["no_weights"]["status"] == "skipped" and "нет весов" in st["no_weights"]["reason"]
    assert st["crash"]["status"] == "error" and "inference failed" in st["crash"]["reason"]
    models = {r["model"] for r in rows}
    assert "crash" not in models and "missing" not in models      # неудачные модели строк не оставляют
    assert {"boxes_oracle", "boxes_classical", "ref:otsu"} <= models


def test_oracle_boxes_detect_everything_but_masks_are_boxy():
    _, rows = C.run_comparison(["boxes_oracle"], [CIRCLES], 2, 0, include_references=False,
                               registry=REGISTRY)
    by, summ = C.aggregate(rows)
    s = summ[0]
    assert s["f1"] == 1.0 and s["count_error"] == 0
    assert 0.6 < s["iou"] < 0.95                                    # прямоугольник ≠ круг
    assert s["prompt_ms"] == 0.0 and s["mode"] == "boxes" and s["prompt_source"] == "oracle"


def test_classical_prompts_cost_time_and_work_without_gt():
    _, rows = C.run_comparison(["boxes_classical"], [CIRCLES], 2, 0, include_references=False,
                               registry=REGISTRY)
    assert all(r["prompt_ms"] > 0 for r in rows)
    assert np.mean([r["f1"] for r in rows]) > 0.9


def test_prompt_jitter_degrades_oracle_boxes():
    kw = dict(scenarios=[CIRCLES], images_per_scenario=2, seed=0, include_references=False, registry=REGISTRY)
    _, clean = C.run_comparison(["boxes_oracle"], prompt_jitter=0.0, **kw)
    _, noisy = C.run_comparison(["boxes_oracle"], prompt_jitter=4.0, **kw)
    assert np.mean([r["iou"] for r in noisy]) < np.mean([r["iou"] for r in clean])


def test_aggregate_robustness_columns():
    _, rows = C.run_comparison([], [CIRCLES, SceneSpec("noisy_c", n_objects=6, elongation=(1.0, 1.0), noise=40.0)],
                               1, 0, include_references=True, registry=REGISTRY)
    _, summ = C.aggregate(rows)
    for s in summ:
        assert s["f1_min"] <= s["f1"] <= 1.0 and s["f1_spread"] >= 0 and s["worst_scenario"]
        assert s["n_scenarios"] == 2


def test_main_with_unavailable_model_writes_reports(tmp_path):
    P.MODEL_REGISTRY["zz_fake_missing"] = spec("zz_fake_missing", P.Segmenter,
                                               requires=("definitely_not_installed_pkg",))
    try:
        rc = C.main(["--models", "zz_fake_missing", "--scenarios", "clean", "--images", "1",
                     "--out", str(tmp_path)])
    finally:
        P.MODEL_REGISTRY.pop("zz_fake_missing")
    assert rc == 0
    md = (tmp_path / "model_comparison.md").read_text(encoding="utf-8")
    assert "пропущена" in md and "definitely_not_installed_pkg" in md and "ref:otsu" in md
    for f in ("model_status.csv", "model_comparison_details.csv",
              "model_comparison_by_scenario.csv", "model_comparison_summary.csv"):
        assert (tmp_path / f).exists()


def test_main_without_any_results_does_not_crash(tmp_path):
    P.MODEL_REGISTRY["zz_fake_missing"] = spec("zz_fake_missing", P.Segmenter,
                                               requires=("definitely_not_installed_pkg",))
    try:
        rc = C.main(["--models", "zz_fake_missing", "--no-references", "--scenarios", "clean",
                     "--images", "1", "--out", str(tmp_path)])
    finally:
        P.MODEL_REGISTRY.pop("zz_fake_missing")
    assert rc == 0 and "Результатов нет" in (tmp_path / "model_comparison.md").read_text(encoding="utf-8")


def test_unknown_model_name_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        C.main(["--models", "no_such_model", "--out", str(tmp_path)])
