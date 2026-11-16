import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark  # noqa: E402

from microtrack.synthetic import DEFAULT_SCENARIOS  # noqa: E402


def test_oracle_algorithm_scores_perfectly_and_empty_scores_zero():
    gts = {}

    def oracle(img):                       # «знает» разметку: запоминаем по кадру
        return gts[img.tobytes()]

    from microtrack.synthetic import generate_dataset
    specs = DEFAULT_SCENARIOS[:2]
    for _, _, img, gt in generate_dataset(specs, 2, 0):
        gts[img.tobytes()] = gt
    empty = lambda img: np.zeros(img.shape, np.int32)

    rows = benchmark.run_benchmark({"oracle": oracle, "empty": empty}, specs, 2, 0)
    assert len(rows) == 2 * 2 * 2
    by, summ = benchmark.aggregate(rows)
    s = {r["algorithm"]: r for r in summ}
    assert s["oracle"]["f1"] == 1.0 and s["oracle"]["iou"] == 1.0 and s["oracle"]["count_error"] == 0
    assert s["empty"]["f1"] == 0.0 and s["empty"]["iou"] == 0.0 and s["empty"]["coverage_pred"] == 0.0
    assert s["oracle"]["time_ms"] >= 0


def test_main_writes_reports(tmp_path):
    rc = benchmark.main(["--out", str(tmp_path), "--images", "1",
                         "--algorithms", "otsu", "adaptive+denoise",
                         "--scenarios", "clean", "touching"])
    assert rc == 0
    for name in ("benchmark_details.csv", "benchmark_by_scenario.csv",
                 "benchmark_summary.csv", "benchmark.md"):
        assert (tmp_path / name).stat().st_size > 0
    md = (tmp_path / "benchmark.md").read_text(encoding="utf-8")
    assert "otsu" in md and "adaptive+denoise" in md and "F1" in md
    lines = (tmp_path / "benchmark_summary.csv").read_text().strip().splitlines()
    assert len(lines) == 3                                 # заголовок + 2 алгоритма


def test_classical_clean_scene_is_nearly_perfect():
    rows = benchmark.run_benchmark({"otsu": benchmark.ALGORITHMS["otsu"]}, DEFAULT_SCENARIOS[:1], 2, 0)
    assert all(r["f1"] == 1.0 and r["iou"] > 0.9 for r in rows)
