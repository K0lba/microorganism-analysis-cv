#!/usr/bin/env python
"""Сравнение baseline-алгоритмов сегментации на синтетической разметке.

Запуск:
    python scripts/benchmark.py                       # все алгоритмы и сценарии
    python scripts/benchmark.py --images 10 --seed 1
    python scripts/benchmark.py --algorithms otsu adaptive --scenarios clean noisy

Результаты (по умолчанию в results/):
    benchmark_details.csv  — метрики по каждому кадру
    benchmark_by_scenario.csv — средние по (алгоритм, сценарий)
    benchmark_summary.csv  — средние по алгоритму (макро-среднее по сценариям)
    benchmark.md           — таблицы для отчёта
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microtrack.preprocess import PreprocessConfig  # noqa: E402
from microtrack.segmentation import SegmentationConfig, segment  # noqa: E402
from microtrack.segmentation.metrics import evaluate  # noqa: E402
from microtrack.synthetic import DEFAULT_SCENARIOS, SceneSpec, generate_dataset  # noqa: E402

# Алгоритм — функция «изображение -> экземплярные метки int32».
Algorithm = Callable[[np.ndarray], np.ndarray]

METRICS = ["iou", "dice", "precision", "recall", "f1", "mean_matched_iou",
           "n_pred", "n_gt", "count_error", "coverage_pred", "coverage_gt", "time_ms"]


def classical(seg: dict, pre: dict | None = None) -> Algorithm:
    seg_cfg = SegmentationConfig.from_dict(seg)
    pre_cfg = PreprocessConfig.from_dict(pre) if pre is not None else None
    return lambda img: segment(img, seg_cfg, preprocess_cfg=pre_cfg).labels


_MIN = {"min_area": 20}
_DENOISE = {"denoise": {"method": "gaussian", "strength": 1.0}}
_ADAPT = {"method": "adaptive", "block_size": 41, "offset": 12.0}
_SPLIT = {"enabled": True, "peak_fraction": 0.8}

ALGORITHMS: Dict[str, Algorithm] = {
    "otsu": classical({"filters": _MIN}),
    "otsu+denoise": classical({"filters": _MIN}, _DENOISE),
    "otsu+clahe+denoise": classical({"filters": _MIN},
                                    {**_DENOISE, "clahe": {"clip_limit": 2.0, "tile_grid": 8}}),
    "adaptive": classical({"threshold": _ADAPT, "filters": _MIN}),
    "adaptive+denoise": classical({"threshold": _ADAPT, "filters": _MIN}, _DENOISE),
    "otsu+denoise+split": classical({"filters": _MIN, "split": _SPLIT}, _DENOISE),
    "adaptive+denoise+split": classical(
        {"threshold": _ADAPT, "filters": _MIN, "split": _SPLIT}, _DENOISE),
}


# --------------------------------------------------------------------------- #
def run_benchmark(
    algorithms: Dict[str, Algorithm],
    scenarios: Sequence[SceneSpec] = DEFAULT_SCENARIOS,
    images_per_scenario: int = 5,
    seed: int = 0,
    iou_threshold: float = 0.5,
    repeats: int = 3,
) -> List[dict]:
    """Возвращает строки метрик: по одной на (алгоритм, сценарий, кадр).

    Время кадра — медиана ``repeats`` запусков (устойчиво к разовым выбросам).
    """
    rows: List[dict] = []
    data = list(generate_dataset(scenarios, images_per_scenario, seed))
    for name, algo in algorithms.items():
        algo(data[0][2])                                  # прогрев (не учитывается во времени)
        for scenario, idx, img, gt in data:
            times = []
            for _ in range(max(1, repeats)):
                t0 = time.perf_counter()
                pred = algo(img)
                times.append(time.perf_counter() - t0)
            m = evaluate(pred, gt, iou_threshold, float(np.median(times)))
            rows.append({"algorithm": name, "scenario": scenario, "image": idx, **m})
    return rows


def _mean(rows: List[dict], keys: Sequence[str]) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def aggregate(rows: List[dict]):
    """(по сценариям, по алгоритмам). Итог по алгоритму — макро-среднее по сценариям."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["algorithm"], r["scenario"])].append(r)
    by_scenario = [
        {"algorithm": a, "scenario": s, "n_images": len(g), **_mean(g, METRICS)}
        for (a, s), g in groups.items()
    ]
    per_algo = defaultdict(list)
    for r in by_scenario:
        per_algo[r["algorithm"]].append(r)
    summary = [
        {"algorithm": a, "n_scenarios": len(g), **_mean(g, METRICS)}
        for a, g in per_algo.items()
    ]
    return by_scenario, summary


# --------------------------------------------------------------------------- #
def _write_csv(path: Path, rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def _md_table(header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _pivot(by_scenario: List[dict], metric: str, algos: Sequence[str], scenarios: Sequence[str]) -> str:
    val = {(r["algorithm"], r["scenario"]): r[metric] for r in by_scenario}
    body = [[a] + [f"{val[(a, s)]:.3f}" for s in scenarios] for a in algos]
    return _md_table(["algorithm"] + list(scenarios), body)


def build_markdown(by_scenario: List[dict], summary: List[dict], args) -> str:
    summary = sorted(summary, key=lambda r: -r["f1"])
    algos = [r["algorithm"] for r in summary]
    scenarios = list(dict.fromkeys(r["scenario"] for r in by_scenario))
    n_img = args.images
    head = ["algorithm", "IoU", "Dice", "Precision", "Recall", "F1", "objects (pred/GT)",
            "coverage (pred/GT)", "time, ms"]
    body = [[r["algorithm"], f"{r['iou']:.3f}", f"{r['dice']:.3f}", f"{r['precision']:.3f}",
             f"{r['recall']:.3f}", f"{r['f1']:.3f}", f"{r['n_pred']:.1f} / {r['n_gt']:.1f}",
             f"{r['coverage_pred']:.3f} / {r['coverage_gt']:.3f}", f"{r['time_ms']:.1f}"]
            for r in summary]
    parts = [
        "# Сравнение baseline-алгоритмов сегментации",
        f"Синтетическая разметка: сценариев — {len(scenarios)}, кадров на сценарий — {n_img}, seed={args.seed}. "
        f"Обнаружение: совпадение при IoU ≥ {args.iou_threshold}. "
        "IoU/Dice — по бинарной маске «все объекты». Итог — макро-среднее по сценариям; "
        "время — медиана на один кадр.",
        "## Итог по алгоритмам (по убыванию F1)",
        _md_table(head, body),
        "## F1 обнаружения по сценариям", _pivot(by_scenario, "f1", algos, scenarios),
        "## IoU масок по сценариям", _pivot(by_scenario, "iou", algos, scenarios),
        "## Ошибка числа объектов (pred − GT) по сценариям",
        "\n".join([_md_table(["algorithm"] + scenarios, [
            [a] + [f"{next(r for r in by_scenario if r['algorithm'] == a and r['scenario'] == s)['count_error']:+.1f}"
                   for s in scenarios] for a in algos])]),
    ]
    return "\n\n".join(parts) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results", help="папка для результатов")
    p.add_argument("--images", type=int, default=5, help="кадров на сценарий")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--repeats", type=int, default=3, help="повторов замера времени на кадр")
    p.add_argument("--algorithms", nargs="+", choices=list(ALGORITHMS), help="по умолчанию все")
    p.add_argument("--scenarios", nargs="+", choices=[s.name for s in DEFAULT_SCENARIOS],
                   help="по умолчанию все")
    args = p.parse_args(argv)

    algos = {k: ALGORITHMS[k] for k in (args.algorithms or ALGORITHMS)}
    scen = [s for s in DEFAULT_SCENARIOS if not args.scenarios or s.name in args.scenarios]
    rows = run_benchmark(algos, scen, args.images, args.seed, args.iou_threshold, args.repeats)
    by_scenario, summary = aggregate(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "benchmark_details.csv", rows)
    _write_csv(out / "benchmark_by_scenario.csv", by_scenario)
    _write_csv(out / "benchmark_summary.csv", sorted(summary, key=lambda r: -r["f1"]))
    md = build_markdown(by_scenario, summary, args)
    (out / "benchmark.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Сохранено в {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
