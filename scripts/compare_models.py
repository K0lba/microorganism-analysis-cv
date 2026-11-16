#!/usr/bin/env python
"""Сравнение готовых моделей сегментации на синтетической разметке.

Для каждой модели из реестра (microtrack.segmentation.pretrained.MODEL_REGISTRY):
  * пробует загрузить её; если нет пакетов/весов/сети — записывает причину и идёт дальше;
  * гоняет по всем типам синтетических изображений (10 сценариев);
  * считает IoU, Dice, precision/recall/F1 обнаружения, число объектов, долю кадра
    под маской, время кадра, пик RSS (загрузка и вывод) и пик памяти GPU.
Режимы сравниваются раздельно: auto (модель сама ищет объекты) и по подсказкам
(точки/рамки) — из эталона (oracle, верхняя граница) и из classical-детектора.
Classical-алгоритмы добавляются как референс.

Запуск:
    python scripts/compare_models.py --list
    python scripts/compare_models.py                       # все модели, 3 кадра на сценарий
    python scripts/compare_models.py --models sam2_auto sam2_boxes_classical cellpose_cpsam
    python scripts/compare_models.py --images 5 --prompt-jitter 2.0

Результаты (папка --out, по умолчанию results/):
    model_comparison.md, model_comparison_summary.csv, model_comparison_by_scenario.csv,
    model_comparison_details.csv, model_status.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark as B  # noqa: E402
from microtrack.profiling import ResourceMonitor  # noqa: E402
from microtrack.segmentation import pretrained as P  # noqa: E402
from microtrack.segmentation.metrics import evaluate  # noqa: E402
from microtrack.synthetic import MODEL_SCENARIOS, SceneSpec, generate_dataset  # noqa: E402

METRICS = ["iou", "dice", "precision", "recall", "f1", "n_pred", "n_gt", "count_error",
           "coverage_pred", "coverage_gt", "time_ms", "prompt_ms", "infer_rss_mb", "gpu_peak_mb"]
REF_PROMPT_ALGO = "adaptive+denoise+split"        # classical-детектор для подсказок
REF_ALGOS = ["otsu", "adaptive+denoise+split"]
_MODE_ORDER = {"auto": 0, "points": 1, "boxes": 2}


# --------------------------------------------------------------------------- #
def _free() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_comparison(
    model_names: Sequence[str],
    scenarios: Sequence[SceneSpec] = MODEL_SCENARIOS,
    images_per_scenario: int = 3,
    seed: int = 0,
    iou_threshold: float = 0.5,
    prompt_jitter: float = 0.0,
    repeats: int = 1,
    include_references: bool = True,
    registry: Optional[Dict[str, P.ModelSpec]] = None,
):
    """Возвращает (статусы моделей, строки метрик по кадрам)."""
    registry = registry or P.MODEL_REGISTRY
    data = list(generate_dataset(scenarios, images_per_scenario, seed))
    rng = np.random.default_rng(seed)

    # classical-детектор: и референс, и источник подсказок (считаем один раз на кадр)
    detector = B.ALGORITHMS[REF_PROMPT_ALGO]
    cache = []
    for _, _, img, _ in data:
        t0 = time.perf_counter()
        lab = detector(img)
        cache.append((lab, (time.perf_counter() - t0) * 1000.0))

    status: List[dict] = []
    rows: List[dict] = []

    def evaluate_runner(name, family, mode, source, fn: Callable) -> List[dict]:
        out = []
        for k, (scenario, idx, img, gt) in enumerate(data):
            prompts, prompt_ms = None, 0.0
            if source == "oracle":
                prompts = P.prompts_from_labels(gt, prompt_jitter, prompt_jitter, rng)
            elif source == "classical":
                prompts = P.prompts_from_labels(cache[k][0])
                prompt_ms = cache[k][1]
            times = []
            for _ in range(max(1, repeats)):
                with ResourceMonitor() as mon:
                    pred = fn(img, prompts)
                times.append(mon.wall_s)
            m = evaluate(pred, gt, iou_threshold, float(np.median(times)))
            m.update(prompt_ms=prompt_ms, infer_rss_mb=mon.rss_delta_mb,
                     gpu_peak_mb=float("nan") if mon.gpu_peak_mb is None else mon.gpu_peak_mb)
            out.append({"model": name, "family": family, "mode": mode,
                        "prompt_source": source or "—", "scenario": scenario, "image": idx, **m})
        return out

    if include_references:
        for ref in REF_ALGOS:
            algo = B.ALGORITHMS[ref]
            rows += evaluate_runner(f"ref:{ref}", "classical", "auto", None, lambda i, p, a=algo: a(i))
            status.append({"model": f"ref:{ref}", "status": "ok", "reason": "", "load_s": 0.0,
                           "load_rss_mb": 0.0, "load_gpu_mb": float("nan")})

    for name in model_names:
        spec = registry[name]
        try:
            with ResourceMonitor() as mon:
                seg = P.load_model(spec)
        except P.ModelUnavailable as e:
            status.append({"model": name, "status": "skipped", "reason": str(e), "load_s": 0.0,
                           "load_rss_mb": 0.0, "load_gpu_mb": float("nan")})
            continue
        except Exception as e:                       # неожиданный сбой при загрузке
            status.append({"model": name, "status": "error", "reason": f"load: {type(e).__name__}: {e}",
                           "load_s": 0.0, "load_rss_mb": 0.0, "load_gpu_mb": float("nan")})
            continue
        try:
            rows += evaluate_runner(name, spec.family, spec.mode, spec.prompt_source, seg)
            status.append({"model": name, "status": "ok", "reason": "", "load_s": mon.wall_s,
                           "load_rss_mb": mon.rss_delta_mb,
                           "load_gpu_mb": float("nan") if mon.gpu_peak_mb is None else mon.gpu_peak_mb})
        except Exception as e:
            rows = [r for r in rows if r["model"] != name]
            status.append({"model": name, "status": "error", "reason": f"inference: {type(e).__name__}: {e}",
                           "load_s": mon.wall_s, "load_rss_mb": mon.rss_delta_mb, "load_gpu_mb": float("nan")})
        finally:
            del seg
            _free()
    return status, rows


# --------------------------------------------------------------------------- #
def _mean(rows, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if not (isinstance(r[k], float) and math.isnan(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def aggregate(rows: List[dict]):
    """(по (модель, сценарий), по модели: макро-среднее + робастность по типам изображений)."""
    g = defaultdict(list)
    meta = {}
    for r in rows:
        g[(r["model"], r["scenario"])].append(r)
        meta[r["model"]] = (r["family"], r["mode"], r["prompt_source"])
    by_scenario = [{"model": m, "scenario": s, "n_images": len(v), **_mean(v, METRICS)}
                   for (m, s), v in g.items()]
    per_model = defaultdict(list)
    for r in by_scenario:
        per_model[r["model"]].append(r)
    summary = []
    for m, v in per_model.items():
        f1s = [x["f1"] for x in v]
        worst = min(v, key=lambda x: x["f1"])
        summary.append({"model": m, "family": meta[m][0], "mode": meta[m][1],
                        "prompt_source": meta[m][2], "n_scenarios": len(v), **_mean(v, METRICS),
                        "f1_min": float(min(f1s)), "f1_spread": float(max(f1s) - min(f1s)),
                        "f1_std": float(np.std(f1s)), "worst_scenario": worst["scenario"]})
    return by_scenario, summary


def _f(v, nd=3, signed=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def _md(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in body])


def build_markdown(status, by_scenario, summary, args, scenario_names) -> str:
    summ = sorted(summary, key=lambda r: -r["f1"])
    names = [r["model"] for r in summ]
    st_body = [[s["model"], {"ok": "запущена", "skipped": "пропущена", "error": "ошибка"}[s["status"]],
                s["reason"].replace("|", "/") or "—",
                _f(s["load_s"], 1) if s["status"] == "ok" else "—",
                _f(s["load_rss_mb"], 0) if s["status"] == "ok" else "—"] for s in status]
    p = [
        "# Сравнение готовых моделей сегментации",
        f"Синтетические данные: типов изображений — {len(scenario_names)} ({', '.join(scenario_names)}), "
        f"кадров на тип — {args.images}, seed={args.seed}. Совпадение объекта при IoU ≥ {args.iou_threshold}. "
        f"IoU/Dice — по бинарной маске «все объекты». Итог — макро-среднее по типам. "
        f"Подсказки oracle: jitter={args.prompt_jitter} px.",
        "## 1. Что удалось запустить",
        _md(["модель", "статус", "причина", "загрузка, с", "память при загрузке, МиБ"], st_body),
    ]
    if summ:
        body = [[r["model"], r["mode"], r["prompt_source"], _f(r["iou"]), _f(r["dice"]),
                 _f(r["precision"]), _f(r["recall"]), _f(r["f1"]),
                 f"{r['n_pred']:.1f} / {r['n_gt']:.1f}", _f(r["time_ms"], 1),
                 _f(r["prompt_ms"], 1), _f(r["infer_rss_mb"], 0), _f(r["gpu_peak_mb"], 0)] for r in summ]
        p += ["## 2. Итог по моделям (по убыванию F1)",
              _md(["модель", "режим", "подсказки", "IoU", "Dice", "Precision", "Recall", "F1",
                   "объекты (pred/GT)", "время модели, мс", "время подсказок, мс",
                   "память вывода, МиБ", "GPU, МиБ"], body),
              "Время подсказок — работа classical-детектора, она добавляется к времени модели в гибридном "
              "режиме. Память вывода — прирост пика RSS процесса при обработке кадра."]

        fam = sorted(summ, key=lambda r: (r["family"], _MODE_ORDER.get(r["mode"], 9), r["prompt_source"]))
        fam = [r for r in fam if sum(1 for x in summ if x["family"] == r["family"]) > 1]
        if fam:
            body = [[r["family"], r["model"], r["mode"], r["prompt_source"], _f(r["iou"]), _f(r["dice"]),
                     _f(r["f1"]), _f(r["count_error"], 1, True),
                     _f(r["time_ms"] + r["prompt_ms"], 1)] for r in fam]
            p += ["## 3. Режимы: автоматический и по точкам/рамкам",
                  _md(["семейство", "модель", "режим", "подсказки", "IoU", "Dice", "F1",
                       "ошибка числа объектов", "время всего, мс"], body),
                  "oracle — подсказки из эталона (верхняя граница, недостижимая без разметки); "
                  "classical — подсказки из classical-детектора, реально применимый гибрид."]

        def pivot(metric, signed=False):
            val = {(r["model"], r["scenario"]): r[metric] for r in by_scenario}
            return _md(["модель"] + list(scenario_names),
                       [[n] + [_f(val.get((n, s), float("nan")), 3 if not signed else 1, signed)
                               for s in scenario_names] for n in names])
        rob = [[r["model"], _f(r["f1"]), _f(r["f1_min"]), r["worst_scenario"], _f(r["f1_spread"]),
                _f(r["f1_std"])] for r in summ]
        p += ["## 4. Зависимость от типа изображения",
              "### F1 обнаружения", pivot("f1"), "### IoU масок", pivot("iou"),
              "### Ошибка числа объектов (pred − GT)", pivot("count_error", True),
              "### Робастность по типам",
              _md(["модель", "F1 средний", "F1 худший тип", "какой тип", "размах F1 (max−min)", "σ F1"], rob)]
    else:
        p += ["## 2–4. Результатов нет", "Ни одна модель не была запущена."]
    return "\n\n".join(p) + "\n"


def _csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["all"], help="имена из реестра или 'all'")
    ap.add_argument("--list", action="store_true", help="показать реестр и доступность, выйти")
    ap.add_argument("--no-references", action="store_true", help="не добавлять classical-референсы")
    ap.add_argument("--scenarios", nargs="+", choices=[s.name for s in MODEL_SCENARIOS])
    ap.add_argument("--images", type=int, default=3, help="кадров на сценарий")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--prompt-jitter", type=float, default=0.0, help="шум oracle-подсказок, px")
    ap.add_argument("--repeats", type=int, default=1, help="повторов замера времени на кадр")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    if args.list:
        for n, s in P.MODEL_REGISTRY.items():
            miss = P.missing_requirements(s)
            print(f"{n:26s} {s.mode:7s} {s.prompt_source or '-':10s} "
                  f"{'готово к загрузке' if not miss else 'нет: ' + ', '.join(miss)}  — {s.description}")
        return 0

    names = list(P.MODEL_REGISTRY) if args.models == ["all"] else args.models
    unknown = [n for n in names if n not in P.MODEL_REGISTRY]
    if unknown:
        ap.error(f"неизвестные модели: {unknown}; доступные: {list(P.MODEL_REGISTRY)}")
    scen = [s for s in MODEL_SCENARIOS if not args.scenarios or s.name in args.scenarios]

    status, rows = run_comparison(names, scen, args.images, args.seed, args.iou_threshold,
                                  args.prompt_jitter, args.repeats, not args.no_references)
    by_scenario, summary = aggregate(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / "model_status.csv", status)
    _csv(out / "model_comparison_details.csv", rows)
    _csv(out / "model_comparison_by_scenario.csv", by_scenario)
    _csv(out / "model_comparison_summary.csv", sorted(summary, key=lambda r: -r["f1"]))
    md = build_markdown(status, by_scenario, summary, args, [s.name for s in scen])
    (out / "model_comparison.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Сохранено в {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
