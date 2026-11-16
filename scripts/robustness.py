#!/usr/bin/env python
"""Таблицы устойчивости трекинга: как меняются метрики при разном шаге кадров и шуме.

Генерирует синтетическую последовательность с известными траекториями,
прогоняет конвейер при разных stride (шаг субдискретизации) и sigma (гауссов шум
на кадр) и записывает таблицу метрик трекинга (IDF1, переключения ID, MOTA).

Запуск:
    python scripts/robustness.py
    python scripts/robustness.py --frames 20 --objects 12 --seed 5
    python scripts/robustness.py --strides 1 2 3 --sigmas 0 8 16 --out results/robust

Результаты (папка --out, по умолчанию results/):
    robustness.csv   метрики по (stride, sigma)
    robustness.md   таблицы для отчёта
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microtrack.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from microtrack.tracking_metrics import evaluate_tracking, report_to_dict  # noqa: E402


def load_gt(data_dir: Path) -> List[dict]:
    """Эталонные объекты из objects.csv."""
    rows = []
    with open(data_dir / "objects.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "object_id": int(r["object_id"]),
                "frame": int(r["frame_index"]),
                "x_px": float(r["x_px"]), "y_px": float(r["y_px"]),
            })
    return rows


def add_noise(image: np.ndarray, sigma: float, rng) -> np.ndarray:
    if sigma <= 0:
        return image
    return np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape),
                    0, 255).astype(np.uint8)


def make_noisy_frames(video: Path, out_dir: Path, sigma: float, rng) -> Path:
    """Перегнать видео в зашумлённые PNG (конвейер умеет читать папку кадров)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"frame_{idx:05d}.png"), add_noise(frame, sigma, rng))
        idx += 1
    cap.release()
    return out_dir


def run_robustness(data_dir: Path, strides: Sequence[int], sigmas: Sequence[float],
                   seed: int) -> List[dict]:
    gt = load_gt(data_dir)
    video = next((data_dir / "videos").glob("*.mp4"), None)
    if video is None:
        raise SystemExit(f"В {data_dir} нет видео")
    rng = np.random.default_rng(seed)
    tmp_root = data_dir.parent / "_robust_tmp"
    rows = []
    for stride in strides:
        for sigma in sigmas:
            run_dir = tmp_root / f"s{stride}_n{int(sigma)}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            source = make_noisy_frames(video, run_dir / "frames", sigma, rng) if sigma > 0 else video
            cfg = PipelineConfig.from_dict({
                "input": {"fps": 5.0, "frame_stride": stride},
                "segmentation": {"min_object_area_px": 20},
                "output": {"save_parquet": False, "save_visualization": False, "save_csv": False},
            })
            t0 = time.perf_counter()
            res = run_pipeline(source, cfg, run_dir / "out")
            elapsed = time.perf_counter() - t0

            pred_rows = [{"track_id": int(r["track_id"]), "frame": int(r["frame"]),
                          "x_px": float(r["x_px"]), "y_px": float(r["y_px"])}
                         for _, r in res.table.iterrows()]
            tr = evaluate_tracking(pred_rows, gt, max_distance=40.0)
            m = report_to_dict(tr)
            m.update(stride=stride, noise_sigma=sigma, n_frames=res.n_frames,
                      n_tracks=len(res.tracker.tracks), n_observations=len(res.table),
                      seconds=round(elapsed, 3))
            rows.append(m)
    shutil.rmtree(tmp_root, ignore_errors=True)
    return rows


def build_markdown(rows: List[dict], args) -> str:
    lines = ["# Устойчивость трекинга", "",
             f"Синтетика: {args.frames} кадров, {args.objects} объектов, seed={args.seed}.",
             f"Шаги кадров: {list(args.strides)}, шум sigma: {list(args.sigmas)}.", ""]
    cols = ["stride", "noise_sigma", "idf1", "n_switches", "switch_rate", "mota",
            "id_precision", "id_recall", "n_pred_tracks", "n_gt_tracks",
            "mostly_tracked", "mostly_lost", "fragments", "n_observations", "seconds"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines += ["", "## Что смотреть", "",
              "- **IDF1** ~ 1 — трекинг идеальный; падает при переключениях ID и пропусках.",
              "- **n_switches** — сколько раз эталонный объект сменил предсказанный ID.",
              "- **stride** — каждый N-й кадр; рост stride должен давать больше переключений.",
              "- **noise_sigma** — гауссов шум на кадр; рост шума снижает IDF1."]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic_demo")
    ap.add_argument("--out", default="results")
    ap.add_argument("--strides", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0, 8, 16])
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--objects", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--regen", action="store_true", help="перегенерировать синтетику")
    args = ap.parse_args(argv)

    data = Path(args.data)
    if args.regen or not (data / "objects.csv").exists():
        sys.path.insert(0, str(ROOT / "src"))
        from generate_synthetic import generate_dataset
        generate_dataset(data, sequences=1, frames=args.frames, width=256, height=192,
                         fps=5.0, objects_per_frame=args.objects, seed=args.seed)

    rows = run_robustness(data, args.strides, args.sigmas, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "robustness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "robustness.md").write_text(build_markdown(rows, args), encoding="utf-8")
    print(build_markdown(rows, args))
    print(f"\nСохранено: {(out / 'robustness.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
