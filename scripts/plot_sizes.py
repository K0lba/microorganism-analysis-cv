#!/usr/bin/env python
"""Графики размеров во времени: измеренные vs эталонные по каждому треку.

Сопоставляет измеренные размеры (из таблицы конвейера) с эталонными (из objects.csv)
и рисует графики площади, длины и ширины по времени для каждого трека.
Также рисует сводный график распределения ошибок по признакам.

Запуск:
    python scripts/plot_sizes.py --table results/demo/observations.csv --gt data/synthetic_demo/objects.csv
    python scripts/plot_sizes.py --table results/demo/observations.csv --gt data/synthetic_demo/objects.csv --out results/plots

Результаты (папка --out, по умолчанию рядом с таблицей):
    plots/per_track/      PNG для каждого трека (track_XX.png)
    plots/error_dist.png  распределение относительных ошибок по признакам
    plots/overview.png    все треки: площадь по времени
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microtrack.evaluation import FEATURE_MAP, gt_area_perimeter, load_gt, match_tracks  # noqa: E402

PLOT_FEATURES = ["area_px2", "length_px", "width_px"]
TITLES = {"area_px2": "Площадь, px²", "length_px": "Длина, px", "width_px": "Ширина, px"}


def plot_per_track(table: pd.DataFrame, gt: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Для каждого сопоставленного трека — график измеренных и эталонных размеров."""
    gt = gt_area_perimeter(gt)
    merged = table.merge(gt, left_on=["gt_id", "frame"], right_on=["gt_id", "frame"],
                          suffixes=("_pred", "_gt"))
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for tid, g in merged.groupby("track_id"):
        g = g.sort_values("frame")
        gid = int(g["gt_id"].iloc[0])
        fig, axes = plt.subplots(len(PLOT_FEATURES), 1, figsize=(8, 2.5 * len(PLOT_FEATURES)),
                                   sharex=True)
        for ax, feat in zip(axes, PLOT_FEATURES):
            pred_col = feat if feat in g.columns else f"{feat}_pred"
            gt_col = feat if feat in g.columns else f"{feat}_gt"
            ax.plot(g["frame"], g[pred_col], "o-", ms=3, lw=1, label="измерено", color="C0")
            ax.plot(g["frame"], g[gt_col], "s--", ms=3, lw=1, label="эталон", color="C1")
            ax.set_ylabel(TITLES[feat])
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("кадр")
        fig.suptitle(f"Трек {int(tid)} (эталон {gid})", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = out_dir / f"track_{int(tid):02d}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        files.append(p)
    return files


def plot_error_distribution(table: pd.DataFrame, gt: pd.DataFrame, out: Path):
    """Сводный график: распределение относительной ошибки по признакам (boxplot)."""
    gt = gt_area_perimeter(gt)
    merged = table.merge(gt, left_on=["gt_id", "frame"], right_on=["gt_id", "frame"],
                          suffixes=("_pred", "_gt"))
    data, labels = [], []
    for gt_name, pred_name in FEATURE_MAP.items():
        pred_col = pred_name if pred_name in merged.columns else f"{pred_name}_pred"
        gt_col = gt_name if gt_name in merged.columns else f"{gt_name}_gt"
        if pred_col not in merged.columns or gt_col not in merged.columns:
            continue
        p = merged[pred_col].astype(float)
        g = merged[gt_col].astype(float).clip(lower=1e-6)
        rel = ((p - g) / g * 100).dropna()
        data.append(rel.values)
        labels.append(pred_name)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_ylabel("относительная ошибка, %")
    ax.set_title("Распределение ошибок признаков")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_overview(table: pd.DataFrame, out: Path):
    """Все треки: площадь по времени на одном графике."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for tid, g in table.groupby("track_id"):
        g = g.sort_values("frame")
        ax.plot(g["frame"], g["area_px2"], "-", lw=1, alpha=0.7, label=f"трек {int(tid)}")
    ax.set_xlabel("кадр")
    ax.set_ylabel("площадь, px²")
    ax.set_title("Площадь по времени, все треки")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="observations.csv из конвейера")
    ap.add_argument("--gt", required=True, help="objects.csv из синтетики")
    ap.add_argument("--out", help="каталог вывода (по умолчанию рядом с таблицей)")
    args = ap.parse_args(argv)

    table = pd.read_csv(args.table)
    gt = load_gt(Path(args.gt))
    matched = match_tracks(table, gt)

    out = Path(args.out) if args.out else Path(args.table).parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    per_track = plot_per_track(matched, gt, out / "per_track")
    plot_error_distribution(matched, gt, out / "error_dist.png")
    plot_overview(matched, out / "overview.png")
    print(f"Графиков по трекам: {len(per_track)} в {(out / 'per_track').resolve()}")
    print(f"Распределение ошибок: {(out / 'error_dist.png').resolve()}")
    print(f"Сводный график: {(out / 'overview.png').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
