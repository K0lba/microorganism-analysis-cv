#!/usr/bin/env python
"""Один кадр → CSV размеров и морфологии и PNG с наложенными масками.

Запуск:
    python scripts/measure_frame.py frame.png
    python scripts/measure_frame.py frame.png --model omnipose_bact_phase
    python scripts/measure_frame.py frame.png --model cellpose_cpsam --min-area 5

Рядом с CSV пишутся test.png (кадр с цветными масками) и masks.png (метки экземпляров).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microtrack.morphometry import morphometry_features  # noqa: E402
from microtrack.pipeline import relabel_min_area  # noqa: E402
from microtrack.segmentation import pretrained as pretrained  # noqa: E402
from microtrack.segmentation.classical import (  # noqa: E402
    FilterConfig, SegmentationConfig, ThresholdConfig, segment,
)

# Модели, которые сами ищут объекты на кадре (без точек и рамок).
AUTO_MODELS = (
    "cellpose_cpsam",
    "cellpose_cyto3",
    "omnipose_bact_phase",
    "stardist_fluo",
    "sam2_auto",
)


def read_image(path: Path) -> np.ndarray:
    """Чтение с путём вне ASCII (cv2.imread на Windows такой путь не открывает)."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise SystemExit(f"не удалось прочитать изображение: {path}")
    return img


def write_image(path: Path, image: np.ndarray) -> None:
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise SystemExit(f"не удалось закодировать изображение: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Цвет каждой метки свой, фон чёрный. BGR uint8."""
    color = np.zeros((*labels.shape, 3), np.uint8)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return color
    hsv = np.zeros((len(ids), 1, 3), np.uint8)
    hsv[:, 0, 0] = (ids.astype(np.uint32) * 47) % 180
    hsv[:, 0, 1] = 200
    hsv[:, 0, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    for i, lab in enumerate(ids):
        color[labels == lab] = bgr[i]
    return color


def render_overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Полупрозрачные маски и белый контур поверх кадра."""
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    color = colorize_labels(labels)
    blended = cv2.addWeighted(bgr, 0.55, color, 0.45, 0)
    vis = bgr.copy()
    mask = labels > 0
    vis[mask] = blended[mask]
    lab = labels.astype(np.int32)
    edge = np.zeros(lab.shape, bool)
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    vis[edge & mask] = (255, 255, 255)
    return vis


def filter_labels(labels: np.ndarray, min_area: float) -> tuple[np.ndarray, int]:
    """Выкинуть мелкие маски и перенумеровать оставшиеся с 1. Возвращает (метки, сколько отброшено)."""
    n_before = int(np.count_nonzero(np.unique(labels)))
    out = relabel_min_area(labels, min_area)
    return out, n_before - int(out.max(initial=0))


def labels_from_model(image: np.ndarray, name: str) -> np.ndarray:
    try:
        seg = pretrained.load_model(name)
    except pretrained.ModelUnavailable as e:
        raise SystemExit(str(e)) from e
    if seg.mode != "auto":
        raise SystemExit(f"{name} ждёт точки или рамки и на весь кадр не запускается")
    return np.asarray(seg(image)).astype(np.int32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="кадр: png, jpg, tif, bmp")
    ap.add_argument("--model", default="classical", choices=["classical", *AUTO_MODELS],
                    help="classical — порог; остальные ищут клетки сами")
    ap.add_argument("--min-area", type=float, default=20, help="минимальная площадь клетки, px")
    ap.add_argument("--objects", choices=["auto", "dark", "bright"], default="auto",
                    help="только для classical: auto — объекты это меньшая часть кадра")
    ap.add_argument("--out", default="results/morphology.csv", help="куда записать CSV")
    args = ap.parse_args(argv)

    img = read_image(Path(args.image))
    if args.model == "classical":
        cfg = SegmentationConfig(
            threshold=ThresholdConfig(method="otsu", objects=args.objects),
            filters=FilterConfig(min_area=args.min_area),
        )
        res = segment(img, cfg)
        labels = res.labels
        note = f"отброшено: {res.rejected or '{}'}"
    else:
        raw = labels_from_model(img, args.model)
        labels, dropped = filter_labels(raw, args.min_area)
        note = f"мельче {args.min_area:g} px отброшено: {dropped}"
    rows = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        row = morphometry_features(labels == lab)
        row["label"] = int(lab)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        front = ["label"]
        df = df[front + [c for c in df.columns if c not in front]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    overlay_path = out.parent / "test.png"
    masks_path = out.parent / "masks.png"
    write_image(overlay_path, render_overlay(img, labels))
    write_image(masks_path, labels.astype(np.uint16))
    print(f"модель: {args.model}")
    print(f"клеток: {len(df)}, {note}")
    print(f"таблица: {out.resolve()}")
    print(f"наложение: {overlay_path.resolve()}")
    print(f"маски: {masks_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
