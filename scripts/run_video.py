#!/usr/bin/env python
"""Видео или последовательность кадров → таблица наблюдений, сводка по трекам, видео с наложением.

Запуск:
    python scripts/run_video.py data/synthetic_demo/videos/seq_000.mp4
    python scripts/run_video.py data/demo/frames --fps 5 --pixel-size-um 0.1 --out results/run1
    python scripts/run_video.py video.avi --model sam2_auto --max-frames 10
    python scripts/run_video.py video.avi --condition medium=PBS --condition temperature_c=37

Источник: видеофайл, папка с кадрами, многостраничный TIFF или одно изображение.
Параметры по умолчанию — из configs/default.yaml; флаги ниже их переопределяют.

Результат в --out: observations.csv (+ .parquet), tracks_summary.csv, overlay.mp4,
preview.png, run.json. Схема таблицы — docs/export_schema.md.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microtrack.io import Calibration  # noqa: E402
from microtrack.pipeline import PipelineConfig, run_pipeline  # noqa: E402


def _parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _conditions(items: Sequence[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--condition ожидает имя=значение, получено: {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = _parse_value(value.strip())
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="видео, папка с кадрами, TIFF-стек или изображение")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="YAML проекта")
    ap.add_argument("--out", help="каталог результатов (по умолчанию results/<имя источника>)")
    ap.add_argument("--model", help="classical или модель: cellpose_cpsam, omnipose_bact_phase, sam2_auto ...")
    ap.add_argument("--fps", type=float, help="частота кадров (для папки кадров обязательно, если важна скорость)")
    ap.add_argument("--pixel-size-um", type=float, help="калибровка, мкм на пиксель")
    ap.add_argument("--stride", type=int, help="обрабатывать каждый N-й кадр")
    ap.add_argument("--max-frames", type=int, help="обработать не больше N кадров")
    ap.add_argument("--experiment-id", help="идентификатор эксперимента")
    ap.add_argument("--sequence-id", help="идентификатор последовательности (по умолчанию имя файла)")
    ap.add_argument("--condition", action="append", default=[], metavar="ИМЯ=ЗНАЧЕНИЕ",
                    help="условие эксперимента, столбец cond_<имя>; можно несколько раз")
    ap.add_argument("--no-video", action="store_true", help="не писать overlay.mp4")
    ap.add_argument("--frames", action="store_true", help="сохранить PNG каждого кадра с наложением")
    ap.add_argument("--include-predicted", action="store_true",
                    help="добавить строки с предсказанным положением на пропусках")
    args = ap.parse_args(argv)

    cfg = PipelineConfig.from_yaml(args.config)
    if args.model:
        cfg.segmentation_method = args.model
    if args.fps:
        cfg.fps = args.fps
    if args.pixel_size_um:
        cfg.calibration = Calibration(args.pixel_size_um)
    if args.stride:
        cfg.frame_stride = args.stride
    if args.max_frames:
        cfg.max_frames = args.max_frames
    if args.experiment_id:
        cfg.experiment_id = args.experiment_id
    if args.condition:
        cfg.conditions = {**cfg.conditions, **_conditions(args.condition)}
    cfg.output = replace(cfg.output,
                         save_visualization=cfg.output.save_visualization and not args.no_video,
                         save_frames=cfg.output.save_frames or args.frames,
                         include_predicted=cfg.output.include_predicted or args.include_predicted)

    source = Path(args.source)
    out = Path(args.out) if args.out else ROOT / cfg.output.directory / source.stem

    def progress(k, frame, seconds):
        print(f"\rкадр {frame}  обработано {k}  {seconds:6.1f} с", end="", flush=True)

    try:
        res = run_pipeline(source, cfg, out, sequence_id=args.sequence_id, progress=progress)
    except Exception as e:
        print()
        print(f"Ошибка: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print()
    print(f"Кадров: {res.n_frames}, треков: {len(res.tracker.tracks)}, "
          f"строк в таблице: {len(res.table)}, время: {res.seconds:.1f} с")
    if res.report.errors:
        print("Проверка таблицы НЕ пройдена:")
        for e in res.report.errors:
            print(f"  ERROR: {e}")
    for w in res.report.warnings:
        print(f"  WARN: {w}")
    for name, path in res.files.items():
        print(f"  {name}: {path}")
    return 0 if res.report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
