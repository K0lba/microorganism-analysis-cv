#!/usr/bin/env python
"""Выбор целевого объекта на первом кадре (точка, рамка или ROI) и сохранение начального состояния.

Интерактивно (окно OpenCV):
    python scripts/init_target.py --source data/demo/video.avi --out results/init
Клавиши: p — точка, b — рамка, r — ROI-полигон, Enter — принять (в ROI — замкнуть),
         u — отмена шага, c — выбрать заново, s — сохранить, q/Esc — выход.
Мышь: клик — точка / углы рамки / вершины ROI; в режиме рамки можно протянуть мышью.

Без окна (скрипты, сервер, проверка):
    python scripts/init_target.py --source frame.png --point 120 85 --out results/init
    python scripts/init_target.py --source video.mp4 --frame 0 --box 90 60 150 120
    python scripts/init_target.py --source frame.png --roi 80 50 160 55 150 130 85 125
    python scripts/init_target.py --source frame.png --box 90 60 150 120 --model sam2-boxes

Результат в --out: init_mask.png, init.json (центр, площадь, контур, выбор, метод), init_overlay.png.
Дальше трекинг читает их через microtrack.initialization.load_target(--out).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microtrack.initialization import (  # noqa: E402
    InitConfig, InitializationError, InitSession, Selection, initialize_target, save_target,
)
from microtrack.io import Calibration, iter_frames  # noqa: E402
from microtrack.segmentation.pretrained import ModelUnavailable, Sam2PromptSegmenter  # noqa: E402

WINDOW = "init_target"


def load_config(path: Optional[str]):
    """(InitConfig, Calibration) из YAML-файла проекта; секции initialization и calibration."""
    if not path:
        return InitConfig(), None
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    calib = Calibration.from_dict(data["calibration"]) if "calibration" in data else None
    return InitConfig.from_dict(data.get("initialization")), calib


def load_model(name: str):
    if name == "none":
        return None
    seg = Sam2PromptSegmenter("points" if name == "sam2-points" else "boxes")
    try:
        seg.load()
    except ModelUnavailable as e:
        print(f"Модель недоступна ({e}) — используется классический алгоритм.", file=sys.stderr)
        return None
    return seg


def _key_name(code: int) -> str:
    code &= 0xFF
    if code in (13, 10):
        return "enter"
    if code == 27:
        return "esc"
    return chr(code).lower() if 32 <= code < 127 else ""


def gui_available() -> bool:
    """На Linux без DISPLAY/WAYLAND_DISPLAY Qt-сборка OpenCV аварийно завершает процесс — проверяем заранее."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def run_gui(session: InitSession, out_dir: Path, max_side: int = 1000) -> bool:
    """Окно OpenCV: события мыши и клавиатуры передаются в InitSession. True — результат сохранён."""
    if not gui_available():
        raise RuntimeError("Нет дисплея для окна OpenCV. Задайте выбор параметрами --point / --box / --roi.")
    h, w = session.image.shape[:2]
    scale = min(4.0, max_side / max(h, w))
    state = {"down": None}

    def on_mouse(event, mx, my, flags, param):
        p = (mx / scale, my / scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["down"] = p
        elif event == cv2.EVENT_LBUTTONUP and state["down"] is not None:
            p0, state["down"] = state["down"], None
            if abs(p[0] - p0[0]) + abs(p[1] - p0[1]) < 4 / scale:
                session.click(*p0)
            else:
                session.drag(p0, p)

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    except cv2.error as e:
        raise RuntimeError("Окно OpenCV недоступно (headless-сборка?). "
                           "Задайте выбор параметрами --point / --box / --roi.") from e
    cv2.setMouseCallback(WINDOW, on_mouse)
    saved = False
    while True:
        frame = session.render()
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
        cv2.rectangle(frame, (0, frame.shape[0] - 22), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, session.message[:150], (6, frame.shape[0] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, frame)
        code = cv2.waitKey(30)
        if code == -1:
            continue
        action = session.key(_key_name(code))
        if action == "save":
            try:
                session.save(out_dir)
                saved = True
                session.message = f"saved to {out_dir}"
            except InitializationError as e:
                session.message = f"error: {e}"
        elif action == "quit":
            break
    cv2.destroyAllWindows()
    return saved


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="изображение, видео, TIFF-стек или папка с кадрами")
    ap.add_argument("--frame", type=int, default=0, help="индекс кадра (по умолчанию первый)")
    ap.add_argument("--fps", type=float, help="частота кадров (для времени кадра)")
    ap.add_argument("--pixel-size-um", type=float, help="калибровка, мкм на пиксель")
    ap.add_argument("--config", help="YAML проекта (секции initialization, calibration)")
    ap.add_argument("--method", choices=["auto", "threshold", "grabcut", "model"], help="переопределить method")
    ap.add_argument("--model", choices=["none", "sam2-points", "sam2-boxes"], default="none",
                    help="готовая модель с подсказкой вместо классического алгоритма")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"))
    sel.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    sel.add_argument("--roi", nargs="+", type=float, metavar="X Y", help="вершины полигона: x1 y1 x2 y2 ...")
    ap.add_argument("--out", default="results/init", help="папка результата")
    args = ap.parse_args(argv)

    cfg, calib = load_config(args.config)
    if args.method:
        cfg.method = args.method
    if args.pixel_size_um:
        calib = Calibration(args.pixel_size_um)
    calib = calib or Calibration.from_dict(None)            # без калибровки площадь только в px
    try:
        frame = next(iter_frames(args.source, start=args.frame, fps=args.fps, calibration=calib))
    except StopIteration:
        ap.error(f"в источнике нет кадра с индексом {args.frame}")
    segmenter = load_model(args.model)

    if args.point or args.box or args.roi:
        if args.roi:
            if len(args.roi) < 6 or len(args.roi) % 2:
                ap.error("--roi: нужно чётное число координат, не меньше трёх вершин")
            selection = Selection.from_roi(list(zip(args.roi[::2], args.roi[1::2])))
        else:
            selection = Selection.from_point(*args.point) if args.point else Selection.from_box(*args.box)
        try:
            target = initialize_target(frame.image, selection, cfg, segmenter, frame.calibration,
                                       frame.index, frame.time_s)
        except (InitializationError, ValueError) as e:
            print(f"Не удалось выделить объект: {e}", file=sys.stderr)
            return 1
        paths = save_target(target, args.out, frame.image)
        print(target.summary())
        for w in target.warnings:
            print("предупреждение:", w)
        print("сохранено:", ", ".join(str(p) for p in paths.values()))
        return 0

    session = InitSession(frame.image, cfg, segmenter, frame.calibration, frame.index, frame.time_s)
    try:
        saved = run_gui(session, Path(args.out))
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 2
    if session.result is not None and not saved:
        print("Результат не сохранён (нажмите s перед выходом).", file=sys.stderr)
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
