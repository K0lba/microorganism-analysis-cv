"""Синтетические сцены с известной эталонной разметкой (экземплярные метки).

По умолчанию объекты — эллипсы (кокки и палочки): этот путь оставлен побитово
прежним, на нём держатся регрессионные тесты. При ``irregularity > 0`` контур
получает радиальные гармоники (вмятины и асимметрия), при ``texture`` и
``phase_halo`` — внутреннее зерно и светлый ободок, как у фазового контраста.
Метка 0 — фон, 1..N — объекты.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class SceneSpec:
    name: str
    size: Tuple[int, int] = (256, 256)          # (высота, ширина)
    n_objects: int = 12                          # всего, включая члены слипшихся пар
    semi_axis: Tuple[float, float] = (7.0, 13.0)  # большая полуось a
    elongation: Tuple[float, float] = (1.0, 2.5)  # a / b
    touching_pairs: int = 0                      # пар слипшихся кругов (входят в n_objects)
    dark: bool = True                            # тёмные объекты на светлом фоне
    background: float = 200.0
    contrast: float = 110.0                      # |фон - объект|
    noise: float = 4.0                           # sigma гауссова шума
    blur: float = 1.0                            # sigma размытия оптикой
    gradient: float = 0.0                        # амплитуда линейного перепада освещения
    margin: int = 4                              # минимальный отступ от края
    irregularity: float = 0.0                    # амплитуда радиальных гармоник контура
    texture: float = 0.0                         # σ внутреннего зерна объекта
    phase_halo: float = 0.0                      # яркость светлого ободка, доли контраста


DEFAULT_SCENARIOS: Tuple[SceneSpec, ...] = (
    SceneSpec("clean"),
    SceneSpec("noisy", noise=22.0, blur=0.8),
    SceneSpec("uneven_light", gradient=55.0),
    SceneSpec("touching", n_objects=14, touching_pairs=5, elongation=(1.0, 1.4)),
    SceneSpec("bright_on_dark", dark=False, background=45.0, contrast=120.0),
    SceneSpec("low_contrast", contrast=40.0, noise=8.0, blur=1.5),
)


# Дополнительные типы искусственных изображений: масштаб объектов, форма, плотность.
EXTRA_SCENARIOS: Tuple[SceneSpec, ...] = (
    SceneSpec("small_objects", n_objects=40, semi_axis=(3.0, 5.0), elongation=(1.0, 1.5)),
    SceneSpec("large_objects", n_objects=4, semi_axis=(20.0, 30.0), elongation=(1.0, 1.5)),
    SceneSpec("rods", n_objects=8, semi_axis=(12.0, 20.0), elongation=(3.0, 6.0)),
    SceneSpec("dense", n_objects=36, semi_axis=(5.0, 8.0), elongation=(1.0, 1.6)),
    SceneSpec("irregular", n_objects=10, semi_axis=(8.0, 14.0), elongation=(1.05, 2.2),
              irregularity=0.14, texture=6.0),
    SceneSpec("phase_contrast", n_objects=8, semi_axis=(8.0, 13.0), elongation=(1.0, 1.8),
              irregularity=0.08, texture=5.0, phase_halo=0.45, blur=1.1, contrast=70.0),
)

# Все типы, на которых сравниваются готовые модели.
MODEL_SCENARIOS: Tuple[SceneSpec, ...] = DEFAULT_SCENARIOS + EXTRA_SCENARIOS


def _norm_dist(shape, cx, cy, a, b, angle_deg):
    """Нормированное расстояние до центра эллипса: <= 1 внутри."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    t = np.deg2rad(angle_deg)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    return (u / a) ** 2 + (v / b) ** 2


def _place_objects(spec: SceneSpec, rng: np.random.Generator) -> List[tuple]:
    h, w = spec.size
    objs: List[tuple] = []          # (cx, cy, a, b, angle)

    def free(cx, cy, r, ignore=()):
        if cx - r < spec.margin or cy - r < spec.margin or \
           cx + r > w - spec.margin or cy + r > h - spec.margin:
            return False
        for i, (ox, oy, oa, _, _) in enumerate(objs):
            if i in ignore:
                continue
            if np.hypot(cx - ox, cy - oy) < (r + oa) * 1.15 + 2:
                return False
        return True

    def attempt(fn, tries=3000):
        for _ in range(tries):
            if fn():
                return
        raise RuntimeError(f"Не удалось разместить объекты для сцены '{spec.name}'")

    for _ in range(spec.touching_pairs):
        def pair():
            r1 = rng.uniform(*spec.semi_axis)
            r2 = r1 * rng.uniform(0.85, 1.15)
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            th = rng.uniform(0, 2 * np.pi)
            d = (r1 + r2) * 0.85                       # заметное перекрытие
            c2 = (cx + d * np.cos(th), cy + d * np.sin(th))
            if free(cx, cy, r1) and free(*c2, r2):
                objs.append((cx, cy, r1, r1, 0.0))
                objs.append((c2[0], c2[1], r2, r2, 0.0))
                return True
            return False
        attempt(pair)

    while len(objs) < spec.n_objects:
        def single():
            a = rng.uniform(*spec.semi_axis)
            b = max(2.5, a / rng.uniform(*spec.elongation))
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            if free(cx, cy, a):
                objs.append((cx, cy, a, b, float(rng.uniform(0, 180))))
                return True
            return False
        attempt(single)
    return objs


def _deformed_contour(cx, cy, a, b, ang, rng: np.random.Generator,
                      irregularity: float, n: int = 96) -> np.ndarray:
    """Контур эллипса с радиальными гармониками. Звёздный относительно центра."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = (a * b) / np.hypot(b * np.cos(theta), a * np.sin(theta))
    wobble = np.zeros(n)
    for k in range(2, 7):
        amp = irregularity / k * float(rng.uniform(0.45, 1.0))
        wobble += amp * np.cos(k * theta + float(rng.uniform(0.0, 2.0 * np.pi)))
    radius = np.maximum(radius * (1.0 + wobble), 1.5)
    t = np.deg2rad(ang)
    x = cx + radius * (np.cos(theta) * np.cos(t) - np.sin(theta) * np.sin(t))
    y = cy + radius * (np.cos(theta) * np.sin(t) + np.sin(theta) * np.cos(t))
    return np.stack([np.rint(x), np.rint(y)], axis=1).astype(np.int32)


def _paint_deformed(shape, objs, rng, irregularity) -> np.ndarray:
    """Метки деформированных контуров; перекрытие отдаётся ближайшему центру."""
    h, w = shape
    labels = np.zeros((h, w), np.int32)
    best = np.full((h, w), np.inf, np.float32)
    stamp = np.zeros((h, w), np.uint8)
    for i, (cx, cy, a, b, ang) in enumerate(objs, start=1):
        pts = _deformed_contour(cx, cy, a, b, ang, rng, irregularity)
        stamp[:] = 0
        cv2.fillPoly(stamp, [pts], 1)
        ys, xs = np.nonzero(stamp)
        if len(xs) == 0:
            continue
        dist = np.hypot(xs - cx, ys - cy).astype(np.float32)
        closer = dist < best[ys, xs]
        labels[ys[closer], xs[closer]] = i
        best[ys[closer], xs[closer]] = dist[closer]
    return labels


def generate_scene(spec: SceneSpec, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Возвращает (изображение uint8, метки int32)."""
    rng = np.random.default_rng(seed)
    h, w = spec.size
    objs = _place_objects(spec, rng)

    if spec.irregularity > 0:
        labels = _paint_deformed((h, w), objs, rng, spec.irregularity)
    else:
        labels = np.zeros((h, w), np.int32)
        best = np.full((h, w), np.inf, np.float32)
        for i, (cx, cy, a, b, ang) in enumerate(objs, start=1):
            d = _norm_dist((h, w), cx, cy, a, b, ang)
            take = (d <= 1.0) & (d < best)           # перекрытие отдаём ближайшему центру
            labels[take] = i
            best[take] = d[take]

    fg = spec.background - spec.contrast if spec.dark else spec.background + spec.contrast
    img = np.where(labels > 0, fg, spec.background).astype(np.float32)
    if spec.texture > 0 and np.any(labels):
        grain = cv2.GaussianBlur(rng.normal(0.0, spec.texture, img.shape).astype(np.float32),
                                 (0, 0), 1.2)
        img[labels > 0] += grain[labels > 0]
    if spec.phase_halo > 0 and np.any(labels):
        body = (labels > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ring = cv2.dilate(body, kernel) - body
        img[ring > 0] += spec.phase_halo * spec.contrast
    if spec.blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), spec.blur)
    if spec.gradient:
        img += np.linspace(-spec.gradient, spec.gradient, w, dtype=np.float32)[None, :]
    if spec.noise > 0:
        img += rng.normal(0, spec.noise, img.shape).astype(np.float32)
    return np.clip(np.round(img), 0, 255).astype(np.uint8), labels


def generate_dataset(
    specs=DEFAULT_SCENARIOS, images_per_scenario: int = 5, seed: int = 0,
) -> Iterator[Tuple[str, int, np.ndarray, np.ndarray]]:
    """Итератор (сценарий, номер, изображение, метки); детерминирован по seed."""
    for si, spec in enumerate(specs):
        for i in range(images_per_scenario):
            s = int(np.random.SeedSequence([seed, si, i]).generate_state(1)[0])
            img, gt = generate_scene(spec, s)
            yield spec.name, i, img, gt
