"""Морфометрия: геометрические признаки бинарной маски одного объекта.

Все значения в пикселях (длины в px, площади в px^2). Перевод в микрометры
делается отдельно (см. ``temporal.Calibration`` и ``scale_features``).

Определения
-----------
area                число пикселей маски (отверстия не входят);
perimeter           длина внешнего контура: дуга контура по центрам граничных
                    пикселей, сглаженного гауссом (sigma=1.5 px, убирает
                    «ступеньки», из-за которых сырой arcLength завышает периметр
                    на 5–6%), плюс pi — поправка на полпикселя до края объекта;
bbox                (x, y, w, h) осевой ограничивающий прямоугольник;
length, width       стороны минимального по площади повёрнутого прямоугольника,
                    описанного вокруг объекта, +1 px на размер пикселя
                    (length >= width);
aspect_ratio        length / width (>= 1);
major_axis,         оси эллипса с теми же вторыми центральными моментами,
minor_axis          что и маска; orientation_deg — угол большой оси;
equivalent_diameter диаметр круга той же площади;
circularity         4*pi*area / perimeter^2 (1 — идеальный круг), обрезана до 1;
roundness           4*area / (pi * major_axis^2) (1 — круг), обрезана до 1;
convex_area         площадь выпуклой оболочки: оболочка центров граничных
                    пикселей, расширенная на полпикселя (формула Штейнера
                    S + P/2 + pi/4);
solidity            area / convex_area (1 — выпуклая фигура);
radial_*            статистика расстояний от центроида до точек контура
                    (с поправкой +0.5 px на половину граничного пикселя);
radial_profile      расстояние от центроида до контура по лучам с равным шагом
                    по углу (имеет смысл для объектов, «звёздных» относительно
                    центроида);
extent              площадь / (длина × ширина) минимального прямоугольника;
convexity           периметр выпуклой оболочки / периметр (1 — выпуклый контур);
radial_skew,        асимметрия и эксцесс углового профиля r(θ);
radial_kurtosis
radial_entropy      нормированная энтропия распределения r(θ), 0..1;
roughness           среднее |r − сглаженный r| / средний радиус;
bending_energy      средняя κ², умноженная на квадрат эквивалентного радиуса
                    (у круга ≈ 1);
fourier_2..4        |F_n| / |F_0| радиального профиля: 2 — вытянутость,
                    3–4 — асимметрия и более мелкие лепестки;
center_shift_*      расстояние между центроидом и центром описанной окружности
                    или повёрнутого прямоугольника, делённое на радиус окружности;
primitive_shift     расстояние между этими двумя центрами, в тех же единицах;
hu_1..hu_7          логарифмированные моменты Ху (инвариантны к сдвигу,
                    масштабу и повороту).

Если в маске несколько связных компонент, берётся самая большая.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Поля, которые масштабируются при переводе px -> µm
LINEAR_FIELDS = frozenset({
    "perimeter", "length", "width", "major", "minor", "major_axis", "minor_axis",
    "equivalent_diameter", "convex_perimeter", "radial_mean", "radial_std",
    "radial_min", "radial_max", "bbox_w", "bbox_h",
})
AREA_FIELDS = frozenset({"area", "convex_area"})


@dataclass
class Morphometry:
    area: int
    perimeter: float
    centroid_x: float
    centroid_y: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    length: float
    width: float
    aspect_ratio: float
    major_axis: float
    minor_axis: float
    orientation_deg: float
    eccentricity: float
    equivalent_diameter: float
    circularity: float
    roundness: float
    convex_area: float
    convex_perimeter: float
    solidity: float
    radial_mean: float
    radial_std: float
    radial_min: float
    radial_max: float
    radial_cv: float
    radial_max_min_ratio: float
    radial_min_max_ratio: float
    extent: float
    convexity: float
    radial_skew: float
    radial_kurtosis: float
    radial_entropy: float
    roughness: float
    bending_energy: float
    fourier_2: float
    fourier_3: float
    fourier_4: float
    center_shift_circ: float
    center_shift_rect: float
    primitive_shift: float
    hu_1: float
    hu_2: float
    hu_3: float
    hu_4: float
    hu_5: float
    hu_6: float
    hu_7: float
    radial_profile: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    hull: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2)))

    def to_dict(self, include_profile: bool = False, include_hull: bool = False) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("radial_profile", "hull")}
        if include_profile:
            for i, r in enumerate(self.radial_profile):
                d[f"radial_profile_{i:02d}"] = float(r)
        if include_hull:
            d["hull"] = self.hull.tolist()
        return d


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------

def _largest_component(mask: np.ndarray) -> np.ndarray:
    m8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 2:
        return m8
    lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == lab).astype(np.uint8)


def _smoothed_arclength(pts: np.ndarray, sigma: float = 1.5) -> float:
    """Длина замкнутой кривой после периодического гауссова сглаживания."""
    n = len(pts)
    rad = int(math.ceil(3 * sigma))
    if n < 2 * rad + 2:  # слишком короткий контур — сглаживать нечего
        return float(cv2.arcLength(pts.astype(np.float32).reshape(-1, 1, 2), True))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    k /= k.sum()
    sm = np.empty_like(pts, dtype=float)
    for j in range(2):
        padded = np.concatenate([pts[-rad:, j], pts[:, j], pts[:rad, j]])
        sm[:, j] = np.convolve(padded, k, mode="valid")
    seg = np.diff(np.vstack([sm, sm[:1]]), axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def _ray_radii(poly: np.ndarray, center: Tuple[float, float], n: int) -> np.ndarray:
    """Расстояние от center до самого дальнего пересечения луча с полигоном."""
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    d = np.stack([np.cos(ang), np.sin(ang)], axis=1)            # (n, 2)
    p = poly - np.asarray(center)                                # вершины от центра
    q = np.roll(p, -1, axis=0)
    e = q - p                                                     # рёбра
    # Решаем  t*d = p + u*e  для каждой пары (луч, ребро)
    cross_de = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (p[None, :, 0] * e[None, :, 1] - p[None, :, 1] * e[None, :, 0]) / cross_de
        u = (p[None, :, 0] * d[:, None, 1] - p[None, :, 1] * d[:, None, 0]) / cross_de
    ok = (np.abs(cross_de) > 1e-12) & (u >= 0) & (u <= 1) & (t > 0)
    t = np.where(ok, t, -np.inf)
    r = t.max(axis=1)
    # Лучи, не нашедшие пересечения (центроид вне объекта) — интерполируем
    bad = ~np.isfinite(r)
    if bad.all():
        return np.zeros(n)
    if bad.any():
        idx = np.arange(n)
        r[bad] = np.interp(idx[bad], idx[~bad], r[~bad], period=n)
    return r


def _periodic_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    n = len(values)
    rad = int(math.ceil(3 * sigma))
    if n < 2 * rad + 2:
        return values.astype(float)
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.concatenate([values[-rad:], values, values[:rad]])
    return np.convolve(padded, k, mode="valid")


def _resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    closed = np.vstack([pts, pts[:1]])
    step = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(step)])
    if s[-1] < 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(target, s, closed[:, j]) for j in range(2)], axis=1)


def _bending_energy(pts: np.ndarray, radius: float) -> float:
    """Безразмерная энергия изгиба: mean(κ²) * R². У круга близка к 1."""
    sm = np.stack([_periodic_smooth(pts[:, j], 4.0) for j in range(2)], axis=1)
    q = _resample_closed(sm, 180)
    closed = np.vstack([q, q[:1]])
    per = float(np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1])).sum())
    if per < 1e-6 or radius <= 0:
        return 0.0
    x, y = q[:, 0], q[:, 1]
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    ds = np.maximum(np.hypot(dx, dy), 1e-6)
    phi = np.arctan2(dy, dx)
    turn = np.angle(np.exp(1j * (np.roll(phi, -1) - phi)))
    kappa = turn / np.maximum(0.5 * (ds + np.roll(ds, -1)), 1e-6)
    return float(np.mean(kappa ** 2) * radius ** 2)


def _radial_shape(profile: np.ndarray) -> dict:
    """Признаки углового профиля r(θ): моменты, энтропия, шероховатость, Фурье."""
    r = np.asarray(profile, dtype=float)
    mu = float(r.mean()) if len(r) else 0.0
    sd = float(r.std()) if len(r) else 0.0
    if mu <= 0 or sd / mu < 1e-3:
        skew = kurt = 0.0
    else:
        z = (r - mu) / sd
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)
    # Фиксированные корзины относительно среднего радиуса: у круга вся масса
    # в одной корзине (энтропия ~ 0), у неровного контура она расходится.
    rel = r / mu if mu > 0 else r
    hist, _ = np.histogram(rel, bins=np.linspace(0.45, 1.7, 17))
    p = hist[hist > 0] / max(int(hist.sum()), 1)
    entropy = float(-(p * np.log(p)).sum() / math.log(16)) if len(p) else 0.0
    smooth = _periodic_smooth(r, 3.0)
    rough = float(np.mean(np.abs(r - smooth)) / mu) if mu > 0 else 0.0
    mag = np.abs(np.fft.rfft(r))
    f0 = float(mag[0]) if len(mag) else 0.0

    def harm(k: int) -> float:
        return float(mag[k] / f0) if f0 > 0 and k < len(mag) else 0.0

    return {
        "radial_skew": skew, "radial_kurtosis": kurt, "radial_entropy": entropy,
        "roughness": rough, "fourier_2": harm(2), "fourier_3": harm(3), "fourier_4": harm(4),
    }


def _hu_logs(moments: dict) -> Tuple[float, ...]:
    hu = cv2.HuMoments(moments).flatten()
    logs = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return tuple(float(v) for v in logs)


# --------------------------------------------------------------------------
# Основная функция
# --------------------------------------------------------------------------

def compute_morphometry(mask: np.ndarray, n_radial: int = 36) -> Morphometry:
    """Посчитать все признаки маски. Бросает ValueError для пустой маски."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Ожидается 2D-маска")
    m8 = _largest_component(mask > 0)
    area = int(m8.sum())
    if area == 0:
        raise ValueError("Пустая маска")

    # --- центроид и моменты ---
    mom = cv2.moments(m8, binaryImage=True)
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    mu20 = mom["mu20"] / mom["m00"] + 1.0 / 12.0  # +1/12: дисперсия пикселя-квадрата
    mu02 = mom["mu02"] / mom["m00"] + 1.0 / 12.0
    mu11 = mom["mu11"] / mom["m00"]
    root = math.sqrt(max(4.0 * mu11 ** 2 + (mu20 - mu02) ** 2, 0.0))
    l1, l2 = (mu20 + mu02 + root) / 2.0, (mu20 + mu02 - root) / 2.0
    major = 4.0 * math.sqrt(max(l1, 0.0))
    minor = 4.0 * math.sqrt(max(l2, 0.0))
    orient = math.degrees(0.5 * math.atan2(2.0 * mu11, mu20 - mu02))
    ecc = math.sqrt(max(0.0, 1.0 - (minor / major) ** 2)) if major > 0 else 0.0

    # --- контур, периметр, bbox ---
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=len)
    cpts = cnt.reshape(-1, 2)
    perimeter = _smoothed_arclength(cpts.astype(float)) + math.pi
    bx, by, bw, bh = cv2.boundingRect(cnt)

    # --- выпуклая оболочка и повёрнутый прямоугольник ---
    hull = cv2.convexHull(cnt).reshape(-1, 2)
    hull_f = hull.astype(np.float32)
    h_area = float(cv2.contourArea(hull_f)) if len(hull) >= 3 else 0.0
    h_per = float(cv2.arcLength(hull_f, True)) if len(hull) >= 2 else 0.0
    convex_area = max(h_area + 0.5 * h_per + math.pi / 4.0, float(area))
    convex_perimeter = h_per + math.pi
    solidity = min(1.0, area / convex_area)
    (rcx, rcy), (rw, rh), _ = cv2.minAreaRect(hull_f)
    length, width = max(rw, rh) + 1.0, min(rw, rh) + 1.0
    aspect = length / width
    (ccx, ccy), circ_r = cv2.minEnclosingCircle(cnt)
    circ_r = max(float(circ_r), 1e-6)

    # --- безразмерные формы ---
    circ = min(1.0, 4.0 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
    roundness = min(1.0, 4.0 * area / (math.pi * major ** 2)) if major > 0 else 0.0
    eq_d = math.sqrt(4.0 * area / math.pi)

    # --- радиальные характеристики ---
    pts = cnt.reshape(-1, 2).astype(float)
    dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) + 0.5  # +0.5 px: полпикселя до края
    r_mean, r_std = float(dist.mean()), float(dist.std())
    r_min, r_max = float(dist.min()), float(dist.max())
    # профиль по лучам: контур смещён наружу на 0.5 px по направлению от центра
    out = pts - np.array([cx, cy])
    norm = np.maximum(np.hypot(out[:, 0], out[:, 1]), 1e-9)[:, None]
    poly = pts + 0.5 * out / norm if len(pts) >= 3 else pts
    dense_n = max(int(n_radial), 360)
    dense = _ray_radii(poly, (cx, cy), dense_n) if len(poly) >= 3 else np.zeros(dense_n)
    pick = (np.arange(n_radial) * dense_n / max(n_radial, 1)).astype(int)
    profile = dense[pick] if len(poly) >= 3 else np.zeros(n_radial)
    radial = _radial_shape(dense)
    eq_r = eq_d / 2.0
    d_circ = math.hypot(cx - ccx, cy - ccy)
    d_rect = math.hypot(cx - rcx, cy - rcy)
    d_prim = math.hypot(ccx - rcx, ccy - rcy)
    hu = _hu_logs(mom)

    return Morphometry(
        area=area, perimeter=perimeter, centroid_x=float(cx), centroid_y=float(cy),
        bbox_x=int(bx), bbox_y=int(by), bbox_w=int(bw), bbox_h=int(bh),
        length=float(length), width=float(width), aspect_ratio=float(aspect),
        major_axis=major, minor_axis=minor, orientation_deg=orient, eccentricity=ecc,
        equivalent_diameter=eq_d, circularity=float(circ), roundness=float(roundness),
        convex_area=convex_area, convex_perimeter=convex_perimeter, solidity=float(solidity),
        radial_mean=r_mean, radial_std=r_std, radial_min=r_min, radial_max=r_max,
        radial_cv=r_std / r_mean if r_mean > 0 else 0.0,
        radial_max_min_ratio=r_max / r_min if r_min > 0 else float("inf"),
        radial_min_max_ratio=r_min / r_max if r_max > 0 else 0.0,
        extent=min(1.0, float(area / (length * width))) if length * width > 0 else 0.0,
        convexity=min(1.0, convex_perimeter / perimeter) if perimeter > 0 else 0.0,
        bending_energy=_bending_energy(pts, eq_r),
        center_shift_circ=d_circ / circ_r,
        center_shift_rect=d_rect / circ_r,
        primitive_shift=d_prim / circ_r,
        hu_1=hu[0], hu_2=hu[1], hu_3=hu[2], hu_4=hu[3],
        hu_5=hu[4], hu_6=hu[5], hu_7=hu[6],
        radial_profile=profile, hull=hull,
        **radial,
    )


def morphometry_features(mask: np.ndarray) -> dict:
    """Плоский словарь признаков — подходит как ``feature_fn`` для Tracker."""
    return compute_morphometry(mask).to_dict()


def scale_features(features: Dict[str, float], um_per_px: Optional[float]) -> Dict[str, float]:
    """Перевести длины и площади из px в µm (имена получают суффиксы _um / _um2)."""
    if um_per_px is None:
        return {}
    if um_per_px <= 0:
        raise ValueError("um_per_px должен быть положительным")
    out = {}
    for k, v in features.items():
        if k in LINEAR_FIELDS:
            out[f"{k}_um"] = v * um_per_px
        elif k in AREA_FIELDS:
            out[f"{k}_um2"] = v * um_per_px ** 2
    return out
