"""Проверка морфометрии на фигурах с известными аналитическими значениями.

Допуски учитывают дискретизацию (пиксельную сетку): площади — доли процента,
периметры — 1–2 %, оси — 1–2 px.
"""
import math

import cv2
import numpy as np
import pytest

from microtrack.morphometry import (compute_morphometry, morphometry_features,
                                    scale_features)

S = 400


def canvas():
    return np.zeros((S, S), np.uint8)


def circle(r=50):
    m = canvas()
    cv2.circle(m, (200, 200), r, 1, -1)
    return m


def ellipse(a=80, b=40, angle=0):
    m = canvas()
    cv2.ellipse(m, (200, 200), (a, b), angle, 0, 360, 1, -1)
    return m


def rectangle(w=200, h=100):
    m = canvas()
    m[150:150 + h, 100:100 + w] = 1
    return m


def rod(length=120, width=21):
    """Палочка: прямоугольник с полукруглыми концами (капсула)."""
    m = canvas()
    r = width // 2
    x0, x1, yc = 140, 140 + length - 1, 200
    cv2.rectangle(m, (x0 + r, yc - r), (x1 - r, yc + r), 1, -1)
    cv2.circle(m, (x0 + r, yc), r, 1, -1)
    cv2.circle(m, (x1 - r, yc), r, 1, -1)
    return m


def ramanujan(a, b):
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


# ---------------------------------------------------------------- круг

def test_circle():
    r = 50
    f = compute_morphometry(circle(r))
    assert f.area == pytest.approx(math.pi * r ** 2, rel=0.005)
    assert f.perimeter == pytest.approx(2 * math.pi * r, rel=0.01)
    assert f.length == pytest.approx(2 * r, abs=1.0)
    assert f.width == pytest.approx(2 * r, abs=1.0)
    assert f.aspect_ratio == pytest.approx(1.0, abs=0.02)
    assert f.circularity > 0.99 and f.circularity <= 1.0
    assert f.roundness > 0.99
    assert f.solidity > 0.98 and f.solidity <= 1.0
    assert f.equivalent_diameter == pytest.approx(2 * r, rel=0.01)
    assert f.eccentricity < 0.1
    assert f.centroid_x == pytest.approx(200, abs=0.5)
    assert f.centroid_y == pytest.approx(200, abs=0.5)
    # радиальные характеристики: все расстояния ≈ r
    assert f.radial_mean == pytest.approx(r, abs=0.6)
    assert f.radial_cv < 0.02
    assert f.radial_max_min_ratio < 1.05
    assert np.allclose(f.radial_profile, r, atol=1.0)


# ---------------------------------------------------------------- эллипс

@pytest.mark.parametrize("angle", [0, 30, 90])
def test_ellipse(angle):
    a, b = 80, 40
    f = compute_morphometry(ellipse(a, b, angle))
    assert f.area == pytest.approx(math.pi * a * b, rel=0.03)  # cv2 рисует чуть шире
    assert f.perimeter == pytest.approx(ramanujan(a, b), rel=0.015)
    assert f.length == pytest.approx(2 * a, abs=2.5)
    assert f.width == pytest.approx(2 * b, abs=2.5)
    assert f.aspect_ratio == pytest.approx(a / b, rel=0.03)
    assert f.roundness == pytest.approx(b / a, abs=0.01)
    expected_circ = 4 * math.pi * (math.pi * a * b) / ramanujan(a, b) ** 2
    assert f.circularity == pytest.approx(expected_circ, abs=0.015)
    assert f.eccentricity == pytest.approx(math.sqrt(1 - (b / a) ** 2), abs=0.01)
    assert f.solidity > 0.98
    assert f.major_axis == pytest.approx(2 * a, abs=2.5)
    assert f.minor_axis == pytest.approx(2 * b, abs=2.5)
    # радиально: от b (малая полуось) до a (большая полуось)
    assert f.radial_min == pytest.approx(b, abs=1.5)
    assert f.radial_max == pytest.approx(a, abs=1.5)
    # ориентация большой оси (с точностью до 180°)
    diff = (f.orientation_deg - angle + 90) % 180 - 90
    assert abs(diff) < 1.5


def test_ellipse_radial_profile_axes():
    a, b = 80, 40
    f = compute_morphometry(ellipse(a, b, 0), n_radial=36)
    assert len(f.radial_profile) == 36
    assert f.radial_profile[0] == pytest.approx(a, abs=1.5)    # 0°   вдоль большой оси
    assert f.radial_profile[9] == pytest.approx(b, abs=1.5)    # 90°  вдоль малой оси
    assert f.radial_profile[18] == pytest.approx(a, abs=1.5)   # 180°
    assert f.radial_profile[27] == pytest.approx(b, abs=1.5)   # 270°
    # 45°: r = ab / sqrt((b cos)^2 + (a sin)^2)
    th = math.radians(45)
    r45 = a * b / math.hypot(b * math.cos(th), a * math.sin(th))
    f2 = compute_morphometry(ellipse(a, b, 0), n_radial=8)
    assert f2.radial_profile[1] == pytest.approx(r45, abs=1.5)


# ---------------------------------------------------------------- прямоугольник

def test_rectangle():
    w, h = 200, 100
    f = compute_morphometry(rectangle(w, h))
    assert f.area == w * h
    assert (f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h) == (100, 150, w, h)
    assert f.perimeter == pytest.approx(2 * (w + h), rel=0.015)
    assert f.length == pytest.approx(w, abs=0.5)
    assert f.width == pytest.approx(h, abs=0.5)
    assert f.aspect_ratio == pytest.approx(2.0, abs=0.01)
    assert f.circularity == pytest.approx(4 * math.pi * w * h / (2 * (w + h)) ** 2, rel=0.025)
    assert f.solidity == pytest.approx(1.0, abs=0.001)
    assert f.convex_area == pytest.approx(w * h, rel=0.002)
    # у равномерного прямоугольника дисперсия по x = w^2/12 -> major = 4*sqrt(w^2/12)
    major = 4 * math.sqrt(w ** 2 / 12)
    assert f.major_axis == pytest.approx(major, rel=0.01)
    assert f.roundness == pytest.approx(4 * w * h / (math.pi * major ** 2), abs=0.005)
    # радиально: ближайший край — h/2, самый дальний — угол
    assert f.radial_min == pytest.approx(h / 2, abs=1.0)
    assert f.radial_max == pytest.approx(math.hypot(w, h) / 2, abs=1.5)
    # профиль: вдоль оси x — w/2, вдоль y — h/2, по диагонали 45° — min(w/2/cos, h/2/sin)
    p = compute_morphometry(rectangle(w, h), n_radial=8).radial_profile
    assert p[0] == pytest.approx(w / 2, abs=1.0)
    assert p[2] == pytest.approx(h / 2, abs=1.0)
    assert p[1] == pytest.approx((h / 2) / math.sin(math.radians(45)), abs=1.5)


def test_rotated_rectangle_is_invariant():
    base = compute_morphometry(rectangle(200, 100))
    img = rectangle(200, 100) * 255
    rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((200, 200), 37, 1.0), (S, S),
                         flags=cv2.INTER_NEAREST) > 127
    f = compute_morphometry(rot)
    assert f.area == pytest.approx(base.area, rel=0.01)
    assert f.length == pytest.approx(base.length, rel=0.02)
    assert f.width == pytest.approx(base.width, rel=0.02)
    assert f.perimeter == pytest.approx(base.perimeter, rel=0.03)
    assert f.solidity > 0.97


# ---------------------------------------------------------------- палочка

def test_rod():
    L, w = 120, 21
    f = compute_morphometry(rod(L, w))
    area_exact = w * (L - w) + math.pi * w ** 2 / 4
    per_exact = 2 * (L - w) + math.pi * w
    assert f.area == pytest.approx(area_exact, rel=0.03)
    assert f.perimeter == pytest.approx(per_exact, rel=0.025)
    assert f.length == pytest.approx(L, abs=1.5)
    assert f.width == pytest.approx(w, abs=1.5)
    assert f.aspect_ratio == pytest.approx(L / w, rel=0.07)
    assert f.circularity == pytest.approx(4 * math.pi * area_exact / per_exact ** 2, abs=0.025)
    assert f.roundness < 0.25            # вытянутый объект — далёк от круга
    assert f.solidity > 0.98             # палочка выпуклая
    assert f.eccentricity > 0.95
    assert f.radial_max == pytest.approx(L / 2, abs=1.5)    # концы палочки
    assert f.radial_min == pytest.approx(w / 2, abs=1.5)    # бока
    p = compute_morphometry(rod(L, w), n_radial=36).radial_profile
    assert p[0] == pytest.approx(L / 2, abs=1.5)
    assert p[9] == pytest.approx(w / 2, abs=1.5)


# ---------------------------------------------------------------- невыпуклая фигура

def test_notched_rectangle_solidity():
    m = rectangle(200, 100)
    m[150:200, 180:220] = 0          # вырез 40x50 с верхнего края
    f = compute_morphometry(m)
    assert f.area == 200 * 100 - 40 * 50
    assert f.convex_area == pytest.approx(200 * 100, rel=0.005)
    assert f.solidity == pytest.approx(18000 / 20000, abs=0.005)
    assert f.perimeter == pytest.approx(2 * (200 + 100) + 2 * 50, rel=0.03)


# ---------------------------------------------------------------- прочее

def test_rotation_invariance_of_shape_numbers():
    ref = compute_morphometry(ellipse(80, 40, 0))
    for ang in (15, 60, 105, 150):
        f = compute_morphometry(ellipse(80, 40, ang))
        assert f.aspect_ratio == pytest.approx(ref.aspect_ratio, rel=0.03)
        assert f.circularity == pytest.approx(ref.circularity, abs=0.015)
        assert f.roundness == pytest.approx(ref.roundness, abs=0.01)
        assert f.solidity == pytest.approx(ref.solidity, abs=0.01)


def test_largest_component_used():
    m = circle(30)
    cv2.circle(m, (60, 60), 5, 1, -1)
    f = compute_morphometry(m)
    assert f.area == pytest.approx(math.pi * 30 ** 2, rel=0.01)


def test_degenerate_masks():
    with pytest.raises(ValueError):
        compute_morphometry(np.zeros((20, 20), bool))
    one = np.zeros((20, 20), bool)
    one[5, 5] = True
    f = compute_morphometry(one)
    assert f.area == 1 and f.length == pytest.approx(1.0, abs=0.01)
    line = np.zeros((20, 30), bool)
    line[10, 5:15] = True
    f = compute_morphometry(line)
    assert f.length == pytest.approx(10.0, abs=0.01) and f.width == pytest.approx(1.0, abs=0.01)
    assert f.solidity == pytest.approx(1.0)


def test_shape_descriptors_separate_circle_ellipse_and_notch():
    circ = compute_morphometry(circle(50))
    ell = compute_morphometry(ellipse(80, 40, 0))
    notch_mask = rectangle(200, 100)
    notch_mask[150:200, 180:220] = 0
    notch = compute_morphometry(notch_mask)
    assert circ.bending_energy == pytest.approx(1.0, abs=0.15)
    assert circ.fourier_2 < 0.02 and ell.fourier_2 > 0.1
    assert circ.radial_entropy < 0.4 and ell.radial_entropy > 0.6
    assert circ.roughness < 0.02 and circ.center_shift_circ < 0.03
    assert circ.extent == pytest.approx(math.pi / 4, abs=0.03)
    assert circ.convexity == pytest.approx(1.0, abs=0.01)
    assert notch.convexity < 0.95
    assert notch.bending_energy > circ.bending_energy
    assert notch.fourier_3 > ell.fourier_3


def test_hu_moments_ignore_rotation():
    ref = compute_morphometry(ellipse(80, 40, 0))
    turned = compute_morphometry(ellipse(80, 40, 37))
    assert turned.hu_1 == pytest.approx(ref.hu_1, abs=0.01)
    assert turned.fourier_2 == pytest.approx(ref.fourier_2, abs=0.02)


def test_to_dict_and_scale():
    f = compute_morphometry(circle(50))
    d = f.to_dict()
    assert "radial_profile" not in d and d["area"] == f.area
    assert "radial_profile_00" in f.to_dict(include_profile=True)
    flat = morphometry_features(circle(50))
    s = scale_features(flat, 0.5)
    assert s["length_um"] == pytest.approx(flat["length"] * 0.5)
    assert s["area_um2"] == pytest.approx(flat["area"] * 0.25)
    assert "circularity_um" not in s                      # безразмерные не масштабируются
    assert scale_features(flat, None) == {}
    with pytest.raises(ValueError):
        scale_features(flat, 0)
