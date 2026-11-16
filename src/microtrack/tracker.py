"""Трекинг объектов между кадрами с устойчивыми ID.

Для каждого кадра на вход подаются маски объектов (список bool-масок или
label-изображение, где 0 — фон). Трекер:

1. считает признаки каждой маски (центр, площадь, оси эллипса, круглость);
2. строит матрицу стоимости «трек — детекция» из трёх частей:
   расстояние до предсказанного центра, пересечение масок (IoU) и сходство формы;
3. решает назначение венгерским алгоритмом (scipy.optimize.linear_sum_assignment);
4. ведёт жизненный цикл трека: active -> lost (пропуски) -> finished;
5. записывает уверенность каждого сопоставления (0..1) и пометку
   неоднозначности, если был почти равноценный другой кандидат.

Режимы:
* многообъектный (по умолчанию) — несопоставленные детекции порождают новые треки;
* одиночная цель — ``init_from_target(TargetInit)``: ведётся только выбранный
  объект, остальные игнорируются (новые треки не создаются).

Ограничения: слияния/разделения объектов не моделируются; предсказание
положения — модель постоянной скорости.

Зависимости: numpy, opencv-python, scipy.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

_BIG = 1e6  # стоимость запрещённой пары


# --------------------------------------------------------------------------
# Признаки маски
# --------------------------------------------------------------------------

@dataclass
class MaskFeatures:
    center: Tuple[float, float]     # центроид (x, y), px
    area: int                       # px
    perimeter: float                # длина внешнего контура, px
    major: float                    # большая ось эквивалентного эллипса, px
    minor: float                    # малая ось, px
    orientation: float              # угол большой оси, рад (-pi/2..pi/2)
    circularity: float              # 4*pi*A/P^2, 0..1
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)

    @property
    def diameter(self) -> float:
        """Диаметр круга той же площади."""
        return math.sqrt(4.0 * self.area / math.pi)

    @property
    def aspect_ratio(self) -> float:
        return self.major / max(self.minor, 1e-6)


def compute_features(mask: np.ndarray) -> Optional[MaskFeatures]:
    """Геометрические признаки бинарной маски (None, если маска пуста)."""
    m8 = np.asarray(mask).astype(np.uint8)
    area = int(m8.sum())
    if area == 0:
        return None
    mom = cv2.moments(m8, binaryImage=True)
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    mu20, mu02, mu11 = (mom["mu20"] / mom["m00"], mom["mu02"] / mom["m00"],
                        mom["mu11"] / mom["m00"])
    # +1/12 — дисперсия квадрата пикселя; без неё тонкие объекты дают 0
    mu20 += 1.0 / 12.0
    mu02 += 1.0 / 12.0
    root = math.sqrt(max(4.0 * mu11 ** 2 + (mu20 - mu02) ** 2, 0.0))
    l1, l2 = (mu20 + mu02 + root) / 2.0, (mu20 + mu02 - root) / 2.0
    major, minor = 4.0 * math.sqrt(max(l1, 0.0)), 4.0 * math.sqrt(max(l2, 0.0))
    orient = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)

    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    per = float(cv2.arcLength(cnt, True))
    circ = min(1.0, 4.0 * math.pi * area / (per ** 2)) if per > 0 else 0.0
    return MaskFeatures(
        center=(float(cx), float(cy)), area=area, perimeter=per,
        major=major, minor=minor, orientation=orient, circularity=float(circ),
        bbox=tuple(int(v) for v in cv2.boundingRect(cnt)),
    )


def masks_from_label_image(labels: np.ndarray) -> List[np.ndarray]:
    """Разложить label-изображение (0 — фон) на список bool-масок."""
    return [labels == v for v in np.unique(labels) if v != 0]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


# --------------------------------------------------------------------------
# Данные трека
# --------------------------------------------------------------------------

@dataclass
class Observation:
    frame: int
    features: MaskFeatures
    observed: bool = True          # False — положение предсказано (пропуск)
    iou: float = 0.0               # IoU со сдвинутой предыдущей маской
    confidence: float = 1.0        # уверенность сопоставления, 0..1
    det_index: Optional[int] = None  # индекс детекции в кадре
    extra: dict = field(default_factory=dict)  # доп. признаки от feature_fn
    ambiguous: bool = False        # был почти равноценный альтернативный кандидат


@dataclass
class Track:
    id: int
    start_frame: int
    status: str = "active"         # active | lost | finished
    missed: int = 0                # подряд пропущенных кадров
    velocity: Tuple[float, float] = (0.0, 0.0)  # px/кадр
    observations: List[Observation] = field(default_factory=list)
    last_mask: Optional[np.ndarray] = field(default=None, repr=False)
    last_features: Optional[MaskFeatures] = field(default=None, repr=False)
    last_frame: int = 0

    def predict_center(self, frame: int) -> Tuple[float, float]:
        dt = frame - self.last_frame
        c = self.last_features.center
        return c[0] + self.velocity[0] * dt, c[1] + self.velocity[1] * dt

    @property
    def end_frame(self) -> int:
        return self.observations[-1].frame

    @property
    def mean_confidence(self) -> float:
        obs = [o.confidence for o in self.observations if o.observed]
        return float(np.mean(obs)) if obs else 0.0

    def trajectory(self, observed_only: bool = True) -> np.ndarray:
        """Массив строк (frame, x, y)."""
        rows = [(o.frame, *o.features.center) for o in self.observations
                if o.observed or not observed_only]
        return np.asarray(rows, dtype=float).reshape(-1, 3)


@dataclass
class TrackerConfig:
    # веса частей стоимости (нормируются на сумму)
    w_distance: float = 0.4
    w_iou: float = 0.4
    w_shape: float = 0.2
    # ворота по расстоянию: max(min_gate, gate_diameters * диаметр объекта)
    min_gate: float = 10.0
    gate_diameters: float = 2.0
    gap_gate_growth: float = 0.5   # рост ворот за каждый пропущенный кадр
    max_cost: float = 0.75         # выше — сопоставление отвергается
    ambiguity_margin: float = 0.1  # второй кандидат равноценен, если его стоимость не хуже выбранной больше чем на это
    max_missed: int = 5            # сколько кадров подряд можно пропустить
    velocity_alpha: float = 0.6    # сглаживание скорости (1 — только последняя)
    min_area: int = 5              # детекции меньше игнорируются


@dataclass
class _Det:
    index: int
    mask: np.ndarray
    f: MaskFeatures


# --------------------------------------------------------------------------
# Трекер
# --------------------------------------------------------------------------

Masks = Union[Sequence[np.ndarray], np.ndarray]


class Tracker:
    def __init__(self, config: Optional[TrackerConfig] = None,
                 spawn_new: bool = True,
                 feature_fn: Optional[Callable[[np.ndarray], dict]] = None):
        """feature_fn(mask) -> dict — необязательные доп. признаки для каждого
        сопоставленного наблюдения (например, ``morphometry_features``).
        Попадают в ``to_rows`` / CSV."""
        self.cfg = config or TrackerConfig()
        self.spawn_new = spawn_new
        self.feature_fn = feature_fn
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._last_frame: Optional[int] = None

    def _extra(self, mask: np.ndarray) -> dict:
        return dict(self.feature_fn(mask)) if self.feature_fn else {}

    # ---- создание треков -------------------------------------------------

    def _new_track(self, frame: int, mask: np.ndarray, f: MaskFeatures,
                   det_index: Optional[int] = None) -> Track:
        tr = Track(id=self._next_id, start_frame=frame, last_mask=mask,
                   last_features=f, last_frame=frame)
        tr.observations.append(Observation(frame, f, True, 1.0, 1.0, det_index,
                                           self._extra(mask)))
        self.tracks[tr.id] = tr
        self._next_id += 1
        return tr

    def init_from_target(self, target, frame: int = 0) -> Track:
        """Начать одиночный трек из ``TargetInit`` (или любого объекта с ``.mask``).

        После вызова новые треки не создаются: ведётся только выбранный объект.
        """
        mask = np.asarray(target.mask, bool)
        f = compute_features(mask)
        if f is None:
            raise ValueError("Пустая маска инициализации")
        self.spawn_new = False
        self._last_frame = frame
        return self._new_track(frame, mask, f)

    # ---- стоимость -------------------------------------------------------

    def _pair_cost(self, tr: Track, frame: int, d: _Det,
                   pred: Tuple[float, float]) -> Tuple[float, float]:
        """Вернуть (cost, iou); cost=_BIG, если пара вне ворот."""
        c = self.cfg
        lf = tr.last_features
        gate = max(c.min_gate, c.gate_diameters * lf.diameter) \
            * (1.0 + c.gap_gate_growth * tr.missed)
        dist = math.hypot(d.f.center[0] - pred[0], d.f.center[1] - pred[1])
        if dist > gate:
            return _BIG, 0.0
        dn = dist / gate

        iou = self._shifted_iou(tr, d, pred)

        ratios = [min(lf.area, d.f.area) / max(lf.area, d.f.area),
                  min(lf.major, d.f.major) / max(lf.major, d.f.major, 1e-6),
                  min(lf.minor, d.f.minor) / max(lf.minor, d.f.minor, 1e-6)]
        shape = 1.0 - float(np.mean(ratios))

        wsum = c.w_distance + c.w_iou + c.w_shape
        cost = (c.w_distance * dn + c.w_iou * (1.0 - iou) + c.w_shape * shape) / wsum
        return float(cost), iou

    @staticmethod
    def _shifted_iou(tr: Track, d: _Det, pred: Tuple[float, float]) -> float:
        """IoU предыдущей маски, сдвинутой в предсказанную позицию, и новой маски."""
        lf = tr.last_features
        dx = int(round(pred[0] - lf.center[0]))
        dy = int(round(pred[1] - lf.center[1]))
        ax, ay, aw, ah = lf.bbox
        bx, by, bw, bh = d.f.bbox
        x0, y0 = max(ax + dx, bx), max(ay + dy, by)
        x1, y1 = min(ax + dx + aw, bx + bw), min(ay + dy + ah, by + bh)
        h, w = d.mask.shape
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        a = tr.last_mask[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        b = d.mask[y0:y1, x0:x1]
        if a.shape != b.shape:  # на границе кадра
            return 0.0
        inter = int(np.count_nonzero(a & b))
        union = lf.area + d.f.area - inter
        return inter / union if union else 0.0

    # ---- шаг по кадру ----------------------------------------------------

    def update(self, frame: int, masks: Masks) -> Dict[int, int]:
        """Обработать кадр. Возвращает {индекс_детекции: id_трека}."""
        if self._last_frame is not None and frame <= self._last_frame:
            raise ValueError(f"Кадры должны идти по возрастанию: {frame} <= {self._last_frame}")
        self._last_frame = frame

        if isinstance(masks, np.ndarray) and masks.ndim == 2:
            # bool-маска — один объект; целочисленная — label-изображение
            masks = [masks] if masks.dtype == bool else masks_from_label_image(masks)
        dets: List[_Det] = []
        for i, m in enumerate(masks):
            m = np.asarray(m, bool)
            f = compute_features(m)
            if f is not None and f.area >= self.cfg.min_area:
                dets.append(_Det(i, m, f))

        live = [t for t in self.tracks.values() if t.status != "finished"]
        assigned: Dict[int, int] = {}
        matched_tracks, matched_dets = set(), set()

        if live and dets:
            cost = np.full((len(live), len(dets)), _BIG)
            ious = np.zeros_like(cost)
            for r, tr in enumerate(live):
                pred = tr.predict_center(frame)
                for k, d in enumerate(dets):
                    cost[r, k], ious[r, k] = self._pair_cost(tr, frame, d, pred)
            rows, cols = linear_sum_assignment(cost)
            for r, k in zip(rows, cols):
                if cost[r, k] > self.cfg.max_cost:
                    continue
                self._apply_match(live[r], frame, dets[k], float(ious[r, k]),
                                  float(np.clip(1.0 - cost[r, k], 0.0, 1.0)),
                                  self._match_ambiguous(cost, r, k))
                assigned[dets[k].index] = live[r].id
                matched_tracks.add(r)
                matched_dets.add(k)

        for r, tr in enumerate(live):
            if r not in matched_tracks:
                self._apply_miss(tr, frame)

        if self.spawn_new:
            for k, d in enumerate(dets):
                if k not in matched_dets:
                    tr = self._new_track(frame, d.mask, d.f, d.index)
                    assigned[d.index] = tr.id
        return assigned

    def _match_ambiguous(self, cost: np.ndarray, r: int, k: int) -> bool:
        """True, если у принятой пары был другой кандидат не хуже выбранного + margin и в пределах max_cost."""
        chosen = float(cost[r, k])
        limit = chosen + self.cfg.ambiguity_margin
        gate = self.cfg.max_cost
        for k2 in range(cost.shape[1]):
            if k2 != k and cost[r, k2] <= gate and cost[r, k2] <= limit:
                return True
        for r2 in range(cost.shape[0]):
            if r2 != r and cost[r2, k] <= gate and cost[r2, k] <= limit:
                return True
        return False

    def _apply_match(self, tr: Track, frame: int, d: _Det, iou: float, conf: float,
                     ambiguous: bool = False):
        dt = frame - tr.last_frame
        old = tr.last_features.center
        v_new = ((d.f.center[0] - old[0]) / dt, (d.f.center[1] - old[1]) / dt)
        a = self.cfg.velocity_alpha
        tr.velocity = (a * v_new[0] + (1 - a) * tr.velocity[0],
                       a * v_new[1] + (1 - a) * tr.velocity[1])
        tr.observations.append(Observation(frame, d.f, True, iou, conf, d.index,
                                           self._extra(d.mask), ambiguous=ambiguous))
        tr.last_mask, tr.last_features, tr.last_frame = d.mask, d.f, frame
        tr.missed, tr.status = 0, "active"

    def _apply_miss(self, tr: Track, frame: int):
        tr.missed += 1
        if tr.missed > self.cfg.max_missed:
            self._finish(tr)
            return
        tr.status = "lost"
        cx, cy = tr.predict_center(frame)
        lf = tr.last_features
        pf = MaskFeatures(center=(cx, cy), area=lf.area, perimeter=lf.perimeter,
                          major=lf.major, minor=lf.minor,
                          orientation=lf.orientation, circularity=lf.circularity,
                          bbox=lf.bbox)
        last_extra = dict(tr.observations[-1].extra)  # на пропуске — значения последнего наблюдения
        tr.observations.append(Observation(frame, pf, False, 0.0, 0.0, None, last_extra))

    def _finish(self, tr: Track):
        tr.status = "finished"
        while tr.observations and not tr.observations[-1].observed:
            tr.observations.pop()  # хвост из предсказаний не считаем частью трека
        tr.last_mask = None

    def finalize(self):
        """Завершить все треки в конце последовательности."""
        for tr in self.tracks.values():
            if tr.status != "finished":
                self._finish(tr)

    # ---- экспорт ---------------------------------------------------------

    def to_rows(self, include_predicted: bool = True) -> List[dict]:
        rows = []
        for tr in sorted(self.tracks.values(), key=lambda t: t.id):
            for o in tr.observations:
                if not o.observed and not include_predicted:
                    continue
                f = o.features
                rows.append({
                    "track_id": tr.id, "frame": o.frame,
                    "x": round(f.center[0], 3), "y": round(f.center[1], 3),
                    "area": f.area, "perimeter": round(f.perimeter, 3),
                    "major": round(f.major, 3), "minor": round(f.minor, 3),
                    "orientation": round(f.orientation, 4),
                    "circularity": round(f.circularity, 4),
                    "observed": int(o.observed), "iou": round(o.iou, 4),
                    "confidence": round(o.confidence, 4), "status": tr.status,
                })
                rows[-1].update(o.extra)  # признаки feature_fn (морфометрия) уточняют базовые
                rows[-1]["ambiguous"] = int(o.ambiguous)
        return rows

    def save_csv(self, path, include_predicted: bool = True) -> str:
        rows = self.to_rows(include_predicted)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        fields = fields or ["track_id", "frame"]
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        return str(p)


def track_sequence(frames: Sequence[Masks], config: Optional[TrackerConfig] = None,
                   first_frame: int = 0) -> Tracker:
    """Прогнать трекер по всей последовательности масок."""
    tracker = Tracker(config)
    for i, masks in enumerate(frames):
        tracker.update(first_frame + i, masks)
    tracker.finalize()
    return tracker
