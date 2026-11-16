"""Оценка качества сегментации без эталонной разметки.

На реальных снимках эталонных масок нет, поэтому качество оценивается по
косвенным признакам, которые считаются для каждой маски:

seg_contrast        |средняя яркость объекта - средняя яркость кольца вокруг|, уровни серого 0..255
seg_cnr             контраст / стандартное отклонение яркости кольца (контраст к шуму)
touches_border      маска касается края кадра (объект обрезан)
seg_flag            "ok" либо список причин через запятую:
                    low_contrast — объект слабо отличается от фона;
                    border       — объект обрезан краем кадра;
                    irregular    — очень неровная форма (solidity ниже порога);
                    tiny         — площадь меньше порога.
Если эталон есть (синтетические данные), используйте ``segmentation.metrics.evaluate``
или ``scripts/benchmark.py`` — там считаются IoU, Dice и F1 обнаружения.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class QualityThresholds:
    min_cnr: float = 3.0          # ниже — low_contrast
    min_contrast: float = 10.0    # уровни серого
    min_solidity: float = 0.80    # ниже — irregular
    min_area: int = 20            # ниже — tiny
    ring_px: int = 4              # ширина кольца для оценки фона


def segmentation_quality(gray: np.ndarray, mask: np.ndarray,
                         morph: Optional[dict] = None,
                         th: Optional[QualityThresholds] = None) -> dict:
    """Косвенные показатели качества маски; ``morph`` — словарь морфометрии (solidity, area)."""
    th = th or QualityThresholds()
    gray = np.asarray(gray)
    m = np.asarray(mask, bool)
    m8 = m.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * th.ring_px + 1,) * 2)
    ring = (cv2.dilate(m8, k) > 0) & ~(cv2.dilate(m8, np.ones((3, 3), np.uint8)) > 0)
    inside = gray[m].astype(float)
    if ring.any() and inside.size:
        outside = gray[ring].astype(float)
        contrast = abs(float(inside.mean()) - float(outside.mean()))
        cnr = contrast / max(float(outside.std()), 1.0)
    else:  # объект занимает весь кадр или маска пуста
        contrast, cnr = 0.0, 0.0
    touches = bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())

    flags = []
    if contrast < th.min_contrast or cnr < th.min_cnr:
        flags.append("low_contrast")
    if touches:
        flags.append("border")
    if morph is not None:
        if morph.get("solidity", 1.0) < th.min_solidity:
            flags.append("irregular")
        if morph.get("area", th.min_area) < th.min_area:
            flags.append("tiny")
    return {"seg_contrast": float(contrast), "seg_cnr": float(cnr),
            "touches_border": touches, "seg_flag": ",".join(flags) if flags else "ok"}
