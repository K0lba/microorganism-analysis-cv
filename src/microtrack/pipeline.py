"""Сквозной конвейер: видео или кадры → таблица наблюдений.

Этапы на каждом кадре: загрузка (``io.iter_frames``) → предобработка →
сегментация (classical или готовая модель) → трекинг с морфометрией и
качеством сегментации каждой маски. В конце — временные признаки, проверка
и экспорт таблицы (``export``), сводка по трекам и видео с наложением.

Параметры читаются из YAML проекта (``configs/default.yaml``), секции:
input, preprocessing, segmentation, tracking, calibration, experiment, output.

Результаты в каталоге вывода:
  observations.csv / .parquet  таблица наблюдений (схема — docs/export_schema.md)
  tracks_summary.csv           итоги по трекам: путь, скорости, прямолинейность
  overlay.mp4, preview.png     кадры с масками, ID и траекториями
  run.json                     параметры запуска, счётчики и результат проверки

Ограничение: маски хранятся в полный размер кадра, поэтому память растёт с
числом объектов в кадре (сотни объектов на кадре ~1 Мп — сотни МБ).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from . import __version__
from .export import ExportMeta, ValidationReport, build_table, export_all, validate_table
from .io import Calibration, is_video, iter_frames, video_info
from .morphometry import morphometry_features
from .preprocess import PreprocessConfig, preprocess, to_grayscale
from .quality import segmentation_quality
from .segmentation import SegmentationConfig, segment
from .temporal import Calibration as TableCalibration
from .temporal import Timeline, compute_temporal, save_rows_csv, summarize_tracks
from .tracker import Tracker, TrackerConfig, masks_from_label_image
from .visualization import OverlayWriter, draw_tracking_frame

Segmenter = Callable[[np.ndarray, Calibration], np.ndarray]

_TRACKING_ALIASES = {"max_displacement_px": "min_gate", "max_missed_frames": "max_missed"}
_INPUT_KEYS = {"frame_stride", "timestamp_source", "fps", "max_frames",
               "supported_images", "supported_videos"}


@dataclass
class OutputConfig:
    directory: str = "results"
    save_csv: bool = True
    save_parquet: bool = False
    save_visualization: bool = True     # overlay.mp4 и preview.png
    save_frames: bool = False           # PNG каждого кадра с наложением
    include_predicted: bool = False     # строки с предсказанным положением на пропусках

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "OutputConfig":
        d = dict(d or {})
        bad = set(d) - {f.name for f in fields(cls)}
        if bad:
            raise ValueError(f"Неизвестные параметры output: {sorted(bad)}")
        return cls(**d)


def tracker_config_from_dict(d: Optional[dict]) -> TrackerConfig:
    """Секция ``tracking``: поля TrackerConfig и короткие имена проекта.

    max_displacement_px → min_gate (ворота не меньше этого расстояния, px за кадр),
    max_missed_frames   → max_missed,
    use_shape_similarity: false → вес формы w_shape = 0.
    """
    d = dict(d or {})
    for old, new in _TRACKING_ALIASES.items():
        if old in d:
            d.setdefault(new, d.pop(old))
    use_shape = d.pop("use_shape_similarity", True)
    bad = set(d) - {f.name for f in fields(TrackerConfig)}
    if bad:
        raise ValueError(f"Неизвестные параметры tracking: {sorted(bad)}")
    cfg = TrackerConfig(**d)
    if not use_shape:
        cfg.w_shape = 0.0
    if cfg.max_missed < 0 or cfg.min_gate <= 0:
        raise ValueError("tracking: max_missed >= 0, min_gate > 0")
    return cfg


@dataclass
class PipelineConfig:
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation_method: str = "classical"      # classical | имя модели из pretrained.MODEL_REGISTRY
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    tracking: TrackerConfig = field(default_factory=TrackerConfig)
    calibration: Calibration = field(default_factory=lambda: Calibration.from_dict(None))
    frame_stride: int = 1
    timestamp_source: str = "fps"               # fps | source_pts
    fps: Optional[float] = None
    max_frames: Optional[int] = None            # ограничить число обработанных кадров
    experiment_id: str = "experiment"
    conditions: Dict[str, object] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PipelineConfig":
        d = dict(d or {})
        inp = dict(d.get("input") or {})
        bad = set(inp) - _INPUT_KEYS
        if bad:
            raise ValueError(f"Неизвестные параметры input: {sorted(bad)}")
        seg = dict(d.get("segmentation") or {})
        method = str(seg.pop("method", "classical"))
        exp = dict(d.get("experiment") or {})
        return cls(
            preprocessing=PreprocessConfig.from_dict(d.get("preprocessing")),
            segmentation_method=method,
            segmentation=SegmentationConfig.from_dict(seg),
            tracking=tracker_config_from_dict(d.get("tracking")),
            calibration=Calibration.from_dict(d.get("calibration")),
            frame_stride=int(inp.get("frame_stride", 1)),
            timestamp_source=str(inp.get("timestamp_source", "fps")),
            fps=inp.get("fps"),
            max_frames=inp.get("max_frames"),
            experiment_id=str(exp.get("id", "experiment")),
            conditions=dict(exp.get("conditions") or {}),
            output=OutputConfig.from_dict(d.get("output")),
        )

    @classmethod
    def from_yaml(cls, path) -> "PipelineConfig":
        import yaml
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass
class PipelineResult:
    table: pd.DataFrame
    tracker: Tracker
    report: ValidationReport
    files: Dict[str, str]
    n_frames: int
    seconds: float


def relabel_min_area(labels: np.ndarray, min_area: float) -> np.ndarray:
    """Удалить метки площадью меньше min_area px и перенумеровать оставшиеся с 1."""
    lab = np.asarray(labels).astype(np.int64)
    if lab.max(initial=0) <= 0:
        return np.zeros(lab.shape, np.int32)
    counts = np.bincount(lab.ravel())
    keep = counts >= max(min_area, 1)
    keep[0] = False
    remap = np.zeros(len(counts), np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1)
    return remap[lab]


def make_segmenter(cfg: PipelineConfig) -> Segmenter:
    """Функция (серый кадр, калибровка) -> метки int32 (0 — фон)."""
    if cfg.segmentation_method == "classical":
        return lambda gray, calib: segment(gray, cfg.segmentation, calib).labels

    from .segmentation import pretrained
    if cfg.segmentation_method not in pretrained.MODEL_REGISTRY:
        raise ValueError(f"segmentation.method: classical или одна из {list(pretrained.MODEL_REGISTRY)}")
    model = pretrained.load_model(cfg.segmentation_method)
    if model.mode != "auto":
        raise ValueError(f"{cfg.segmentation_method} работает по точкам/рамкам; "
                         "для всего кадра нужна модель с mode=auto")
    flt = cfg.segmentation.filters

    def run(gray, calib):
        min_area = flt.min_area
        if flt.unit == "um" and calib.calibrated:
            min_area = min_area / calib.pixel_size_um ** 2
        return relabel_min_area(model(gray), min_area)
    return run


def _timestamp_source(source: Path, cfg: PipelineConfig) -> str:
    if is_video(source) and cfg.timestamp_source == "source_pts" and not cfg.fps:
        return "video"
    return "fps"


def _fps_nominal(source: Path, cfg: PipelineConfig) -> Optional[float]:
    if cfg.fps:
        return float(cfg.fps)
    if is_video(source):
        fps = video_info(source).get("fps")
        return float(fps) if fps and fps > 0 else None
    return None


def run_pipeline(source, cfg: Optional[PipelineConfig] = None, out_dir=None,
                 sequence_id: Optional[str] = None,
                 progress: Optional[Callable[[int, int, float], None]] = None) -> PipelineResult:
    """Обработать видео, папку кадров, TIFF-стек или одно изображение.

    ``progress(k, frame_index, seconds)`` вызывается после каждого кадра.
    """
    cfg = cfg or PipelineConfig()
    source = Path(source)
    out = Path(out_dir or cfg.output.directory)
    t_start = time.perf_counter()
    segmenter = make_segmenter(cfg)

    state: Dict[str, np.ndarray] = {}

    def feature_fn(mask):
        d = morphometry_features(mask)
        d.update(segmentation_quality(state["gray"], mask, d))
        return d

    tracker = Tracker(cfg.tracking, feature_fn=feature_fn)
    fps_nominal = _fps_nominal(source, cfg)
    vis = None
    if cfg.output.save_visualization or cfg.output.save_frames:
        play_fps = fps_nominal / cfg.frame_stride if fps_nominal else 5.0
        vis = OverlayWriter(out, fps=play_fps, save_video=cfg.output.save_visualization,
                            save_frames=cfg.output.save_frames)

    pre_cfg = replace(cfg.preprocessing, grayscale=True)
    times: Dict[int, float] = {}
    calib = cfg.calibration
    n_frames = 0
    frames = iter_frames(source, step=cfg.frame_stride, fps=cfg.fps, calibration=cfg.calibration,
                         timestamp_source=cfg.timestamp_source)
    try:
        for frame in frames:
            if cfg.max_frames is not None and n_frames >= cfg.max_frames:
                break
            gray, calib = preprocess(frame.image, pre_cfg, frame.calibration)
            gray = to_grayscale(gray)
            masks = masks_from_label_image(segmenter(gray, calib))
            state["gray"] = gray
            assigned = tracker.update(frame.index, masks)
            times[frame.index] = frame.time_s
            if vis is not None:
                vis.add(draw_tracking_frame(gray, masks, assigned, tracker, frame.index, frame.time_s),
                        frame.index)
            n_frames += 1
            if progress:
                progress(n_frames, frame.index, time.perf_counter() - t_start)
    finally:
        files = vis.close() if vis is not None else {}
    if n_frames == 0:
        raise ValueError(f"В источнике нет кадров: {source}")
    tracker.finalize()

    table_cal = TableCalibration(calib.pixel_size_um if calib.calibrated else None)
    timeline = Timeline(times)
    meta = ExportMeta(
        experiment_id=cfg.experiment_id,
        sequence_id=sequence_id or source.stem,
        calibration=table_cal,
        calibration_source="config" if calib.calibrated else "none",
        timestamp_source=_timestamp_source(source, cfg),
        fps_nominal=fps_nominal,
        frame_step=cfg.frame_stride,
        conditions=cfg.conditions,
        pipeline_version=__version__,
    )
    table = build_table(tracker, timeline, meta, include_predicted=cfg.output.include_predicted)
    report = validate_table(table)

    formats = [f for f, on in (("csv", cfg.output.save_csv), ("parquet", cfg.output.save_parquet)) if on]
    if formats:
        files.update(export_all(table, out, formats=formats, strict=False))
    temporal = compute_temporal(tracker.to_rows(include_predicted=False), timeline, table_cal)
    files["tracks_summary"] = save_rows_csv(summarize_tracks(temporal, table_cal),
                                            out / "tracks_summary.csv")

    seconds = time.perf_counter() - t_start
    info = {
        "source": str(source), "sequence_id": meta.sequence_id, "experiment_id": cfg.experiment_id,
        "segmentation_method": cfg.segmentation_method, "pipeline_version": __version__,
        "n_frames": n_frames, "n_tracks": len(tracker.tracks), "n_observations": int(len(table)),
        "seconds": round(seconds, 3),
        "validation": {"ok": report.ok, "errors": report.errors, "warnings": report.warnings},
        "config": _config_dict(cfg), "files": files,
    }
    out.mkdir(parents=True, exist_ok=True)
    run_json = out / "run.json"
    run_json.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    files["run"] = str(run_json)
    return PipelineResult(table, tracker, report, files, n_frames, seconds)


def _config_dict(cfg: PipelineConfig) -> dict:
    d = asdict(cfg)
    d["calibration"] = {"pixel_size_um": cfg.calibration.pixel_size_um if cfg.calibration.calibrated else None,
                        "unit": cfg.calibration.unit}
    return d
