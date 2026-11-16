"""Generate synthetic microscopy images/videos with instance masks and persistent tracks.

Формы не сводятся к эллипсам: кокки и овалы получают радиальные гармоники
(асимметрия и вмятины), палочки рисуются как капсулы, диплококки — как два
перекрывающихся кокка с общим контуром. Эталонные размеры в objects.csv
считаются по истинной маске, а не по параметрам эллипса. Оптика по умолчанию
ближе к фазовому контрасту: тёмное тело, светлый ободок, зерно и виньетка.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from microtrack.morphometry import compute_morphometry


def make_tracks(rng, width, height, n_objects, sequence_id):
    """Create persistent object IDs and smooth linear trajectories for one sequence."""
    tracks = []
    next_id = 1
    remaining = n_objects
    while remaining > 0:
        group_size = min(remaining, int(rng.choice([1, 1, 1, 2, 3, 4])))
        remaining -= group_size
        gx = float(rng.uniform(25, max(26, width - 25)))
        gy = float(rng.uniform(25, max(26, height - 25)))
        group_angle = float(rng.uniform(0, 180))
        rad = math.radians(group_angle)
        spacing = float(rng.uniform(0.60, 0.92))
        group_vx = float(rng.uniform(-5.0, 5.0))
        group_vy = float(rng.uniform(-5.0, 5.0))
        for j in range(group_size):
            cell_type = str(rng.choice(
                ["coccus", "oval", "rod", "diplo"], p=[0.34, 0.22, 0.28, 0.16]))
            cell_width = float(rng.uniform(8, 18))
            if cell_type == "coccus":
                cell_length = cell_width
            elif cell_type == "oval":
                cell_length = cell_width * float(rng.uniform(1.25, 1.65))
            elif cell_type == "diplo":
                cell_length = cell_width * float(rng.uniform(1.55, 1.9))
            else:
                cell_length = cell_width * float(rng.uniform(2.2, 3.6))
            offset = (j - (group_size - 1) / 2) * cell_width * spacing
            tracks.append({
                "object_id": next_id,
                "cell_type": cell_type,
                "x0": gx + math.cos(rad) * offset,
                "y0": gy + math.sin(rad) * offset,
                "vx": group_vx + float(rng.uniform(-0.5, 0.5)),
                "vy": group_vy + float(rng.uniform(-0.5, 0.5)),
                "length": cell_length,
                "width": cell_width,
                "angle": group_angle if cell_type != "coccus" else float(rng.uniform(0, 180)),
                "harmonics": _harmonics(rng, cell_type),
            })
            next_id += 1
    return tracks


def _harmonics(rng, cell_type):
    """Амплитуды радиальных гармоник, фиксированные на всё время жизни клетки."""
    bands = {
        "coccus": [(2, 0.02, 0.06), (3, 0.025, 0.07), (4, 0.015, 0.04), (5, 0.01, 0.03)],
        "oval": [(3, 0.02, 0.06), (4, 0.015, 0.045), (5, 0.01, 0.03)],
        "rod": [(3, 0.02, 0.05), (4, 0.015, 0.04), (5, 0.01, 0.03)],
        "diplo": [(3, 0.03, 0.07), (4, 0.02, 0.05), (5, 0.015, 0.035)],
    }[cell_type]
    return [(k, float(rng.uniform(lo, hi)), float(rng.uniform(0.0, 2.0 * math.pi)))
            for k, lo, hi in bands]


def _modulate(theta, radii, harmonics):
    factor = np.ones_like(radii)
    for k, amp, phase in harmonics:
        factor += amp * np.cos(k * theta + phase)
    return np.maximum(radii * factor, 1.0)


def _ellipse_radii(theta, length, width):
    a = max(length / 2.0, 1.0)
    b = max(width / 2.0, 1.0)
    return (a * b) / np.hypot(b * np.cos(theta), a * np.sin(theta))


def _stadium_radii(theta, length, width):
    """Радиус капсулы (палочка с полукруглыми концами) из её центра."""
    radius = max(width / 2.0, 1.0)
    half = max(length / 2.0 - radius, 0.0)
    if half <= 0:
        return np.full(theta.shape, radius)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    out = np.full(theta.shape, np.inf)
    side = np.abs(sin_t) > 1e-8
    t_side = np.divide(radius, np.abs(sin_t), out=np.full_like(sin_t, np.inf), where=side)
    hit_side = side & (np.abs(t_side * cos_t) <= half + 1e-6)
    out = np.where(hit_side, t_side, out)
    disc = radius ** 2 - (half * sin_t) ** 2
    good = disc >= -1e-8
    root = np.sqrt(np.clip(disc, 0.0, None))
    for sign in (1.0, -1.0):
        for pm in (1.0, -1.0):
            t = sign * half * cos_t + pm * root
            on_cap = good & (t > 1e-6) & (sign * t * cos_t >= half - 1e-3)
            out = np.where(on_cap, np.minimum(out, t), out)
    return np.where(np.isfinite(out), out, radius)


def _place_poly(mask, cx, cy, angle_deg, radii, theta):
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    ox = radii * np.cos(theta)
    oy = radii * np.sin(theta)
    x = np.rint(cx + ox * c - oy * s).astype(np.int32)
    y = np.rint(cy + ox * s + oy * c).astype(np.int32)
    cv2.fillPoly(mask, [np.stack([x, y], axis=1)], 1)


def draw_cell_mask(shape, x, y, length, width, angle, cell_type, harmonics=()):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    theta = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
    if cell_type == "diplo":
        lobe = max(width / 2.0, 2.0)
        sep = max(length - width, lobe) / 2.0
        rad = _modulate(theta, np.full(theta.shape, lobe), harmonics)
        t = math.radians(angle)
        for sign in (-1.0, 1.0):
            _place_poly(mask, x + sign * sep * math.cos(t), y + sign * sep * math.sin(t),
                        angle, rad, theta)
        return mask
    if cell_type == "rod":
        radii = _stadium_radii(theta, length, width)
    else:
        radii = _ellipse_radii(theta, length, width)
    _place_poly(mask, x, y, angle, _modulate(theta, radii, harmonics), theta)
    return mask


def _mask_record(mask, track, x, y):
    """Истинные размеры по нарисованной маске. При пустой маске — параметры модели."""
    if int(mask.sum()) < 4:
        length, width = track["length"], track["width"]
        return {
            "x_px": round(x, 3), "y_px": round(y, 3),
            "length_px": round(length, 3), "width_px": round(width, 3),
            "area_px2": round(math.pi * length * width / 4.0, 3),
            "perimeter_px": round(math.pi * (length + width) / 2.0, 3),
            "aspect_ratio": round(length / max(width, 1e-6), 4),
            "circularity": 0.0, "solidity": 0.0, "extent": 0.0,
            "roughness": 0.0, "bending_energy": 0.0, "radial_cv": 0.0,
        }
    m = compute_morphometry(mask)
    return {
        "x_px": round(m.centroid_x, 3), "y_px": round(m.centroid_y, 3),
        "length_px": round(m.length, 3), "width_px": round(m.width, 3),
        "area_px2": round(float(m.area), 3), "perimeter_px": round(m.perimeter, 3),
        "aspect_ratio": round(m.aspect_ratio, 4), "circularity": round(m.circularity, 4),
        "solidity": round(m.solidity, 4), "extent": round(m.extent, 4),
        "roughness": round(m.roughness, 4), "bending_energy": round(m.bending_energy, 4),
        "radial_cv": round(m.radial_cv, 4),
    }


def build_scene(rng, width, height, tracks, frame_index, fps, sequence_id, config):
    """Render a frame from persistent tracks. IDs and trajectories are stable over time."""
    base = rng.normal(config["background_mean"], config["background_noise"],
                      (height, width)).astype(np.float32)
    coarse = rng.normal(0, config["background_texture"],
                        (max(2, height // 32), max(2, width // 32))).astype(np.float32)
    texture = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:height, 0:width]
    vignette = 1.0 - config["vignette"] * (
        ((xx - width / 2) / (width / 2)) ** 2 +
        ((yy - height / 2) / (height / 2)) ** 2
    )
    image = base + texture + (vignette - 1.0) * 35
    labels = np.zeros((height, width), dtype=np.uint16)
    objects = []
    dt = frame_index / fps

    # Painter's order gives realistic partial occlusion. Labels retain the visible pixels.
    placements = []
    for tr in tracks:
        x = float(np.clip(tr["x0"] + tr["vx"] * dt, 3, width - 4))
        y = float(np.clip(tr["y0"] + tr["vy"] * dt, 3, height - 4))
        placements.append((y, tr, x, y))
    placements.sort(key=lambda row: row[0])

    for _, tr, x, y in placements:
        full_mask = draw_cell_mask((height, width), x, y, tr["length"], tr["width"],
                                   tr["angle"], tr["cell_type"], tr.get("harmonics", ()))
        visible = full_mask.astype(bool)
        labels[visible] = tr["object_id"]
        image[visible] += float(rng.uniform(config["cell_intensity_min"],
                                             config["cell_intensity_max"]))
        measured = _mask_record(full_mask, tr, x, y)
        objects.append({
            "sequence_id": sequence_id,
            "frame_index": frame_index,
            "timestamp_sec": round(dt, 6),
            "object_id": tr["object_id"],
            "cell_type": tr["cell_type"],
            "x_px": measured["x_px"],
            "y_px": measured["y_px"],
            "length_px": measured["length_px"],
            "width_px": measured["width_px"],
            "area_px2": measured["area_px2"],
            "perimeter_px": measured["perimeter_px"],
            "aspect_ratio": measured["aspect_ratio"],
            "circularity": measured["circularity"],
            "solidity": measured["solidity"],
            "extent": measured["extent"],
            "roughness": measured["roughness"],
            "bending_energy": measured["bending_energy"],
            "radial_cv": measured["radial_cv"],
            "length_model_px": round(tr["length"], 3),
            "width_model_px": round(tr["width"], 3),
            "angle_deg": round(tr["angle"], 3),
            "vx_px_s": round(tr["vx"], 3),
            "vy_px_s": round(tr["vy"], 3),
        })

    texture_sigma = float(config.get("cell_texture", 0.0))
    if texture_sigma > 0 and np.any(labels):
        grain = cv2.GaussianBlur(
            rng.normal(0.0, texture_sigma, image.shape).astype(np.float32), (0, 0), 1.2)
        body = labels > 0
        image[body] += grain[body]
    halo = float(config.get("phase_halo", 0.0))
    if halo > 0 and np.any(labels):
        body = (labels > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ring = cv2.dilate(body, kernel) - body
        image[ring > 0] += halo

    image = np.clip(image, 0, 255)
    alpha = float(rng.uniform(config["contrast_min"], config["contrast_max"]))
    beta = float(rng.uniform(-config["brightness_shift"], config["brightness_shift"]))
    image = (image - 127.5) * alpha + 127.5 + beta
    image = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < config["blur_probability"]:
        k = int(rng.choice([3, 5]))
        image = cv2.GaussianBlur(image, (k, k), sigmaX=float(rng.uniform(0.4, 1.3)))
    image = np.clip(image.astype(np.float32) + rng.normal(0, config["sensor_noise"], image.shape),
                    0, 255).astype(np.uint8)

    ids = [int(v) for v in np.unique(labels) if v != 0]
    masks = np.stack([(labels == oid).astype(np.uint8) for oid in ids], axis=0) if ids else np.zeros((0, height, width), dtype=np.uint8)
    return image, labels, masks, objects


def generate_dataset(output_dir, sequences=3, frames=20, width=512, height=384,
                     fps=5.0, objects_per_frame=18, seed=42):
    output = Path(output_dir)
    images_dir, masks_dir, videos_dir = output / "images", output / "masks", output / "videos"
    for directory in (images_dir, masks_dir, videos_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = {
        "background_mean": 188, "background_noise": 4.0, "background_texture": 5.0,
        "vignette": 0.10, "cell_intensity_min": -62, "cell_intensity_max": -34,
        "cell_texture": 7.0, "phase_halo": 26.0,
        "contrast_min": 0.85, "contrast_max": 1.15, "brightness_shift": 10,
        "blur_probability": 0.45, "sensor_noise": 3.5,
    }
    rng = np.random.default_rng(seed)
    all_objects, frame_rows = [], []

    for seq in range(sequences):
        sequence_id = f"seq_{seq:03d}"
        tracks = make_tracks(rng, width, height, objects_per_frame, sequence_id)
        video_path = videos_dir / f"{sequence_id}.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height), isColor=False)
        if not writer.isOpened():
            raise RuntimeError(f"Не удалось создать видео: {video_path}")
        try:
            for frame_idx in range(frames):
                image, labels, masks, objects = build_scene(
                    rng, width, height, tracks, frame_idx, fps, sequence_id, config
                )
                stem = f"{sequence_id}_frame_{frame_idx:05d}"
                cv2.imwrite(str(images_dir / f"{stem}.png"), image)
                cv2.imwrite(str(masks_dir / f"{stem}_labels.png"), labels)
                ids = [int(v) for v in np.unique(labels) if v != 0]
                np.savez_compressed(masks_dir / f"{stem}_instances.npz",
                                    instance_ids=np.array(ids, dtype=np.uint16), masks=masks)
                writer.write(image)
                all_objects.extend(objects)
                frame_rows.append({
                    "sequence_id": sequence_id, "frame_index": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 6),
                    "image_path": f"images/{stem}.png",
                    "label_mask_path": f"masks/{stem}_labels.png",
                    "instance_masks_path": f"masks/{stem}_instances.npz",
                    "height": height, "width": width, "fps": fps,
                })
        finally:
            writer.release()

    def write_csv(path, rows):
        if rows:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    write_csv(output / "frames.csv", frame_rows)
    write_csv(output / "objects.csv", all_objects)
    manifest = {
        "description": "Synthetic microscopy-like dataset; not real laboratory data.",
        "seed": seed, "sequences": sequences, "frames_per_sequence": frames,
        "width": width, "height": height, "fps": fps,
        "objects_per_sequence": objects_per_frame,
        "annotations": {
            "frames.csv": "frame index, timestamp, file paths and dimensions",
            "objects.csv": "one row per object per frame; x/y are the mask centroid, "
                           "length/width/area/perimeter are measured on the true mask",
            "masks/*_labels.png": "instance label map: 0=background, positive values=object IDs",
            "masks/*_instances.npz": "instance_ids and masks (N,H,W), one visible binary mask per ID",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--sequences", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--objects", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output, args.sequences, args.frames, args.width, args.height,
                     args.fps, args.objects, args.seed)


if __name__ == "__main__":
    main()
