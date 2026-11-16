import numpy as np
import pytest

from microtrack.io import Calibration, Frame
from microtrack.preprocess import (
    ClaheConfig,
    DenoiseConfig,
    PreprocessConfig,
    adjust_sharpness,
    apply_clahe,
    apply_median,
    denoise,
    ensure_uint8,
    preprocess,
    preprocess_frame,
)


def _low_contrast(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(100, 130, (64, 64), dtype=np.uint8)


def _noisy_flat(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((64, 64), 128, np.float32) + rng.normal(0, 15, (64, 64))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_default_config_is_noop_for_gray():
    img = _low_contrast()
    out, calib = preprocess(img)
    assert np.array_equal(out, img)
    assert calib.pixel_size_um == 1.0


def test_grayscale_conversion():
    img = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    out, _ = preprocess(img, PreprocessConfig(grayscale=True))
    assert out.ndim == 2
    out, _ = preprocess(img, PreprocessConfig(grayscale=False))
    assert out.ndim == 3


def test_clahe_increases_contrast():
    img = _low_contrast()
    out = apply_clahe(img, clip_limit=4.0, tile_grid=4)
    assert out.std() > img.std() * 1.5
    assert out.dtype == np.uint8 and out.shape == img.shape


def test_clahe_color_keeps_shape():
    img = np.random.default_rng(1).integers(80, 120, (32, 32, 3), dtype=np.uint8)
    out = apply_clahe(img)
    assert out.shape == img.shape


def test_median_removes_salt_and_pepper():
    img = np.full((40, 40), 100, np.uint8)
    img[10, 10] = 255
    img[20, 20] = 0
    out = apply_median(img, 3)
    assert out[10, 10] == 100 and out[20, 20] == 100
    with pytest.raises(ValueError):
        apply_median(img, 4)


@pytest.mark.parametrize("method", ["gaussian", "bilateral", "nlm"])
def test_denoise_reduces_noise(method):
    img = _noisy_flat()
    out = denoise(img, method, strength=2.0)
    assert out.std() < img.std()


def test_sharpness_factor():
    img = np.full((40, 40), 50, np.uint8)
    img[:, 20:] = 200
    assert np.array_equal(adjust_sharpness(img, 1.0), img)
    sharper = adjust_sharpness(img, 2.0)
    softer = adjust_sharpness(img, 0.0)
    # резкость выше: на границе появляется "недолёт"/"перелёт"
    assert sharper[:, 19].mean() < 50 and sharper[:, 20].mean() > 200
    # резкость ниже: край размыт
    assert 50 < softer[:, 19].mean() and softer[:, 20].mean() < 200


def test_ensure_uint8():
    u16 = np.array([[0, 65535]], np.uint16)
    assert ensure_uint8(u16).tolist() == [[0, 255]]
    f = np.array([[0.0, 0.5, 1.0]], np.float32)
    assert ensure_uint8(f).tolist() == [[0, 128, 255]]
    flat = np.full((3, 3), 7.0, np.float32)
    assert ensure_uint8(flat).max() == 0


def test_rescale_updates_calibration():
    img = _low_contrast()
    out, calib = preprocess(
        img, PreprocessConfig(scale=0.5), Calibration(pixel_size_um=0.2)
    )
    assert out.shape == (32, 32)
    assert calib.pixel_size_um == pytest.approx(0.4)


def test_full_chain_and_order():
    img = _noisy_flat()
    cfg = PreprocessConfig(
        denoise=DenoiseConfig("gaussian", 1.0),
        median_ksize=3,
        clahe=ClaheConfig(2.0, 4),
        sharpness=1.5,
    )
    out, _ = preprocess(img, cfg)
    assert out.shape == img.shape and out.dtype == np.uint8
    reordered = PreprocessConfig(
        median_ksize=3, clahe=ClaheConfig(), order=("clahe", "median")
    )
    assert preprocess(img, reordered)[0].shape == img.shape


def test_from_dict_and_validation():
    cfg = PreprocessConfig.from_dict(
        {"clahe": {"clip_limit": 3.0}, "denoise": "bilateral", "median_ksize": 5}
    )
    assert cfg.clahe.clip_limit == 3.0
    assert cfg.denoise.method == "bilateral"
    with pytest.raises(ValueError):
        PreprocessConfig.from_dict({"median_ksize": 4})
    with pytest.raises(ValueError):
        PreprocessConfig.from_dict({"unknown": 1})
    with pytest.raises(ValueError):
        PreprocessConfig.from_dict({"order": ["nope"]})


def test_preprocess_frame_keeps_time_and_index():
    frame = Frame(_low_contrast(), index=12, time_s=1.2, calibration=Calibration(0.3))
    out = preprocess_frame(frame, PreprocessConfig(scale=2.0, clahe=ClaheConfig()))
    assert out.index == 12 and out.time_s == 1.2
    assert out.image.shape == (128, 128)
    assert out.calibration.pixel_size_um == pytest.approx(0.15)
    assert frame.image.shape == (64, 64)  # исходный кадр не изменён
