import cv2
import numpy as np
import pytest

from microtrack.io import Calibration, iter_frames, load_frames, read_image, video_info


def _img(h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return img


@pytest.mark.parametrize("ext", [".png", ".jpg", ".tiff", ".bmp"])
def test_read_image_formats(tmp_path, ext):
    path = tmp_path / f"a{ext}"
    cv2.imwrite(str(path), _img())
    img = read_image(path)
    assert img.shape == (48, 64, 3)
    gray = read_image(path, grayscale=True)
    assert gray.shape == (48, 64)


def test_read_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        read_image(bad)


def test_single_image_is_one_frame(tmp_path):
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), _img())
    frames = load_frames(path)
    assert len(frames) == 1
    assert frames[0].index == 0 and frames[0].time_s == 0.0


def test_image_folder_step_keeps_time(tmp_path):
    for i in range(10):
        cv2.imwrite(str(tmp_path / f"f{i:03d}.png"), _img(seed=i))
    frames = load_frames(tmp_path, step=3, fps=2.0)
    assert [f.index for f in frames] == [0, 3, 6, 9]
    assert [f.time_s for f in frames] == pytest.approx([0.0, 1.5, 3.0, 4.5])


def test_multipage_tiff(tmp_path):
    pages = [np.full((20, 30), 10 * i, np.uint8) for i in range(5)]
    path = tmp_path / "stack.tiff"
    assert cv2.imwritemulti(str(path), pages)
    frames = load_frames(path, step=2, fps=5.0)
    assert [f.index for f in frames] == [0, 2, 4]
    assert frames[1].time_s == pytest.approx(0.4)
    assert int(frames[2].image.mean()) == 40


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((48, 64, 3), 10 * i, np.uint8))
    writer.release()
    return path


def test_video_info(video_path):
    info = video_info(video_path)
    assert info["fps"] == pytest.approx(10.0)
    assert info["n_frames"] == 20
    assert (info["width"], info["height"]) == (64, 48)


def test_video_all_frames(video_path):
    frames = load_frames(video_path)
    assert len(frames) == 20
    assert frames[7].time_s == pytest.approx(0.7)


def test_video_every_nth_frame_keeps_original_time(video_path):
    frames = load_frames(video_path, step=5)
    assert [f.index for f in frames] == [0, 5, 10, 15]
    assert [f.time_s for f in frames] == pytest.approx([0.0, 0.5, 1.0, 1.5])
    # содержимое соответствует исходному кадру (MJPG — с небольшими потерями)
    assert abs(float(frames[2].image.mean()) - 100) < 6


def test_video_range_and_fps_override(video_path):
    frames = load_frames(video_path, step=2, start=4, end=10, fps=20.0)
    assert [f.index for f in frames] == [4, 6, 8]
    assert [f.time_s for f in frames] == pytest.approx([0.2, 0.3, 0.4])


def test_video_grayscale(video_path):
    f = next(iter_frames(video_path, grayscale=True))
    assert f.image.ndim == 2


def test_invalid_arguments(tmp_path, video_path):
    with pytest.raises(ValueError):
        list(iter_frames(video_path, step=0))
    with pytest.raises(ValueError):
        list(iter_frames(video_path, start=5, end=2))
    with pytest.raises(FileNotFoundError):
        list(iter_frames(tmp_path / "nope.avi"))
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    with pytest.raises(ValueError):
        list(iter_frames(txt))


def test_calibration_is_attached_to_frames(video_path):
    calib = Calibration(pixel_size_um=0.5)
    f = next(iter_frames(video_path, calibration=calib))
    assert f.calibration.px_to_um(10) == pytest.approx(5.0)


def test_calibration_math():
    c = Calibration.from_reference(length_um=100.0, length_px=200.0)
    assert c.pixel_size_um == pytest.approx(0.5)
    assert c.um_to_px(10) == pytest.approx(20)
    assert c.area_px_to_um2(100) == pytest.approx(25)
    assert c.rescaled(0.5).pixel_size_um == pytest.approx(1.0)
    with pytest.raises(ValueError):
        Calibration(pixel_size_um=0)
