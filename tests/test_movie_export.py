from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import quantem.widget as qw
from quantem.gpu import movie as gpu_movie
from quantem.widget import movie


def _stack(offset: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[:10, :12].astype(np.float32)
    return np.stack([(xx + frame) + yy * 0.5 + offset for frame in range(3)]).astype(np.float32)


def test_movie_module_is_available_from_package() -> None:
    assert qw.movie.save_mp4 is movie.save_mp4


def test_save_gif_accepts_single_stack(tmp_path: Path) -> None:
    out = movie.save_gif(_stack(), tmp_path / "single.gif", fps=6, label_height=0)

    assert out.exists()
    with Image.open(out) as img:
        assert img.is_animated
        assert img.n_frames == 3
        assert img.size == (12, 10)


def test_save_gif_accepts_four_dimensional_data(tmp_path: Path) -> None:
    data = np.stack([_stack(0), _stack(100)], axis=0)

    out = movie.save_gif(data, tmp_path / "grid.gif", labels=["raw", "tv"], cols=2, gap=2, label_height=0)

    assert out.exists()
    with Image.open(out) as img:
        assert img.n_frames == 3
        assert img.size == (12 * 2 + 2, 10)


def test_save_movie_dispatches_by_suffix(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_write_mp4(frames, path, fps, *, crf=18):
        captured["frames"] = len(frames)
        captured["fps"] = fps
        captured["crf"] = crf
        path = Path(path)
        path.write_bytes(b"mp4")
        return path

    monkeypatch.setattr(gpu_movie, "_write_mp4", fake_write_mp4)

    out = movie.save_movie(_stack(), tmp_path / "movie.mp4", fps=7, crf=21, backend="cpu")

    assert out.read_bytes() == b"mp4"
    assert captured == {"frames": 3, "fps": 7.0, "crf": 21}


def test_save_mp4_accepts_rendered_pil_frames(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_write_mp4(frames, path, fps, *, crf=18):
        captured["sizes"] = [frame.size for frame in frames]
        path = Path(path)
        path.write_bytes(b"mp4")
        return path

    monkeypatch.setattr(gpu_movie, "_write_mp4", fake_write_mp4)
    frames = [Image.new("RGB", (11, 9), (idx, idx, idx)) for idx in range(2)]

    out = movie.save_mp4(frames, tmp_path / "frames.mp4", fps=12)

    assert out.read_bytes() == b"mp4"
    assert captured["sizes"] == [(11, 9), (11, 9)]


def test_save_mp4_rejects_unknown_backend(tmp_path: Path) -> None:
    try:
        movie.save_mp4(_stack(), tmp_path / "bad.mp4", backend="bad")
    except ValueError as exc:
        assert "unknown movie backend" in str(exc)
    else:
        raise AssertionError("save_mp4 should reject unavailable backends")


def test_save_mp4_auto_uses_cuda_backend_when_available(tmp_path: Path, monkeypatch) -> None:
    from quantem.gpu.movie import cuda_mp4

    captured = {}

    def fake_cuda_writer(stacks, path, **kwargs):
        captured["shape"] = stacks[0].shape
        captured["labels"] = kwargs["labels"]
        captured["qp"] = kwargs["qp"]
        path = Path(path)
        path.write_bytes(b"cuda")
        return path

    monkeypatch.setattr(cuda_mp4, "is_available", lambda: True)
    monkeypatch.setattr(cuda_mp4, "save_mp4", fake_cuda_writer)

    data = np.zeros((3, 300, 300), dtype=np.float32)
    out = movie.save_mp4(data, tmp_path / "auto.mp4", labels=["raw"], crf=22)

    assert out.read_bytes() == b"cuda"
    assert captured == {"shape": (3, 300, 300), "labels": ["raw"], "qp": 22}


def test_save_mp4_auto_falls_back_when_cuda_unavailable(tmp_path: Path, monkeypatch) -> None:
    from quantem.gpu.movie import cuda_mp4

    captured = {}

    def fake_write_mp4(frames, path, fps, *, crf=18):
        captured["frames"] = len(frames)
        path = Path(path)
        path.write_bytes(b"cpu")
        return path

    monkeypatch.setattr(cuda_mp4, "is_available", lambda: False)
    monkeypatch.setattr(gpu_movie, "_write_mp4", fake_write_mp4)

    out = movie.save_mp4(_stack(), tmp_path / "fallback.mp4")

    assert out.read_bytes() == b"cpu"
    assert captured == {"frames": 3}


def test_save_mp4_auto_skips_cuda_for_tiny_movie(tmp_path: Path, monkeypatch) -> None:
    from quantem.gpu.movie import cuda_mp4

    captured = {"cuda_calls": 0}

    def fake_cuda_writer(_stacks, _path, **_kwargs):
        captured["cuda_calls"] += 1
        raise RuntimeError("NVENC init failed for this frame size")

    def fake_write_mp4(frames, path, fps, *, crf=18):
        captured["frames"] = len(frames)
        captured["fps"] = fps
        path = Path(path)
        path.write_bytes(b"cpu")
        return path

    monkeypatch.setattr(cuda_mp4, "is_available", lambda: True)
    monkeypatch.setattr(cuda_mp4, "save_mp4", fake_cuda_writer)
    monkeypatch.setattr(gpu_movie, "_write_mp4", fake_write_mp4)

    out = movie.save_mp4(_stack(), tmp_path / "fallback_after_cuda_error.mp4", fps=11)

    assert out.read_bytes() == b"cpu"
    assert captured == {"cuda_calls": 0, "frames": 3, "fps": 11.0}


def test_save_mp4_auto_falls_back_when_cuda_writer_fails(tmp_path: Path, monkeypatch) -> None:
    from quantem.gpu.movie import cuda_mp4

    captured = {}

    def fake_cuda_writer(_stacks, _path, **_kwargs):
        captured["cuda_called"] = True
        raise RuntimeError("NVENC init failed after encoder probe")

    def fake_write_mp4(frames, path, fps, *, crf=18):
        captured["frames"] = len(frames)
        path = Path(path)
        path.write_bytes(b"cpu")
        return path

    monkeypatch.setattr(cuda_mp4, "is_available", lambda: True)
    monkeypatch.setattr(cuda_mp4, "save_mp4", fake_cuda_writer)
    monkeypatch.setattr(gpu_movie, "_write_mp4", fake_write_mp4)

    data = np.zeros((3, 300, 300), dtype=np.float32)
    out = movie.save_mp4(data, tmp_path / "fallback_after_large_cuda_error.mp4")

    assert out.read_bytes() == b"cpu"
    assert captured == {"cuda_called": True, "frames": 3}


def test_save_mp4_cuda_backend_keeps_cuda_writer_errors(tmp_path: Path, monkeypatch) -> None:
    from quantem.gpu.movie import cuda_mp4

    def fake_cuda_writer(_stacks, _path, **_kwargs):
        raise RuntimeError("NVENC init failed for this frame size")

    monkeypatch.setattr(cuda_mp4, "is_available", lambda: True)
    monkeypatch.setattr(cuda_mp4, "save_mp4", fake_cuda_writer)

    try:
        movie.save_mp4(_stack(), tmp_path / "cuda_error.mp4", backend="cuda")
    except RuntimeError as exc:
        assert "NVENC init failed" in str(exc)
    else:
        raise AssertionError("backend='cuda' should expose CUDA writer failures")


def test_cuda_backend_rejects_rendered_frames(tmp_path: Path) -> None:
    frames = [Image.new("RGB", (11, 9), (idx, idx, idx)) for idx in range(2)]

    try:
        movie.save_mp4(frames, tmp_path / "frames.mp4", backend="cuda")
    except ValueError as exc:
        assert "requires array movie data" in str(exc)
    else:
        raise AssertionError("CUDA backend should reject pre-rendered frames")
