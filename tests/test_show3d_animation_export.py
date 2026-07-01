from __future__ import annotations

import json
import pathlib

import numpy as np
from PIL import Image

from quantem.widget import Show3D
from quantem.widget.render import gif as gif_utils


def _stack(offset: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[:12, :14].astype(np.float32)
    frames = []
    for i in range(4):
        frames.append(np.sin((xx + i) / 3.0) + np.cos((yy - i) / 4.0) + offset)
    return np.stack(frames).astype(np.float32)


def test_show3d_save_gif_single_panel(tmp_path: pathlib.Path) -> None:
    widget = Show3D(
        _stack(),
        cmap="viridis",
        fps=8,
        show_controls=False,
        show_scale_bar=False,
        show_panel_titles=False,
    )

    out = widget.save_gif(tmp_path / "single.gif", quality="high", playback="forward")

    assert out.exists()
    with Image.open(out) as img:
        assert img.is_animated
        assert img.n_frames == 4
        assert img.size == (14, 12)


def test_show3d_save_gif_multi_panel_grid_respects_hidden_panels(tmp_path: pathlib.Path) -> None:
    widget = Show3D(
        _stack(0),
        _stack(1000),
        _stack(2000),
        panel_titles=["raw", "denoised", "residual"],
        hidden_panels=["denoised"],
        max_cols=2,
        panel_gap=3,
        cmap="magma",
        fps=8,
        show_controls=False,
        show_scale_bar=False,
        show_panel_titles=False,
    )

    out = widget.save_gif(tmp_path / "multi.gif", quality="high", playback="forward")

    assert out.exists()
    with Image.open(out) as img:
        assert img.is_animated
        assert img.n_frames == 4
        # Two visible 14x12 panels in one row with a 3 px gap.
        assert img.size == (14 * 2 + 3, 12)


def test_show3d_animation_grid_uses_dark_background_by_default() -> None:
    widget = Show3D(
        np.zeros((2, 4, 4), dtype=np.float32),
        np.ones((2, 4, 4), dtype=np.float32),
        max_cols=2,
        panel_gap=2,
        show_controls=False,
        show_scale_bar=False,
        show_panel_titles=False,
    )

    frame = widget._render_animation_frames(
        quality="high",
        playback="forward",
        show_frame_labels=False,
        background="dark",
    )[0]

    assert frame.getpixel((4, 0)) == (12, 12, 12)


def test_show3d_save_gif_bounce_order_omits_duplicate_endpoints(tmp_path: pathlib.Path) -> None:
    widget = Show3D(
        _stack(),
        show_controls=False,
        show_scale_bar=False,
        show_panel_titles=False,
    )

    out = widget.save_gif(tmp_path / "bounce.gif", quality="high", playback="bounce")

    with Image.open(out) as img:
        assert img.n_frames == 6


def test_show3d_save_mp4_uses_panel_only_renderer(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_write_mp4(frames, path, fps, *, crf=18):
        captured["n_frames"] = len(frames)
        captured["size"] = frames[0].size
        captured["fps"] = fps
        captured["crf"] = crf
        path = pathlib.Path(path)
        path.write_bytes(b"mp4")
        return path

    monkeypatch.setattr(gif_utils, "write_mp4", fake_write_mp4)
    widget = Show3D(
        _stack(0),
        _stack(1000),
        panel_titles=["raw", "denoised"],
        max_cols=1,
        panel_gap=4,
        fps=12,
        show_controls=False,
        show_scale_bar=False,
        show_panel_titles=False,
    )

    out = widget.save_mp4(tmp_path / "multi.mp4", quality="high", crf=21)

    assert out.read_bytes() == b"mp4"
    assert captured == {
        "n_frames": 4,
        "size": (14, 12 * 2 + 4),
        "fps": 12.0,
        "crf": 21,
    }


def test_show3d_animation_frame_labels_are_opt_in(monkeypatch) -> None:
    captured: list[list[str] | None] = []

    def fake_compose_panel_grid(images, **kwargs):
        captured.append(kwargs["frame_labels"])
        return images[0]

    monkeypatch.setattr(gif_utils, "compose_panel_grid", fake_compose_panel_grid)
    widget = Show3D(
        _stack(),
        _stack(1000),
        panel_titles=["raw", "denoised"],
        panel_frame_labels=[
            ["raw frame 1", "raw frame 2", "raw frame 3", "raw frame 4"],
            ["den frame 1", "den frame 2", "den frame 3", "den frame 4"],
        ],
        show_controls=False,
        show_scale_bar=False,
    )

    widget._render_animation_frames(
        quality="high",
        playback="forward",
        show_frame_labels=False,
        background="dark",
    )
    assert captured[-1] is None

    widget._render_animation_frames(
        quality="high",
        playback="forward",
        show_frame_labels=True,
        background="dark",
    )
    assert captured[-1] == ["raw frame 4", "den frame 4"]


def test_show3d_frontend_gif_export_request_creates_payload() -> None:
    widget = Show3D(
        _stack(),
        show_controls=False,
        show_scale_bar=False,
        show_panel_titles=False,
    )
    filename = "frontend.gif"

    widget.export_request = json.dumps({
        "id": "gif-request",
        "mode": "gif",
        "quality": "low",
        "filename": filename,
        "download": True,
    })

    assert widget.export_payload_id == "gif-request"
    assert widget.export_filename == filename
    assert widget.export_payload.startswith(b"GIF")
    assert widget.export_status.startswith(f"Ready {filename}")


def test_show3d_frontend_mp4_export_request_creates_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_save_mp4(self, path, **_kwargs):
        captured.update(_kwargs)
        path = pathlib.Path(path)
        path.write_bytes(b"fake mp4")
        return path

    monkeypatch.setattr(Show3D, "save_mp4", fake_save_mp4)
    widget = Show3D(
        _stack(),
        show_controls=False,
        show_scale_bar=False,
        show_panel_titles=False,
    )
    filename = "frontend.mp4"

    widget.export_request = json.dumps({
        "id": "mp4-request",
        "mode": "mp4",
        "quality": "low",
        "filename": filename,
        "download": True,
    })

    assert widget.export_payload_id == "mp4-request"
    assert widget.export_filename == filename
    assert widget.export_payload == b"fake mp4"
    assert widget.export_status.startswith(f"Ready {filename}")
    assert captured["quality"] == "low"
    assert captured["crf"] == 24


def test_show3d_animation_export_rejects_unknown_quality() -> None:
    widget = Show3D(
        _stack(),
        show_controls=False,
        show_scale_bar=False,
        show_panel_titles=False,
    )

    widget.export_request = json.dumps({
        "id": "bad-quality",
        "mode": "gif",
        "quality": "giant",
        "filename": "bad.gif",
        "download": True,
    })

    assert widget.export_status.startswith("Export failed:")
    assert "animation quality" in widget.export_status
