"""Show1D behaviour against references: series normalisation and stats, axis and
UI state, the line profile, the JSONL monitor file, snapshot groups and the
Show2D hand-off, state round trip, HTML export structure."""

import base64
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from quantem.widget import Show1D, Show2D
from quantem.widget.show1d import sample_line_profile


def _traces(seed: int = 0, n_points: int = 50) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"loss": rng.random(n_points).astype(np.float32) + 1, "validation": rng.random(n_points).astype(np.float32) * 10}


def _decode(buffer: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(buffer[: 4 * int(np.prod(shape))], dtype=np.float32).reshape(shape)


def _embedded_state(path: Path) -> dict:
    match = re.search(r'<script type="application/vnd.jupyter.widget-state\+json">\n(.*?)\n</script>', path.read_text(), re.S)
    models = json.loads(match.group(1))["state"].values()
    return next(model["state"] for model in models if model["model_name"] == "AnyModel")


def _write_monitor(path: Path, n_events: int = 6) -> Path:
    rng = np.random.default_rng(1)
    for iteration in range(n_events):
        event = {
            "iteration": iteration,
            "losses": {"lambda 1": 10.0 / (iteration + 1), "lambda 10": 12.0 / (iteration + 1)},
            "metrics": {"lambda_1": {"rmse": 0.1 * (iteration + 1)}, "lambda_10": {"rmse": 0.2}},
        }
        if iteration % 2 == 1:
            snapshot_path = path.parent / "snapshots" / f"lambda_1_i{iteration:03d}.npy"
            snapshot_path.parent.mkdir(exist_ok=True)
            np.save(snapshot_path, rng.random((6, 8)).astype(np.float32))
            event["label"] = f"iter {iteration}"
            event["snapshots"] = {"reference": "snapshots/" + snapshot_path.name, "lambda_1": str(snapshot_path)}
        if iteration == n_events - 1:
            event["warnings"] = ["lambda 10 flickers"]
            event["starred"] = ["lambda_1"]
            event["hidden"] = ["lambda_10"]
            event["notes"] = {"lambda_1": "best balance"}
            event["tags"] = {"lambda_1": ["best lambda", "best lambda"]}
        Show1D.append_monitor_event(path, event)
    return path


# --- series and stats --------------------------------------------------------


def test_mapping_input_labels_bytes_and_stats_match_numpy():
    traces = _traces()
    x = np.arange(50, dtype=np.float32) * 2
    widget = Show1D(traces, x=x, title="loss")
    stacked = np.stack(list(traces.values()))
    assert (widget.n_traces, widget.n_points, widget.labels) == (2, 50, ["loss", "validation"])
    np.testing.assert_array_equal(_decode(widget.y_bytes, (2, 50)), stacked)
    np.testing.assert_array_equal(_decode(widget.x_bytes, (50,)), x)
    np.testing.assert_allclose(widget.stats_mean, stacked.mean(axis=1), rtol=1e-6)
    np.testing.assert_allclose(widget.stats_std, stacked.std(axis=1), rtol=1e-5)
    np.testing.assert_allclose(widget.stats_min, stacked.min(axis=1))
    np.testing.assert_allclose(widget.stats_max, stacked.max(axis=1))
    assert repr(widget) == "Show1D(2 traces x 50 points)"
    assert widget.review_mode == "trace" and widget.trial_sort_key == "label"
    assert [row["label"] for row in widget.trial_rankings] == ["loss", "validation"]  # trace review sorts by label
    assert widget.best_trial_label == ""  # scientific series are not ranked as losses


def test_plain_list_is_one_trace_and_nested_lists_are_many():
    assert Show1D([1.0, 2.0, 3.0]).labels == ["Data"]
    widget = Show1D([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert (widget.n_traces, widget.labels) == (3, ["Data 1", "Data 2", "Data 3"])
    with pytest.raises(ValueError, match="same length"):
        Show1D({"a": [1.0, 2.0], "b": [1.0]})
    with pytest.raises(ValueError, match="x has 2 points"):
        Show1D([1.0, 2.0, 3.0], x=[0, 1])


def test_live_append_and_extend_backfill_nan_and_rank_losses():
    monitor = Show1D.live(["training", "validation"], title="run", x_label="epoch", x_integer=True)
    assert monitor.n_points == 0 and monitor.review_mode == "optimization" and monitor.trial_sort_key == "final_loss"
    monitor.append(0, training=2.0, validation=3.0)
    monitor.append(1, training=1.5)
    monitor.extend([2, 3], training=[1.0, 0.5], validation=[2.0, 1.0], selection=[0.3, 0.2])
    data = _decode(monitor.y_bytes, (3, 4))
    assert monitor.labels == ["training", "validation", "selection"]
    np.testing.assert_array_equal(_decode(monitor.x_bytes, (4,)), [0, 1, 2, 3])
    np.testing.assert_array_equal(data[0], [2.0, 1.5, 1.0, 0.5])
    assert np.isnan(data[1, 1]) and np.isnan(data[2, :2]).all()
    assert monitor.best_trial_label == "selection"  # lowest final loss
    assert [row["rank"] for row in monitor.trial_rankings] == [1, 2, 3]
    monitor.append(training=0.4)  # x continues from the last sample
    assert monitor.n_points == 5 and _decode(monitor.x_bytes, (5,))[-1] == 4
    with pytest.raises(ValueError, match="same length"):
        monitor.extend(training=[1.0], validation=[1.0, 2.0])


def test_set_data_replaces_traces_and_keeps_display_settings():
    widget = Show1D(_traces(), log_scale=True, plot_height_px=400)
    widget.set_data(np.ones((3, 4), dtype=np.float32), labels=["a", "b", "c"])
    assert (widget.n_traces, widget.n_points, widget.labels) == (3, 4, ["a", "b", "c"])
    assert widget.log_scale and widget.plot_height_px == 400


# --- axis and UI state -------------------------------------------------------


def test_ui_mode_presets_explicit_toggles_and_unknown_kwargs():
    widget = Show1D(_traces(), log_scale=True, ui_mode="minimal", show_legend=True)
    assert widget.log_scale and widget.show_legend
    assert not widget.show_title and not widget.show_grid and not widget.show_controls
    report = Show1D(_traces(), ui_mode="report", show_review=True)
    assert not report.show_controls and report.show_review and report.show_title
    with pytest.raises(TypeError, match="unexpected keyword argument 'show_denoise'"):
        Show1D(_traces(), show_denoise=True)


def test_validators_clamp_sizes_and_reject_bad_values():
    widget = Show1D(_traces(), plot_height_px=10, snapshot_columns=20, snapshot_histogram_width=5)
    assert (widget.plot_height_px, widget.snapshot_columns, widget.snapshot_histogram_width) == (220, 8, 110)
    widget.snapshot_fps = 99
    widget.snapshot_real_space_zoom = 100
    assert (widget.snapshot_fps, widget.snapshot_real_space_zoom) == (24, 32.0)
    widget.snapshot_contrast_range = [0.5, 2]
    assert widget.snapshot_contrast_range == [0.5, 2.0]
    for name, value in [("image_cmap", "nope"), ("snapshot_contrast_range", [2, 1]), ("review_mode", "loss"), ("trial_sort_key", "speed"), ("pixel_size", -1)]:
        with pytest.raises(ValueError):
            setattr(widget, name, value)


# --- line profile ------------------------------------------------------------


def test_sample_line_profile_is_exact_on_a_bilinear_plane():
    rows, cols = np.mgrid[0:20, 0:30]
    plane = (2.0 * rows + 3.0 * cols).astype(np.float32)
    values = sample_line_profile(plane, ((2, 1), (8, 25)))
    n_samples = int(np.ceil(np.hypot(6, 24))) + 1
    expected = 2.0 * np.linspace(2, 8, n_samples) + 3.0 * np.linspace(1, 25, n_samples)
    np.testing.assert_allclose(values, expected, atol=1e-5)
    wide = sample_line_profile(plane, ((5, 2), (5, 12)), profile_width=3)  # parallel lines average to the centre line on a plane
    np.testing.assert_allclose(wide, 2.0 * 5 + 3.0 * np.arange(2, 13), atol=1e-5)
    with pytest.raises(ValueError, match="row0, col0"):
        sample_line_profile(plane, ((0, 0),))


def test_from_image_embeds_profile_context_with_physical_distance():
    image = np.random.default_rng(3).random((16, 24)).astype(np.float32)
    widget = Show1D.from_image(image, line=((2, 1), (2, 9)), profile_width=3, sampling=0.5, x_unit="nm", title="profile", image_cmap="gray")
    assert (widget.labels, widget.x_label, widget.x_unit, widget.title) == (["profile"], "distance", "nm", "profile")
    assert (widget.pixel_size, widget.pixel_unit, widget.image_cmap) == (0.5, "nm", "gray")
    assert (widget.profile_image_height, widget.profile_image_width, widget.profile_width) == (16, 24, 3)
    assert widget.profile_line == [{"row": 2.0, "col": 1.0}, {"row": 2.0, "col": 9.0}]
    np.testing.assert_array_equal(_decode(widget.profile_image_bytes, image.shape), image)
    x = _decode(widget.x_bytes, (widget.n_points,))
    assert x[0] == 0 and x[-1] == pytest.approx(8 * 0.5) and widget.n_points == 9
    np.testing.assert_allclose(_decode(widget.y_bytes, (9,)), image[1:4, 1:10].mean(axis=0), atol=1e-6)


# --- snapshots, monitor file -------------------------------------------------


def test_snapshot_groups_selection_and_stars():
    widget = Show1D.live(["loss"])
    widget.append(0, loss=1.0).append(5, loss=0.5)
    widget.snapshot(0, object=np.ones((4, 5), np.float32), probe=np.zeros((3, 3), np.float32))
    widget.snapshot(5, label="final", object=np.full((4, 5), 2.0, np.float32))
    assert (widget.n_snapshots, widget.n_snapshot_groups) == (3, 2)
    assert widget.snapshot_group_labels == ["iter 0", "final"] and widget.snapshot_group_indices == [0, 0, 1]
    assert (widget.snapshot_height, widget.snapshot_width, widget.snapshot_heights, widget.snapshot_widths) == (4, 5, [4, 3, 4], [5, 3, 5])
    stack = _decode(widget.snapshot_bytes, (3, 4, 5))
    assert np.isnan(stack[1, 3, :]).all() and stack[1, :3, :3].sum() == 0  # smaller probe is NaN padded
    assert (widget.selected_snapshot_group_idx, widget.selected_snapshot_idx) == (1, 2)
    widget.goto_snapshot(0)
    assert (widget.selected_snapshot_group_idx, widget.selected_snapshot_idx) == (0, 0)
    widget.star_snapshot_group("final").star_snapshot_group(0).star_snapshot_group(0)
    assert widget.bookmarked_snapshot_groups == [0, 1]
    with pytest.raises(ValueError, match="unknown snapshot group"):
        widget.star_snapshot_group("missing")
    with pytest.raises(ValueError, match="must be 2D"):
        widget.snapshot(6, object=np.ones(4, np.float32))


def test_monitor_file_rebuilds_losses_snapshots_metrics_and_review_state(tmp_path):
    monitor_path = _write_monitor(tmp_path / "run" / "show1d_monitor.jsonl")
    widget = Show1D.from_monitor_file(tmp_path / "run", title="reopened", log_scale=False, snapshot_columns=2)
    assert (widget.labels, widget.n_points, widget.n_snapshot_groups, widget.n_snapshots) == (["lambda 1", "lambda 10"], 6, 3, 6)
    assert widget.snapshot_group_labels == ["iter 1", "iter 3", "iter 5"]
    assert widget.snapshot_image_labels[:2] == ["lambda_1", "reference"]  # the monitor file stores sorted keys
    assert widget.title == "reopened" and not widget.log_scale and widget.snapshot_columns == 2
    np.testing.assert_allclose(_decode(widget.y_bytes, (2, 6))[0], 10.0 / np.arange(1, 7))
    assert widget.hidden_snapshot_image_labels == ["lambda_10"] and widget.starred_snapshot_image_labels == ["lambda_1"]
    assert widget.trial_notes == {"lambda_1": "best balance"} and widget.trial_tags == {"lambda_1": ["best lambda"]}
    by_label = {row["label"]: row for row in widget.trial_rankings}
    assert by_label["lambda 1"]["rmse"] == pytest.approx(0.6) and by_label["lambda 10"]["hidden"]  # metrics merge per trial key
    assert by_label["lambda 1"]["starred"] and by_label["lambda 1"]["tags"] == ["best lambda"] and widget.best_trial_label == "lambda 1"
    assert any(alert["kind"] == "monitor_warning" and alert["message"] == "lambda 10 flickers" for alert in widget.trial_alerts)
    first_line = json.loads(monitor_path.read_text().splitlines()[0])
    assert first_line["iteration"] == 0 and list(first_line) == sorted(first_line)


def test_watch_run_tails_complete_lines_only(tmp_path):
    monitor_path = _write_monitor(tmp_path / "show1d_monitor.jsonl", n_events=2)
    widget = Show1D.watch_run(monitor_path, refresh_s=0.2)
    try:
        assert widget.n_points == 2
        with monitor_path.open("a") as handle:
            handle.write(json.dumps({"iteration": 2, "losses": {"lambda 1": 1.0, "lambda 10": 2.0}}) + "\n")
            handle.write('{"iteration": 3, "losses": {"lambda 1": 0.5')  # writer still flushing
        deadline = time.monotonic() + 3
        while widget.n_points < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert widget.n_points == 3  # the half-written line waits
        with monitor_path.open("a") as handle:
            handle.write(', "lambda 10": 1.5}}\n')
        deadline = time.monotonic() + 3
        while widget.n_points < 4 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert widget.n_points == 4
        np.testing.assert_array_equal(_decode(widget.x_bytes, (4,)), [0, 1, 2, 3])
    finally:
        widget.stop_monitor()


def test_to_show2d_carries_group_images_colormap_scale_bar_and_stars():
    widget = Show1D({"frame-by-frame": [3.0, 2.0], "lambda 10": [4.0, 5.0]}, image_cmap="magma", sampling=0.25, units="nm", show_snapshot_fft=True)
    widget.snapshot(0, reference=np.zeros((4, 5), np.float32), **{"frame-by-frame": np.ones((4, 5), np.float32), "lambda_10": np.full((3, 4), 2.0, np.float32)})
    widget.starred_snapshot_image_labels = ["frame-by-frame"]
    widget.hidden_snapshot_image_labels = ["lambda 10"]
    gallery = widget.to_show2d()
    assert isinstance(gallery, Show2D) and gallery.labels == ["reference", "frame-by-frame"]  # hidden trial excluded
    assert (gallery.cmap, gallery.pixel_size, gallery.pixel_unit, gallery.show_fft) == ("magma", 0.25, "nm", True)
    assert gallery.starred == [0, 1] and gallery.link_zoom and gallery.title == "Show1D · iter 0"  # Show2D keeps per-panel star flags
    np.testing.assert_array_equal(gallery._data[1], np.ones((4, 5), np.float32))
    chosen = widget.to_show2d(group="iter 0", images=["lambda_10", 0], title="picked")
    assert chosen.labels == ["lambda_10", "reference"] and chosen.title == "picked"
    with pytest.raises(ValueError, match="not in the selected group"):
        widget.to_show2d(images=["probe"])
    widget.handoff_request = json.dumps({"mode": "show2d", "id": "req", "group": 0, "images": ["reference"]})
    assert widget.prepared_view_widget is widget.prepared_view and widget.prepared_view.labels == ["reference"]
    assert widget.handoff_status == "Showing 2D with 1 panel"
    widget.handoff_request = json.dumps({"mode": "clear", "id": "clear"})
    assert widget.prepared_view is None and widget.prepared_view_widget is None and widget.handoff_status == ""


# --- state round trip and exports --------------------------------------------


def test_state_round_trip_through_save_file_and_state_kwarg(tmp_path):
    widget = Show1D(_traces(), title="saved", log_scale=True, plot_height_px=500, snapshot_columns=3)
    widget.snapshot(1, object=np.ones((4, 4), np.float32))
    widget.star_snapshot_group(0)
    widget.plot_width_px = 700
    widget.side_panel_width_px = 420
    widget.snapshot_contrast_preset = "1-99"
    widget.trial_notes = {"loss": "note"}
    widget.save(tmp_path / "view.json")
    envelope = json.loads((tmp_path / "view.json").read_text())
    assert envelope["widget_name"] == "Show1D" and "snapshot_bytes" not in envelope["state"]
    restored = Show1D(_traces(), state=tmp_path / "view.json")
    for key in ("title", "log_scale", "plot_height_px", "plot_width_px", "side_panel_width_px", "snapshot_columns", "snapshot_contrast_preset", "bookmarked_snapshot_groups", "trial_notes"):
        assert getattr(restored, key) == getattr(widget, key), key
    assert restored.state_dict() == widget.state_dict()
    with pytest.raises(ValueError, match="cannot load into"):
        Show1D(_traces(), state={"widget_name": "Show2D", "state": {}})
    # a plain widget keeps the trace bytes but not the image buffers in the saved notebook
    saved = widget.get_state()
    assert "y_bytes" in saved and "snapshot_bytes" not in saved and saved["_static_fallback_jpeg"]
    assert "snapshot_bytes" in Show1D(_traces(), save_state=True).get_state()


def test_export_html_embeds_state_and_downsamples_only_linked_images(tmp_path):
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    widget = Show1D.from_image(image, line=((0, 0), (0, 7)), sampling=0.5, x_unit="nm", title="Export me")
    widget.snapshot(0, object=image, probe=image[:4, :6])
    widget.snapshot_real_space_center = [4, 4]
    widget.snapshot_profile_line = [{"row": 1, "col": 1}, {"row": 7, "col": 7}]
    out = widget.export_html(tmp_path / "full.html")
    full = _embedded_state(out)
    assert full["_anywidget_id"] == "quantem.widget.show1d.widget.Show1D" and full["_export_light"] and not full["export_enabled"]
    assert (full["n_snapshots"], full["snapshot_height"], full["pixel_size"], full["labels"]) == (2, 8, 0.5, ["profile"])
    assert widget.export_status.startswith("Exported full.html") and widget.export_status.endswith("full float32)")
    small = _embedded_state(widget.export_html(tmp_path / "small.html", downsample=2))
    assert (small["snapshot_height"], small["snapshot_width"], small["snapshot_heights"], small["snapshot_widths"]) == (4, 4, [4, 2], [4, 3])
    assert (small["pixel_size"], small["profile_image_height"], small["snapshot_real_space_center"]) == (1.0, 4, [2.0, 2.0])
    assert small["profile_line"] == [{"row": 0.0, "col": 0.0}, {"row": 0.0, "col": 3.5}] and small["snapshot_profile_line"][1] == {"row": 3.5, "col": 3.5}
    assert small["n_points"] == full["n_points"]  # trace samples are never binned
    assert small["x_unit"] == "nm" and small["title"] == "Export me"
    default_name = widget._default_html_export_path(downsample=4).name
    assert default_name == "export_me_1x8_4xdownsampled_single.html"
    with pytest.raises(NotImplementedError, match="folder"):
        widget.export_html(tmp_path / "x.html", mode="folder")
    with pytest.raises(NotImplementedError, match="uint8"):
        widget.export_html(tmp_path / "x.html", encoding="uint8")
    with pytest.raises(ValueError, match="one of"):
        widget.export_html(tmp_path / "x.html", downsample=3)
    widget.export_request = json.dumps({"mode": "single", "encoding": "full", "downsample": 2, "id": "req", "filename": "dl.html", "download": True})
    assert widget.export_payload_id == "req" and widget.export_filename == "dl.html" and b"snapshot_heights" in widget.export_payload
    assert widget.export_status.startswith("Ready dl.html") and "2x downsampled images" in widget.export_status


def test_static_preview_and_save_image(tmp_path):
    widget = Show1D(_traces(), title="preview", x_label="iteration", y_label="loss", log_scale=True)
    preview = Image.open(io.BytesIO(base64.b64decode(widget._static_png_b64())))
    assert preview.size == (512, 317) and preview.mode == "RGBA"
    assert Show1D.live(["loss"])._static_png_b64() is None  # nothing to draw yet
    png = widget.save_image(tmp_path / "traces.png")
    pdf = widget.save_image(tmp_path / "traces.pdf")
    assert Image.open(png).size[0] > 600 and pdf.read_bytes().startswith(b"%PDF")
    with pytest.raises(ValueError, match="Unsupported format"):
        widget.save_image(tmp_path / "traces.svg")
