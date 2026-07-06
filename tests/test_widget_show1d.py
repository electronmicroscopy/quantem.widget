from __future__ import annotations

import json
import pathlib

import numpy as np

from quantem.widget.show1d import Show1D, sample_line_profile


def test_show1d_live_append_adds_traces_and_exports_csv(tmp_path: pathlib.Path) -> None:
    widget = Show1D.live(["loss"], title="Adam loss")

    widget.append(0, loss=3.0)
    widget.append(1, loss=1.5, step_size=0.01)
    widget.add_marker(1, label="checkpoint")

    assert widget.n_traces == 2
    assert widget.n_points == 2
    assert widget.labels == ["loss", "step_size"]
    assert np.isnan(widget._data[1, 0])
    assert widget.stats_min[0] == 1.5
    assert widget.markers == [{"x": 1.0, "label": "checkpoint", "kind": "checkpoint"}]

    csv_path = widget.export_csv(tmp_path / "loss.csv")
    rows = csv_path.read_text().splitlines()
    assert rows[0] == "x,loss,step_size"
    assert rows[1].startswith("0.0,3.0,")
    assert rows[2] == "1.0,1.5,0.01"


def test_show1d_from_loss_runs_flattens_nested_loss_families() -> None:
    widget = Show1D.from_loss_runs(
        {
            "lambda 1": {
                "data": [3.0, 2.0, 1.0],
                "temporal": [0.5, 0.25, 0.1],
            },
            "lambda 10": {
                "data": [4.0, 2.5, 1.5],
                "temporal": [0.2, 0.15, 0.08],
            },
        },
        x=[0, 10, 20],
        losses=["data", "temporal"],
        label_template="{run} / {loss}",
        title="lambda sweep",
    )

    assert widget.title == "lambda sweep"
    assert widget.labels == [
        "lambda 1 / data",
        "lambda 1 / temporal",
        "lambda 10 / data",
        "lambda 10 / temporal",
    ]
    np.testing.assert_allclose(widget._x, [0, 10, 20])
    np.testing.assert_allclose(widget._data[2], [4.0, 2.5, 1.5])


def test_show1d_from_loss_runs_supports_bare_single_loss_traces() -> None:
    widget = Show1D.from_loss_runs(
        {"lambda 1": [3.0, 2.0], "lambda 3": [4.0, 1.0]},
        losses=["final"],
        label_template="{run} {loss}",
    )

    assert widget.labels == ["lambda 1 final", "lambda 3 final"]
    np.testing.assert_allclose(widget._data, [[3.0, 2.0], [4.0, 1.0]])

    with np.testing.assert_raises(ValueError):
        Show1D.from_loss_runs({"lambda 1": [3.0, 2.0]}, losses=["data", "regularizer"])


def test_show1d_from_loss_runs_reports_missing_nested_loss() -> None:
    with np.testing.assert_raises(ValueError):
        Show1D.from_loss_runs({"lambda 1": {"data": [1.0, 0.5]}}, losses=["temporal"])


def test_sample_line_profile_uses_row_col_coordinates() -> None:
    image = np.arange(20, dtype=np.float32).reshape(4, 5)

    profile = sample_line_profile(image, ((0, 0), (0, 4)))

    np.testing.assert_allclose(profile, [0, 1, 2, 3, 4])


def test_show1d_from_image_embeds_profile_context() -> None:
    image = np.arange(20, dtype=np.float32).reshape(4, 5)

    widget = Show1D.from_image(image, line=((1, 0), (1, 4)), profile_width=1)

    assert widget.n_traces == 1
    assert widget.n_points == 5
    assert widget.profile_image_height == 4
    assert widget.profile_image_width == 5
    assert widget.profile_line == [{"row": 1.0, "col": 0.0}, {"row": 1.0, "col": 4.0}]
    np.testing.assert_allclose(widget._data[0], [5, 6, 7, 8, 9])


def test_show1d_snapshots_keep_iteration_labels() -> None:
    widget = Show1D.live(["loss"])

    widget.snapshot(0, np.ones((3, 4), dtype=np.float32), label="start")
    widget.snapshot(10, np.full((3, 4), 2.0, dtype=np.float32), label="checkpoint")

    assert widget.n_snapshots == 2
    assert widget.snapshot_height == 3
    assert widget.snapshot_width == 4
    assert widget.snapshot_iterations == [0.0, 10.0]
    assert widget.snapshot_labels == ["start", "checkpoint"]
    assert widget.snapshot_group_indices == [0, 1]
    assert widget.snapshot_group_iterations == [0.0, 10.0]
    assert widget.snapshot_group_labels == ["start", "checkpoint"]
    assert widget.n_snapshot_groups == 2
    assert widget.selected_snapshot_idx == 1
    assert widget.selected_snapshot_group_idx == 1


def test_show1d_display_public_api_state_and_validation() -> None:
    widget = Show1D(
        np.arange(4, dtype=np.float32),
        plot_height_px=500,
        side_panel_width_px=420,
        image_cmap="viridis",
        snapshot_contrast_preset="1-99",
        snapshot_thumbnail_size=80,
        snapshot_columns=3,
        sampling=(2.0, 0.5),
        units=("nm", "A"),
        show_scale_bar=True,
        show_snapshot_histogram=False,
        show_snapshot_fft=True,
        snapshot_fft_window=False,
        snapshot_fft_cmap="inferno",
        starred_snapshot_image_labels=["lambda_10"],
        hidden_snapshot_image_labels=["lambda_300"],
        prefer_webgpu=False,
    )

    assert widget.colors[0] == "#0072B2"
    assert widget.plot_height_px == 500
    assert widget.side_panel_width_px == 420
    assert widget.image_cmap == "viridis"
    assert widget.snapshot_contrast_preset == "1-99"
    assert widget.snapshot_thumbnail_size == 80
    assert widget.snapshot_columns == 3
    assert widget.pixel_size == 0.5
    assert widget.pixel_unit == "A"
    assert widget.scale_bar_visible is True
    assert widget.show_snapshot_histogram is False
    assert widget.show_snapshot_fft is True
    assert widget.snapshot_fft_window is False
    assert widget.snapshot_fft_cmap == "inferno"
    assert widget.starred_snapshot_image_labels == ["lambda_10"]
    assert widget.hidden_snapshot_image_labels == ["lambda_300"]
    assert widget.prefer_webgpu is False

    state = widget.state_dict()
    assert state["plot_height_px"] == 500
    assert state["side_panel_width_px"] == 420
    assert state["image_cmap"] == "viridis"
    assert state["snapshot_contrast_preset"] == "1-99"
    assert state["snapshot_columns"] == 3
    assert state["pixel_size"] == 0.5
    assert state["pixel_unit"] == "A"
    assert state["scale_bar_visible"] is True
    assert state["show_snapshot_fft"] is True
    assert state["snapshot_fft_window"] is False
    assert state["snapshot_fft_cmap"] == "inferno"
    assert state["starred_snapshot_image_labels"] == ["lambda_10"]
    assert state["hidden_snapshot_image_labels"] == ["lambda_300"]
    assert state["prefer_webgpu"] is False

    widget.snapshot_columns = 99
    assert widget.snapshot_columns == 4

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), image_cmap="not-a-cmap")

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), snapshot_contrast_preset="middle-ish")

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), show_scale_bar=True, scale_bar_visible=False)

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), sampling=-1.0)


def test_show1d_trial_review_helpers_persist_state() -> None:
    widget = Show1D({"lambda 10": [3.0, 2.0], "lambda 300": [4.0, 4.5]})

    assert widget.star_trial("lambda_10") is widget
    widget.star_trial("lambda 10")
    widget.hide_trial("lambda_300")

    assert widget.starred_snapshot_image_labels == ["lambda_10"]
    assert widget.hidden_snapshot_image_labels == ["lambda_300"]

    widget.hide_trial("lambda 10")
    assert widget.starred_snapshot_image_labels == []
    assert widget.hidden_snapshot_image_labels == ["lambda_300", "lambda 10"]

    state = widget.state_dict()
    restored = Show1D(np.zeros((2, 2), dtype=np.float32), state=state)
    assert restored.hidden_snapshot_image_labels == ["lambda_300", "lambda 10"]
    assert restored.show_trial("lambda 300") is restored
    assert restored.hidden_snapshot_image_labels == ["lambda 10"]
    restored.unstar_trial("lambda_10").show_all_trials().clear_starred_trials()
    assert restored.hidden_snapshot_image_labels == []
    assert restored.starred_snapshot_image_labels == []


def test_show1d_ranking_notes_tags_alerts_and_summary_export(tmp_path: pathlib.Path) -> None:
    widget = Show1D(
        {
            "lambda 1": [5.0, 3.0, 1.0],
            "lambda 3": [np.nan, 4.0, 2.0],
            "lambda 10": [5.0, 6.0, 7.0],
        },
        trial_sort_key="final_loss",
    )

    assert widget.best_trial_label == "lambda 1"
    assert [row["label"] for row in widget.trial_rankings] == ["lambda 1", "lambda 3", "lambda 10"]

    widget.set_trial_note("lambda 1", "best lambda").tag_trial("lambda 1", "best")
    widget.star_best_trial().hide_worst_trials()

    assert widget.starred_snapshot_image_labels == ["lambda 1"]
    assert widget.hidden_snapshot_image_labels == ["lambda 10"]
    assert widget.trial_notes == {"lambda 1": "best lambda"}
    assert widget.trial_tags == {"lambda 1": ["best"]}
    assert any(alert["kind"] == "nonfinite" and alert["label"] == "lambda 3" for alert in widget.trial_alerts)
    assert any(alert["kind"] == "worse_final" and alert["label"] == "lambda 10" for alert in widget.trial_alerts)

    out = widget.export_run_summary(tmp_path / "summary.json")
    summary = json.loads(out.read_text())
    assert summary["best_trial"] == "lambda 1"
    assert summary["starred_trials"] == ["lambda 1"]
    assert summary["hidden_trials"] == ["lambda 10"]
    assert summary["trial_notes"] == {"lambda 1": "best lambda"}


def test_show1d_review_state_exports_strict_json_without_nan(tmp_path: pathlib.Path) -> None:
    widget = Show1D(
        {
            "frame-by-frame": [3.0, 2.0, 1.0],
            "lambda 10": [4.0, 3.0, 2.0],
        },
        title="review export",
    )

    assert widget.trial_rankings[0]["lambda"] is None
    assert widget.run_summary["rankings"][0]["lambda"] is None

    summary_path = widget.export_run_summary(tmp_path / "summary.json")
    summary_text = summary_path.read_text()
    assert "NaN" not in summary_text
    json.loads(summary_text)

    html_path = widget.export_html(tmp_path / "show1d_review.html")
    html = html_path.read_text()
    state_json = html.split('<script type="application/vnd.jupyter.widget-state+json">', 1)[1].split("</script>", 1)[0]
    json.loads(
        state_json,
        parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(f"non-finite JSON constant {value}")),
    )


def test_show1d_ui_mode_presets_and_overrides() -> None:
    data = np.arange(4, dtype=np.float32)

    presentation = Show1D(data, ui_mode="presentation")
    assert presentation.show_title is True
    assert presentation.show_controls is True
    assert presentation.controls_collapsed is True
    assert presentation.show_stats is False
    assert presentation.show_legend is True
    assert presentation.show_grid is True

    report = Show1D(data, ui_mode="report")
    assert report.show_title is True
    assert report.show_controls is False
    assert report.controls_collapsed is False
    assert report.show_stats is False
    assert report.show_legend is True
    assert report.show_grid is True

    minimal = Show1D(data, ui_mode="minimal")
    assert minimal.show_title is False
    assert minimal.show_controls is False
    assert minimal.controls_collapsed is False
    assert minimal.show_stats is False
    assert minimal.show_legend is False
    assert minimal.show_grid is False

    override = Show1D(
        data,
        ui_mode="minimal",
        show_title=True,
        show_controls=True,
        controls_collapsed=True,
        show_stats=True,
        show_legend=True,
        show_grid=True,
    )
    assert override.show_title is True
    assert override.show_controls is True
    assert override.controls_collapsed is True
    assert override.show_stats is True
    assert override.show_legend is True
    assert override.show_grid is True
    assert override.expand_controls() is override
    assert override.controls_collapsed is False
    assert override.collapse_controls() is override
    assert override.controls_collapsed is True
    assert override.toggle_controls() is override
    assert override.controls_collapsed is False


def test_show1d_snapshot_group_accepts_multiple_mixed_size_images() -> None:
    widget = Show1D.live(["loss"])

    widget.snapshot(
        5,
        object=np.ones((3, 4), dtype=np.float32),
        probe=np.full((2, 2), 2.0, dtype=np.float32),
    )

    assert widget.n_snapshots == 2
    assert widget.n_snapshot_groups == 1
    assert widget.snapshot_height == 3
    assert widget.snapshot_width == 4
    assert widget.snapshot_heights == [3, 2]
    assert widget.snapshot_widths == [4, 2]
    assert widget.snapshot_group_indices == [0, 0]
    assert widget.snapshot_image_labels == ["object", "probe"]
    assert widget.snapshot_group_iterations == [5.0]
    assert widget.selected_snapshot_idx == 0
    assert widget.selected_snapshot_group_idx == 0
    widget.snapshot_thumbnail_size = 72
    assert widget.state_dict()["snapshot_thumbnail_size"] == 72

    widget.play()
    assert widget.snapshot_playing is True
    widget.pause()
    assert widget.snapshot_playing is False
    widget.goto_snapshot(10)
    assert widget.selected_snapshot_group_idx == 0
    assert widget.selected_snapshot_idx == 0


def test_show1d_from_joint_time_report_loads_metrics_and_snapshots(tmp_path: pathlib.Path) -> None:
    summary = {
        "data": "ducky",
        "num_frames": 4,
        "num_iters": 12,
        "electrons_per_pattern": 1000,
        "joint_init": "frame_average",
        "loss_type": "poisson",
        "metrics": {
            "joint_lambda_0.1": {
                "rmse_per_frame_mask": 0.4,
                "rmse_time_average_mask": 0.3,
                "temporal_flicker_mask": 0.2,
                "mean_phase_std_mask": 0.1,
                "elapsed_s": 2.0,
            },
            "frame_by_frame": {
                "rmse_per_frame_mask": 0.8,
                "rmse_time_average_mask": 0.7,
                "temporal_flicker_mask": 0.6,
                "mean_phase_std_mask": 0.5,
                "elapsed_s": 1.0,
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    arrays_path = tmp_path / "reconstructions.npz"
    np.savez(
        arrays_path,
        reference_phase=np.zeros((3, 4), dtype=np.float32),
        frame_by_frame=np.ones((4, 3, 4), dtype=np.float32),
        **{"joint_lambda_0.1": np.full((4, 3, 4), 2.0, dtype=np.float32)},
    )

    widget = Show1D.from_joint_time_report(summary_path)

    assert widget.method_labels == ["frame_by_frame", "joint_lambda_0.1"]
    assert widget.labels[:2] == ["per-frame RMSE", "average RMSE"]
    assert widget.report_metadata["data"] == "ducky"
    assert widget.n_snapshots == 3
    assert widget.snapshot_labels == ["clean reference", "frame_by_frame average", "joint_lambda_0.1 average"]


def test_show1d_from_joint_time_report_frame_by_frame_loads_full_sweep(tmp_path: pathlib.Path) -> None:
    summary = {
        "data": "ducky",
        "num_frames": 4,
        "num_iters": 12,
        "metrics": {
            "frame_by_frame": {
                "final_losses": [4.0, 3.0, 2.0, 1.0],
                "rmse_per_frame_mask": 0.8,
            },
            "joint_lambda_0.1": {
                "final_losses": [5.0, 4.0, 2.5, 1.5],
                "rmse_per_frame_mask": 0.4,
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    np.savez(
        tmp_path / "reconstructions.npz",
        reference_phase=np.zeros((4, 6), dtype=np.float32),
        frame_by_frame=np.ones((4, 4, 6), dtype=np.float32),
        **{"joint_lambda_0.1": np.full((4, 4, 6), 2.0, dtype=np.float32)},
    )

    widget = Show1D.from_joint_time_report(
        summary_path,
        frame_by_frame=True,
        snapshot_downsample=2,
    )

    assert widget.labels == ["frame-by-frame", "lambda 0.1"]
    np.testing.assert_allclose(widget._data, [[4, 3, 2, 1], [5, 4, 2.5, 1.5]])
    assert widget.x_label == "frame"
    assert widget.method_labels == ["0", "1", "2", "3"]
    assert widget.report_metadata["frame_by_frame"] is True
    assert widget.report_metadata["metrics_by_trial"]["lambda 0.1"]["rmse_per_frame_mask"] == 0.4
    assert widget.n_snapshot_groups == 4
    assert widget.n_snapshots == 12
    assert widget.snapshot_image_labels[:3] == ["reference", "frame-by-frame", "lambda_0.1"]
    assert widget.snapshot_height == 2
    assert widget.snapshot_width == 3


def test_show1d_monitor_file_reloads_losses_snapshots_and_review_state(tmp_path: pathlib.Path) -> None:
    image0 = tmp_path / "lambda1_0.npy"
    image1 = tmp_path / "lambda1_1.npy"
    np.save(image0, np.ones((3, 3), dtype=np.float32))
    np.save(image1, np.full((3, 3), 2.0, dtype=np.float32))
    monitor = tmp_path / "show1d_monitor.jsonl"

    Show1D.append_monitor_event(
        monitor,
        {
            "iteration": 0,
            "losses": {"lambda 1": 3.0, "lambda 10": 5.0},
            "snapshots": {"lambda_1": image0.name},
            "warnings": ["started from coarse object"],
        },
    )
    Show1D.append_monitor_event(
        monitor,
        {
            "iteration": 1,
            "losses": {"lambda 1": 1.0, "lambda 10": 6.0},
            "snapshots": {"lambda_1": image1.name},
            "starred": ["lambda 1"],
            "notes": {"lambda 1": "best so far"},
            "tags": {"lambda 1": ["best"]},
        },
    )

    widget = Show1D.from_monitor_file(monitor)

    assert widget.monitor_path == str(monitor)
    assert widget.labels == ["lambda 1", "lambda 10"]
    assert widget.n_points == 2
    assert widget.n_snapshot_groups == 2
    assert widget.n_snapshots == 2
    assert widget.starred_snapshot_image_labels == ["lambda 1"]
    assert widget.trial_notes == {"lambda 1": "best so far"}
    assert widget.trial_tags == {"lambda 1": ["best"]}
    assert any(alert["kind"] == "monitor_warning" for alert in widget.trial_alerts)

    widget.hide_trial("lambda 10")
    Show1D.append_monitor_event(monitor, {"iteration": 2, "losses": {"lambda 1": 0.5, "lambda 10": 7.0}})
    widget.refresh_monitor()

    assert widget.n_points == 3
    assert widget.hidden_snapshot_image_labels == ["lambda 10"]
    assert widget.trial_notes == {"lambda 1": "best so far"}
