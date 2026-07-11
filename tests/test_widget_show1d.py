from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from quantem.widget.show1d import Show1D, sample_line_profile


def _wait_until(predicate, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition was not reached before timeout")


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


def test_show1d_detect_jumps_uses_raw_points_and_preserves_manual_markers() -> None:
    data = np.asarray(
        [
            [0, 0, 0, 0, 50, 0, 0, 0, 0, 0, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 10, 10, 10, -60, -60, -60, -60],
        ],
        dtype=np.float32,
    )
    widget = Show1D(
        data,
        x=np.arange(data.shape[1]),
        labels=["spike", "drop"],
        y_unit="nm",
    )
    widget.add_marker(6, label="manual", kind="group")

    events = widget.detect_jumps(threshold=8, min_separation=1)

    assert [(event["trace_label"], event["point_index"]) for event in events] == [
        ("spike", 4),
        ("drop", 8),
    ]
    assert [event["direction"] for event in events] == ["increase", "decrease"]
    assert [event["delta"] for event in events] == [50.0, -70.0]
    assert widget.markers[0] == {
        "x": 6.0,
        "label": "manual",
        "kind": "group",
    }
    assert [marker["source"] for marker in widget.markers[1:]] == [
        "auto-jump",
        "auto-jump",
    ]
    assert widget.report_metadata["jump_detection"] == {
        "method": "first-difference-median-mad",
        "threshold": 8.0,
        "min_abs_change": 0.0,
        "min_separation": 1,
        "uses_raw_points": True,
        "event_count": 2,
    }

    widget.clear_detected_jumps()

    assert widget.markers == [{"x": 6.0, "label": "manual", "kind": "group"}]
    assert "detected_jumps" not in widget.report_metadata


def test_show1d_detect_jumps_validates_configuration() -> None:
    widget = Show1D(np.arange(8, dtype=np.float32))

    with np.testing.assert_raises(ValueError):
        widget.detect_jumps(threshold=0)
    with np.testing.assert_raises(ValueError):
        widget.detect_jumps(min_abs_change=-1)
    with np.testing.assert_raises(ValueError):
        widget.detect_jumps(min_separation=-1)


def test_show1d_live_extend_batches_reactive_trace_updates() -> None:
    widget = Show1D.live(["loss"], title="Adam loss")

    widget.extend([0, 1, 2], loss=[3.0, 2.0, 1.0], reg=[0.3, 0.2, 0.1])
    widget.append_many([3, 4], loss=[0.8, 0.6])

    assert widget.labels == ["loss", "reg"]
    assert widget.n_points == 5
    np.testing.assert_allclose(widget._x, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(widget._data[0], [3.0, 2.0, 1.0, 0.8, 0.6])
    np.testing.assert_allclose(widget._data[1, :3], [0.3, 0.2, 0.1])
    assert np.isnan(widget._data[1, 3])
    assert np.isnan(widget._data[1, 4])

    with np.testing.assert_raises(ValueError):
        widget.extend([5, 6], loss=[0.4])


def test_show1d_plain_numeric_list_is_one_trace_but_nested_lists_remain_multiple() -> None:
    single = Show1D([1.0, 2.0, 3.0])
    multiple = Show1D([[1.0, 2.0], [3.0, 4.0]])

    assert single.n_traces == 1
    assert single.n_points == 3
    assert single.labels == ["Data"]
    np.testing.assert_allclose(single._data, [[1.0, 2.0, 3.0]])

    assert multiple.n_traces == 2
    assert multiple.n_points == 2
    assert multiple.labels == ["Data 1", "Data 2"]
    np.testing.assert_allclose(multiple._data, [[1.0, 2.0], [3.0, 4.0]])


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


def test_show1d_from_image_preserves_physical_profile_coordinates() -> None:
    image = np.arange(20, dtype=np.float32).reshape(4, 5)

    widget = Show1D.from_image(
        image,
        line=((1, 0), (1, 4)),
        sampling=0.25,
        x_unit="nm",
    )

    np.testing.assert_allclose(widget._x, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(widget._data[0], [5, 6, 7, 8, 9])
    assert widget.pixel_size == 0.25
    assert widget.pixel_unit == "nm"
    assert widget.profile_line == [{"row": 1.0, "col": 0.0}, {"row": 1.0, "col": 4.0}]


def test_show1d_profile_distance_uses_geometric_diagonal_and_fractional_length() -> None:
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    widget = Show1D.from_image(
        image,
        line=((0.0, 0.0), (3.0, 4.0)),
        sampling=0.2,
        x_unit="nm",
    )

    assert widget.n_points == 6
    np.testing.assert_allclose(widget._x, np.linspace(0.0, 1.0, 6))

    fractional_line = ((0.5, 1.25), (3.0, 4.75))
    widget.set_profile_image(image, line=fractional_line)
    expected_length_nm = np.hypot(2.5, 3.5) * 0.2
    np.testing.assert_allclose(
        widget._x,
        np.linspace(0.0, expected_length_nm, widget.n_points),
    )
    assert widget._x[-1] == np.float32(expected_length_nm)
    assert widget.profile_line == [
        {"row": 0.5, "col": 1.25},
        {"row": 3.0, "col": 4.75},
    ]


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


def test_show1d_snapshot_group_stars_persist_and_validate() -> None:
    widget = Show1D.live(["loss"], bookmarked_snapshot_groups=[3, 3, 1])
    widget.snapshot(0, image=np.ones((3, 4), dtype=np.float32), label="start")
    widget.snapshot(10, image=np.full((3, 4), 2.0, dtype=np.float32), label="checkpoint")

    assert widget.bookmarked_snapshot_groups == [1, 3]
    assert widget.star_snapshot_group("start") is widget
    assert widget.bookmarked_snapshot_groups == [0, 1, 3]
    assert widget.toggle_snapshot_group_star("checkpoint") is widget
    assert widget.bookmarked_snapshot_groups == [0, 3]
    assert widget.toggle_snapshot_group_star(1) is widget
    assert widget.bookmarked_snapshot_groups == [0, 1, 3]
    assert widget.unstar_snapshot_group() is widget
    assert widget.bookmarked_snapshot_groups == [0, 3]

    restored = Show1D.live(["loss"])
    restored.snapshot(0, image=np.ones((3, 4), dtype=np.float32), label="start")
    restored.snapshot(10, image=np.full((3, 4), 2.0, dtype=np.float32), label="checkpoint")
    restored.load_state_dict(widget.state_dict())
    assert restored.bookmarked_snapshot_groups == [0, 3]
    assert restored.clear_snapshot_group_stars() is restored
    assert restored.bookmarked_snapshot_groups == []

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), bookmarked_snapshot_groups=[-1])
    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), bookmarked_snapshot_groups=["bad"])


def test_show1d_frontend_stars_current_snapshot_group_contract() -> None:
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "js" / "show1d" / "index.tsx"
    ).read_text(encoding="utf-8")

    assert 'useModelState<number[]>("bookmarked_snapshot_groups")' in source
    assert "const toggleCurrentSnapshotGroupBookmark" in source
    assert "aria-pressed={currentSnapshotGroupBookmarked}" in source
    assert 'currentSnapshotGroupBookmarked ? "Unstar" : "Star"' in source
    assert 'bgcolor: "#ffc107"' in source


def test_show1d_frontend_resizes_snapshot_tiles_like_show2d_contract() -> None:
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "js" / "show1d" / "index.tsx"
    ).read_text(encoding="utf-8")

    assert 'data-testid={`show1d-snapshot-panel-resize-${imageIdx}`}' in source
    assert 'title="Resize all snapshot panels"' in source
    assert "deltaTile * start.columns" in source
    assert "window.requestAnimationFrame" in source
    assert "const maximumResizableWidth" in source
    assert "setSidePanelWidthPx(Math.round(clampValue(" in source
    assert "nextWidth > sidePanelWidth" not in source
    assert 'data-testid="show1d-snapshot-grid-resize"' not in source


def test_show1d_display_public_api_state_and_validation() -> None:
    widget = Show1D(
        np.arange(4, dtype=np.float32),
        plot_height_px=500,
        side_panel_width_px=420,
        image_cmap="viridis",
        snapshot_contrast_preset="1-99",
        snapshot_contrast_range=(0.1, 0.9),
        snapshot_thumbnail_size=80,
        snapshot_columns=3,
        snapshot_overlay_position="bottom-left",
        snapshot_fft_layout="below",
        snapshot_real_space_zoom=2.5,
        snapshot_real_space_center=(12.0, 18.5),
        snapshot_fft_zoom=3.0,
        snapshot_fft_center=(64.0, 65.5),
        sampling=(2.0, 0.5),
        units=("nm", "A"),
        show_scale_bar=True,
        show_snapshot_histogram=False,
        show_snapshot_fft=True,
        snapshot_fft_window=False,
        snapshot_fft_cmap="inferno",
        show_snapshot_profile=True,
        snapshot_profile_line=((2, 3), (8, 9)),
        snapshot_profile_height=88,
        snapshot_histogram_width=320,
        snapshot_histogram_height=50,
        snapshot_loop=False,
        snapshot_bounce=True,
        starred_snapshot_image_labels=["lambda_10"],
        hidden_snapshot_image_labels=["lambda_300"],
        show_trial_notes=True,
        prefer_webgpu=False,
    )

    assert widget.colors[0] == "#0072B2"
    assert widget.show_stats is False
    assert widget.show_review is False
    assert widget.plot_height_px == 500
    assert widget.side_panel_width_px == 420
    assert widget.image_cmap == "viridis"
    assert widget.snapshot_contrast_preset == "1-99"
    assert widget.snapshot_contrast_range == [0.1, 0.9]
    assert widget.snapshot_thumbnail_size == 80
    assert widget.snapshot_columns == 3
    assert widget.snapshot_overlay_position == "bottom-left"
    assert widget.snapshot_fft_layout == "below"
    assert widget.snapshot_real_space_zoom == 2.5
    assert widget.snapshot_real_space_center == [12.0, 18.5]
    assert widget.snapshot_fft_zoom == 3.0
    assert widget.snapshot_fft_center == [64.0, 65.5]
    assert widget.pixel_size == 0.5
    assert widget.pixel_unit == "A"
    assert widget.scale_bar_visible is True
    assert widget.show_snapshot_histogram is False
    assert widget.show_snapshot_fft is True
    assert widget.snapshot_fft_window is False
    assert widget.snapshot_fft_cmap == "inferno"
    assert widget.show_snapshot_profile is True
    assert widget.snapshot_profile_line == [{"row": 2.0, "col": 3.0}, {"row": 8.0, "col": 9.0}]
    assert widget.snapshot_profile_height == 88
    assert widget.snapshot_histogram_width == 320
    assert widget.snapshot_histogram_height == 50
    assert widget.snapshot_loop is False
    assert widget.snapshot_bounce is True
    assert widget.starred_snapshot_image_labels == ["lambda_10"]
    assert widget.hidden_snapshot_image_labels == ["lambda_300"]
    assert widget.show_trial_notes is True
    assert widget.prefer_webgpu is False
    widget.snapshot_fps = 5
    assert widget.snapshot_fps == 5
    widget.load_state_dict({
        "snapshot_fps": 4.25,
        "snapshot_fft_zoom": 99,
        "snapshot_fft_center": (7.0, 8.0),
        "snapshot_profile_line": [{"row": 1.5, "col": 2.5}],
        "snapshot_profile_height": 999,
        "snapshot_histogram_width": 1000,
        "snapshot_histogram_height": 20,
        "snapshot_loop": True,
        "snapshot_bounce": False,
    })
    assert widget.snapshot_fps == 4
    assert widget.snapshot_fft_zoom == 32.0
    assert widget.snapshot_fft_center == [7.0, 8.0]
    assert widget.snapshot_profile_line == [{"row": 1.5, "col": 2.5}]
    assert widget.snapshot_profile_height == 220
    assert widget.snapshot_histogram_width == 640
    assert widget.snapshot_histogram_height == 36
    assert widget.snapshot_loop is True
    assert widget.snapshot_bounce is False

    state = widget.state_dict()
    assert state["show_stats"] is False
    assert state["show_review"] is False
    assert state["plot_height_px"] == 500
    assert state["side_panel_width_px"] == 420
    assert state["image_cmap"] == "viridis"
    assert state["snapshot_contrast_preset"] == "1-99"
    assert state["snapshot_contrast_range"] == [0.1, 0.9]
    assert state["snapshot_columns"] == 3
    assert state["snapshot_overlay_position"] == "bottom-left"
    assert state["snapshot_fft_layout"] == "below"
    assert state["snapshot_real_space_zoom"] == 2.5
    assert state["snapshot_real_space_center"] == [12.0, 18.5]
    assert state["snapshot_fft_zoom"] == 32.0
    assert state["snapshot_fft_center"] == [7.0, 8.0]
    assert state["show_snapshot_profile"] is True
    assert state["snapshot_profile_line"] == [{"row": 1.5, "col": 2.5}]
    assert state["snapshot_profile_height"] == 220
    assert state["snapshot_histogram_width"] == 640
    assert state["snapshot_histogram_height"] == 36
    assert state["snapshot_fps"] == 4
    assert state["snapshot_loop"] is True
    assert state["snapshot_bounce"] is False
    assert state["pixel_size"] == 0.5
    assert state["pixel_unit"] == "A"
    assert state["scale_bar_visible"] is True
    assert state["show_snapshot_fft"] is True
    assert state["snapshot_fft_window"] is False
    assert state["snapshot_fft_cmap"] == "inferno"
    assert state["starred_snapshot_image_labels"] == ["lambda_10"]
    assert state["hidden_snapshot_image_labels"] == ["lambda_300"]
    assert state["show_trial_notes"] is True
    assert state["prefer_webgpu"] is False

    widget.snapshot_columns = 99
    assert widget.snapshot_columns == 8

    widget.snapshot_columns = -1
    assert widget.snapshot_columns == 0

    wide_widget = Show1D(
        np.arange(4, dtype=np.float32),
        side_panel_width_px=5000,
        snapshot_panel_width_px=5000,
    )
    assert wide_widget.side_panel_width_px == 4096
    assert wide_widget.snapshot_panel_width_px == 4096
    wide_widget.side_panel_width_px = 2200
    wide_widget.snapshot_panel_width_px = 2600
    assert wide_widget.side_panel_width_px == 2200
    assert wide_widget.snapshot_panel_width_px == 2600

    tall_widget = Show1D(np.arange(4, dtype=np.float32), plot_height_px=2000)
    assert tall_widget.plot_height_px == 960
    tall_widget.plot_height_px = 900
    assert tall_widget.plot_height_px == 900

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), image_cmap="not-a-cmap")

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), snapshot_contrast_preset="middle-ish")

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), snapshot_contrast_range=(1.0,))

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), snapshot_profile_line=((0, 0), (1, 1), (2, 2)))

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), snapshot_contrast_range=(1.0, 0.0))

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), snapshot_overlay_position="center")

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), snapshot_fft_layout="right")

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), snapshot_real_space_zoom=float("nan"))

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), snapshot_fft_center=(1.0,))

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), show_scale_bar=True, scale_bar_visible=False)

    with np.testing.assert_raises(ValueError):
        Show1D(np.arange(4, dtype=np.float32), sampling=-1.0)


def test_show1d_real_space_colormap_defaults_to_viridis_and_preserves_override() -> None:
    assert Show1D(np.arange(4, dtype=np.float32)).image_cmap == "viridis"
    assert Show1D(np.arange(4, dtype=np.float32), image_cmap="cividis").image_cmap == "cividis"


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


def test_show1d_scientific_trace_review_does_not_rank_values_as_losses() -> None:
    widget = Show1D(
        {
            "3.6Mx C10": [20.0, -80.0],
            "5.1Mx C10": [30.0, -120.0],
        },
        y_label="C10 defocus",
        y_unit="nm",
    )

    assert widget.review_mode == "trace"
    assert widget.trial_sort_key == "label"
    assert widget.best_trial_label == ""
    assert widget.run_summary["best_trial"] == ""
    assert widget.run_summary["review_mode"] == "trace"
    assert [row["label"] for row in widget.trial_rankings] == [
        "3.6Mx C10",
        "5.1Mx C10",
    ]
    assert all(row["score"] is None for row in widget.trial_rankings)
    assert all(row["final_loss"] is None for row in widget.trial_rankings)
    assert not any(
        alert["kind"] in {"nonfinite", "worse_final", "flat_loss"}
        for alert in widget.trial_alerts
    )

    with np.testing.assert_raises_regex(ValueError, "review_mode='optimization'"):
        widget.set_trial_sort("final_loss")
    with np.testing.assert_raises_regex(ValueError, r"star_trial\(label\)"):
        widget.star_best_trial()
    with np.testing.assert_raises_regex(ValueError, r"hide_trial\(label\)"):
        widget.hide_worst_trials()


def test_show1d_explicit_loss_sort_selects_optimization_review() -> None:
    widget = Show1D(
        {"lambda 1": [4.0, 1.0], "lambda 10": [3.0, 2.0]},
        trial_sort_key="final_loss",
    )

    assert widget.review_mode == "optimization"
    assert widget.best_trial_label == "lambda 1"
    assert [row["label"] for row in widget.trial_rankings] == [
        "lambda 1",
        "lambda 10",
    ]


def test_show1d_best_and_worst_helpers_ignore_descending_presentation() -> None:
    widget = Show1D(
        {
            "best": [4.0, 1.0],
            "middle": [4.0, 2.0],
            "worst": [4.0, 3.0],
        },
        trial_sort_key="final_loss",
        trial_sort_descending=True,
    )

    assert [row["label"] for row in widget.trial_rankings] == [
        "worst",
        "middle",
        "best",
    ]
    assert widget.best_trial_label == "best"
    assert widget.run_summary["best_trial"] == "best"

    widget.star_best_trial().hide_worst_trials()

    assert widget.starred_snapshot_image_labels == ["best"]
    assert widget.hidden_snapshot_image_labels == ["worst"]


def test_show1d_review_mode_and_sort_are_validated_at_all_boundaries() -> None:
    with np.testing.assert_raises_regex(ValueError, "review_mode='trace'"):
        Show1D(
            np.arange(4, dtype=np.float32),
            review_mode="trace",
            trial_sort_key="final_loss",
        )

    trace = Show1D(np.arange(4, dtype=np.float32))
    with np.testing.assert_raises_regex(ValueError, "review_mode='trace'"):
        trace.trial_sort_key = "rmse"

    optimization = Show1D(
        np.arange(4, dtype=np.float32),
        review_mode="optimization",
    )
    with np.testing.assert_raises_regex(ValueError, "review_mode='trace'"):
        optimization.review_mode = "trace"

    contradictory = optimization.state_dict()
    contradictory["review_mode"] = "trace"
    contradictory["trial_sort_key"] = "final_loss"
    with np.testing.assert_raises_regex(ValueError, "review_mode='trace'"):
        optimization.load_state_dict(contradictory)
    assert optimization.review_mode == "optimization"
    assert optimization.trial_sort_key == "final_loss"


def test_show1d_legacy_state_migrates_metrics_and_live_review_modes() -> None:
    legacy_metric = Show1D(
        [[0.4, 0.2], [2.0, 1.5]],
        labels=["RMSE", "elapsed"],
        title="Joint-Time Ptychography Metrics",
        y_label="metric",
    ).state_dict()
    legacy_metric.pop("review_mode")
    legacy_metric["trial_sort_key"] = "final_loss"
    legacy_metric["report_metadata"] = {"frame_by_frame": False}

    metric = Show1D(np.zeros((2, 2), dtype=np.float32))
    metric.load_state_dict(legacy_metric)
    assert metric.review_mode == "trace"
    assert metric.trial_sort_key == "label"
    assert metric.best_trial_label == ""

    legacy_live = Show1D.live(["loss"]).state_dict()
    legacy_live.pop("review_mode")
    live = Show1D(np.zeros((1, 1), dtype=np.float32))
    live.load_state_dict(legacy_live)
    assert live.review_mode == "optimization"
    assert live.trial_sort_key == "final_loss"


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

    default = Show1D(data)
    assert default.show_stats is False
    assert default.show_review is False

    presentation = Show1D(data, ui_mode="presentation")
    assert presentation.show_title is True
    assert presentation.show_controls is True
    assert presentation.controls_collapsed is True
    assert presentation.show_stats is False
    assert presentation.show_review is False
    assert presentation.show_legend is True
    assert presentation.show_grid is True

    report = Show1D(data, ui_mode="report")
    assert report.show_title is True
    assert report.show_controls is False
    assert report.controls_collapsed is False
    assert report.show_stats is False
    assert report.show_review is False
    assert report.show_legend is True
    assert report.show_grid is True

    minimal = Show1D(data, ui_mode="minimal")
    assert minimal.show_title is False
    assert minimal.show_controls is False
    assert minimal.controls_collapsed is False
    assert minimal.show_stats is False
    assert minimal.show_review is False
    assert minimal.show_legend is False
    assert minimal.show_grid is False

    override = Show1D(
        data,
        ui_mode="minimal",
        show_title=True,
        show_controls=True,
        controls_collapsed=True,
        show_stats=True,
        show_review=True,
        show_legend=True,
        show_grid=True,
    )
    assert override.show_title is True
    assert override.show_controls is True
    assert override.controls_collapsed is True
    assert override.show_stats is True
    assert override.show_review is True
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

    widget.play(loop=False, bounce=True)
    assert widget.snapshot_playing is True
    assert widget.snapshot_loop is False
    assert widget.snapshot_bounce is True
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
    assert widget.n_snapshots == 2
    assert widget.hidden_snapshot_image_labels == ["lambda 10"]
    assert widget.trial_notes == {"lambda 1": "best so far"}
    assert widget.report_metadata["monitor_events"] == 3

    partial_offset = widget._monitor_offset
    with monitor.open("a", encoding="utf-8") as handle:
        handle.write('{"iteration": 3, "losses": {"lambda 1": 0.25')
    widget.refresh_monitor()
    assert widget.n_points == 3
    assert widget._monitor_offset == partial_offset

    with monitor.open("a", encoding="utf-8") as handle:
        handle.write(', "lambda 10": 8.0}}\n')
    widget.refresh_monitor()
    assert widget.n_points == 4
    np.testing.assert_allclose(widget._data[:, -1], [0.25, 8.0])


def test_show1d_from_example_ducky_uses_public_dataset_loader(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    import quantem.widget.datasets as tutorial_datasets

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    image = snapshots / "lambda1_frame0.npy"
    np.save(image, np.ones((4, 4), dtype=np.float32))
    monitor = tmp_path / "show1d_monitor.jsonl"
    Show1D.append_monitor_event(
        monitor,
        {
            "iteration": 0,
            "losses": {"lambda 1": 2.0},
            "snapshots": {"lambda_1": image.relative_to(tmp_path)},
        },
    )
    calls = []

    def fake_show1d_ducky(**kwargs):
        calls.append(kwargs)
        return tmp_path

    monkeypatch.setattr(tutorial_datasets, "show1d_ducky", fake_show1d_ducky)

    widget = Show1D.from_example("ducky", size="small", show_snapshot_fft=False)

    assert calls == [
        {
            "size": "small",
            "cache_dir": None,
            "revision": None,
            "force_download": False,
            "verbose": False,
        }
    ]
    assert widget.title == "Real ducky joint iterative ptychography"
    assert widget.x_label == "frame"
    assert widget.y_label == "final loss"
    assert widget.log_scale is False
    assert widget.snapshot_columns == 4
    assert widget.show_snapshot_fft is False
    assert widget.n_snapshots == 1


def test_show1d_html_export_configures_anywidget_requirejs(tmp_path: pathlib.Path) -> None:
    widget = Show1D([[1.0, 0.5]], labels=["loss"], title="loss export")

    html = widget.export_html(tmp_path / "show1d.html").read_text(encoding="utf-8")

    assert 'id="quantem-widget-anywidget-requirejs"' in html
    assert "anywidget@0.11.0/dist/index.min" in html


def test_show1d_html_export_downsamples_only_linked_images(tmp_path: pathlib.Path) -> None:
    trace = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float32)
    x = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    widget = Show1D(
        trace,
        x=x,
        sampling=0.25,
        units="nm",
        snapshot_real_space_center=(4.0, 6.0),
        snapshot_fft_center=(2.0, 4.0),
        snapshot_profile_line=((0.0, 0.0), (4.0, 6.0)),
    )
    widget.snapshot(0, image=image, label="phase")
    widget.set_profile_image(image, line=None)
    widget.profile_line = [{"row": 0.0, "col": 0.0}, {"row": 4.0, "col": 6.0}]

    clone = widget._clone_for_html_export(downsample=2)
    try:
        np.testing.assert_allclose(clone._data, widget._data)
        np.testing.assert_allclose(clone._x, widget._x)
        assert clone.snapshot_heights == [3]
        assert clone.snapshot_widths == [4]
        assert clone.profile_image_height == 3
        assert clone.profile_image_width == 4
        assert clone.pixel_size == 0.5
        assert clone.snapshot_real_space_center == [2.0, 3.0]
        assert clone.snapshot_fft_center == [1.0, 2.0]
        assert clone.snapshot_profile_line == [
            {"row": 0.0, "col": 0.0},
            {"row": 2.0, "col": 3.0},
        ]
        assert clone.profile_line == [
            {"row": 0.0, "col": 0.0},
            {"row": 2.0, "col": 3.0},
        ]
        assert clone.report_metadata["html_export"] == {
            "mode": "single",
            "encoding": "full",
            "downsample": 2,
            "trace_samples_preserved": True,
        }
    finally:
        clone.close()

    output = widget.export_html(tmp_path / "show1d-downsampled.html", downsample=2)
    assert output.exists()
    assert "2x downsampled images" in widget.export_status
    assert widget.snapshot_heights == [5]
    assert widget.snapshot_widths == [7]


def test_show1d_html_export_rejects_unimplemented_storage_and_encoding() -> None:
    widget = Show1D(np.arange(4, dtype=np.float32))

    with np.testing.assert_raises_regex(NotImplementedError, "mode='single'.*downsample"):
        widget.export_html(mode="folder")
    with np.testing.assert_raises_regex(NotImplementedError, "frontend decoder"):
        widget.export_html(encoding="uint8")
    with np.testing.assert_raises_regex(ValueError, "one of 1, 2, 4, or 8"):
        widget.export_html(downsample=3)
    for value in (2.5, np.nan, np.inf, -np.inf):
        with np.testing.assert_raises_regex(ValueError, "finite integer factor"):
            widget.export_html(downsample=value)


def test_show1d_watch_run_polls_joint_ptycho_monitor_with_object_probe_snapshots(tmp_path: pathlib.Path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    monitor = tmp_path / "show1d_monitor.jsonl"
    y, x = np.mgrid[-1:1:16j, -1:1:16j]
    duck = np.exp(-((x + 0.1) ** 2 + (y + 0.05) ** 2) * 5).astype(np.float32)

    obj0 = snapshots / "lambda1_i000_object.npy"
    probe0 = snapshots / "lambda1_i000_probe.npy"
    np.save(obj0, duck)
    np.save(probe0, np.exp(-(x**2 + y**2) * 20).astype(np.float32))
    Show1D.append_monitor_event(
        monitor,
        {
            "iteration": 0,
            "losses": {"lambda 1": 3.0, "lambda 10": 4.0, "lambda 30": 5.0},
            "metrics": {
                "lambda 1": {"rmse": 0.5, "flicker": 0.1},
                "lambda 10": {"rmse": 0.7, "flicker": 0.2},
            },
            "snapshots": {"lambda_1": obj0.relative_to(tmp_path), "lambda_1_probe": probe0.relative_to(tmp_path)},
            "warnings": ["coarse initialization"],
        },
    )

    widget = Show1D.watch_run(monitor, refresh_s=0.25, start=True, show_review=True, snapshot_columns=2)
    try:
        assert widget.n_points == 1
        assert widget.n_snapshot_groups == 1
        assert widget.snapshot_image_labels == ["lambda_1", "lambda_1_probe"]

        obj1 = snapshots / "lambda1_i001_object.npy"
        obj30 = snapshots / "lambda30_i001_object.npy"
        np.save(obj1, (duck * 1.2).astype(np.float32))
        np.save(obj30, np.zeros((16, 16), dtype=np.float32))
        Show1D.append_monitor_event(
            monitor,
            {
                "iteration": 1,
                "losses": {"lambda 1": 1.2, "lambda 10": 8.0, "lambda 30": None},
                "metrics": {
                    "lambda 1": {"rmse": 0.2, "flicker": 0.15},
                    "lambda 10": {"rmse": 0.9, "flicker": 0.85},
                    "lambda 30": {"rmse": 1.2},
                },
                "snapshots": {"lambda_1": obj1.relative_to(tmp_path), "lambda_30": obj30.relative_to(tmp_path)},
                "starred": ["lambda 1"],
                "tags": {"lambda 1": ["best"], "lambda 30": ["bad start"]},
                "notes": {"lambda 10": "exploded after first update"},
            },
        )

        _wait_until(
            lambda: widget.n_points == 2
            and widget.n_snapshot_groups == 2
            and widget.report_metadata.get("monitor_events") == 2
        )
        assert widget.report_metadata["monitor_events"] == 2
        assert widget.best_trial_label == "lambda 1"
        assert widget.starred_snapshot_image_labels == ["lambda 1"]
        assert widget.trial_tags["lambda 30"] == ["bad start"]
        assert widget.trial_notes["lambda 10"] == "exploded after first update"
        assert any(alert["kind"] == "monitor_warning" for alert in widget.trial_alerts)
        assert any(alert["kind"] == "worse_final" and alert["label"] == "lambda 10" for alert in widget.trial_alerts)
        assert any(alert["kind"] == "nonfinite" and alert["label"] == "lambda 30" for alert in widget.trial_alerts)
        assert any(alert["kind"] == "image_collapse" and alert["label"] == "lambda 30" for alert in widget.trial_alerts)
    finally:
        widget.stop_monitor()
