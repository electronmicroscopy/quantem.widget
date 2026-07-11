"""Progressive live-page behavior for folder-backed Show4DSTEM."""

from __future__ import annotations

import threading
import time

import torch

from quantem.widget import Show4DSTEM
from quantem.widget.data.dataset5dstem import Dataset5dstem


def _progressive_widget(*, cold_delay: float = 0.02):
    shape = (4, 4, 6, 6)
    calls: list[int] = []

    def loader(index: int):
        def load():
            calls.append(index)
            if index >= 3 and cold_delay > 0:
                time.sleep(cold_delay)
            return torch.full(shape, index + 1, dtype=torch.uint8)

        return load

    data = Dataset5dstem.from_lazy_loaders(
        [loader(index) for index in range(6)],
        shape=(6, *shape),
        dtype=torch.uint8,
    )
    widget = Show4DSTEM(
        data,
        page_budget=2,
        page_device="cpu",
        view_mode="multiple",
        compare_max_panels=3,
        compare_cache_pages=4,
        precompute_virtual_images=False,
        verbose=False,
    )
    # from_folder attaches this after the initial synchronous first paint.
    widget._folder_source = {
        "preload_all_if_fits": False,
        "warm_cache": False,
    }
    widget.compare_page_progressive_enabled = True
    return widget, data, calls


def _capture_messages(monkeypatch, widget):
    messages: list[tuple[dict, list | None]] = []

    def capture(content, buffers=None):
        messages.append((dict(content), buffers))

    monkeypatch.setattr(widget, "send", capture)
    return messages


def test_progressive_page_returns_immediately_and_streams_stable_panels(monkeypatch):
    widget, data, calls = _progressive_widget(cold_delay=0.025)
    messages = _capture_messages(monkeypatch, widget)
    try:
        started = time.perf_counter()
        widget.set_compare_page(1)
        setter_seconds = time.perf_counter() - started

        assert setter_seconds < 0.08
        assert messages[0][0]["type"] == "compare_page_start"
        assert messages[0][0]["indices"] == [3, 4, 5]

        widget.wait_for_compare_page(timeout=5)

        generation = widget.compare_page_generation
        panel_messages = [
            (content, buffers)
            for content, buffers in messages
            if content.get("type") == "compare_panel"
            and content.get("generation") == generation
        ]
        assert [content["frame_idx"] for content, _ in panel_messages] == [3, 4, 5]
        assert [content["slot"] for content, _ in panel_messages] == [0, 1, 2]
        assert all(buffers and len(buffers[0]) == 4 * 4 * 4 for _, buffers in panel_messages)
        assert all(
            isinstance(buffers[0], memoryview)
            for _, buffers in panel_messages
            if buffers
        )
        assert any(
            content.get("type") == "compare_page_complete"
            and content.get("generation") == generation
            for content, _ in messages
        )
        assert widget.compare_panel_indices == [3, 4, 5]
        assert widget.compare_page_loading is False
        assert widget.compare_page_loaded_count == 3
        assert widget.compare_page_panel_sequence == 3
        assert widget.compare_page_panel_frame_idx == 5
        assert widget.compare_page_panel_slot == 2
        assert len(widget.compare_page_panel_bytes) == 4 * 4 * 4
        assert widget.compare_page_total_ms >= widget.compare_page_first_panel_ms > 0
        assert len(data.loaded_indices()) <= 2
        # Wave-time DP caching prevents the final average diffraction update
        # from re-reading the just-completed cold page.
        assert [calls.count(index) for index in (3, 4, 5)] == [1, 1, 1]
    finally:
        widget.close()


def test_new_page_generation_cancels_stale_results(monkeypatch):
    widget, _, _ = _progressive_widget(cold_delay=0.05)
    messages = _capture_messages(monkeypatch, widget)
    try:
        widget.set_compare_page(1)
        widget.set_compare_page(0)
        widget.wait_for_compare_page(timeout=5)

        current = widget.compare_page_generation
        starts = [
            position
            for position, (content, _) in enumerate(messages)
            if content.get("type") == "compare_page_start"
            and content.get("generation") == current
        ]
        assert starts
        assert all(
            content.get("generation") == current
            for content, _ in messages[starts[-1] :]
            if content.get("type") in {
                "compare_page_start",
                "compare_panel",
                "compare_page_complete",
            }
        )
        assert widget.compare_page_idx == 0
        assert widget.compare_panel_indices == [0, 1, 2]
        assert widget.compare_status.endswith("page 1/2")
    finally:
        widget.close()


def test_warm_page_publishes_before_waiting_for_background_maintenance(monkeypatch):
    widget, _, _ = _progressive_widget(cold_delay=0.01)
    messages = _capture_messages(monkeypatch, widget)
    try:
        widget.set_compare_page(1)
        widget.wait_for_compare_page(timeout=5)

        complete = threading.Event()

        def capture(content, buffers=None):
            messages.append((dict(content), buffers))
            if content.get("type") == "compare_page_complete":
                complete.set()

        monkeypatch.setattr(widget, "send", capture)
        original_stop = widget.stop_dataset_preload

        def slow_stop(*, wait=False):
            if wait:
                time.sleep(0.35)
            return widget

        monkeypatch.setattr(widget, "stop_dataset_preload", slow_stop)
        widget.set_compare_page(0)

        assert complete.wait(0.2), "warm cached page waited for unrelated maintenance"
        monkeypatch.setattr(widget, "stop_dataset_preload", original_stop)
        widget.stop_compare_page_load(wait=True)
        assert widget.compare_panel_indices == [0, 1, 2]
    finally:
        widget.close()


def test_close_joins_folder_maintenance_before_it_can_restart_warming(monkeypatch):
    widget, _, _ = _progressive_widget(cold_delay=0)
    preload_started = threading.Event()
    release_preload = threading.Event()
    warm_calls: list[bool] = []

    widget._folder_source = {
        "preload_all_if_fits": True,
        "warm_cache": True,
    }
    stop = threading.Event()
    widget._compare_page_stop = stop

    def start_preload(*, background=False):
        assert background is True
        preload_started.set()
        return widget

    def wait_for_preload(timeout=None):
        release_preload.wait(timeout=5)
        return widget

    def stop_preload(*, wait=False):
        if widget._compare_maintenance_thread is not None:
            release_preload.set()
        return widget

    def start_warm(*, background=False):
        warm_calls.append(background)
        return widget

    monkeypatch.setattr(widget, "preload_all_datasets", start_preload)
    monkeypatch.setattr(widget, "wait_for_dataset_preload", wait_for_preload)
    monkeypatch.setattr(widget, "stop_dataset_preload", stop_preload)
    monkeypatch.setattr(widget, "warm_compare_cache", start_warm)

    widget._resume_folder_compare_maintenance(
        generation=widget._compare_page_generation_counter,
        stop=stop,
    )
    assert preload_started.wait(1)
    maintenance = widget._compare_maintenance_thread
    assert maintenance is not None and maintenance.is_alive()

    widget.close()

    assert not maintenance.is_alive()
    assert widget._compare_maintenance_thread is None
    assert warm_calls == []


def test_progressive_failure_retains_diagnostic_for_verification(monkeypatch):
    widget, _, _ = _progressive_widget(cold_delay=0)
    try:
        def fail_page(*args, **kwargs):
            raise RuntimeError("synthetic progressive failure")

        monkeypatch.setattr(widget, "_load_progressive_compare_page", fail_page)
        widget.set_compare_page(1)
        widget.wait_for_compare_page(timeout=5)

        assert widget.compare_page_loading is False
        assert widget.compare_status == (
            "Multiple grid unavailable: page could not be loaded."
        )
        assert widget._compare_page_last_error == (
            "RuntimeError: synthetic progressive failure"
        )
    finally:
        widget.close()


def test_fully_hidden_page_clears_progressive_slot_state():
    widget, _, _ = _progressive_widget(cold_delay=0)
    try:
        widget.set_compare_page(1)
        widget.wait_for_compare_page(timeout=5)
        assert widget.compare_page_expected_indices == [3, 4, 5]

        widget.set_compare_hidden_panels([3, 4, 5])
        widget.wait_for_compare_page(timeout=5)

        assert widget.compare_page_idx == 1
        assert widget.compare_page_expected_indices == []
        assert widget.compare_panel_indices == []
        assert widget.compare_page_loading is False
        assert widget.compare_page_panel_bytes == b""
        assert widget.compare_status.endswith("hidden · page 2/2")
    finally:
        widget.close()
