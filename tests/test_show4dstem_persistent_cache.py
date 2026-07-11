"""Persistent Show4DSTEM preview-cache and refresh-protocol regressions."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import torch

from quantem.widget.show4dstem import Show4DSTEM
from quantem.widget.show4dstem_preview_cache import Show4DSTEMPreviewCache


PREVIEW_CONTRACT = {
    "preset": "bf",
    "shape": [3, 4],
    "detector_shape": [8, 8],
    "center": [3.5, 3.5],
    "bf_radius": 2.0,
    "mask_sha256": "bf-mask-v1",
    "normalization": "detector-mean-v1",
    "output_dtype": "<f4",
}


def _source(path: Path, payload: bytes = b"master-v1") -> Path:
    path.write_bytes(payload)
    return path


def _image(value: float, shape: tuple[int, int] = (3, 4)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def _cache(
    folder: Path,
    cache_root: Path,
    **kwargs,
) -> Show4DSTEMPreviewCache:
    return Show4DSTEMPreviewCache(
        folder,
        cache_dir=cache_root,
        load_contract={"det_bin": 4, "dtype": "uint16"},
        **kwargs,
    )


def test_preview_cache_flush_reopen_returns_exact_float32_hit(tmp_path: Path) -> None:
    # C1: a completed asynchronous write is reopened, expect an exact disk hit.
    folder = tmp_path / "data"
    folder.mkdir()
    master = _source(folder / "scan_00_master.h5")
    expected = _image(7.25)
    cache_root = tmp_path / "cache"
    first = _cache(folder, cache_root)
    try:
        assert first.store(master, "bf", PREVIEW_CONTRACT, expected)
        first.flush()
        assert first.info["writes"] == 1
        assert first.info["entries"] == 1
    finally:
        first.close()

    reopened = _cache(folder, cache_root)
    try:
        actual = reopened.load(master, "bf", PREVIEW_CONTRACT)
        assert actual is not None
        assert actual.dtype == np.float32
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, expected)
        assert reopened.info["hits"] == 1
        assert reopened.info["bytes_read"] >= expected.nbytes
    finally:
        reopened.close()


def test_preview_cache_off_never_creates_or_queues_entries(tmp_path: Path) -> None:
    # C1: persistent previews are disabled, expect no path, writes, or hits.
    folder = tmp_path / "data"
    folder.mkdir()
    master = _source(folder / "scan_00_master.h5")
    cache = Show4DSTEMPreviewCache(folder, cache=False)
    try:
        assert cache.enabled is False
        assert cache.path is None
        assert cache.store(master, "bf", PREVIEW_CONTRACT, _image(1.0)) is False
        cache.flush()
        assert cache.load(master, "bf", PREVIEW_CONTRACT) is None
        info = cache.info
        assert info["enabled"] is False
        assert info["mode"] == "off"
        assert info["path"] is None
        assert info["entries"] == 0
        assert info["current_bytes"] == 0
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["writes"] == 0
    finally:
        cache.close()


def test_preview_cache_path_modes_rebuild_and_zero_byte_disable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # C1: automatic and folder modes are selected, expect scoped durable paths.
    folder = tmp_path / "data"
    folder.mkdir()
    master = _source(folder / "scan_00_master.h5")
    user_root = tmp_path / "user-cache"
    monkeypatch.setenv("QUANTEM_WIDGET_CACHE", str(user_root))

    automatic = Show4DSTEMPreviewCache(folder, cache="auto")
    folder_local = Show4DSTEMPreviewCache(folder, cache="folder")
    try:
        assert automatic.mode == "user"
        assert automatic.path is not None
        assert automatic.path.parent.parent.parent == user_root
        assert folder_local.mode == "folder"
        assert folder_local.path == folder / ".quantem/show4dstem-cache/v1"
    finally:
        automatic.close()
        folder_local.close()

    # C2: a zero-byte public budget is requested, expect persistence disabled.
    zero = Show4DSTEMPreviewCache(
        folder,
        cache_dir=tmp_path / "zero",
        max_bytes=0,
    )
    try:
        assert zero.enabled is False
        assert zero.store(master, "bf", PREVIEW_CONTRACT, _image(1.0)) is False
        assert zero.info["entries"] == 0
        assert zero.info["pending_writes"] == 0
    finally:
        zero.close()

    # C3: rebuild opens an existing namespace, expect old entries ignored.
    cache_root = tmp_path / "rebuild"
    populated = _cache(folder, cache_root)
    try:
        assert populated.store(master, "bf", PREVIEW_CONTRACT, _image(2.0))
        populated.flush()
        assert populated.info["entries"] == 1
    finally:
        populated.close()

    rebuilt = Show4DSTEMPreviewCache(
        folder,
        cache_dir=cache_root,
        rebuild=True,
        load_contract={"det_bin": 4, "dtype": "uint16"},
    )
    try:
        assert rebuilt.info["entries"] == 0
        assert rebuilt.load(master, "bf", PREVIEW_CONTRACT) is None
        assert rebuilt.info["hits"] == 0
        assert rebuilt.info["misses"] == 1
    finally:
        rebuilt.close()


def test_source_change_invalidates_only_changed_master(tmp_path: Path) -> None:
    # C1: one of two masters changes, expect one miss and one unaffected hit.
    folder = tmp_path / "data"
    folder.mkdir()
    masters = [
        _source(folder / "scan_00_master.h5", b"zero-v1"),
        _source(folder / "scan_01_master.h5", b"one-v1"),
    ]
    cache = _cache(folder, tmp_path / "cache")
    try:
        for idx, master in enumerate(masters):
            assert cache.store(
                master,
                "bf",
                PREVIEW_CONTRACT,
                _image(float(idx)),
            )
        cache.flush()

        masters[0].write_bytes(b"zero-v2-with-a-different-size")

        assert cache.load(masters[0], "bf", PREVIEW_CONTRACT) is None
        unchanged = cache.load(masters[1], "bf", PREVIEW_CONTRACT)
        assert unchanged is not None
        np.testing.assert_array_equal(unchanged, _image(1.0))
        assert cache.info["misses"] == 1
        assert cache.info["invalidations"] == 1
        assert cache.info["hits"] == 1
    finally:
        cache.close()


def test_corrupt_entry_is_nonfatal_and_can_be_repaired(tmp_path: Path) -> None:
    # C1: an NPZ is truncated, expect a miss followed by atomic replacement.
    folder = tmp_path / "data"
    folder.mkdir()
    master = _source(folder / "scan_00_master.h5")
    expected = _image(4.0)
    cache = _cache(folder, tmp_path / "cache")
    try:
        assert cache.store(master, "bf", PREVIEW_CONTRACT, expected)
        cache.flush()
        assert cache.path is not None
        entry = next(cache.path.glob("*.npz"))
        entry.write_bytes(b"not-an-npz")

        assert cache.load(master, "bf", PREVIEW_CONTRACT) is None
        assert cache.info["corruptions"] == 1

        assert cache.store(master, "bf", PREVIEW_CONTRACT, expected)
        cache.flush()
        repaired = cache.load(master, "bf", PREVIEW_CONTRACT)
        assert repaired is not None
        np.testing.assert_array_equal(repaired, expected)
        assert cache.info["writes"] == 2
    finally:
        cache.close()


def test_processing_contract_changes_cannot_reuse_preview(tmp_path: Path) -> None:
    # C1: mask geometry or detector binning changes, expect provenance misses.
    folder = tmp_path / "data"
    folder.mkdir()
    master = _source(folder / "scan_00_master.h5")
    cache_root = tmp_path / "cache"
    original = _cache(folder, cache_root)
    try:
        assert original.store(master, "bf", PREVIEW_CONTRACT, _image(2.0))
        original.flush()
        changed_mask = {**PREVIEW_CONTRACT, "mask_sha256": "bf-mask-v2"}
        assert original.load(master, "bf", changed_mask) is None
        assert original.load(master, "bf", PREVIEW_CONTRACT) is not None
    finally:
        original.close()

    changed_load = Show4DSTEMPreviewCache(
        folder,
        cache_dir=cache_root,
        load_contract={"det_bin": 2, "dtype": "uint16"},
    )
    try:
        assert changed_load.load(master, "bf", PREVIEW_CONTRACT) is None
        assert changed_load.info["misses"] == 1
    finally:
        changed_load.close()


def test_external_hdf5_chunk_change_invalidates_master_preview(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    # C1: linked detector bytes change behind a stable master, expect a miss.
    folder = tmp_path / "data"
    folder.mkdir()
    master = folder / "scan_00_master.h5"
    chunk = folder / "scan_00_data_000001.h5"
    with h5py.File(chunk, "w") as handle:
        handle.require_group("entry/data").create_dataset(
            "data",
            data=np.arange(16, dtype=np.uint16).reshape(4, 4),
        )
    with h5py.File(master, "w") as handle:
        handle.require_group("entry/data")["data_000001"] = h5py.ExternalLink(
            chunk.name,
            "/entry/data/data",
        )

    cache = _cache(folder, tmp_path / "cache")
    try:
        assert cache.store(master, "bf", PREVIEW_CONTRACT, _image(3.0))
        cache.flush()
        assert cache.load(master, "bf", PREVIEW_CONTRACT) is not None

        prior_stat = chunk.stat()
        with h5py.File(chunk, "r+") as handle:
            handle["entry/data/data"][0, 0] = np.uint16(999)
            handle.flush()
        # Filesystems with coarse metadata timestamps still get a deterministic
        # signature change for this validity test.
        os.utime(
            chunk,
            ns=(prior_stat.st_atime_ns, prior_stat.st_mtime_ns + 1_000_000_000),
        )

        assert cache.load(master, "bf", PREVIEW_CONTRACT) is None
        assert cache.info["misses"] == 1
    finally:
        cache.close()


def test_byte_cap_prunes_entries_and_clear_resets_scope(tmp_path: Path) -> None:
    # C1: an entry exceeds the configured cap, expect immediate cache eviction.
    folder = tmp_path / "data"
    folder.mkdir()
    master = _source(folder / "scan_00_master.h5")
    capped = _cache(folder, tmp_path / "capped", max_bytes=1)
    try:
        assert capped.store(master, "bf", PREVIEW_CONTRACT, _image(1.0))
        capped.flush()
        info = capped.info
        assert info["entries"] == 0
        assert info["current_bytes"] <= info["max_bytes"]
        assert info["evictions"] == 1
    finally:
        capped.close()

    # C2: a populated scope is cleared, expect entries and telemetry reset only.
    cache = _cache(folder, tmp_path / "clearable")
    try:
        assert cache.store(master, "bf", PREVIEW_CONTRACT, _image(1.0))
        assert cache.store(master, "adf", PREVIEW_CONTRACT, _image(2.0))
        cache.flush()
        assert cache.info["entries"] == 2

        cache.clear()

        info = cache.info
        assert info["enabled"] is True
        assert info["entries"] == 0
        assert info["current_bytes"] == 0
        assert info["writes"] == 0
        assert info["bytes_written"] == 0
        assert master.exists()
    finally:
        cache.close()


def test_concurrent_stores_are_serialized_without_lost_entries(tmp_path: Path) -> None:
    # C1: callers queue writes concurrently, expect one valid entry per master.
    folder = tmp_path / "data"
    folder.mkdir()
    masters = [
        _source(folder / f"scan_{idx:02d}_master.h5", f"master-{idx}".encode())
        for idx in range(8)
    ]
    cache = _cache(folder, tmp_path / "cache")
    barrier = threading.Barrier(len(masters))
    accepted = [False] * len(masters)

    def store_one(idx: int) -> None:
        barrier.wait()
        accepted[idx] = cache.store(
            masters[idx],
            "bf",
            PREVIEW_CONTRACT,
            _image(float(idx)),
        )

    threads = [
        threading.Thread(target=store_one, args=(idx,))
        for idx in range(len(masters))
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
        assert not any(thread.is_alive() for thread in threads)
        assert all(accepted)

        cache.flush()

        assert cache.info["entries"] == len(masters)
        assert cache.info["writes"] == len(masters)
        assert cache.info["write_errors"] == 0
        for idx, master in enumerate(masters):
            actual = cache.load(master, "bf", PREVIEW_CONTRACT)
            assert actual is not None
            np.testing.assert_array_equal(actual, _image(float(idx)))
    finally:
        cache.close()


def test_close_drains_writer_and_rejects_late_store(tmp_path: Path) -> None:
    # C1: close overlaps a queued write, expect one drain and no post-close job.
    folder = tmp_path / "data"
    folder.mkdir()
    first = _source(folder / "scan_00_master.h5", b"first")
    late = _source(folder / "scan_01_master.h5", b"late")
    cache = _cache(folder, tmp_path / "cache")
    write_started = threading.Event()
    release_write = threading.Event()
    original_write = cache._write_job

    def slow_write(*args):
        write_started.set()
        assert release_write.wait(2.0)
        return original_write(*args)

    cache._write_job = slow_write
    assert cache.store(first, "bf", PREVIEW_CONTRACT, _image(1.0))
    assert write_started.wait(2.0)

    closer = threading.Thread(target=cache.close)
    closer.start()
    deadline = time.monotonic() + 2.0
    while not cache.info["closed"] and time.monotonic() < deadline:
        time.sleep(0.001)
    assert cache.info["closed"] is True
    assert cache.store(late, "bf", PREVIEW_CONTRACT, _image(2.0)) is False

    release_write.set()
    closer.join(timeout=3.0)
    assert not closer.is_alive()
    assert cache.info["pending_writes"] == 0
    assert cache._writer is not None and not cache._writer.is_alive()


def test_from_folder_second_widget_paints_cache_before_raw_refresh(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # C1: a fresh widget reopens three masters, expect cached tiles before raw
    # page completion.
    import quantem.widget as qw
    import quantem.widget.io as widget_io
    import quantem.widget.show4dstem_factory as factory

    masters = [
        _source(tmp_path / f"scan_{idx:02d}_master.h5", f"master-{idx}".encode())
        for idx in range(3)
    ]
    monkeypatch.setattr(
        widget_io,
        "discover_masters",
        lambda *args, **kwargs: list(masters),
    )
    monkeypatch.setattr(widget_io, "is_master_ready", lambda path: True)
    monkeypatch.setattr(
        factory,
        "_largest_compatible_master_group",
        lambda values, **kwargs: list(values),
    )
    calls = 0

    def load_first(path, **kwargs):
        nonlocal calls
        calls += 1
        idx = masters.index(Path(path))
        return torch.full((4, 4, 8, 8), idx + 1, dtype=torch.uint16)

    monkeypatch.setattr(qw, "load", load_first)
    cache_root = tmp_path / "cache"
    first = qw.Show4DSTEM.from_folder(
        tmp_path,
        watch=False,
        preload_all_if_fits=False,
        page_size=3,
        preview_cache_dir=cache_root,
        precompute_virtual_images=False,
    )
    try:
        first.wait_for_compare_page(timeout=10)
    finally:
        first.close()
    assert first.preview_cache_info["entries"] == 3

    raw_gate = threading.Event()
    calls = 0
    original_reduce = Show4DSTEM._compare_virtual_images_for_indices

    def blocked_reduce(widget, indices, mask):
        assert raw_gate.wait(5.0)
        return original_reduce(widget, indices, mask)

    monkeypatch.setattr(
        Show4DSTEM,
        "_compare_virtual_images_for_indices",
        blocked_reduce,
    )

    def load_second(path, **kwargs):
        nonlocal calls
        calls += 1
        idx = masters.index(Path(path))
        return torch.full((4, 4, 8, 8), idx + 1, dtype=torch.uint16)

    monkeypatch.setattr(qw, "load", load_second)
    second = qw.Show4DSTEM.from_folder(
        tmp_path,
        watch=False,
        preload_all_if_fits=False,
        page_size=3,
        preview_cache_dir=cache_root,
        precompute_virtual_images=False,
    )
    try:
        deadline = time.monotonic() + 5.0
        while (
            len(second.compare_page_cached_indices) < 3
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert second.compare_page_cached_indices == [0, 1, 2]
        assert second.compare_page_cache_state == "cached"
        assert second.compare_page_loading is True
        assert second.compare_panel_indices == [0, 1, 2]
        assert calls >= 1  # calibration/raw decode may overlap; reduction is blocked
        assert second.compare_page_first_panel_ms > 0

        raw_gate.set()
        second.wait_for_compare_page(timeout=10)
        assert second.compare_page_cache_state == "fresh"
        assert second.compare_page_cached_indices == []
    finally:
        raw_gate.set()
        second.close()


def test_from_folder_rejects_conflicting_cuda_placement_options(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # C1: public gpus and a private load device conflict, expect corrective
    # failure before decode.
    import quantem.widget.io as widget_io
    import quantem.widget.show4dstem_factory as factory
    import quantem.widget as qw

    master = _source(tmp_path / "scan_00_master.h5")
    monkeypatch.setattr(
        widget_io,
        "discover_masters",
        lambda *args, **kwargs: [master],
    )
    monkeypatch.setattr(widget_io, "is_master_ready", lambda path: True)
    monkeypatch.setattr(
        factory,
        "_largest_compatible_master_group",
        lambda values, **kwargs: list(values),
    )

    with pytest.raises(ValueError, match="gpus= controls.*remove"):
        qw.Show4DSTEM.from_folder(
            tmp_path,
            gpus=[0, 1],
            load_kwargs={"device": 0},
            watch=False,
        )


def test_running_folder_reloads_replaced_master_instead_of_host_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # C1: a known source changes after a warm page, expect raw release/reload
    # and fresh pixels.
    import quantem.widget as qw
    import quantem.widget.io as widget_io
    import quantem.widget.show4dstem_factory as factory

    masters = [
        _source(tmp_path / f"scan_{idx:02d}_master.h5", bytes([value]))
        for idx, value in enumerate((2, 3))
    ]
    monkeypatch.setattr(
        widget_io,
        "discover_masters",
        lambda *args, **kwargs: list(masters),
    )
    monkeypatch.setattr(widget_io, "is_master_ready", lambda path: True)
    monkeypatch.setattr(
        factory,
        "_largest_compatible_master_group",
        lambda values, **kwargs: list(values),
    )
    calls: list[tuple[str, int]] = []

    def load_source(path, **kwargs):
        source = Path(path)
        value = int(source.read_bytes()[0])
        calls.append((source.name, value))
        return torch.full((3, 3, 8, 8), value, dtype=torch.uint16)

    monkeypatch.setattr(qw, "load", load_source)
    widget = qw.Show4DSTEM.from_folder(
        tmp_path,
        watch=False,
        preload_all_if_fits=False,
        page_size=2,
        preview_cache_dir=tmp_path / "cache",
        precompute_virtual_images=False,
        center=(4.0, 4.0),
        bf_radius=2.0,
    )
    try:
        widget.wait_for_compare_page(timeout=10)
        masters[1].write_bytes(bytes([9]))

        widget._refresh_compare_virtual_images()
        widget.wait_for_compare_page(timeout=10)

        stack = np.frombuffer(
            widget.compare_virtual_image_bytes,
            dtype=np.float32,
        ).reshape(2, 3, 3)
        assert calls.count((masters[1].name, 9)) == 1
        np.testing.assert_allclose(stack[1], 9.0)
        assert widget.preview_cache_info["invalidations"] >= 1
        assert widget._compare_loaded_source_signatures[1] == (
            widget._compare_preview_cache.source_signature(masters[1])
        )
    finally:
        widget.close()


def _protocol_widget() -> Show4DSTEM:
    data = np.zeros((2, 2, 3, 4, 4), dtype=np.uint16)
    return Show4DSTEM(
        data,
        view_mode="multiple",
        compare_max_panels=2,
        center=(2.0, 2.0),
        bf_radius=1.0,
        precompute_virtual_images=False,
        verbose=False,
    )


def _prepare_protocol_request(widget: Show4DSTEM) -> tuple[int, threading.Event]:
    class EnabledProtocolCache:
        enabled = True

        @staticmethod
        def close() -> None:
            return None

    widget._compare_preview_cache = EnabledProtocolCache()
    widget._compare_virtual_page_cache.clear()
    widget._compare_virtual_page_cache_bytes = 0
    generation = int(widget._compare_page_generation_counter) + 1
    stop = threading.Event()
    widget._compare_page_generation_counter = generation
    widget._compare_page_stop = stop
    widget.compare_page_loading = True
    widget.compare_page_loaded_count = 0
    widget.compare_page_cached_indices = []
    widget.compare_page_cache_state = "miss"
    widget.compare_page_first_panel_ms = 0.0
    widget.compare_page_first_fresh_ms = 0.0
    widget.compare_page_total_ms = 0.0
    return generation, stop


def _stub_protocol_maintenance(monkeypatch, widget: Show4DSTEM) -> None:
    monkeypatch.setattr(
        widget,
        "stop_compare_maintenance",
        lambda *, wait=False: widget,
    )
    monkeypatch.setattr(
        widget,
        "stop_dataset_preload",
        lambda *, wait=False: widget,
    )
    monkeypatch.setattr(
        widget,
        "stop_compare_cache_warm",
        lambda *, wait=False: widget,
    )
    monkeypatch.setattr(widget, "_current_detector_mask", lambda: None)
    monkeypatch.setattr(widget, "_current_frame_bytes", lambda: None)
    monkeypatch.setattr(
        widget,
        "_prefetch_neighbor_compare_pages",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        widget,
        "_resume_folder_compare_maintenance",
        lambda **kwargs: None,
    )


def test_near_preset_roi_never_reads_or_writes_persistent_cache() -> None:
    # C1: a circle 0.9 px from BF matches only the fuzzy RAM preset, expect no disk IO.
    widget = _protocol_widget()
    calls: list[str] = []

    class RecordingCache:
        enabled = True

        def load(self, *args, **kwargs):
            calls.append("load")
            return _image(1.0, (2, 3))

        def store(self, *args, **kwargs):
            calls.append("store")
            return True

    try:
        widget._compare_preview_cache = RecordingCache()
        widget._compare_preview_sources = ["unused-master.h5", "unused-1.h5"]
        widget.roi_mode = "circle"
        widget.roi_center_row = float(widget.center_row)
        widget.roi_center_col = float(widget.center_col)
        widget.roi_radius = float(widget.bf_radius) + 0.9

        assert widget._preset_cache_name() == "bf"
        assert widget._persistent_compare_preset_name() is None
        assert widget._load_persistent_compare_previews([0]) == {}
        widget._persist_compare_previews(
            _image(2.0, (2, 3)).tobytes(),
            [0],
        )
        assert calls == []
    finally:
        widget._compare_preview_cache = None
        widget.close()


def test_cached_panels_are_marked_until_fresh_panels_replace_them(
    monkeypatch,
) -> None:
    # C1: a warm reopen has validated previews, expect cached then fresh panels.
    widget = _protocol_widget()
    messages: list[dict[str, object]] = []
    cached = {0: _image(1.0, (2, 3)), 1: _image(2.0, (2, 3))}
    fresh = {0: _image(11.0, (2, 3)), 1: _image(12.0, (2, 3))}
    cached_ready = threading.Event()
    try:
        generation, stop = _prepare_protocol_request(widget)
        _stub_protocol_maintenance(monkeypatch, widget)
        monkeypatch.setattr(
            widget,
            "_send_compare_page_message",
            lambda content, *, buffer=None: messages.append(dict(content)),
        )
        def load_cached(indices, **kwargs):
            on_hit = kwargs.get("on_hit")
            for count, idx in enumerate(indices, start=1):
                if callable(on_hit):
                    on_hit(idx, cached[idx], count)
            cached_ready.set()
            return cached

        monkeypatch.setattr(
            widget,
            "_load_persistent_compare_previews",
            load_cached,
        )
        monkeypatch.setattr(
            widget,
            "_get_cached_compare_preset",
            lambda indices: None,
        )

        def load_fresh(indices, mask, **kwargs):
            assert cached_ready.wait(2.0)
            deadline = time.monotonic() + 2.0
            while (
                widget.compare_page_cache_state != "cached"
                and time.monotonic() < deadline
            ):
                time.sleep(0.001)
            assert widget.compare_page_cache_state == "cached"
            assert widget.compare_page_cached_indices == [0, 1]
            for loaded_count, idx in enumerate(indices, start=1):
                widget._emit_progressive_compare_panel(
                    generation=kwargs["generation"],
                    stop=kwargs["stop"],
                    page_idx=kwargs["page_idx"],
                    frame_idx=idx,
                    slot=list(indices).index(idx),
                    image=fresh[idx],
                    started=kwargs["started"],
                    loaded_count=loaded_count,
                    cached=False,
                )
            return fresh

        monkeypatch.setattr(widget, "_load_progressive_compare_page", load_fresh)

        widget._run_progressive_compare_page(
            generation=generation,
            stop=stop,
            page_idx=0,
            indices=(0, 1),
            all_groups=False,
            started=time.perf_counter(),
        )

        panel_messages = [
            message for message in messages if message["type"] == "compare_panel"
        ]
        assert [message["cached"] for message in panel_messages] == [
            True,
            True,
            False,
            False,
        ]
        assert widget.compare_page_cache_state == "fresh"
        assert widget.compare_page_cached_indices == []
        assert widget.compare_page_panel_cached is False
        assert widget.compare_page_loading is False
        assert widget.compare_page_loaded_count == 2
        actual = np.frombuffer(
            widget.compare_virtual_image_bytes,
            dtype=np.float32,
        ).reshape(2, 2, 3)
        np.testing.assert_array_equal(actual, np.stack([fresh[0], fresh[1]]))
        assert messages[-1]["type"] == "compare_page_complete"
        assert messages[-1]["cache_state"] == "fresh"
    finally:
        widget.close()


def test_refresh_error_retains_validated_cached_panels(monkeypatch) -> None:
    # C1: authoritative refresh fails after cache paint, expect cache retention.
    widget = _protocol_widget()
    messages: list[dict[str, object]] = []
    cached = {0: _image(5.0, (2, 3)), 1: _image(6.0, (2, 3))}
    fresh_zero = _image(50.0, (2, 3))
    cached_ready = threading.Event()
    try:
        generation, stop = _prepare_protocol_request(widget)
        _stub_protocol_maintenance(monkeypatch, widget)
        monkeypatch.setattr(
            widget,
            "_send_compare_page_message",
            lambda content, *, buffer=None: messages.append(dict(content)),
        )
        def load_cached(indices, **kwargs):
            on_hit = kwargs.get("on_hit")
            for count, idx in enumerate(indices, start=1):
                if callable(on_hit):
                    on_hit(idx, cached[idx], count)
            cached_ready.set()
            return cached

        monkeypatch.setattr(
            widget,
            "_load_persistent_compare_previews",
            load_cached,
        )
        monkeypatch.setattr(
            widget,
            "_get_cached_compare_preset",
            lambda indices: None,
        )

        def fail_refresh(indices, mask, **kwargs):
            assert cached_ready.wait(2.0)
            widget._emit_progressive_compare_panel(
                generation=kwargs["generation"],
                stop=kwargs["stop"],
                page_idx=kwargs["page_idx"],
                frame_idx=0,
                slot=0,
                image=fresh_zero,
                started=kwargs["started"],
                loaded_count=1,
                cached=False,
            )
            raise RuntimeError("synthetic decode failure")

        monkeypatch.setattr(widget, "_load_progressive_compare_page", fail_refresh)

        widget._run_progressive_compare_page(
            generation=generation,
            stop=stop,
            page_idx=0,
            indices=(0, 1),
            all_groups=False,
            started=time.perf_counter(),
        )

        assert widget.compare_page_cache_state == "warning"
        assert widget.compare_page_cached_indices == [1]
        assert widget.compare_panel_indices == [0, 1]
        assert widget.compare_page_loading is False
        assert "Cached preview" in widget.compare_status
        assert (
            "RuntimeError: synthetic decode failure"
            in widget._compare_page_last_error
        )
        actual = np.frombuffer(
            widget.compare_virtual_image_bytes,
            dtype=np.float32,
        ).reshape(2, 2, 3)
        np.testing.assert_array_equal(actual, np.stack([fresh_zero, cached[1]]))
        assert messages[-1]["type"] == "compare_page_complete"
        assert messages[-1]["cache_state"] == "warning"
        assert messages[-1]["cached_indices"] == [1]
    finally:
        widget.close()
