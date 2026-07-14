from __future__ import annotations

import numpy as np
import pytest


mps = pytest.importorskip("quantem.gpu.io.backends.mps")


def _plan(
    *,
    frames: int = 512 * 512,
    detector_shape: tuple[int, int] = (192, 192),
) -> "mps.MPSMasterPlan":
    return mps.MPSMasterPlan(
        master_path="/tmp/example_master.h5",
        detector_shape=detector_shape,
        dtype=np.dtype("uint16"),
        ntrigger=frames,
        chunk_files=("/tmp/example_data_000001.h5",),
        chunk_n_frames=(frames,),
    )


def test_mps_memory_guard_rejects_no_bin_load_with_det_bin_recommendation(monkeypatch) -> None:
    plan = _plan()
    monkeypatch.setattr(mps, "_mps_recommended_working_set_bytes", lambda: 18 << 30)
    monkeypatch.setattr(mps, "_mps_max_buffer_bytes", lambda: 14 << 30)

    with pytest.raises(MemoryError, match="det_bin=2"):
        mps._check_mps_memory_guard(plan, det_bin=1)


def test_mps_memory_guard_allows_safe_binned_load(monkeypatch) -> None:
    plan = _plan()
    monkeypatch.setattr(mps, "_mps_recommended_working_set_bytes", lambda: 18 << 30)
    monkeypatch.setattr(mps, "_mps_max_buffer_bytes", lambda: 14 << 30)

    mps._check_mps_memory_guard(plan, det_bin=4)


def test_mps_memory_guard_warns_for_borderline_load(monkeypatch) -> None:
    plan = _plan(frames=160_000)
    monkeypatch.setattr(mps, "_mps_recommended_working_set_bytes", lambda: 18 << 30)
    monkeypatch.setattr(mps, "_mps_max_buffer_bytes", lambda: 14 << 30)

    with pytest.warns(RuntimeWarning, match="MPS load memory check"):
        mps._check_mps_memory_guard(plan, det_bin=1)


def test_mps_memory_guard_override_warns_but_allows(monkeypatch) -> None:
    plan = _plan()
    monkeypatch.setattr(mps, "_mps_recommended_working_set_bytes", lambda: 18 << 30)
    monkeypatch.setattr(mps, "_mps_max_buffer_bytes", lambda: 14 << 30)

    with pytest.warns(RuntimeWarning, match="Proceeding"):
        mps._check_mps_memory_guard(plan, det_bin=1, skip_memory_check=True)
