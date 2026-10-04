"""CPU-only scientist workflow through a real widget and tiny source fixtures."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantem.gpu.io._resident import CudaResidentSource
from quantem.gpu.io.models import Dataset4dstemGPU
from quantem.widget import Show4DSTEM
from quantem.widget.show4dstem_factory import show4dstem_backend_kind


def fixture_source():
    counts = np.arange(66*2*3*18*18, dtype=np.uint16).reshape(66, 2, 3, 18, 18)
    counts[0] = 60000
    counts[0, 0, 0, 0, 0] = 60001
    counts[..., -1, -1] = 65535
    owner = SimpleNamespace(calls=[], counts=counts, active=False)
    def execute(task):
        assert not owner.active
        owner.active = True
        try:
            return task()
        finally:
            owner.active = False
    def integrate(current, mask):
        assert current is owner and owner.active
        owner.calls.append("VI")
        return counts[..., mask.astype(bool)].sum(axis=-1, dtype=np.uint32)
    def patterns(current, index):
        assert current is owner and owner.active
        owner.calls.append("DP")
        return np.ascontiguousarray(counts.reshape(66, 6, 18, 18)[:, index])
    valid = np.ones((18, 18), bool);valid[-1, -1] = False
    source = CudaResidentSource(owner, shape=counts.shape, device="cuda:1", generation="widget-fixture",
        storage_format="fixture-v1", source_codec="numpy-uint16-v1", sparse_codec="none-v1",
        resident_bytes=123456, valid_pixels=valid, execute=execute, integrate=integrate, patterns=patterns)
    return source, owner


@pytest.fixture
def no_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU resident widget tests must not probe or initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)


def test_public_factory_all66_exact_batched_updates_and_export(no_gpu, tmp_path):
    source, owner = fixture_source()
    original = owner.counts.copy()
    assert show4dstem_backend_kind(source) == "resident"
    widget = Show4DSTEM(Dataset4dstemGPU(source, {"file_names":[f"acq-{i}" for i in range(66)]}),
                       center=(8.5, 8.5), bf_radius=30, verbose=False)
    assert widget._data is source
    assert str(widget._device) == "cpu" and source.device == "cuda:1"
    assert widget.compare_panel_count == 66
    assert widget.resident_batch_info["dtype"] == "<u4"
    assert widget.resident_batch_info["shape"] == [66, 2, 3]
    assert widget.resident_batch_info["generation"] == "widget-fixture"
    assert widget.resident_batch_info["storage_format"] == "fixture-v1"
    assert widget.resident_batch_info["source_codec"] == "numpy-uint16-v1"
    assert widget.resident_batch_info["sparse_codec"] == "none-v1"
    expected = original[..., source.valid_pixels].sum(axis=-1, dtype=np.uint32)
    np.testing.assert_array_equal(np.frombuffer(widget.compare_virtual_image_bytes,"<u4").reshape(66,2,3), expected)
    assert owner.calls.count("VI") == 1
    initial_id = widget.resident_batch_info["request_id"]
    before = owner.calls.count("VI")
    widget.roi_center = [4.125, 6.375]
    widget.roi_radius = 4.7
    mask = widget._current_detector_mask().astype(bool)
    np.testing.assert_array_equal(np.frombuffer(widget.compare_virtual_image_bytes,"<u4").reshape(66,2,3),
                                  original[...,mask].sum(axis=-1,dtype=np.uint32))
    assert owner.calls.count("VI") - before <= 2
    assert widget.resident_batch_info["request_id"] > initial_id
    filename = widget.save_virtual_images(tmp_path/"exact.npy")
    np.testing.assert_array_equal(np.load(filename), widget._resident_images)
    assert np.load(filename).dtype == np.uint32
    with pytest.raises(FileExistsError):
        widget.save_virtual_images(filename)
    np.testing.assert_array_equal(owner.counts, original)
    widget.close()


def test_empty_fractional_annulus_and_dp_scan_and_average(no_gpu):
    source, owner = fixture_source()
    widget = Show4DSTEM(source, center=(8.5, 8.5), bf_radius=3, verbose=False)
    with widget.hold_trait_notifications():
        widget.roi_mode = "annular"
        widget.roi_center = [8.125, 8.125]
        widget.roi_radius_inner = 0
        widget.roi_radius = 0
    assert widget.resident_batch_info["mask_area"] == 0
    assert not np.frombuffer(widget.compare_virtual_image_bytes,"<u4").any()
    assert not np.frombuffer(widget.virtual_image_bytes,"<f4").any()
    before = owner.calls.count("DP")
    widget.pos_row = 1;widget.pos_col = 2
    assert owner.calls.count("DP") - before <= 2
    np.testing.assert_array_equal(widget._resident_patterns, owner.counts[:,1,2])
    widget.compare_dp_mode = "average"
    expected = (owner.counts[:,1,2].sum(axis=0,dtype=np.uint64)/66).astype(np.float32)
    # The displayed pattern excludes the authenticated invalid pixel (raw counts stay in frame_batch).
    displayed = expected.copy(); displayed[-1, -1] = 0
    np.testing.assert_array_equal(np.frombuffer(widget.frame_bytes,np.float32).reshape(18,18),displayed)
    with pytest.raises(NotImplementedError, match="Multi-position"):
        widget._compute_vi_roi_dp()
    with pytest.raises(NotImplementedError, match="materialization"):
        widget._export_data_array(dtype="uint16",det_bin=1)
    widget.close()


def test_resident_classifier_precedes_generic_chunks(no_gpu):
    source, _ = fixture_source()
    class AmbiguousSource(type(source)):
        @property
        def chunks(self):
            raise AssertionError("Resident owner must not be routed through Metal chunk inspection")
    source.__class__ = AmbiguousSource
    assert show4dstem_backend_kind(source) == "resident"


def test_remote_owner_location_is_explicit_in_widget(no_gpu):
    source, _ = fixture_source()
    source.remote_owner_metadata = dict(
        pid=357, hostname="cpu-fixture", device="cuda:1", physical_gpu_index=1,
        physical_gpu_uuid="CPU-FIXTURE-NOT-A-GPU", source_location="remote owner fixture",
    )
    widget = Show4DSTEM(source, verbose=False)
    assert widget.resident_batch_info["source_location"] == source.remote_owner_metadata
    assert "remote cpu-fixture GPU1 (cuda:1), PID 357" in widget.gpu_memory_label
    widget.close()
