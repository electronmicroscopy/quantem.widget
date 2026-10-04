"""CPU-only: rans_url plumbing on the widget and the browser export's folder contract."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from quantem.widget import Show4DSTEM
from quantem.widget.show4dstem_webgpu_export import export_show4dstem_rans_viewer


def test_rans_url_marks_a_webgpu_browser_source():
    widget = Show4DSTEM(np.zeros((1, 1, 1, 1), dtype=np.uint8), rans_url="../rans/", rans_count=3,
                        scan_shape=(4, 5), detector_shape=(6, 7), backend="webgpu", precompute_virtual_images=False, verbose=False)
    assert widget._rans_url == "../rans/" and widget._h5_url == "" and widget._h5_urls == ""
    assert widget.gpu_memory_label == "Browser WebGPU lossless encoded source"
    assert widget.offline is True and widget._webgpu_h5_source
    assert widget.shape_rows == 4 and widget.shape_cols == 5 and widget.det_rows == 6 and widget.det_cols == 7
    with pytest.raises(ValueError):
        Show4DSTEM(np.zeros((1, 1, 1, 1), dtype=np.uint8), rans_url="../rans/", h5_urls=["a_master.h5"],
                   scan_shape=(4, 5), detector_shape=(6, 7), backend="webgpu", verbose=False)


def _tiny_encoded_series(root: pathlib.Path, tilts: int = 2, det: int = 2, scan: int = 4, frames: int = 8):
    """Synthetic artifact folders with the encoder's file layout; stream contents are not decoded here."""
    K = det * det
    records = []
    for t in range(tilts):
        art = root / f"tilt-{t + 1}"
        art.mkdir(parents=True)
        blocks = scan * scan // frames
        payload = np.arange(37 * blocks, dtype=np.uint8)
        payload.tofile(art / "payload.bin")
        np.save(art / "block_starts.npy", np.arange(0, 37 * (blocks + 1), 37, dtype=np.uint64))
        np.save(art / "offsets.npy", np.tile(np.arange(K + 1, dtype=np.uint32) * 9, (blocks, 1)))
        np.savez(art / "model.npz", context_offsets=np.arange(K + 1, dtype=np.uint32) * 3, symbols=np.tile(np.arange(3, dtype=np.uint16), K),
                 cumulative=np.tile(np.array([0, 16384, 24576], dtype=np.uint16), K), frequencies=np.tile(np.array([16384, 8192, 8192], dtype=np.uint16), K),
                 literal=np.zeros(K, dtype=np.uint8))
        records.append(dict(tilt_ordinal=t, shape=[scan, scan, det, det], scan_block=frames, model_frames=scan * scan, scale_bits=15,
                            payload_sha256="0" * 64, models=[{"path": "model.npz"}], artifacts=str(art)))
    build = root / "build-result.json"
    build.write_text(json.dumps({"tilts": records}))
    return build


def test_export_writes_range_manifest_and_bundle(tmp_path):
    build = _tiny_encoded_series(tmp_path / "encoded")
    out = tmp_path / "viewer"
    valid = np.ones((2, 2), bool); valid[1, 1] = False
    html = export_show4dstem_rans_viewer(build, out, title="tiny", frame_labels=["a", "b"], valid_pixels=valid)
    assert html.is_file() and (out / "Show4DSTEM.command").is_file()
    manifest = json.loads((out / "rans" / "manifest.json").read_text())
    assert manifest["schema"].startswith("quantem.show4dstem-rans-browser/") and manifest["bad_pixels"] == [3]
    assert manifest["scan_shape"] == [4, 4] and manifest["detector_shape"] == [2, 2]
    assert len(manifest["tilts"]) == 2
    first = manifest["tilts"][0]
    assert first["payload_url"] == "t0-payload.bin" and (out / "rans" / "t0-payload.bin").stat().st_size == 37 * 2
    assert first["blocks_meta"][1] == {"index": 1, "byte_start": 37, "byte_end": 74, "bytes": 37, "model": 0}
    for name in ("t0-entries-0.u32", "t0-lut-0.u8", "t0-ctx-0.u32", "t0-literal-0.u8", "t0-offsets-00.u32", "t0-offsets-01.u32"):
        assert (out / "rans" / name).is_file(), name
    assert (out / "rans" / "t0-lut-0.u8").stat().st_size == 4 * 256
    text = html.read_text()
    assert '"_rans_url": "../rans/"' in text and '"view_mode": "multiple"' in text


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_export_canonical_series_retains_order_geometry_and_local_grant(tmp_path, monkeypatch, dtype):
    from quantem.gpu.io import save
    from quantem.gpu.io import _qem_reference

    counts = np.arange(4 * 5 * 2 * 3, dtype=dtype).reshape(4, 5, 2, 3)
    paths = [tmp_path / "first.qem", tmp_path / "second.qem"]
    for ordinal, path in enumerate(paths):
        save(path, counts + ordinal, format="quantem", compression="ans", backend="cpu",
             metadata={"scan_sampling_A": [0.4, 0.6], "voltage_kV": 300.0})
        restored, _ = _qem_reference.load_array(path)
        np.testing.assert_array_equal(restored, counts + ordinal)

    def no_count_decode(*args, **kwargs):
        raise AssertionError("Viewer export must not decode native count arrays.")

    monkeypatch.setattr(_qem_reference, "load_array", no_count_decode)
    valid = np.ones((2, 3), bool)
    valid[1, 2] = False
    html = export_show4dstem_rans_viewer(
        paths[::-1], tmp_path / "viewer", frame_labels=["second", "first"], valid_pixels=valid,
    )
    root = html.parent.parent / "rans"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == "quantem.show4dstem-qem-browser/v1"
    assert manifest["local_grant_required"] is True
    assert manifest["scan_shape"] == [4, 5] and manifest["detector_shape"] == [2, 3]
    assert manifest["dtype"] == np.dtype(dtype).name and manifest["bad_pixels"] == [5]
    assert [item["url"] for item in manifest["sources"]] == ["t0-counts.qem", "t1-counts.qem"]
    for item, original in zip(manifest["sources"], paths[::-1]):
        assert (root / item["url"]).samefile(original)
        assert item["file_bytes"] == original.stat().st_size
        assert item["scientific_metadata"]["axes"][0]["sampling"]["value"] == 0.4
        assert item["scientific_metadata"]["axes"][1]["sampling"]["value"] == 0.6
    state = html.read_text()
    assert '"_rans_format": "qem-v1"' in state
    assert '"_rans_dtype": "' + np.dtype(dtype).name + '"' in state
    filenames = json.dumps(json.dumps(["t0-counts.qem", "t1-counts.qem"]))
    assert '"_rans_files": ' + filenames in state
    assert '"n_frames": 2' in state
    assert '"_offline_bad_px": "[5]"' in state
    assert '"view_mode": "multiple"' in state


def test_export_single_canonical_file_requires_homogeneous_series(tmp_path):
    from quantem.gpu.io import save

    first, second = tmp_path / "first.qem", tmp_path / "second.qem"
    save(first, np.zeros((3, 4, 2, 2), np.uint16), format="quantem", compression="ans", backend="cpu")
    save(second, np.zeros((3, 5, 2, 2), np.uint16), format="quantem", compression="ans", backend="cpu")
    html = export_show4dstem_rans_viewer(first, tmp_path / "single")
    assert '"_rans_format": "qem-v1"' in html.read_text()
    with pytest.raises(ValueError, match="share native"):
        export_show4dstem_rans_viewer([first, second], tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
