"""bslz4 offline pack round-trip: Show4DSTEM(..., offline_codec='bslz4') writes a
chunked companion folder the browser WebGPU decoder inverts bit-exactly.

We decode the chunks in Python with the SAME algorithm the WGSL decoder uses (LZ4
block decode + plane-major LSB-first inverse bitshuffle, uint8 clip) and assert it
matches clip(data, 0, 255). Locks the on-disk format + the JS decoder contract.
Skips cleanly without lz4.
"""
import json
import os
import numpy as np
import pytest

pytest.importorskip("lz4.block")
pytest.importorskip("hdf5plugin")
import lz4.block as _lz4


def _inverse_bitshuffle_block(planes, block_elems, nbits=8):
    # nbits=8: packer encodes uint8 (8 bit-planes). plane byte = blockElems/8.
    plane_bytes = block_elems // 8
    out = np.zeros(block_elems, dtype=np.uint16)
    for e in range(block_elems):
        v = 0
        for bit in range(nbits):
            v |= ((int(planes[bit * plane_bytes + (e >> 3)]) >> (e & 7)) & 1) << bit
        out[e] = v
    return out


def _decode_bslz4_volume(out_dir, chunks, n_frames, det):
    decoded = np.zeros((n_frames, det), dtype=np.uint16)
    for c in chunks:
        raw = np.fromfile(out_dir / c["bin"], dtype=np.uint8)
        bm = np.fromfile(out_dir / c["meta"], dtype=np.uint32)
        be, nb = c["blockElems"], c["nBlocksPerFrame"]
        for lf in range(c["nScan"]):
            gf = c["startScan"] + lf
            for b in range(nb):
                coff = int(bm[(lf * nb + b) * 2])
                clen = int(bm[(lf * nb + b) * 2 + 1])
                planes = np.frombuffer(
                    _lz4.decompress(bytes(raw[coff:coff + clen]), uncompressed_size=be),
                    np.uint8,
                )
                decoded[gf, b * be:(b + 1) * be] = _inverse_bitshuffle_block(planes, be)
    return decoded


def test_bslz4_offline_chunked_roundtrip(tmp_path):
    from quantem.widget import Show4DSTEM
    rng = np.random.default_rng(1)
    data = rng.integers(0, 40, size=(8, 8, 64, 64), dtype=np.uint16)
    out_dir = tmp_path / "off"
    w = Show4DSTEM(data, scan_shape=(8, 8), offline=True, offline_codec="bslz4", data_url=str(out_dir))
    meta = json.loads(w._offline_bslz4)
    det = 64 * 64
    decoded = _decode_bslz4_volume(out_dir, meta["chunks"], 64, det)
    expected = np.clip(data.reshape(64, -1), 0, 255).astype(np.uint16)
    np.testing.assert_array_equal(np.clip(decoded, 0, 255), expected)  # GPU-decoder contract: bit-exact uint8


def test_bslz4_offline_multi_volume_roundtrip(tmp_path):
    from quantem.widget import Show4DSTEM
    rng = np.random.default_rng(2)
    data = rng.integers(0, 300, size=(2, 4, 4, 32, 32), dtype=np.uint16)
    out_dir = tmp_path / "multi"
    w = Show4DSTEM(
        data,
        frame_dim_label="Dataset",
        frame_labels=["a", "b"],
        offline=True,
        offline_codec="bslz4",
        data_url=str(out_dir),
        verbose=False,
    )
    meta = json.loads(w._offline_bslz4)
    assert w.offline is True
    assert w._offline_url == ""
    assert w._offline_stack == b""
    assert w._offline_chunks == ""
    assert meta["srcDtype"] == "uint8"
    assert len(meta["volumes"]) == 2
    det = 32 * 32
    for idx, vol in enumerate(meta["volumes"]):
        assert vol["base"] == f"vol{idx}/"
        decoded = _decode_bslz4_volume(out_dir / f"vol{idx}", vol["chunks"], 16, det)
        expected = np.clip(data[idx].reshape(16, -1), 0, 255).astype(np.uint16)
        np.testing.assert_array_equal(np.clip(decoded, 0, 255), expected)
