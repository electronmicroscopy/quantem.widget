"""Tests for the standalone interactive HTML export of ShowPDF."""

import base64
import inspect
import json
import re

import numpy as np
import pytest
import traitlets

# ShowPDF needs the PairDistributionFunction API of quantem PR #248 (calculate_Gr with density
# and max_iter); skip until the installed quantem has it.
_diffraction = pytest.importorskip("quantem.diffraction")
_PDF = getattr(_diffraction, "PairDistributionFunction", None)
if _PDF is None or not {"density", "max_iter"} <= set(inspect.signature(_PDF.calculate_Gr).parameters):
    pytest.skip(
        "ShowPDF needs PairDistributionFunction.calculate_Gr(density=..., max_iter=...)",
        allow_module_level=True,
    )

from quantem.core.datastructures.dataset4dstem import Dataset4dstem  # noqa: E402
from quantem.diffraction import PairDistributionFunction  # noqa: E402
from quantem.widget import ShowPDF  # noqa: E402


@pytest.fixture
def small_pdf():
    """PairDistributionFunction of a 4x4 scan of 32x32 patterns with two broad rings and a little noise.

    The origin is fixed at the detector center (find_origin=False, for speed).
    """
    scan_y, scan_x, det_h, det_w = 4, 4, 32, 32
    rr, cc = np.mgrid[0:det_h, 0:det_w]
    center_r, center_c = (det_h - 1) / 2.0, (det_w - 1) / 2.0
    radius = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)

    dp = np.exp(-((radius - 5) ** 2) / 4) + 0.3 * np.exp(-((radius - 10) ** 2) / 8)
    dp = dp.astype(np.float32)

    rng = np.random.RandomState(42)
    data = np.empty((scan_y, scan_x, det_h, det_w), dtype=np.float32)
    for iy in range(scan_y):
        for ix in range(scan_x):
            data[iy, ix] = dp + 0.01 * rng.randn(det_h, det_w).astype(np.float32)

    ds = Dataset4dstem.from_array(data)
    return PairDistributionFunction.from_data(
        ds,
        find_origin=False,
        origin_row=center_r,
        origin_col=center_c,
        num_annular_bins=36,
        radial_step=1.0,
        device="cpu",
    )


def _varied_pdf():
    """PairDistributionFunction of a 4x4 scan whose ring moves outwards and brightens across the scan.

    Every k bin then spans a wide range of intensities over the scan positions.
    """
    scan_y, scan_x, det = 4, 4, 32
    rr, cc = np.mgrid[0:det, 0:det]
    center = (det - 1) / 2.0
    radius = np.hypot(rr - center, cc - center)
    data = np.empty((scan_y, scan_x, det, det), dtype=np.float32)
    for iy in range(scan_y):
        for ix in range(scan_x):
            i = iy * scan_x + ix
            data[iy, ix] = (1.0 + 0.5 * i) * np.exp(-((radius - (5.0 + 0.4 * i)) ** 2) / 4)
    return PairDistributionFunction.from_data(
        Dataset4dstem.from_array(data),
        find_origin=False,
        origin_row=center,
        origin_col=center,
        num_annular_bins=36,
        radial_step=1.0,
        device="cpu",
    )


def _reject_constant(name):
    raise ValueError(f"{name} in the widget state is not valid JSON")


def _widget_state(html):
    """The ShowPDF model entry of an exported file's widget-state JSON.

    The JSON is parsed strictly: NaN or Infinity, which the browser's JSON.parse rejects (the
    page then never renders), fails the test.
    """
    match = re.search(
        r'<script type="application/vnd.jupyter.widget-state\+json">\s*(\{.*?\})\s*</script>',
        html,
        re.S,
    )
    assert match, "no widget state in export"
    models = json.loads(match.group(1), parse_constant=_reject_constant)["state"]
    entries = [m for m in models.values() if m.get("state", {}).get("scan_rows") is not None]
    assert len(entries) == 1
    return entries[0]


def _buffers(entry):
    """The binary buffers of a model entry, by trait name."""
    return {b["path"][0]: base64.b64decode(b["data"]) for b in entry.get("buffers", [])}


def _export_entry(w, path, **kwargs):
    """Export ``w`` to ``path`` and return the ShowPDF model entry of the written file."""
    return _widget_state(w.export_html(path, **kwargs).read_text(errors="ignore"))


# ---------------------------------------------------------------------------
# export_html: the standalone file
# ---------------------------------------------------------------------------


def test_showpdf_export_html_file(small_pdf, tmp_path):
    w = ShowPDF(small_pdf, title="Amorphous PDF")
    out = w.export_html(tmp_path / "pdf_view.html")
    assert out.exists()
    html = out.read_text()
    # The anywidget bundle and the ipywidgets manager state are baked into the file.
    assert "application/vnd.jupyter.widget-state+json" in html
    assert "anywidget" in html or "AnyModel" in html
    assert 'name="viewport"' in html  # standalone shell (ensure_mobile_viewport)
    entry = _widget_state(html)
    state = entry["state"]
    # The file carries the widget's state and data, with its own export button disabled: a
    # kernel-less page cannot serve another export.
    assert state["scan_rows"] == 4
    assert state["export_enabled"] is False
    assert state["offline"] is True
    assert state["export_status"] == ""
    assert "_offline" not in state
    buffers = _buffers(entry)
    assert buffers["export_payload"] == b""
    # Every export carries the exact float64 k axis for the browser recompute.
    np.testing.assert_array_equal(np.frombuffer(buffers["offline_qq"], np.float64), small_pdf.qq)
    size_mb = out.stat().st_size / 1048576
    assert w.export_status == f"Exported pdf_view.html ({size_mb:.1f} MB, exact float32 curves)"
    # The live widget is left as it was.
    assert w.export_enabled is True
    assert w.offline is False
    assert w.export_payload == b""
    assert w.export_request == ""
    assert w.offline_qq == b""


def test_showpdf_export_keeps_what_a_saved_notebook_leaves_out(small_pdf, tmp_path):
    # With save_state=False a saved notebook leaves out the scan image, mask and line-scan stack;
    # the exported page needs them, and its saved-notebook preview is dropped.
    w = ShowPDF(small_pdf)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2] = True
    w.set_mask(mask)
    w.set_line(0, 0, 3, 3, width=3)
    w.line_mode = "linescan"
    assert "nav_image_bytes" not in w.get_state()
    entry = _export_entry(w, tmp_path / "scan.html")
    buffers = _buffers(entry)
    for name in ("nav_image_bytes", "mask_bytes", "linescan_bytes"):
        assert buffers.get(name), name
    assert entry["state"]["mask_b64"] == w.mask_b64 != ""
    assert entry["state"]["_static_fallback_jpeg"] == ""


def test_showpdf_export_html_default_path(small_pdf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = ShowPDF(small_pdf, title="My PDF")
    out = w.export_html()
    assert out.exists()
    assert out.name == "my_pdf_4x4_exact.html"
    assert out.parent == tmp_path
    assert w.export_html(encoding="uint8").name == "my_pdf_4x4_quantized.html"


def test_showpdf_export_uint8_profiles_are_within_half_a_quantization_step(tmp_path):
    # A "uint8" export holds every position's radial profile, each within half an 8-bit step of
    # that k bin's range over the scan: it stores round((profile - lo) / (hi - lo) * 255). The live
    # widget never holds the (possibly large) profiles, and a "full" export has none.
    w = ShowPDF(_varied_pdf())
    entry = _export_entry(w, tmp_path / "varied.html", encoding="uint8")
    state, buffers = entry["state"], _buffers(entry)
    assert state["offline_recompute"] is True
    assert w.export_status.endswith("exact float32 curves + quantized uint8 profiles)")
    assert (w.offline_profiles, w.offline_recompute) == (b"", False)
    rows, cols, n_k = state["offline_rows"], state["offline_cols"], state["offline_nk"]
    levels = np.frombuffer(buffers["offline_profiles"], np.uint8).reshape(rows, cols, n_k)
    lo = np.frombuffer(buffers["offline_profiles_lo"], np.float32).astype(np.float64)
    hi = np.frombuffer(buffers["offline_profiles_hi"], np.float32).astype(np.float64)
    profiles = lo + levels / 255.0 * (hi - lo)
    assert (hi - lo).max() > 0.1 * np.abs(profiles).max()  # the positions really differ

    half_step = (hi - lo) / 255.0 / 2.0

    def radial_mean(mask):
        return np.asarray(w._pdf.calculate_radial_mean(mask_realspace=mask, returnval=True), float)

    for row in range(rows):
        for col in range(cols):
            mask = np.zeros((rows, cols), dtype=bool)
            mask[row, col] = True
            exact = radial_mean(mask)
            error = np.abs(profiles[row, col] - exact)
            assert np.all(error <= half_step + 1e-5 * np.abs(exact).max()), (row, col, error.max())
    # The page averages the selected positions' profiles, which gives the library's radial mean.
    mask = np.zeros((rows, cols), dtype=bool)
    mask[: rows // 2] = True
    exact = radial_mean(mask)
    assert np.all(np.abs(profiles[mask].mean(axis=0) - exact) <= half_step + 1e-5 * np.abs(exact).max())

    entry = _export_entry(w, tmp_path / "full.html", encoding="full")
    assert entry["state"]["offline_recompute"] is False
    assert _buffers(entry).get("offline_profiles", b"") == b""


def test_showpdf_export_html_rejects_unsupported_options(small_pdf, tmp_path):
    w = ShowPDF(small_pdf)
    with pytest.raises(ValueError, match="mode='single'"):
        w.export_html(tmp_path / "a.html", mode="folder")
    with pytest.raises(ValueError, match="encoding"):
        w.export_html(tmp_path / "b.html", encoding="float16")
    with pytest.raises(NotImplementedError, match="downsample"):
        w.export_html(tmp_path / "c.html", downsample=2)
    with pytest.raises(TypeError):
        w.export_html(tmp_path / "d.html", quantized=True)  # use encoding="uint8"


# ---------------------------------------------------------------------------
# Export requests from the frontend's Export menu
# ---------------------------------------------------------------------------


def test_showpdf_export_request_download_roundtrip(small_pdf):
    # A download request builds the file into export_payload; a clear request releases it.
    w = ShowPDF(small_pdf, title="PDF")
    w.export_request = json.dumps(
        {"mode": "single", "download": True, "id": "abc123", "filename": "pdf.html", "encoding": "full"}
    )
    assert w.export_payload_id == "abc123"
    assert w.export_filename == "pdf.html"
    assert len(w.export_payload) > 0
    assert b"application/vnd.jupyter.widget-state+json" in w.export_payload
    assert w.export_status.startswith("Ready pdf.html (")
    assert w.export_status.endswith(", exact float32 curves)")
    # The live widget stays interactive after serving the export.
    assert w.export_enabled is True
    first_size = len(w.export_payload)

    w.export_request = json.dumps({"mode": "clear", "id": "abc123-clear"})
    assert w.export_payload == b""
    assert w.export_payload_id == ""
    assert w.export_filename == ""

    # A second export does not embed the first one's payload, even while that is still held.
    w.export_request = json.dumps(
        {"mode": "single", "download": True, "id": "def456", "filename": "pdf.html"}
    )
    w.export_request = json.dumps(
        {"mode": "single", "download": True, "id": "ghi789", "filename": "pdf.html"}
    )
    assert abs(len(w.export_payload) - first_size) < first_size * 0.1

    # A uint8 request bakes the profiles into the file.
    w.export_request = json.dumps(
        {"mode": "single", "download": True, "id": "u8", "filename": "pdf.html", "encoding": "uint8"}
    )
    assert w.export_status.endswith("exact float32 curves + quantized uint8 profiles)")
    entry = _widget_state(w.export_payload.decode("utf-8", errors="ignore"))
    assert entry["state"]["offline_recompute"] is True


def test_showpdf_export_request_without_download_writes_file(small_pdf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = ShowPDF(small_pdf, title="PDF")
    w.export_request = json.dumps({"mode": "single", "id": "file1"})
    assert (tmp_path / "pdf_4x4_exact.html").exists()
    assert w.export_status.startswith("Exported pdf_4x4_exact.html")
    assert w.export_payload == b""


def test_showpdf_export_request_failure_is_reported(small_pdf):
    w = ShowPDF(small_pdf)
    w.export_request = json.dumps({"mode": "single", "download": True, "id": "x", "encoding": "png"})
    assert w.export_status.startswith("Export failed:")


# ---------------------------------------------------------------------------
# What the exported state holds after failed computes and non-finite values
# ---------------------------------------------------------------------------


def test_showpdf_export_html_has_no_curves_of_another_selection(small_pdf, tmp_path):
    # A compute that fails for a new selection clears the curves, so a full export cannot
    # recompute G(r) from the previous selection's I(k): the file has none.
    w = ShowPDF(small_pdf)
    left = np.zeros((4, 4), dtype=bool)
    left[:, :2] = True
    w.set_mask(left)
    w.r_max = -1.0  # refused: the curves are kept, they belong to the shown mask
    entry = _export_entry(w, tmp_path / "kept.html")
    assert entry["state"]["status_message"].startswith("Error")
    assert _buffers(entry)["ik_y_bytes"] == w.ik_y_bytes != b""
    w.set_mask(~left)  # refused too, for other scan positions
    entry = _export_entry(w, tmp_path / "cleared.html")
    assert entry["state"]["status_message"].startswith("Error")
    assert entry["state"]["mask_b64"] == w.mask_b64
    assert _buffers(entry)["ik_y_bytes"] == b""


@pytest.mark.parametrize(
    "pixels, expected",
    [
        ({(1, 2): np.nan}, (0.0, 15.0)),
        ({(1, 2): np.nan, (0, 0): np.inf, (3, 3): -np.inf}, (1.0, 14.0)),
        (None, (0.0, 0.0)),
    ],
    ids=["nan", "nan and inf", "all nan"],
)
def test_showpdf_export_with_nonfinite_nav_pixels(small_pdf, tmp_path, pixels, expected):
    # nav_data_min and nav_data_max are the range of the finite pixels (as findDataRange in the
    # frontend computes it), so the state JSON stays valid; the pixels are exported unchanged.
    nav = np.arange(16, dtype=np.float32).reshape(4, 4)
    if pixels is None:
        nav[:] = np.nan
    else:
        for (row, col), value in pixels.items():
            nav[row, col] = value
    w = ShowPDF(small_pdf, nav_image=nav)
    assert (w.nav_data_min, w.nav_data_max) == expected
    for encoding in ("full", "uint8"):
        entry = _export_entry(w, tmp_path / f"{encoding}.html", encoding=encoding)
        assert (entry["state"]["nav_data_min"], entry["state"]["nav_data_max"]) == expected
        exported = np.frombuffer(_buffers(entry)["nav_image_bytes"], np.float32)
        np.testing.assert_array_equal(exported, nav.ravel())
    w.set_data(small_pdf, nav_image=nav[::-1])
    assert (w.nav_data_min, w.nav_data_max) == expected


def test_showpdf_export_refuses_nonfinite_float_traits(small_pdf, tmp_path):
    # NaN or infinity would make the state JSON invalid. Parameters reject them when set; any other
    # float trait holding one stops the export with an error naming it, before anything is written.
    w = ShowPDF(small_pdf)
    floats = sorted(name for name, trait in w.traits(sync=True).items() if isinstance(trait, traitlets.Float))
    refused = []
    for name in floats:
        old = getattr(w, name)
        for bad in (float("nan"), float("inf")):
            try:
                setattr(w, name, bad)
            except traitlets.TraitError:
                continue  # a parameter, rejected when set
            path = tmp_path / f"{name}-{bad}" / "sub" / "x.html"
            message = rf"^Cannot export: {name} is {bad!r}, which is not valid JSON"
            with pytest.raises(ValueError, match=message):
                w.export_html(path)
            assert not path.parent.parent.exists()
            w.export_request = json.dumps(
                {"mode": "single", "download": True, "id": f"{name}-{bad}", "filename": "x.html"}
            )
            assert w.export_status.startswith(f"Export failed: Cannot export: {name} is {bad!r}")
            assert w.export_payload == b""
            setattr(w, name, old)
            refused.append(name)
    assert sorted(set(refused)) == [
        "k_max_available", "k_min_available", "mask_fraction", "nav_data_max", "nav_data_min",
        "nav_pixel_size",
    ]
    # With every trait finite again, the export's JSON parses strictly.
    _export_entry(w, tmp_path / "finite.html")
