"""Tests for DiffractionSim: crystal payload, validation, state and HTML export."""

from __future__ import annotations

import base64
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import traitlets

from quantem.widget import _diffraction_sim_crystal

REPO = Path(__file__).resolve().parents[2]

try:
    _diffraction_sim_crystal.require_crystal_tools()
    _MISSING = ""
except ImportError as exc:
    _MISSING = str(exc)

needs_crystal = pytest.mark.skipif(bool(_MISSING), reason=_MISSING)


def _decode(payload: dict, key: str) -> np.ndarray:
    dtype = np.int16 if key == "hkl_i16" else np.float32
    return np.frombuffer(base64.b64decode(payload[key]), dtype=dtype)


def _presets_script():
    spec = importlib.util.spec_from_file_location(
        "diffraction_sim_presets", REPO / "scripts" / "diffraction_sim_presets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sim(*args, **kwargs):
    from quantem.widget import DiffractionSim

    return DiffractionSim(*args, **kwargs)


def _beam_direction(widget, uvw) -> np.ndarray:
    w, x, y, z = widget.orientation
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    direction = np.asarray(uvw, float) @ widget.crystal.lat_real.numpy()
    return rot @ (direction / np.linalg.norm(direction))


# ---------------------------------------------------------------------------
# no quantem Crystal needed


def test_missing_crystal_tools_raise_with_requirement(monkeypatch):
    monkeypatch.setitem(sys.modules, "ase", None)
    with pytest.raises(ImportError, match=r"PR #297, first release after 0\.1\.9") as info:
        _diffraction_sim_crystal.require_crystal_tools()
    assert "abtem" not in str(info.value).lower()
    assert "[crystal]" not in str(info.value)


def test_export_html_signature_follows_protocol():
    from quantem.widget import DiffractionSim
    from quantem.widget.export import HTML_EXPORT_TRAITS, supports_html_export

    assert supports_html_export(DiffractionSim)
    signature = inspect.signature(DiffractionSim.export_html)
    assert signature.parameters["path"].default is None
    assert signature.parameters["title"].default is None
    assert signature.parameters["title"].kind is inspect.Parameter.KEYWORD_ONLY
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    assert signature.return_annotation in {"pathlib.Path", Path}
    assert set(HTML_EXPORT_TRAITS) <= set(DiffractionSim.class_trait_names())


def test_constructor_is_keyword_only_after_crystal():
    from quantem.widget import DiffractionSim

    params = list(inspect.signature(DiffractionSim.__init__).parameters.values())[1:]
    assert params[0].name == "crystal"
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params[1:])
    names = {p.name for p in params}
    assert {"energy_keV", "semiangle_mrad", "panel_width_px", "save_state", "state"} <= names
    assert not {"energy_ev", "semiconv_mrad", "size"} & names


# ---------------------------------------------------------------------------
# crystal payload


@needs_crystal
def test_prepare_crystal_silicon_reflection_list():
    data = _diffraction_sim_crystal.prepare_crystal(
        _diffraction_sim_crystal.preset_crystal("Si (diamond cubic)"), 200e3, 3.0
    )

    n = data["n_reflections"]
    hkl = _decode(data, "hkl_i16").reshape(-1, 3)
    assert hkl.shape == (n, 3)
    for key in ("F2", "U_re", "U_im"):
        assert _decode(data, key).shape == (n,)
    assert data["name"] == "Si"
    assert data["absorptive"] is True
    assert data["k_max"] == 3.0
    assert data["wavelength"] == pytest.approx(0.0250793, abs=1e-6)
    assert len(data["numbers"]) == 8

    recip = np.asarray(data["recip"])
    assert np.linalg.norm(hkl @ recip, axis=1).max() <= 3.0 + 1e-9

    f2 = _decode(data, "F2")
    index = {tuple(h): i for i, h in enumerate(hkl.tolist())}
    assert f2[index[(2, 0, 0)]] == 0  # diamond glide
    assert f2[index[(1, 1, 1)]] > 0
    assert json.loads(json.dumps(data)) == data


@needs_crystal
def test_website_presets_match_python_payload():
    script = _presets_script()
    committed = script.parse_ts((REPO / "js" / "diffractionsim-web" / "presets.ts").read_text())
    assert list(committed) == script.WEB_PRESETS

    name = "Si (diamond cubic)"
    regenerated = {
        name: _diffraction_sim_crystal.prepare_crystal(
            _diffraction_sim_crystal.preset_crystal(name), script.ENERGY_EV, script.K_MAX
        )
    }
    assert script.compare(regenerated, {name: committed[name]}) == []


# ---------------------------------------------------------------------------
# widget


@needs_crystal
def test_defaults_and_zone_axis():
    from quantem.widget import DiffractionSim
    from quantem.widget.utils.state_io import resolve_widget_version

    widget = _sim(zone_axis=[1, 1, 0])

    assert widget.preset == "Si (diamond cubic)"
    assert widget.crystal.name == "Si"
    assert widget.presets == DiffractionSim.presets_available()
    assert widget.pattern_range_inv_A == pytest.approx(3.0)
    assert widget.energy_keV == 200.0
    assert widget.title == "Diffraction Simulator"
    assert widget.widget_version == resolve_widget_version()
    assert json.loads(widget.crystal_json)["n_reflections"] > 0
    assert json.loads(widget.crystal_json)["energy_ev"] == 200e3
    np.testing.assert_allclose(np.abs(_beam_direction(widget, [1, 1, 0])), [0, 0, 1], atol=1e-9)


@needs_crystal
@pytest.mark.parametrize("value", ["kikuchi", "kossel", "Kikuchi"])
def test_mode_stores_kikuchi(value):
    widget = _sim(mode=value)
    assert widget.mode == "kikuchi"
    widget.mode = "cbed"
    assert widget.mode == "cbed"
    with pytest.raises(traitlets.TraitError, match="mode must be one of"):
        widget.mode = "spots"


@needs_crystal
def test_k_max_change_recomputes_reflections():
    widget = _sim("Fe (bcc)", k_max_inv_A=3.0)
    n_before = json.loads(widget.crystal_json)["n_reflections"]
    widget.k_max_inv_A = 2.0
    data = json.loads(widget.crystal_json)
    assert data["k_max"] == 2.0
    assert data["n_reflections"] < n_before
    assert widget.pattern_range_inv_A <= 2.0


@needs_crystal
def test_energy_change_recomputes_wavelength():
    widget = _sim("Al (fcc)")
    widget.energy_keV = 300
    data = json.loads(widget.crystal_json)
    assert data["energy_ev"] == 300e3
    assert data["wavelength"] == pytest.approx(0.0196875, abs=1e-6)


# ---------------------------------------------------------------------------
# validation


@needs_crystal
@pytest.mark.parametrize(
    ("trait", "value", "match"),
    [
        ("energy_keV", 0.0, "energy_keV"),
        ("energy_keV", float("nan"), "energy_keV"),
        ("k_max_inv_A", 20.0, "k_max_inv_A"),
        ("thickness_A", -5.0, "thickness_A"),
        ("thickness_A", float("inf"), "thickness_A"),
        ("semiangle_mrad", 0.0, "semiangle_mrad"),
        ("precession_deg", -1.0, "precession_deg"),
        ("sg_max_inv_A", 0.0, "sg_max_inv_A"),
        ("sigma_excitation_inv_A", 0.0, "sigma_excitation_inv_A"),
        ("field_mrad", 0.0, "field_mrad"),
        ("power", 2.0, "power"),
        ("vmax_pct", 120.0, "vmax_pct"),
        ("pattern_range_inv_A", 5.0, "pattern_range_inv_A"),
        ("pattern_range_inv_A", 0.1, "pattern_range_inv_A"),
        ("n_precession", 1, "n_precession"),
        ("n_cells", [1, 1, 7], "n_cells"),
        ("n_cells", [1, 1], "n_cells"),
        ("panel_width_px", 100, "panel_width_px"),
        ("orientation", [0.0, 0.0, 0.0, 0.0], "orientation"),
        ("cmap", "not_a_map", "Unknown colormap"),
        ("render", "dots", "render"),
        ("quality", "ultra", "quality"),
        ("scaling", "sqrt", "scaling"),
        ("view_from", "side", "view_from"),
        ("preset", "Unobtainium (fcc)", "Unknown preset"),
    ],
)
def test_invalid_values_raise(trait, value, match):
    widget = _sim()
    with pytest.raises(traitlets.TraitError, match=match):
        setattr(widget, trait, value)


@needs_crystal
def test_constructor_validates_arguments():
    with pytest.raises(traitlets.TraitError, match="thickness_A"):
        _sim(thickness_A=0)
    with pytest.raises(ValueError, match="neither a preset nor an existing CIF"):
        _sim("not_a_file.cif")
    with pytest.raises(TypeError, match="crystal must be"):
        _sim(42)
    with pytest.raises(TypeError):
        _sim(size=420)


@needs_crystal
@pytest.mark.parametrize("zone_axis", [[0, 0, 0], [1, 1], [1, float("nan"), 0]])
def test_set_zone_axis_rejects_bad_directions(zone_axis):
    widget = _sim()
    with pytest.raises(ValueError, match="zone_axis"):
        widget.set_zone_axis(zone_axis)


@needs_crystal
def test_orientation_is_normalized_and_cmap_accepts_reversed_and_quantem_maps():
    widget = _sim()
    widget.orientation = [2.0, 0.0, 0.0, 0.0]
    assert widget.orientation == [1.0, 0.0, 0.0, 0.0]
    for name in ("inferno", "gray_r", "turbo"):
        widget.cmap = name
        assert widget.cmap == name
    if widget.cmap_luts:
        widget.cmap = "turbo_black_r"
        assert widget.cmap == "turbo_black_r"


@needs_crystal
def test_turbo_black_luts_come_from_quantem():
    pytest.importorskip("quantem.core.visualization")
    from quantem.core.visualization import turbo_black

    from quantem.widget.diffraction_sim import quantem_colormap_luts

    luts = quantem_colormap_luts()
    ref = np.round(255 * turbo_black(256)(np.linspace(0, 1, 256))[:, :3]).astype(int)
    np.testing.assert_array_equal(np.asarray(luts["turbo_black"]).reshape(256, 3), ref)
    np.testing.assert_array_equal(np.asarray(luts["turbo_black_r"]).reshape(256, 3), ref[::-1])

    widget = _sim(cmap="turbo_black")
    assert widget.cmap_luts == luts
    assert "cmap_luts" not in widget.state_dict()


# ---------------------------------------------------------------------------
# state

HTML_KEYS = (
    "export_request",
    "export_status",
    "export_enabled",
    "export_payload",
    "export_payload_id",
    "export_filename",
)


@needs_crystal
def test_state_dict_is_settings_only():
    widget = _sim("Al (fcc)", mode="cbed", thickness_A=750)
    state = widget.state_dict()
    for derived in ("crystal_json", "kossel_json", "presets", "cmap_luts", "status", "widget_version"):
        assert derived not in state
    for key in HTML_KEYS:
        assert key not in state
    assert state["preset"] == "Al (fcc)"
    assert state["mode"] == "cbed"
    assert state["thickness_A"] == 750
    assert json.loads(json.dumps(state)) == state


@needs_crystal
def test_save_and_load_roundtrip(tmp_path):
    from quantem.widget import DiffractionSim

    widget = _sim(
        "Ti (hcp)", zone_axis=[1, 0, 0], mode="cbed", energy_keV=120, k_max_inv_A=3.0,
        pattern_range_inv_A=2.5, thickness_A=640, cmap="viridis", n_cells=[2, 1, 3],
    )
    path = tmp_path / "nested" / "ti.json"
    widget.save(path)
    envelope = json.loads(path.read_text())
    assert envelope["widget_name"] == "DiffractionSim"
    assert "metadata_version" in envelope and "widget_version" in envelope

    restored = DiffractionSim(state=path)
    assert restored.preset == "Ti (hcp)"
    saved, got = widget.state_dict(), restored.state_dict()
    assert set(got) == set(saved)
    for key, value in saved.items():
        if isinstance(value, (bool, str)):
            assert got[key] == value, key
        else:
            assert got[key] == pytest.approx(value), key
    assert json.loads(restored.crystal_json)["energy_ev"] == 120e3
    assert json.loads(restored.crystal_json)["k_max"] == 3.0

    from_dict = _sim(state=widget.state_dict())
    assert from_dict.mode == "cbed"
    assert from_dict.orientation == pytest.approx(widget.orientation)


@needs_crystal
def test_load_state_dict_switches_preset_and_ignores_unknown_keys():
    widget = _sim("Si (diamond cubic)")
    widget.load_state_dict({"preset": "Cu (fcc)", "thickness_A": 300.0, "not_a_trait": 1})
    assert widget.preset == "Cu (fcc)"
    assert widget.crystal.name == "Cu"
    assert widget.thickness_A == 300.0


@needs_crystal
def test_explicit_crystal_wins_over_state_preset():
    state = _sim("Au (fcc)", thickness_A=900).state_dict()
    widget = _sim("Fe (bcc)", state=state)
    assert widget.crystal.name == "Fe"
    assert widget.thickness_A == 900


@needs_crystal
def test_state_from_other_widget_is_rejected(tmp_path):
    from quantem.widget.utils.state_io import save_state_file

    path = tmp_path / "other.json"
    save_state_file(path, "ShowDiffraction", {"title": "x"})
    with pytest.raises(ValueError, match="cannot load into 'DiffractionSim'"):
        _sim(state=path)


# ---------------------------------------------------------------------------
# saved notebook


@needs_crystal
def test_full_snapshot_drops_kossel_reference_unless_save_state():
    widget = _sim()
    widget.kossel_json = json.dumps({"shape": [2, 2], "data": "AAAA"})
    assert "kossel_json" in widget.get_state("kossel_json")
    assert "kossel_json" not in widget.get_state()
    kept = _sim(save_state=True)
    kept.kossel_json = widget.kossel_json
    assert "kossel_json" in kept.get_state()


@needs_crystal
def test_static_preview_is_the_frontend_capture():
    widget = _sim()
    assert widget._static_png_b64() is None
    widget._static_fallback_jpeg = "iVBORw0KGgo="
    assert widget._static_png_b64() == "iVBORw0KGgo="


# ---------------------------------------------------------------------------
# HTML export


def _embedded_state(html: str, widget_title: str) -> dict:
    marker = '<script type="application/vnd.jupyter.widget-state+json">'
    start = html.index(marker) + len(marker)
    state = json.loads(html[start : html.index("</script>", start)])["state"]
    return next(m["state"] for m in state.values() if m["state"].get("title") == widget_title)


@needs_crystal
def test_export_html_embeds_state_presets_and_luts(tmp_path):
    widget = _sim("Si (diamond cubic)", zone_axis=[1, 1, 0], mode="cbed", title="Si 110 export")
    if widget.cmap_luts:
        widget.cmap = "turbo_black"
    widget.kossel_json = json.dumps({"shape": [1, 1], "step": 3.0, "thicknesses": [300.0], "data": "AAAAAA=="})

    path = widget.export_html(tmp_path / "nested" / "si.html", title="Si <110>", presets=["Al (fcc)"])

    html = path.read_text(encoding="utf-8")
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert "maximum-scale" not in html
    assert 'id="quantem-widget-export-layout"' in html
    assert "<title>Si &lt;110&gt;</title>" in html or "<title>Si <110></title>" in html
    state = _embedded_state(html, "Si 110 export")
    assert state["mode"] == "cbed"
    assert state["orientation"] == pytest.approx(widget.orientation)
    assert state["offline"] is True
    assert state["export_enabled"] is False
    assert set(state["_offline_presets"]) == {"Al (fcc)", "Si (diamond cubic)"}
    assert state["kossel_json"] == widget.kossel_json
    assert state["cmap_luts"] == widget.cmap_luts
    assert widget.export_status.startswith(f"Exported {path.name}")


@needs_crystal
def test_export_html_rejects_options_that_do_not_apply(tmp_path):
    widget = _sim()
    with pytest.raises(ValueError, match="encoding"):
        widget.export_html(tmp_path / "a.html", encoding="uint8")
    with pytest.raises(ValueError, match="downsample"):
        widget.export_html(tmp_path / "a.html", downsample=2)
    with pytest.raises(TypeError, match="quantized"):
        widget.export_html(tmp_path / "a.html", quantized=True)
    with pytest.raises(ValueError, match="Unknown presets"):
        widget.export_html(tmp_path / "a.html", presets=["Nope"])


@needs_crystal
def test_frontend_export_request_builds_payload():
    widget = _sim("Al (fcc)", title="Bridge test")
    widget.export_request = json.dumps(
        {"mode": "single", "encoding": "full", "id": "req-1", "filename": "al.html", "download": True}
    )
    assert widget.export_payload_id == "req-1"
    assert widget.export_filename == "al.html"
    assert b"Bridge test" in widget.export_payload
    assert widget.export_status.startswith("Ready al.html")
    widget.export_request = json.dumps({"mode": "clear"})
    assert widget.export_payload == b""


def test_frontend_colormap_list_matches_quantem_gpu():
    gpu_display = pytest.importorskip("quantem.gpu.display")
    from quantem.widget.diffraction_sim import _FRONTEND_COLORMAPS

    assert tuple(gpu_display.colormap_names()) == _FRONTEND_COLORMAPS
