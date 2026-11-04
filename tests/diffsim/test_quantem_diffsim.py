import json

import pytest

from quantem import diffsim

bulk = pytest.importorskip("ase.build").bulk
needs_bundle = pytest.mark.skipif(not diffsim.BUNDLE_PATH.is_file(), reason="run `npm run build:web` first")


def test_settings_carry_the_structure_as_cif_and_reject_unknown_names(tmp_path):
    si = bulk("Si", cubic=True)
    data = diffsim.settings(si, zone_axis=(0, 0, 1), mode="nanobeam", thickness_A=300)
    assert data["zone_axis"] == [0, 0, 1] and data["mode"] == "nanobeam" and data["structure_name"] == "Si"
    assert "_cell_length_a" in data["cif"] and data["cif"].count("Si") >= 8
    json.dumps(data)
    path = tmp_path / "si.cif"
    path.write_text(data["cif"])
    assert diffsim.structure_to_cif(path) == diffsim.structure_to_cif(str(path)) == data["cif"]
    assert diffsim.structure_to_cif(data["cif"]) == data["cif"]
    with pytest.raises(ValueError, match="unknown settings"):
        diffsim.settings(si, zone="001")
    with pytest.raises(ValueError, match="mode"):
        diffsim.settings(si, mode="sad")
    with pytest.raises(TypeError, match="pymatgen Structure"):
        diffsim.structure_to_cif(42)


@needs_bundle
def test_html_page_inlines_the_bundle_or_loads_it_from_a_url(tmp_path):
    si = bulk("Si", cubic=True)
    page = diffsim.save_html(tmp_path / "si.html", si, title="Si", zone_axis=[1, 1, 0]).read_text()
    assert '<meta name="viewport"' in page and "<title>Si</title>" in page
    assert 'id="quantem-diffsim-bundle"' in page and '"zone_axis": [1, 1, 0]' in page
    linked = diffsim.html(si, bundle_url="https://example.org/diffraction-sim.js")
    assert "quantem-diffsim-bundle" not in linked and '"https://example.org/diffraction-sim.js"' in linked
    assert len(linked) < 20_000 < len(page)


@needs_bundle
def test_dash_serves_the_bundle_and_draws_from_the_settings_store():
    dash = pytest.importorskip("dash")
    from quantem.diffsim import dash as diffsim_dash

    app = dash.Dash(__name__)
    url = diffsim_dash.register(app)
    assert diffsim_dash.register(app) == url == f"/_quantem_diffsim/diffraction-sim-{diffsim.__version__}.js"
    view = diffsim_dash.layout("tem", bulk("Si", cubic=True), zone_axis=[0, 0, 1])
    app.layout = view
    response = app.server.test_client().get(url)
    assert response.status_code == 200 and response.mimetype == "text/javascript"
    assert response.data == diffsim.BUNDLE_PATH.read_bytes()
    store, mount = view.children
    assert store.id == diffsim_dash.settings_id("tem") and store.data["zone_axis"] == [0, 0, 1]
    assert mount.id == {"type": diffsim_dash.MOUNT_TYPE, "index": "tem"}
