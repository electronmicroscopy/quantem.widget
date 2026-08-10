"""Tests for the ``widget`` CLI: content detection + image rendering end-to-end.

4D-STEM rendering needs a GPU + real master files, so it is exercised manually
(see docs); here we cover the routing logic and the image paths, which run on CPU.
"""
import json
import pathlib
import re
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from quantem.widget import cli


def _png(path, shape=(32, 32)):
    Image.fromarray((np.random.rand(*shape) * 255).astype("uint8")).save(path)


def test_embed_jpeg_adds_image_to_widget_only_output(tmp_path):
    png = tmp_path / "shot.png"
    _png(png, (24, 24))
    cell = {
        "cell_type": "code",
        "outputs": [{
            "output_type": "display_data",
            "metadata": {},
            "data": {
                "application/vnd.jupyter.widget-view+json": {
                    "model_id": "abc",
                    "version_major": 2,
                    "version_minor": 1,
                }
            },
        }],
    }

    assert cli._embed_jpeg(cell, png.read_bytes(), quality=80)
    data = cell["outputs"][0]["data"]
    assert "image/jpeg" in data
    assert "application/vnd.jupyter.widget-view+json" in data


def test_github_widget_cell_detector_includes_showeds():
    assert "ShowEDS(" in cli._WIDGET_CELL


def test_github_prepare_reuses_existing_image_outputs(tmp_path, monkeypatch):
    notebook = tmp_path / "show2d_github.ipynb"
    notebook.write_text(
        """{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "output_type": "display_data",
     "metadata": {},
     "data": {
      "text/plain": "<quantem.widget.show2d.Show2D>",
      "image/jpeg": "/9j/4AAQSkZJRgABAQAAAQABAAD/2w=="
     }
    }
   ],
   "source": [
    "from quantem.widget import Show2D\\n",
    "Show2D(data)"
   ]
  }
 ],
 "metadata": {
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {}
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
""",
        encoding="utf-8",
    )

    def fail_capture(*args, **kwargs):
        raise AssertionError("existing image outputs should not trigger browser capture")

    monkeypatch.setattr(cli, "_capture_full_ui", fail_capture)
    args = type("Args", (), {
        "path": str(notebook),
        "no_execute": True,
        "quality": 90,
        "timeout": 600,
    })()

    assert cli._prepare_github(args) == 0
    text = notebook.read_text(encoding="utf-8")
    assert "image/jpeg" in text
    assert "application/vnd.jupyter.widget-state+json" not in text


# ---------------------------------------------------------------------------
def test_detect_single_image(tmp_path):
    p = tmp_path / "a.png"
    _png(p)
    assert cli._detect(p, "auto") == "image"


def test_detect_image_folder(tmp_path):
    for i in range(3):
        _png(tmp_path / f"f{i}.png")
    assert cli._detect(tmp_path, "auto") == "images"


def test_detect_master_folder(tmp_path):
    (tmp_path / "scan_master.h5").write_bytes(b"\x00")
    assert cli._detect(tmp_path, "auto") == "4dstem"


def test_detect_master_wins_over_images(tmp_path):
    _png(tmp_path / "a.png")
    (tmp_path / "scan_master.h5").write_bytes(b"\x00")
    assert cli._detect(tmp_path, "auto") == "4dstem"


def _showptycho_folder(tmp_path):
    folder = tmp_path / "logic013_512_bfr24"
    folder.mkdir()
    source = folder / "source"
    source.mkdir()
    (source / "scan_master.h5").write_bytes(b"master")
    (source / "scan_data_000001.h5").write_bytes(b"data")
    (folder / "index.html").write_text("<!doctype html><title>ShowPtycho</title>", encoding="utf-8")
    snapshots = folder / "snapshots"
    snapshots.mkdir()
    (snapshots / "manifest.json").write_text(
        """{
  "schema_version": 2,
  "format": "quantem.showptycho.webgpu.folder.v2",
  "title": "ShowPtycho smoke",
  "source": {
    "kind": "hdf5",
    "master": "source/scan_master.h5",
    "data_files": ["source/scan_data_000001.h5"],
    "link_mode": ["hardlink"]
  },
  "arrays": {}
}
""",
        encoding="utf-8",
    )
    return folder


def test_detect_showptycho_folder_export(tmp_path):
    folder = _showptycho_folder(tmp_path)

    assert cli._detect(folder, "auto") == "showptycho"
    assert cli._detect(folder / "index.html", "auto") == "showptycho"
    assert cli._detect(folder, "showptycho") == "showptycho"


def test_showptycho_folder_rejects_retired_top_level_manifest(tmp_path):
    folder = tmp_path / "retired-export"
    folder.mkdir()
    (folder / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (folder / "manifest.json").write_text(
        json.dumps({"format": "quantem.showptycho.webgpu.folder.v2"}),
        encoding="utf-8",
    )

    assert cli._is_showptycho_folder_export(folder) is False
    with pytest.raises(ValueError, match="snapshots/manifest.json"):
        cli._showptycho_folder(folder)


def test_detect_showptycho_master_when_forced(tmp_path):
    """C1: explicit showptycho on a master builds ptychography, not Show4DSTEM."""
    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"\x00")

    assert cli._detect(master, "auto") == "4dstem"
    assert cli._detect(master, "showptycho") == "showptycho-master"


def test_showptycho_folder_builds_user_owned_anonymous_project(
    tmp_path,
    monkeypatch,
):
    """A folder command owns one catalog and one isolated result per master."""

    for name in ("first_master.h5", "second_master_wrapper.h5"):
        (tmp_path / name).write_bytes(b"\x00")

    def fake_render(master, args, *, out_dir=None):
        assert out_dir is not None
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(
            "<!doctype html><title>ShowPtycho</title>", encoding="utf-8"
        )
        snapshots = out_dir / "snapshots"
        snapshots.mkdir()
        (snapshots / "cal.json").write_text("{}\n", encoding="utf-8")
        (out_dir / "ssb_fit.json").write_text(
            json.dumps({"backend": "mps", "num_bf": 42, "loss": 0.125}),
            encoding="utf-8",
        )
        return out_dir

    def fake_raw_viewer(master, folder, *, label, target_stem=None):
        assert target_stem is not None
        viewer = folder / "show4dstem" / ".viewer" / "Show4DSTEM.html"
        viewer.parent.mkdir(parents=True)
        viewer.write_text("<!doctype html><title>Show4DSTEM</title>")
        return viewer

    served = {}
    monkeypatch.setattr(cli, "_render_showptycho_master", fake_render)
    monkeypatch.setattr(cli, "_write_show4dstem_viewer", fake_raw_viewer)
    monkeypatch.setattr(
        cli,
        "_serve_showptycho_folder",
        lambda folder, **kwargs: served.update(folder=folder, **kwargs),
    )
    output_root = tmp_path / "user" / "QuantEM" / "showptycho"
    monkeypatch.setattr(cli, "_default_showptycho_root", lambda: output_root)

    assert cli.main([
        "showptycho",
        str(tmp_path),
        "--trials", "0",
        "--anonymize",
        "--no-open",
    ]) == 0

    root = output_root / tmp_path.name
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert served["folder"] == root
    assert manifest["format"] == "quantem.showptycho.collection.v1"
    assert [item["label"] for item in manifest["datasets"]] == [
        "Dataset 001",
        "Dataset 002",
    ]
    assert all("source" not in item for item in manifest["datasets"])
    assert (root / "dataset-001" / "index.html").is_file()
    assert (root / "dataset-002" / "index.html").is_file()
    assert manifest["datasets"][0]["calibration"] == (
        "dataset-001/snapshots/cal.json"
    )
    assert (root / manifest["datasets"][0]["calibration"]).is_file()
    raw_viewer = root / manifest["datasets"][0]["show4dstem"]
    assert raw_viewer == (
        root / "dataset-001" / "show4dstem" / ".viewer" / "Show4DSTEM.html"
    )
    assert raw_viewer.is_file()
    assert (root / "ShowPtycho.command").is_file()
    assert cli._detect(root, "showptycho") == "showptycho-collection"


def test_showptycho_master_cli_uses_native_bin_default(tmp_path, monkeypatch):
    """C2: ptychography master generation keeps native detector pixels by default."""
    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"\x00")
    folder = tmp_path / "project"
    seen = {}

    def fake_collection(masters, args, *, source_dir=None):
        seen["path"] = masters[0]
        seen["det_bin"] = cli._effective_det_bin(args, default=1)
        return folder

    def fake_serve(path, *, bind, port, no_open):
        seen["served"] = path
        seen["no_open"] = no_open

    monkeypatch.setattr(cli, "_render_showptycho_collection", fake_collection)
    monkeypatch.setattr(cli, "_serve_showptycho_folder", fake_serve)

    assert cli.main(["showptycho", str(master), "--no-open"]) == 0
    assert seen["path"] == master.resolve()
    assert seen["det_bin"] == 1
    assert seen["served"] == folder
    assert seen["no_open"] is True


def test_showptycho_cli_routes_exact_mps_optimization_options(tmp_path, monkeypatch):
    """C2b: CLI exposes the canonical 200-trial + Nelder-Mead workflow."""
    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"\x00")
    seen = {}

    def fake_collection(masters, args, *, source_dir=None):
        seen["path"] = masters[0]
        seen["trials"] = args.trials
        seen["refinement"] = args.refinement
        seen["backend"] = args.backend
        seen["drag_bf"] = args.drag_bf
        return tmp_path / "project"

    monkeypatch.setattr(cli, "_render_showptycho_collection", fake_collection)
    monkeypatch.setattr(cli, "_serve_showptycho_folder", lambda *args, **kwargs: None)

    assert cli.main([
        "showptycho",
        str(master),
        "--trials", "200",
        "--refinement", "nelder-mead",
        "--backend", "mps",
        "--no-open",
    ]) == 0
    assert seen == {
        "path": master.resolve(),
        "trials": 200,
        "refinement": "nelder-mead",
        "backend": "mps",
        "drag_bf": 1.0,
    }


def test_showptycho_replaces_old_ptycho_command(capsys):
    """C3: stale CLI name, expect argparse to reject it without an alias."""

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["ptycho"])

    assert exc_info.value.code == 2
    assert "showptycho" in capsys.readouterr().err


def test_cli_exposes_only_widget_and_export_commands(capsys):
    """C4: top-level help, expect no acquisition or infrastructure commands."""

    assert cli.main([]) == 0
    output = capsys.readouterr().out

    assert "showptycho" in output
    assert "data-transfer" not in output
    assert "jupyter" not in output
    assert "screen" not in output


def test_showptycho_in_place_is_explicit(tmp_path, monkeypatch):
    """C5: shared source, expect writes beside it only with --in-place."""

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"\x00")
    args = SimpleNamespace(out=None, in_place=True)

    target = cli._showptycho_collection_output_dir([master], args, None)

    assert target == tmp_path / "quantem" / "showptycho"


def test_showptycho_rejects_out_with_in_place(tmp_path):
    """C6: conflicting ownership options, expect a deterministic error."""

    master = tmp_path / "scan_master.h5"
    args = SimpleNamespace(out=str(tmp_path / "results"), in_place=True)

    with pytest.raises(ValueError, match="either --out or --in-place"):
        cli._showptycho_collection_output_dir([master], args, None)


def test_showptycho_writes_direct_show4dstem_viewer(tmp_path, monkeypatch):
    """C7: raw-data companion, expect one direct browser Show4DSTEM viewer."""

    from quantem.widget import show4dstem_webgpu_export

    master = tmp_path / "scan_master.h5"
    folder = tmp_path / "project" / "scan"
    snapshots = folder / "snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "cal.json").write_text(json.dumps({
        "scan_region": {"shape": [512, 512]},
        "detector_shape": [192, 192],
    }))
    seen = {}

    def fake_export(path, out_dir, **kwargs):
        seen.update(master=path, out_dir=out_dir, **kwargs)
        viewer = out_dir / ".viewer" / "Show4DSTEM.html"
        viewer.parent.mkdir(parents=True)
        viewer.write_text("<!doctype html><title>Show4DSTEM</title>")
        return viewer

    monkeypatch.setattr(
        show4dstem_webgpu_export,
        "export_show4dstem_hdf5_viewer",
        fake_export,
    )
    viewer = cli._write_show4dstem_viewer(master, folder, label="scan")

    assert viewer.is_file()
    assert seen["master"] == master
    assert seen["out_dir"] == folder / "show4dstem"
    assert seen["scan_shape"] == (512, 512)
    assert seen["detector_shape"] == (192, 192)
    assert seen["target_stem"] is None


def test_show4dstem_cli_count_defaults_to_full_detector(tmp_path, monkeypatch):
    """C3: Show4DSTEM CLI count gates use native detector pixels by default."""
    for idx in range(2):
        (tmp_path / f"scan_{idx}_master.h5").write_bytes(b"\x00")
    seen = {}

    def fake_discover(path, verbose=False):
        seen["discover_path"] = path
        return [str(tmp_path / "scan_0_master.h5"), str(tmp_path / "scan_1_master.h5")]

    def fake_render(masters, label, args, *, source_path=None):
        seen["masters"] = masters
        seen["label"] = label
        seen["det_bin"] = args.det_bin
        seen["backend"] = args.backend
        seen["source_path"] = source_path
        return tmp_path / "viewer.ipynb"

    def fake_launch(notebook, *, no_open):
        seen["notebook"] = notebook
        seen["no_open"] = no_open

    monkeypatch.setattr("quantem.gpu.io.discover", fake_discover)
    monkeypatch.setattr(cli, "_render_4dstem_notebook", fake_render)
    monkeypatch.setattr(cli, "_launch_notebook", fake_launch)

    assert cli.main(["show4dstem", str(tmp_path), "--count", "1", "--backend", "mps", "--no-open"]) == 0
    assert seen["discover_path"] == str(tmp_path.resolve())
    assert seen["masters"] == [str(tmp_path / "scan_0_master.h5")]
    assert seen["label"] == tmp_path.name
    assert seen["det_bin"] == 1
    assert seen["backend"] == "mps"
    assert seen["source_path"] == tmp_path.resolve()
    assert seen["notebook"] == tmp_path / "viewer.ipynb"
    assert seen["no_open"] is True


def test_show4dstem_cli_count_requires_enough_masters(tmp_path, monkeypatch):
    """C4: a seven-tilt command fails instead of silently running fewer tilts."""
    (tmp_path / "scan_0_master.h5").write_bytes(b"\x00")

    def fake_discover(path, verbose=False):
        return [str(tmp_path / "scan_0_master.h5")]

    monkeypatch.setattr("quantem.gpu.io.discover", fake_discover)

    assert cli.main(["show4dstem", str(tmp_path), "--count", "7", "--no-open"]) == 1


def test_show4dstem_webgpu_cli_opens_generated_command(tmp_path, monkeypatch):
    """C5: WebGPU CLI uses the browser HDF5-backed export entry path."""
    (tmp_path / "scan_0_master.h5").write_bytes(b"\x00")
    out = tmp_path / "artifact" / "index.html"
    out.parent.mkdir()
    command = out.parent / "Show4DSTEM.command"
    command.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    seen = {}

    def fake_discover(path, verbose=False):
        return [str(tmp_path / "scan_0_master.h5")]

    def fake_render(masters, label, args):
        seen["masters"] = masters
        seen["label"] = label
        seen["det_bin"] = args.det_bin
        seen["backend"] = args.backend
        return out

    def fake_open(path, *, no_open):
        seen["opened"] = path
        seen["no_open"] = no_open

    monkeypatch.setattr("quantem.gpu.io.discover", fake_discover)
    monkeypatch.setattr(cli, "_render_4dstem_webgpu_h5", fake_render)
    monkeypatch.setattr(cli, "_open_show4dstem_command", fake_open)

    assert cli.main([
        "show4dstem",
        str(tmp_path),
        "--backend",
        "webgpu",
        "--html",
        "--count",
        "1",
        "--no-open",
    ]) == 0
    assert seen["masters"] == [str(tmp_path / "scan_0_master.h5")]
    assert seen["label"] == tmp_path.name
    assert seen["det_bin"] == 1
    assert seen["backend"] == "webgpu"
    assert seen["opened"] == command
    assert seen["no_open"] is True


@pytest.mark.parametrize(
    ("count", "view_mode", "dp_mode"),
    [(1, "single", "average"), (2, "multiple", "selected"), (7, "multiple", "selected")],
)
def test_render_show4dstem_webgpu_h5_uses_anonymous_h5_urls(
    tmp_path, monkeypatch, count, view_mode, dp_mode
):
    """C6: WebGPU CLI export links source H5 masters instead of preprocessing them."""
    import quantem.widget as qw

    masters = []
    for idx in range(count):
        master = tmp_path / f"private_source_{idx}_master.h5"
        master.write_bytes(f"private-{idx}".encode())
        (tmp_path / f"private_source_{idx}_data_000001.h5").write_bytes(f"chunk-{idx}".encode())
        masters.append(str(master))
    seen = {}

    def fake_contract(master):
        return {"scan_shape": (4, 4), "detector_shape": (8, 8), "n_frames": 16}

    def fake_export(widget, out_dir, *, title=None, h5_decode_dtype=None):
        seen["bundle"] = {
            "out_dir": pathlib.Path(out_dir),
            "title": title,
            "h5_decode_dtype": h5_decode_dtype,
        }
        root = pathlib.Path(out_dir)
        (root / ".viewer").mkdir()
        (root / ".viewer" / "Show4DSTEM.html").write_text("<!doctype html>", encoding="utf-8")
        (root / "Show4DSTEM.command").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    class FakeShow4DSTEM:
        def __init__(self, data, **kwargs):
            seen["kwargs"] = kwargs

    monkeypatch.setattr("quantem.widget.show4dstem_factory._master_file_contract", fake_contract)
    monkeypatch.setattr(
        "quantem.widget.show4dstem_webgpu_export.export_show4dstem_webgpu_bundle",
        fake_export,
    )
    monkeypatch.setattr(qw, "Show4DSTEM", FakeShow4DSTEM)
    args = SimpleNamespace(det_bin=1, dtype="u8", out=str(tmp_path / "out"), title=None, verbose=False)

    html = cli._render_4dstem_webgpu_h5(masters, "private_folder", args)

    assert html == tmp_path / "out" / "private_folder_show4dstem_webgpu" / "index.html"
    assert "lazy_urls" not in seen["kwargs"]
    assert seen["kwargs"]["h5_urls"] == [
        f"../data/dataset_{idx:02d}_master.h5" for idx in range(count)
    ]
    assert seen["kwargs"]["backend"] == "webgpu"
    assert seen["kwargs"]["scan_shape"] == (4, 4)
    assert seen["kwargs"]["detector_shape"] == (8, 8)
    assert seen["kwargs"]["view_mode"] == view_mode
    assert seen["kwargs"]["compare_max_panels"] == count
    assert seen["kwargs"]["compare_group_mode"] == "all"
    assert seen["kwargs"]["compare_dp_mode"] == dp_mode
    assert seen["bundle"]["h5_decode_dtype"] == "uint8"
    assert seen["bundle"]["out_dir"] == html.parent
    assert (html.parent / "data" / "dataset_00_master.h5").is_symlink()
    assert (html.parent / "data" / "dataset_00_master.h5").resolve() == pathlib.Path(masters[0])
    assert (html.parent / "data" / "dataset_00_data_000001.h5").is_symlink()
    assert (html.parent / "data" / "dataset_00_data_000001.h5").resolve() == tmp_path / "private_source_0_data_000001.h5"
    assert not (html.parent / "data" / "dataset_00_lazy").exists()


def test_show4dstem_generated_master_links_are_not_input_candidates(tmp_path):
    """A rerun must ignore the anonymous links in its owned export folder."""
    source = tmp_path / "scan_master.h5"
    source.write_bytes(b"source")
    generated = tmp_path / "scan_show4dstem_webgpu"
    generated.mkdir()
    link = generated / "dataset_00_master.h5"
    link.symlink_to(source)

    assert not cli._is_show4dstem_generated_master_link(source)
    assert cli._is_show4dstem_generated_master_link(link)


def test_show4dstem_dataset_label_uses_coordinates_when_available():
    master = "experiment_-8.5x_14.72y_run_master.h5"
    second_master = "experiment_17.0x_0.0y_run_master.h5"

    assert cli._show4dstem_dataset_label(master, 2) == "Tilt (-8.5, +14.72)"
    assert cli._show4dstem_dataset_label(second_master, 2) == "Tilt (+17.0, +0.0)"
    assert cli._show4dstem_dataset_label("unknown_master.h5", 2) == "Dataset 3"


def test_render_show4dstem_folder_notebook_records_backend_count_and_devices(tmp_path):
    """C7: generated CUDA folder notebooks preserve the seven-entry gate options."""
    args = SimpleNamespace(
        backend="cuda",
        det_bin=1,
        dtype="u8",
        gpus="0,1",
        page_budget="auto",
        out=str(tmp_path),
    )

    notebook = cli._render_4dstem_notebook(
        [str(tmp_path / f"tilt_{idx:02d}_master.h5") for idx in range(7)],
        "seven",
        args,
        source_path=tmp_path,
    )

    text = notebook.read_text(encoding="utf-8")
    assert "Show4DSTEM.from_folder(" in text
    assert "backend='cuda'" in text
    assert "max_masters=7" in text
    assert "min_masters=7" in text
    assert "det_bin=1" in text
    assert "dtype='u8'" in text
    assert "gpus = [0, 1]" in text


def test_showptycho_auto_calibration_selects_matching_source(tmp_path):
    """C7: automatic calibration search picks the matching microscope source."""
    master = tmp_path / "reference_512_master.h5"
    master.write_bytes(b"\x00")
    cal_dir = tmp_path / "quantem" / "showptycho" / "reference_512"
    cal_dir.mkdir(parents=True)
    cal_path = cal_dir / "calibration.json"
    cal_path.write_text(
        """[
  {
    "source_stem": "reference_511",
    "rotation_angle_deg": 1,
    "aberrations": {"C10": 2, "C12": 3, "phi12": 0.1},
    "loss": 9
  },
  {
    "source_stem": "reference_512",
    "rotation_angle_deg": 158.9,
    "aberrations": {"C10": 78.1, "C12": 17.4, "phi12": 0.58},
    "semiangle_mrad": 30,
    "scan_sampling_A": 0.264,
    "voltage_kV": 300,
    "loss": 0.01
  }
]""",
        encoding="utf-8",
    )
    args = type("Args", (), {"calibration": "auto"})()

    calibration, path = cli._resolve_showptycho_calibration(master, args)

    assert path == cal_path
    assert calibration.source_stem == "reference_512"
    assert calibration.rotation_angle_deg == 158.9
    assert calibration.semiangle_mrad == 30


def test_ptycho_geometry_defaults_when_calibration_missing():
    args = SimpleNamespace(
        semiangle_mrad=None,
        scan_sampling_A=None,
        voltage_kv=None,
        det_sampling_mrad_px=None,
    )

    semiangle, scan_sampling, voltage, det_sampling, warnings = (
        cli._resolve_showptycho_geometry(args, None, {})
    )

    assert semiangle == cli.DEFAULT_PTYCHO_SEMIANGLE_MRAD
    assert scan_sampling == cli.DEFAULT_PTYCHO_SCAN_SAMPLING_A
    assert voltage == cli.DEFAULT_PTYCHO_VOLTAGE_KV
    assert det_sampling is None
    assert len(warnings) == 3
    assert "--semiangle" in warnings[0]
    assert "--scan-sampling" in warnings[1]
    assert "--voltage-kv" in warnings[2]


def test_ptycho_geometry_prefers_cli_then_calibration_then_metadata():
    args = SimpleNamespace(
        semiangle_mrad=None,
        scan_sampling_A=0.31,
        voltage_kv=None,
        det_sampling_mrad_px=None,
    )
    calibration = SimpleNamespace(
        semiangle_mrad=28,
        scan_sampling_A=0.27,
        voltage_kV=200,
        det_sampling_mrad_px=None,
    )
    meta = {
        "semiangle_mrad": 22,
        "voltage_kV": 120,
        "det_sampling_mrad_px": 0.05,
    }

    semiangle, scan_sampling, voltage, det_sampling, warnings = (
        cli._resolve_showptycho_geometry(args, calibration, meta)
    )

    assert semiangle == 28
    assert scan_sampling == 0.31
    assert voltage == 200
    assert det_sampling == 0.05
    assert warnings == []


def test_ptycho_geometry_rejects_bad_explicit_value():
    args = SimpleNamespace(
        semiangle_mrad=None,
        scan_sampling_A=0,
        voltage_kv=None,
        det_sampling_mrad_px=None,
    )

    with pytest.raises(ValueError, match="--scan-sampling"):
        cli._resolve_showptycho_geometry(args, None, {})


def test_detect_forced_4dstem(tmp_path):
    _png(tmp_path / "a.png")
    assert cli._detect(tmp_path, "4dstem") == "4dstem"


def test_detect_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        cli._detect(tmp_path, "auto")


def test_detect_unsupported_file_raises(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hi")
    with pytest.raises(ValueError):
        cli._detect(p, "auto")


# ---------------------------------------------------------------------------
def test_show_single_image_writes_html(tmp_path):
    p = tmp_path / "img.png"
    _png(p, (48, 48))
    dest = tmp_path / "out"
    assert cli.main(["show", str(p), "--no-open", "--out", str(dest) + "/"]) == 0
    out = dest / "img_show2d.html"
    assert out.exists() and out.stat().st_size > 50_000


def test_show_same_size_folder_is_show3d(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    for i in range(4):
        _png(src / f"frame_{i}.png", (40, 40))
    dest = tmp_path / "out"
    assert cli.main(["show", str(src), "--no-open", "--out", str(dest) + "/"]) == 0
    out = dest / "frames_show3d.html"
    assert out.exists() and out.stat().st_size > 50_000


def test_show_mixed_size_folder_is_gallery(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    _png(src / "a.png", (32, 32))
    _png(src / "b.png", (64, 48))
    dest = tmp_path / "out"
    assert cli.main(["show", str(src), "--no-open", "--out", str(dest) + "/"]) == 0
    out = dest / "frames_gallery.html"
    assert out.exists() and out.stat().st_size > 50_000


def test_4dstem_default_writes_notebook(tmp_path):
    src = tmp_path / "data"
    src.mkdir()
    (src / "scan_master.h5").write_bytes(b"\x00")
    dest = tmp_path / "out"
    # --no-open avoids launching jupyter; we only check the notebook is written + valid.
    assert cli.main(["show", str(src), "--no-open", "--out", str(dest)]) == 0
    notebooks = list(dest.glob("*.ipynb"))
    assert len(notebooks) == 1
    import json
    nb = json.loads(notebooks[0].read_text())
    code = "".join(nb["cells"][1]["source"])
    assert "Show4DSTEM.from_folder(" in code
    assert "det_bin=1" in code
    assert "max_masters=1" in code


def test_multiple_masters_one_5d_notebook(tmp_path):
    m1 = tmp_path / "a_master.h5"
    m2 = tmp_path / "b_master.h5"
    m1.write_bytes(b"\x00")
    m2.write_bytes(b"\x00")
    dest = tmp_path / "out"
    assert cli.main(["show", str(m1), str(m2), "--no-open", "--out", str(dest)]) == 0
    notebooks = list(dest.glob("*.ipynb"))
    assert len(notebooks) == 1
    import json
    code = "".join(json.loads(notebooks[0].read_text())["cells"][1]["source"])
    # Both explicit masters stay in one load call -> one 5D viewer.
    assert "masters = [" in code and "a_master.h5" in code and "b_master.h5" in code
    assert "det_bin=1" in code


def test_multiple_images_one_gallery(tmp_path):
    _png(tmp_path / "a.png", (32, 32))
    _png(tmp_path / "b.png", (40, 40))
    dest = tmp_path / "out"
    assert cli.main(["show", str(tmp_path / "a.png"), str(tmp_path / "b.png"),
                     "--no-open", "--out", str(dest) + "/"]) == 0
    assert (dest / "gallery.html").exists()


def test_show3d_subcommand_forces_stack(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    for i in range(3):
        _png(src / f"f{i}.png", (36, 36))
    dest = tmp_path / "out"
    assert cli.main(["show3d", str(src), "--no-open", "--out", str(dest) + "/"]) == 0
    assert (dest / "frames_show3d.html").exists()


def test_show2d_subcommand_folder_is_gallery(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    for i in range(3):
        _png(src / f"f{i}.png", (36, 36))  # same size, but show2d forces a gallery
    dest = tmp_path / "out"
    assert cli.main(["show2d", str(src), "--no-open", "--out", str(dest) + "/"]) == 0
    assert (dest / "frames_gallery.html").exists()


def test_show2d_folder_watch_writes_live_notebook(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    _png(src / "f0.png", (36, 36))
    dest = tmp_path / "out"

    assert cli.main([
        "show2d",
        str(src),
        "--watch",
        "--watch-interval",
        "0.5",
        "--no-open",
        "--out",
        str(dest),
    ]) == 0

    notebooks = list(dest.glob("*_show2d_live.ipynb"))
    assert len(notebooks) == 1
    import json

    code = "".join(json.loads(notebooks[0].read_text())["cells"][1]["source"])
    assert "ShowFolder(" in code
    assert "open_show2d(all_images=True)" in code
    assert "folder.watch(interval=0.5)" in code


def test_show3d_folder_watch_writes_live_notebook(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    _png(src / "f0.png", (36, 36))
    dest = tmp_path / "out"

    assert cli.main(["show3d", str(src), "--watch", "--no-open", "--out", str(dest)]) == 0

    notebooks = list(dest.glob("*_show3d_live.ipynb"))
    assert len(notebooks) == 1
    import json

    code = "".join(json.loads(notebooks[0].read_text())["cells"][1]["source"])
    assert "ShowFolder(" in code
    assert "open_show3d(all_images=True)" in code
    assert "folder.watch(interval=2.0)" in code


def test_show4dstem_subcommand_writes_notebook(tmp_path):
    (tmp_path / "scan_master.h5").write_bytes(b"\x00")
    dest = tmp_path / "out"
    assert cli.main(["show4dstem", str(tmp_path / "scan_master.h5"), "--no-open", "--out", str(dest)]) == 0
    assert list(dest.glob("*.ipynb"))


def test_show4dstem_folder_watch_writes_live_notebook(tmp_path):
    source = tmp_path / "live"
    source.mkdir()
    (source / "scan_000_master.h5").write_bytes(b"\x00")
    dest = tmp_path / "out"

    assert cli.main([
        "show4dstem",
        str(source),
        "--watch",
        "--bin",
        "4",
        "--gpus",
        "0,1",
        "--page-budget",
        "2",
        "--watch-interval",
        "1.5",
        "--no-open",
        "--out",
        str(dest),
    ]) == 0

    notebooks = list(dest.glob("*_live.ipynb"))
    assert len(notebooks) == 1
    import json

    code = "".join(json.loads(notebooks[0].read_text())["cells"][1]["source"])
    assert "ShowFolder(" in code
    assert "attach_selection_panel()" in code
    assert "open_show4dstem(" in code
    assert "gpus=[0, 1]" in code
    assert "page_budget=2" in code
    assert "det_bin=4" in code
    assert "folder.watch(interval=1.5)" in code


def test_show4dstem_watch_requires_live_folder_notebook(tmp_path):
    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"\x00")

    assert cli.main(["show4dstem", str(master), "--watch", "--no-open"]) == 1
    assert cli.main(["show4dstem", str(tmp_path), "--watch", "--html", "--no-open"]) == 1


def test_showptycho_cli_validates_folder_without_opening(tmp_path, capsys):
    folder = _showptycho_folder(tmp_path)

    assert cli.main(["showptycho", str(folder), "--no-open"]) == 0

    out = capsys.readouterr().out
    assert "ShowPtycho folder:" in out
    assert "compressed HDF5" in out
    assert "browser source: compressed_hdf5" in out
    assert "no persistent BF-G cache" in out
    assert "ready: run without --no-open" in out


def test_show_auto_routes_showptycho_folder(tmp_path, capsys):
    folder = _showptycho_folder(tmp_path)

    assert cli.main(["show", str(folder), "--no-open"]) == 0

    out = capsys.readouterr().out
    assert "ShowPtycho folder:" in out


def test_showptycho_range_parser_accepts_first_bytes():
    assert cli._parse_http_range("bytes=0-3", 16) == (0, 3)
    assert cli._parse_http_range("bytes=4-", 16) == (4, 15)
    assert cli._parse_http_range("bytes=-4", 16) == (12, 15)
    assert cli._parse_http_range("bytes=99-100", 16) is None


def test_showptycho_range_handler_serves_bf_column_partial_content(tmp_path):
    """C6: ShowPtycho folder server, expect real byte-range BF-column reads."""
    import http.client
    import http.server
    import threading

    folder = tmp_path / "showptycho-folder"
    source = folder / "source"
    source.mkdir(parents=True)
    payload = bytes(range(16))
    (source / "bf_columns.u8").write_bytes(payload)

    handler = type("TestRangeHandler", (cli._RangeRequestHandler,), {"root": folder})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    conn = None
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request("GET", "/source/bf_columns.u8", headers={"Range": "bytes=2-5"})
        response = conn.getresponse()
        body = response.read()

        assert response.status == 206
        assert response.getheader("Accept-Ranges") == "bytes"
        assert response.getheader("Content-Range") == "bytes 2-5/16"
        assert body == payload[2:6]
    finally:
        if conn is not None:
            conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_showptycho_range_handler_writes_snapshots_only(tmp_path):
    """C6: ShowPtycho folder server, expect persisted snapshots without saves/."""
    import http.client
    import http.server
    import threading

    folder = tmp_path / "showptycho-folder"
    folder.mkdir()

    handler = type("TestRangeHandler", (cli._RangeRequestHandler,), {"root": folder})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    conn = None
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request("PUT", "/snapshots/snapshots.json", body=b'[{"C10": 1}]')
        response = conn.getresponse()
        assert response.status == 204
        response.read()
        assert (folder / "snapshots" / "snapshots.json").read_bytes() == b'[{"C10": 1}]'
        assert not (folder / "saves").exists()

        conn.request(
            "PUT",
            "/dataset-001/snapshots/snapshots.json",
            body=b'[{"C10": 2}]',
        )
        response = conn.getresponse()
        assert response.status == 204
        response.read()
        assert (
            folder / "dataset-001" / "snapshots" / "snapshots.json"
        ).read_bytes() == b'[{"C10": 2}]'

        conn.request("PUT", "/snapshots/snapshot_test.jpg", body=b"jpeg")
        response = conn.getresponse()
        assert response.status == 204
        response.read()
        assert (folder / "snapshots" / "snapshot_test.jpg").read_bytes() == b"jpeg"

        conn.request("DELETE", "/snapshots/snapshot_test.jpg")
        response = conn.getresponse()
        assert response.status == 204
        response.read()
        assert not (folder / "snapshots" / "snapshot_test.jpg").exists()

        conn.request("PUT", "/source/bad.txt", body=b"bad")
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        assert not (folder / "source" / "bad.txt").exists()
    finally:
        if conn is not None:
            conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_show4dstem_html_cli_threads_full_dtype_to_load_and_export() -> None:
    """C1: CLI full export docs, expect --dtype uint16 to reach load and export."""
    import inspect

    source = inspect.getsource(cli._render_4dstem)
    loader_source = inspect.getsource(cli._master_to_binned_numpy)

    assert "export_dtype = _show4dstem_export_dtype(args)" in source
    assert "_master_to_binned_numpy(master, args.det_bin, args.dtype)" in source
    assert "widget.export_html(str(out), title=args.title or stem, dtype=export_dtype)" in source
    assert "load(master, det_bin=det_bin, dtype=dtype)" in loader_source
    assert cli._show4dstem_export_dtype(SimpleNamespace(dtype="uint16")) == "uint16"
    assert cli._show4dstem_export_dtype(SimpleNamespace(dtype="u16")) == "uint16"
    assert cli._show4dstem_export_dtype(SimpleNamespace(dtype="uint8")) == "uint8"


def test_showptycho_cli_threads_explicit_dtype_to_ssb_open() -> None:
    """C1b: ShowPtycho optimization must honor its requested load dtype."""
    import inspect

    source = inspect.getsource(cli._render_showptycho_master)

    assert "dtype=_showptycho_decode_dtype(args)" in source


def test_showptycho_fit_records_compute_and_ui_provenance(monkeypatch) -> None:
    """C1c: fit records identify all packages without local source paths."""
    import inspect

    seen = []

    def fake_source_state(*, version, module_file):
        seen.append((version, module_file))
        return {"version": version, "commit": "abc123", "dirty": False}

    monkeypatch.setattr(cli, "_package_source_state", fake_source_state)

    provenance = cli._showptycho_software_provenance()
    render_source = inspect.getsource(cli._render_showptycho_master)

    assert set(provenance) == {"quantem", "quantem.gpu", "quantem.widget"}
    assert all(state["commit"] == "abc123" for state in provenance.values())
    assert all("/" not in key for key in provenance)
    assert len(seen) == 3
    assert "software = _showptycho_software_provenance()" in render_source
    assert '"software": software' in render_source
    assert '"export_software": software' in render_source


def test_showptycho_anonymization_preserves_science_and_redacts_sources() -> None:
    payload = {
        "source_path": "/private/session/sample_master.h5",
        "loss": 0.125,
        "calibration": {
            "source_file": "/private/session/sample_master.h5",
            "aberrations": {"C10": 73.0},
        },
        "trials": [{"value": 0.2}],
    }

    redacted = cli._anonymize_showptycho_payload(payload)
    assert redacted["source_path"] == "redacted_local_source"
    assert redacted["calibration"]["source_file"] == "redacted_local_source"
    assert redacted["loss"] == payload["loss"]
    assert redacted["calibration"]["aberrations"] == {"C10": 73.0}
    assert redacted["trials"] == payload["trials"]


def test_showptycho_reused_fit_records_current_export_software(
    tmp_path,
    monkeypatch,
) -> None:
    fit_record = tmp_path / "ssb_fit.json"
    fit_record.write_text(json.dumps({
        "source_path": "/private/acquisition/master.h5",
        "software": {"quantem.gpu": {"commit": "fit-commit"}},
        "loss": 0.125,
    }))
    current = {"quantem.gpu": {"commit": "export-commit", "dirty": False}}
    monkeypatch.setattr(cli, "_showptycho_software_provenance", lambda: current)

    payload = cli._showptycho_reused_fit_payload(
        fit_record,
        anonymize=True,
    )

    assert payload["software"]["quantem.gpu"]["commit"] == "fit-commit"
    assert payload["export_software"] == current
    assert payload["source_path"] == "redacted_local_source"


def test_show4dstem_html_cli_rejects_float32_export_dtype() -> None:
    """C1: CLI HTML export, expect float32 to stay a live-notebook workflow."""
    with pytest.raises(ValueError, match="Use a live notebook for float32 analysis"):
        cli._show4dstem_export_dtype(SimpleNamespace(dtype="float32"))


def test_out_path_explicit_file(tmp_path):
    p = tmp_path / "img.png"
    _png(p)
    dest = tmp_path / "custom" / "viewer.html"
    assert cli.main(["show", str(p), "--no-open", "--out", str(dest)]) == 0
    assert dest.exists()


# ---------------------------------------------------------------------------
def _ring_pattern(size=256, radii=(60.0, 90.0)):
    center = (size - 1) / 2
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    r = np.hypot(rows - center, cols - center)
    pattern = 300.0 * np.exp(-(r**2) / (2 * 8.0**2)) + 20.0 * np.exp(-r / 40.0)
    for radius in radii:
        pattern += 30.0 * np.exp(-((r - radius) ** 2) / (2 * 2.5**2))
    return pattern.astype(np.float32)


def test_showdiffraction_writes_html(tmp_path, monkeypatch):
    # a 2D pattern, a 3D stack, and the --demo path all export
    p = tmp_path / "pattern.npy"
    np.save(p, _ring_pattern())
    dest = tmp_path / "out"
    assert cli.main(["showdiffraction", str(p), "--no-open", "--out", str(dest) + "/"]) == 0
    out = dest / "pattern_showdiffraction.html"
    assert out.exists() and out.stat().st_size > 50_000

    stack = tmp_path / "stack.npy"
    np.save(stack, np.stack([_ring_pattern(), _ring_pattern(radii=(50.0, 80.0))]))
    argv = ["showdiffraction", str(stack), "--no-auto", "--no-open", "--out", str(dest) + "/"]
    assert cli.main(argv) == 0
    assert (dest / "stack_showdiffraction.html").exists()

    monkeypatch.setattr(
        "quantem.widget.data.tutorials.showdiffraction_fe3o4",
        lambda **kwargs: _ring_pattern(),
    )
    assert cli.main(["showdiffraction", "--demo", "--no-open", "--out", str(dest) + "/"]) == 0
    demo = dest / "fe3o4_saed_showdiffraction.html"
    assert demo.exists() and demo.stat().st_size > 50_000


def test_showdiffraction_phase_modes(tmp_path, capsys):
    # --phase calibrates and indexes, an explicit --k-pixel-size survives it,
    # and --no-auto still preselects the phase
    from quantem.widget import library_phase

    au = library_phase("Au")
    size = 512
    center = (size - 1) / 2
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    r = np.hypot(rows - center, cols - center)
    pattern = 300.0 * np.exp(-(r**2) / (2 * 8.0**2)) + 20.0 * np.exp(-r / 40.0)
    for refl in au.reflections(d_min=1.2):
        pattern += 30.0 * np.exp(-((r - 1.0 / (refl["d"] * 0.004)) ** 2) / (2 * 2.5**2))
    p = tmp_path / "au.npy"
    np.save(p, pattern.astype(np.float32))
    dest = tmp_path / "out"

    argv = ["showdiffraction", str(p), "--phase", "Au", "--max-rings", "4",
            "--no-open", "--out", str(dest) + "/"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert re.search(r"0\.00(39|40|41) 1/Å", out)
    assert re.search(r"Au \(fcc\): \d/4 matched", out)
    assert (dest / "au_showdiffraction.html").exists()

    argv = ["showdiffraction", str(p), "--phase", "Au", "--k-pixel-size", "0.005",
            "--no-open", "--out", str(dest) + "/"]
    assert cli.main(argv) == 0
    assert "0.0050 1/Å" in capsys.readouterr().out

    ring = tmp_path / "pattern.npy"
    np.save(ring, _ring_pattern())
    argv = ["showdiffraction", str(ring), "--no-auto", "--phase", "Fe3O4",
            "--no-open", "--out", str(dest) + "/"]
    assert cli.main(argv) == 0
    assert "Fe3O4" in (dest / "pattern_showdiffraction.html").read_text(encoding="utf-8")


def test_showdiffraction_bad_inputs_error_cleanly(tmp_path, capsys):
    assert cli.main(["showdiffraction"]) == 1
    assert "--demo" in capsys.readouterr().err

    p = tmp_path / "pattern.npy"
    np.save(p, _ring_pattern())
    assert cli.main(["showdiffraction", str(p), "--demo", "--no-open"]) == 1
    assert "not both" in capsys.readouterr().err

    assert cli.main(["showdiffraction", str(p), "--phase", "Nope", "--no-open"]) == 1
    assert "unknown library phase" in capsys.readouterr().err

    notes = tmp_path / "notes.txt"
    notes.write_text("hi")
    assert cli.main(["showdiffraction", str(notes), "--no-open"]) == 1
    assert "unsupported file type" in capsys.readouterr().err

    assert cli.main(["showdiffraction", str(tmp_path), "--no-open"]) == 1
    assert "not a file" in capsys.readouterr().err

    empty = tmp_path / "empty.npy"
    empty.write_bytes(b"")
    assert cli.main(["showdiffraction", str(empty), "--no-open"]) == 1
    assert "could not read" in capsys.readouterr().err
