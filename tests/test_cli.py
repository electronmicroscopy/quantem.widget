"""Tests for the ``widget`` CLI: content detection + image rendering end-to-end.

4D-STEM rendering needs a GPU + real master files, so it is exercised manually
(see docs); here we cover the routing logic and the image paths, which run on CPU.
"""
import numpy as np
import pytest
from PIL import Image

from quantem.widget import cli


def _png(path, shape=(32, 32)):
    Image.fromarray((np.random.rand(*shape) * 255).astype("uint8")).save(path)


def test_jupyter_launch_enables_widget_state_save(tmp_path, monkeypatch):
    monkeypatch.setenv("JUPYTERLAB_SETTINGS_DIR", str(tmp_path / "lab-settings"))
    settings_path = cli._enable_jupyterlab_widget_state_save()

    assert settings_path == (
        tmp_path
        / "lab-settings"
        / "@jupyter-widgets"
        / "jupyterlab-manager"
        / "plugin.jupyterlab-settings"
    )
    assert '"saveState": true' in settings_path.read_text()

    settings_path.write_text('// comment from JupyterLab\n{"other": 3, "saveState": false}\n')
    cli._enable_jupyterlab_widget_state_save()
    assert '"other": 3' in settings_path.read_text()
    assert '"saveState": true' in settings_path.read_text()


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
    assert "Show4DSTEM(load(" in code and "det_bin=8" in code


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
    # Both masters in a single load([...]) -> one 5D viewer.
    assert "load([" in code and "a_master.h5" in code and "b_master.h5" in code


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


def test_data_transfer_cli_plan_inspect_copy(tmp_path, monkeypatch):
    import quantem.widget.io.hdf5 as hdf5

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "scan_000_master.h5").write_bytes(b"master")
    (source / "scan_000_data_000001.h5").write_bytes(b"data")
    manifest = tmp_path / "transfer.json"

    monkeypatch.setattr(hdf5, "disk_of", lambda path: "disk")
    monkeypatch.setattr(hdf5, "is_master_ready", lambda path: True)

    assert cli.main([
        "data-transfer",
        "plan",
        str(source),
        str(target),
        "--manifest",
        str(manifest),
    ]) == 0
    assert manifest.exists()

    assert cli.main([
        "data-transfer",
        "inspect",
        "--manifest",
        str(manifest),
    ]) == 0

    assert cli.main([
        "data-transfer",
        "copy",
        "--manifest",
        str(manifest),
    ]) == 0
    assert not (target / "scan_000_master.h5").exists()

    assert cli.main([
        "data-transfer",
        "copy",
        "--manifest",
        str(manifest),
        "--execute",
    ]) == 0
    assert (target / "scan_000_master.h5").read_bytes() == b"master"


def test_out_path_explicit_file(tmp_path):
    p = tmp_path / "img.png"
    _png(p)
    dest = tmp_path / "custom" / "viewer.html"
    assert cli.main(["show", str(p), "--no-open", "--out", str(dest)]) == 0
    assert dest.exists()
