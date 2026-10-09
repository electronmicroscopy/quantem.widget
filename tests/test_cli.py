"""Tests for the ``quantem`` CLI: Show4DSTEM routing and the GitHub notebook copy.

4D-STEM rendering needs a GPU + real master files, so it is exercised manually
(see docs); here we cover the routing logic, which runs on CPU.
"""
import json
import pathlib
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
    output = cell["outputs"][0]
    data = output["data"]
    assert "image/jpeg" in data
    assert "application/vnd.jupyter.widget-view+json" in data
    assert output["metadata"]["quantem.widget"]["github_full_ui"] is True
    assert output["metadata"]["quantem.widget"]["github_quality"] == 80
    assert output["metadata"]["quantem.widget"]["github_width"] == 24


def test_scientific_pixel_gate_rejects_uniform_canvas(tmp_path):
    from PIL import Image

    blank = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), "white").save(blank)

    assert not cli._image_has_scientific_pixels(blank.read_bytes())


def test_scientific_pixel_gate_rejects_blank_canvas_with_resize_handle(tmp_path):
    from PIL import Image

    blank = tmp_path / "blank-with-handle.png"
    image = Image.new("RGB", (128, 128), "white")
    for row in range(4):
        for col in range(4):
            image.putpixel((64 + col, 120 + row), (66, 153, 225))
    image.save(blank)

    assert not cli._image_has_scientific_pixels(blank.read_bytes())


def test_scientific_pixel_gate_accepts_image_content(tmp_path):
    from PIL import Image

    content = tmp_path / "content.png"
    image = Image.new("L", (64, 64))
    image.putdata([(row * 4 + col * 2) % 256 for row in range(64) for col in range(64)])
    image.save(content)

    assert cli._image_has_scientific_pixels(content.read_bytes())


def test_promote_static_fallback_marks_single_github_preview():
    cell = {
        "cell_type": "code",
        "source": ["viewer"],
        "outputs": [{
            "output_type": "display_data",
            "metadata": {"quantem.widget": {"static_fallback": True}},
            "data": {"image/jpeg": "fallback"},
        }],
    }

    assert cli._promote_static_fallback(cell)
    metadata = cell["outputs"][0]["metadata"]["quantem.widget"]
    assert metadata == {"github_static_preview": True}
    assert cli._cell_has_static_preview_output(cell)


def test_promote_static_fallback_does_not_mutate_stream_outputs():
    stream = {
        "output_type": "stream",
        "name": "stdout",
        "text": "progress",
    }
    cell = {
        "cell_type": "code",
        "source": ["viewer"],
        "outputs": [
            stream,
            {
                "output_type": "display_data",
                "metadata": {"quantem.widget": {"static_fallback": True}},
                "data": {"image/jpeg": "fallback"},
            },
        ],
    }

    assert cli._promote_static_fallback(cell)
    assert stream == {
        "output_type": "stream",
        "name": "stdout",
        "text": "progress",
    }


def test_github_widget_cell_detector_uses_runtime_widget_output_for_public_api():
    cell = {
        "cell_type": "code",
        "source": ["drift.show(mode='interactive')"],
        "outputs": [{
            "output_type": "display_data",
            "metadata": {},
            "data": {
                "application/vnd.jupyter.widget-view+json": {"model_id": "abc"},
                "image/jpeg": "fallback",
            },
        }],
    }
    notebook = {"cells": [cell]}

    assert cli._github_widget_cells(notebook) == [cell]
    assert cli._github_capture_cells(notebook) == [cell]


def test_github_capture_reuses_only_marked_full_ui_output():
    cell = {
        "cell_type": "code",
        "source": ["drift.show(mode='interactive')"],
        "outputs": [{
            "output_type": "display_data",
            "metadata": {"quantem.widget": {"github_full_ui": True}},
            "data": {"image/jpeg": "full-ui"},
        }],
    }
    notebook = {"cells": [cell]}

    assert cli._github_widget_cells(notebook) == [cell]
    assert cli._github_capture_cells(notebook) == []


def test_widget_model_closure_includes_layout_dependency():
    state = {
        "root": {"state": {"layout": "IPY_MODEL_layout"}},
        "layout": {"state": {}},
        "unrelated": {"state": {}},
    }

    assert cli._widget_model_closure(state, ["root"]) == {"root", "layout"}


def test_widget_capture_notebook_keeps_only_required_models():
    view = {"model_id": "root", "version_major": 2, "version_minor": 0}
    cell = {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "source": ["drift.show()"],
        "outputs": [{
            "output_type": "display_data",
            "metadata": {},
            "data": {cli._WIDGET_VIEW_MIME: view, "image/jpeg": "fallback"},
        }],
    }
    notebook = {
        "cells": [cell],
        "metadata": {"widgets": {cli._WIDGET_STATE_MIME: {
            "version_major": 2,
            "version_minor": 0,
            "state": {
                "root": {"state": {"layout": "IPY_MODEL_layout"}},
                "layout": {"state": {}},
                "unrelated": {"state": {}},
            },
        }}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    capture = cli._widget_capture_notebook(notebook, cell)
    payload = capture["metadata"]["widgets"][cli._WIDGET_STATE_MIME]
    assert set(payload["state"]) == {"root", "layout"}
    assert capture["cells"][0]["source"] == []
    assert capture["cells"][0]["outputs"][0]["data"] == {
        cli._WIDGET_VIEW_MIME: view
    }


def test_prune_widget_fallbacks_keeps_only_full_ui_visual():
    cell = {
        "cell_type": "code",
        "source": ["drift.show()"],
        "outputs": [
            {
                "output_type": "display_data",
                "metadata": {"quantem.widget": {"github_full_ui": True}},
                "data": {"image/jpeg": "ui", "text/html": "redundant"},
            },
            {
                "output_type": "display_data",
                "metadata": {"quantem.widget": {"static_fallback": True}},
                "data": {"image/jpeg": "fallback", "text/html": "fallback"},
            },
        ],
    }

    assert cli._prune_widget_fallbacks({"cells": [cell]}) == 1
    assert len(cell["outputs"]) == 1
    assert cell["outputs"][0]["data"] == {"image/jpeg": "ui"}


def test_github_validation_rejects_duplicate_fallback():
    cell = {
        "cell_type": "code",
        "source": ["drift.show()"],
        "outputs": [
            {
                "output_type": "display_data",
                "metadata": {"quantem.widget": {"github_full_ui": True}},
                "data": {"image/jpeg": "ui"},
            },
            {
                "output_type": "display_data",
                "metadata": {"quantem.widget": {"static_fallback": True}},
                "data": {"image/jpeg": "fallback"},
            },
        ],
    }

    with pytest.raises(ValueError, match="fallbacks=1"):
        cli._validate_github_widget_outputs([cell])

    assert cli._prune_widget_fallbacks({"cells": [cell]}) == 1
    cli._validate_github_widget_outputs([cell])


def test_github_prepare_reuses_existing_full_ui_output(tmp_path, monkeypatch):
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
     "metadata": {
      "quantem.widget": {
       "github_full_ui": true,
       "github_quality": 90,
       "github_width": 1200
      }
     },
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
        raise AssertionError("existing full-UI output should not trigger capture")

    monkeypatch.setattr(cli, "_capture_full_ui", fail_capture)
    args = type("Args", (), {
        "path": str(notebook),
        "no_execute": True,
        "quality": 90,
        "max_width": 1200,
        "timeout": 600,
    })()

    assert cli._prepare_github(args) == 0
    text = notebook.read_text(encoding="utf-8")
    assert "image/jpeg" in text
    assert text.count("github_full_ui") == 1
    assert "application/vnd.jupyter.widget-state+json" not in text


def test_github_prepare_prefers_static_scientific_preview(tmp_path, monkeypatch):
    notebook = tmp_path / "show2d_github.ipynb"
    notebook.write_text(
        json.dumps({
            "cells": [{
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": ["viewer"],
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {
                            "application/vnd.jupyter.widget-view+json": {
                                "model_id": "root"
                            }
                        },
                    },
                    {
                        "output_type": "display_data",
                        "metadata": {
                            "quantem.widget": {"static_fallback": True}
                        },
                        "data": {"image/jpeg": "scientific-preview"},
                    },
                ],
            }],
            "metadata": {
                "widgets": {cli._WIDGET_STATE_MIME: {"state": {}}}
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }),
        encoding="utf-8",
    )

    def fail_capture(*args, **kwargs):
        raise AssertionError("native static preview should avoid browser capture")

    monkeypatch.setattr(cli, "_capture_notebook_widget_uis", fail_capture)
    args = type("Args", (), {
        "path": str(notebook),
        "no_execute": True,
        "quality": 90,
        "max_width": 1200,
        "timeout": 600,
    })()

    assert cli._prepare_github(args) == 0
    prepared = json.loads(notebook.read_text(encoding="utf-8"))
    outputs = prepared["cells"][0]["outputs"]
    assert len(outputs) == 1
    assert outputs[0]["data"] == {"image/jpeg": "scientific-preview"}
    assert outputs[0]["metadata"]["quantem.widget"] == {
        "github_static_preview": True
    }


# ---------------------------------------------------------------------------


def test_cli_exposes_only_show4dstem_and_github(capsys):
    """Top-level help, expect exactly the two kept subcommands; no command is a usage error."""

    with pytest.raises(SystemExit) as missing:
        cli.main([])
    assert missing.value.code == 2
    assert "required: command" in capsys.readouterr().err
    with pytest.raises(SystemExit) as shown:
        cli.main(["--help"])
    assert shown.value.code == 0
    output = capsys.readouterr().out

    assert "show4dstem" in output
    assert "github" in output
    for retired in ("showptycho", "show2d", "show3d", "showdiffraction", "html"):
        assert f" {retired} " not in output


def test_user_errors_print_one_line_and_bugs_keep_their_traceback(tmp_path, monkeypatch, capsys):
    """A refused dense read, a missing package or a missing file is one ``quantem:`` line and exit 1."""
    from quantem.widget.adapters import gpu as gpu_adapter
    from quantem.widget.show4dstem import reader

    np.save(tmp_path / "scan.npy", np.ones((4, 4, 8, 8), np.uint16))
    monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    monkeypatch.setattr(reader, "ceiling", lambda device=None: 1000)
    assert cli.main(["show4dstem", str(tmp_path / "scan.npy"), "--html", "--no-open", "--out", str(tmp_path / "out")]) == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith("quantem: skipped scan: scan.npy: the dense (4, 4, 8, 8) array needs")
    assert lines[1:] == ["quantem: every master failed to export (see messages above)"]
    assert cli.main(["show4dstem", str(tmp_path / "missing_master.h5"), "--no-open"]) == 1
    assert capsys.readouterr().err.startswith("quantem: ")

    def raises(error):
        def command(args):
            raise error
        return command

    monkeypatch.setattr(cli, "_prepare_github", raises(ImportError("quantem github needs Playwright: pip install playwright")))
    assert cli.main(["github", str(tmp_path / "x.ipynb")]) == 1
    assert capsys.readouterr().err == "quantem: quantem github needs Playwright: pip install playwright\n"
    monkeypatch.setattr(cli, "_prepare_github", raises(MemoryError()))
    assert cli.main(["github", str(tmp_path / "x.ipynb")]) == 1
    assert capsys.readouterr().err == "quantem: MemoryError\n"
    for bug in (TypeError("unsupported operand"), ImportError("cannot import name 'x'", name="quantem.widget.show2d")):
        monkeypatch.setattr(cli, "_prepare_github", raises(bug))
        with pytest.raises(type(bug)):
            cli.main(["github", str(tmp_path / "x.ipynb")])


def test_show4dstem_cli_count_defaults_to_full_detector(tmp_path, monkeypatch):
    """C3: Show4DSTEM CLI count gates use native detector pixels by default."""
    for idx in range(2):
        (tmp_path / f"scan_{idx}_master.h5").write_bytes(b"\x00")
    seen = {}

    def fake_discover(path, **kwargs):
        seen["discover_path"] = str(path)
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

    monkeypatch.setattr("quantem.widget.adapters.gpu.discover_masters", fake_discover)
    monkeypatch.setattr("quantem.widget.show4dstem.reader.find_masters", fake_discover)
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

    def fake_discover(path, **kwargs):
        return [str(tmp_path / "scan_0_master.h5")]

    monkeypatch.setattr("quantem.widget.adapters.gpu.discover_masters", fake_discover)
    monkeypatch.setattr("quantem.widget.show4dstem.reader.find_masters", fake_discover)

    assert cli.main(["show4dstem", str(tmp_path), "--count", "7", "--no-open"]) == 1


def test_show4dstem_webgpu_folder_names_the_launcher_for_its_platform(tmp_path, monkeypatch, capsys):
    """The .command launcher is macOS-only; Linux prints the folder's Range server command."""
    command = tmp_path / "Show4DSTEM.command"
    command.write_text("#!/bin/zsh\n", encoding="utf-8")
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli, "_free_port", lambda start: start)
    cli._open_show4dstem_command(command, no_open=False)
    printed = capsys.readouterr().out
    assert f"python3 {tmp_path / '.viewer' / 'serve_range.py'} --root {tmp_path} --port 8794" in printed
    assert "http://127.0.0.1:8794/" in printed and ".command" not in printed
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    cli._open_show4dstem_command(command, no_open=True)
    assert capsys.readouterr().out == f"wrote {command}\n"


def test_show4dstem_webgpu_cli_opens_generated_command(tmp_path, monkeypatch):
    """C5: WebGPU CLI uses the browser HDF5-backed export entry path."""
    (tmp_path / "scan_0_master.h5").write_bytes(b"\x00")
    out = tmp_path / "artifact" / "index.html"
    out.parent.mkdir()
    command = out.parent / "Show4DSTEM.command"
    command.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    seen = {}

    def fake_discover(path, **kwargs):
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

    monkeypatch.setattr("quantem.widget.adapters.gpu.discover_masters", fake_discover)
    monkeypatch.setattr("quantem.widget.show4dstem.reader.find_masters", fake_discover)
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

    monkeypatch.setattr("quantem.widget.show4dstem.widget.master_contract", fake_contract)
    monkeypatch.setattr("quantem.widget.show4dstem.export.write_webgpu_bundle", fake_export)
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


def test_render_show4dstem_folder_notebook_records_backend_and_count(tmp_path):
    """C7: generated CUDA folder notebooks preserve the seven-entry gate options."""
    args = SimpleNamespace(backend="cuda", det_bin=1, out=str(tmp_path))

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
    assert "watch=False" in text
    assert "det_bin" not in text and "dtype" not in text



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
        "--scan-size",
        "512",
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
    assert "Show4DSTEM.from_folder(" in code
    assert "scan_size=512" in code
    assert "backend='auto'" in code
    assert "watch=True, watch_interval=1.5" in code


def test_show4dstem_watch_requires_live_folder_notebook(tmp_path):
    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"\x00")

    assert cli.main(["show4dstem", str(master), "--watch", "--no-open"]) == 1
    assert cli.main(["show4dstem", str(tmp_path), "--watch", "--html", "--no-open"]) == 1



def test_show4dstem_html_cli_threads_export_dtype() -> None:
    """C1: CLI full export docs, expect --dtype uint16 to reach the export."""
    import inspect

    source = inspect.getsource(cli._render_4dstem)

    assert "export_dtype = _show4dstem_export_dtype(args)" in source
    assert "widget.export_html(str(out), title=args.title or stem, dtype=pack_dtype)" in source
    assert cli._show4dstem_export_dtype(SimpleNamespace(dtype="auto")) == "auto"
    assert cli._show4dstem_export_dtype(SimpleNamespace(dtype="uint16")) == "uint16"
    assert cli._show4dstem_export_dtype(SimpleNamespace(dtype="u16")) == "uint16"
    assert cli._show4dstem_export_dtype(SimpleNamespace(dtype="uint8")) == "uint8"


@pytest.mark.parametrize("path", ["encoded", "dense"])
def test_show4dstem_html_bins_master_by_rounded_mean(tmp_path, monkeypatch, path) -> None:
    """C1a: a real 4x4x8x8 master, expect --bin 1 exact counts and --bin 2 rounded block means,
    from quantem.gpu's encoded acquisition and from the widget's dense reader alike."""
    import h5py
    import hdf5plugin

    from quantem.widget.adapters import gpu as gpu_adapter

    if path == "encoded" and not gpu_adapter.accelerator_ready():
        pytest.skip("encoded acquisitions need quantem.gpu and a CUDA or MPS GPU")
    if path == "dense":
        monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    counts = np.random.default_rng(5).integers(0, 1000, (4, 4, 8, 8), dtype=np.uint16)
    with h5py.File(tmp_path / "scan_data_000001.h5", "w") as handle:
        handle.create_dataset(
            "entry/data/data", data=counts.reshape(16, 8, 8), chunks=(1, 8, 8),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        handle.require_group("entry/data")["data_000001"] = h5py.ExternalLink(
            "scan_data_000001.h5", "entry/data/data"
        )
        specific = handle.require_group("entry/instrument/detector/detectorSpecific")
        specific.create_dataset("ntrigger", data=16)
        specific.create_dataset("nimages", data=1)

    np.testing.assert_array_equal(cli._master_to_binned_numpy(str(master), 1), counts)
    blocks = counts.reshape(4, 4, 4, 2, 4, 2).sum(axis=(3, 5), dtype=np.uint64)
    np.testing.assert_array_equal(
        cli._master_to_binned_numpy(str(master), 2),
        np.round(blocks / 4).astype(np.float32),
    )



def test_show4dstem_html_refuses_a_payload_no_page_can_hold_before_reading(tmp_path, monkeypatch, capsys) -> None:
    """A master whose packed counts exceed the page limit, expect one refusal naming a --bin that fits,
    without reading the acquisition (512 x 512 x 192 x 192 at --bin 1 read for 3 minutes, then failed)."""
    import h5py
    import hdf5plugin

    counts = np.ones((16, 8, 8), dtype=np.uint16)
    with h5py.File(tmp_path / "scan_data_000001.h5", "w") as handle:
        handle.create_dataset("entry/data/data", data=counts, chunks=(1, 8, 8), **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    with h5py.File(tmp_path / "scan_master.h5", "w") as handle:
        handle.require_group("entry/data")["data_000001"] = h5py.ExternalLink("scan_data_000001.h5", "entry/data/data")
    monkeypatch.setattr(cli, "_HTML_PAYLOAD_LIMIT", 16 * 8 * 8 - 1)
    monkeypatch.setattr(cli, "_master_to_binned_numpy", lambda *args, **kwargs: pytest.fail("read before refusing"))
    command = ["show4dstem", str(tmp_path / "scan_master.h5"), "--html", "--no-open", "--out", str(tmp_path / "out")]

    assert cli.main(command) == 1
    err = capsys.readouterr().err
    assert "an offline HTML at --bin 1 packs 0.0 GB of uint8 counts" in err
    assert "pass --bin 2, use --backend webgpu" in err


def test_show4dstem_cli_max_gb_reads_what_the_ceiling_refused(tmp_path, monkeypatch, capsys):
    """The refusal names --max-gb, and --max-gb raises the dense-read ceiling for that run."""
    from quantem.widget.adapters import gpu as gpu_adapter
    from quantem.widget.show4dstem import reader

    np.save(tmp_path / "scan.npy", np.ones((4, 4, 8, 8), np.uint16))
    monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    real_ceiling = reader.ceiling  # the dense file read sees a tiny ceiling; the viewer's GPU copy the real one
    monkeypatch.setattr(reader, "ceiling", lambda device=None: 1000 if str(device or "cpu") == "cpu" else real_ceiling(device))
    command = ["show4dstem", str(tmp_path / "scan.npy"), "--html", "--no-open", "--out", str(tmp_path / "out")]
    assert cli.main(command) == 1
    assert "--max-gb for the quantem command" in capsys.readouterr().err
    assert cli.main([*command, "--max-gb", "1"]) == 0
    assert list((tmp_path / "out").glob("*.html"))


def test_show4dstem_html_serve_reports_every_file_and_the_url_before_blocking(tmp_path, monkeypatch, capsys):
    """--serve blocks until Ctrl-C, so the file count and the URL print (flushed) first."""
    pages = [tmp_path / "scanA.html", tmp_path / "scanB.html"]
    for page in pages:
        page.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(cli, "_render_4dstem", lambda masters, args: pages)

    class Server:
        server_address = ("127.0.0.1", 8123)

        def __init__(self, *args):
            pass

        def serve_forever(self):
            pass

        def shutdown(self):
            pass

    printed_before_blocking = []

    class Interrupted:
        def wait(self):
            printed_before_blocking.append(capsys.readouterr().out)
            raise KeyboardInterrupt

    monkeypatch.setattr(cli.socketserver, "TCPServer", Server)
    monkeypatch.setattr(cli.threading, "Thread", lambda target, daemon: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(cli.threading, "Event", Interrupted)
    monkeypatch.delenv("DISPLAY", raising=False)
    args = SimpleNamespace(det_bin=1, backend="auto", html=True, serve=True, no_open=False)
    assert cli._do_4dstem(["a_master.h5", "b_master.h5"], "two", args) == 0
    assert "wrote 2 HTML files" in printed_before_blocking[0]
    assert "serving http://127.0.0.1:8123/scanA.html" in printed_before_blocking[0]


def test_show4dstem_serve_without_html_is_refused_in_one_line(tmp_path, capsys):
    """--serve only applies to an --html export; a notebook run must not ignore it silently."""
    assert cli.main(["show4dstem", str(tmp_path), "--serve", "--no-open"]) == 1
    assert capsys.readouterr().err == "quantem: --serve opens an --html export over HTTP; add --html.\n"


def test_show4dstem_cli_html_bin_announces_the_detector_binning(tmp_path, monkeypatch, capsys):
    """--bin reduces the detector, so the export says so and how to keep the full detector."""
    from quantem.widget.adapters import gpu as gpu_adapter

    np.save(tmp_path / "scan.npy", np.ones((4, 4, 8, 8), np.uint16))
    monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    command = ["show4dstem", str(tmp_path / "scan.npy"), "--html", "--bin", "2", "--no-open", "--out", str(tmp_path / "out")]
    assert cli.main(command) == 0
    assert "detector binned 2x2 (mean of each block), 8x8 -> 4x4; pass --bin 1 for the full detector" in capsys.readouterr().out


def test_show4dstem_cli_html_auto_dtype_keeps_counts_above_255(tmp_path, monkeypatch, capsys):
    """Pre-binned data with counts above 255 is packed as uint16 by default, not clipped to a flat uint8 image."""
    from quantem.widget.adapters import gpu as gpu_adapter

    counts = np.full((4, 4, 8, 8), 300, np.uint16)
    counts[..., 3:5, 3:5] = np.arange(16, dtype=np.uint16).reshape(4, 4)[..., None, None] + 400
    np.save(tmp_path / "binned.npy", counts)
    monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    command = ["show4dstem", str(tmp_path / "binned.npy"), "--html", "--no-open", "--out", str(tmp_path / "out")]
    assert cli.main(command) == 0
    out = capsys.readouterr().out
    assert "counts reach 415, packed as uint16" in out
    assert "clipped" not in out
    assert cli.main([*command, "--dtype", "u8"]) == 0
    assert "--dtype u16 or auto on the command line" in capsys.readouterr().out


def test_show4dstem_cli_out_refuses_a_file_name(tmp_path, capsys):
    """--out is a folder; `--out x.html` used to create a folder named x.html."""
    np.save(tmp_path / "scan.npy", np.ones((4, 4, 8, 8), np.uint16))
    command = ["show4dstem", str(tmp_path / "scan.npy"), "--html", "--no-open", "--out", str(tmp_path / "x.html")]
    assert cli.main(command) == 1
    assert "--out is the output folder; x.html looks like a file name" in capsys.readouterr().err
    assert not (tmp_path / "x.html").exists()
