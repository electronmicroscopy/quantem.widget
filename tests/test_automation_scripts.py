from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _write_notebook(path: Path, output_text: str = "") -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": output_text,
                    }
                ],
                "source": "print('ok')\n",
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


def test_notebook_size_guard_accepts_small_notebook(tmp_path: Path) -> None:
    notebook = tmp_path / "small.ipynb"
    _write_notebook(notebook, "small output")

    result = _run(sys.executable, "scripts/check_notebook_sizes.py", str(notebook), "--max-mb", "1")

    assert result.returncode == 0, result.stdout
    assert "Notebook size guard passed" in result.stdout


def test_notebook_size_guard_rejects_large_embedded_output(tmp_path: Path) -> None:
    notebook = tmp_path / "large.ipynb"
    _write_notebook(notebook, "x" * 4096)

    result = _run(
        sys.executable,
        "scripts/check_notebook_sizes.py",
        str(notebook),
        "--max-mb",
        "1",
        "--max-output-mb",
        "0.001",
    )

    assert result.returncode == 1
    assert "embeds" in result.stdout


def test_large_file_guard_accepts_small_explicit_file(tmp_path: Path) -> None:
    data = tmp_path / "small.npy"
    data.write_bytes(b"0" * 128)

    result = _run(
        sys.executable,
        "scripts/check_large_files.py",
        str(data),
        "--max-mb",
        "1",
        "--data-max-mb",
        "1",
    )

    assert result.returncode == 0, result.stdout
    assert "Tracked file size guard passed" in result.stdout


def test_large_file_guard_rejects_large_data_artifact(tmp_path: Path) -> None:
    data = tmp_path / "large.npy"
    data.write_bytes(b"0" * 4096)

    result = _run(
        sys.executable,
        "scripts/check_large_files.py",
        str(data),
        "--max-mb",
        "1",
        "--data-max-mb",
        "0.001",
    )

    assert result.returncode == 1
    assert "data/rendered artifact" in result.stdout


def test_automation_documentation_names_entrypoints() -> None:
    doc = (ROOT / "docs/maintainer/automation.md").read_text(encoding="utf-8")

    for path in [
        "scripts/widget_local_signoff.sh",
        "scripts/docs_preview.sh",
        "scripts/widget_html_smoke.py",
        "scripts/widget_performance_smoke.py",
        "scripts/check_large_files.py",
        "scripts/check_notebook_sizes.py",
        ".github/workflows/widget-ci.yml",
        "--artifact-dir",
        "index.html",
        "browser-plan.json",
    ]:
        assert path in doc


def test_widget_html_smoke_writes_visual_report(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "html-smoke"

    result = _run(
        sys.executable,
        "scripts/widget_html_smoke.py",
        "--artifact-dir",
        str(artifact_dir),
        "--max-total-mb",
        "25",
    )

    assert result.returncode == 0, result.stdout
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    plan = json.loads((artifact_dir / "browser-plan.json").read_text(encoding="utf-8"))
    index = (artifact_dir / "index.html").read_text(encoding="utf-8")

    assert len(report["exports"]) == 7
    assert {item["widget"] for item in report["exports"]} == {
        "show2d",
        "show3d",
        "show3dslices",
        "show4dstem",
        "showeds",
        "showdiffraction",
        "showfolder",
    }
    assert "show2d.html" in index
    assert "showfolder.html" in index
    assert {page["widget"] for page in plan["pages"]} == {
        "show2d",
        "show3d",
        "show3dslices",
        "show4dstem",
        "showeds",
        "showdiffraction",
        "showfolder",
    }


def test_widget_performance_smoke_writes_browser_plan(tmp_path: Path) -> None:
    data_dir = tmp_path / "real"
    artifact_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    rng = np.random.default_rng(8)
    for idx in range(2):
        image = rng.random((48, 48), dtype=np.float32)
        tifffile.imwrite(data_dir / f"real_{idx}.tif", image)

    result = _run(
        sys.executable,
        "scripts/widget_performance_smoke.py",
        "--quick",
        "--search-root",
        str(data_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--show2d-panels",
        "2",
        "--show2d-size",
        "32",
        "--show3d-panels",
        "2",
        "--show3d-frames",
        "3",
        "--show3d-size",
        "32",
        "--max-single-mb",
        "20",
    )

    assert result.returncode == 0, result.stdout
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    plan = json.loads((artifact_dir / "browser-plan.json").read_text(encoding="utf-8"))
    assert (artifact_dir / "index.html").exists()
    assert report["targets"]["show2d"] == {"panels": 2, "size": 32}
    assert report["targets"]["show3d"] == {"panels": 2, "frames": 3, "size": 32}
    assert len(report["exports"]) == 4
    assert plan["target_fps"] == 30
    assert {page["widget"] for page in plan["pages"]} == {"show2d", "show3d"}
