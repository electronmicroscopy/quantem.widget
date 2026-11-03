"""The two size guards the release workflow runs on every tracked file and notebook."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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


def test_large_file_guard_allows_show4dstem_readme_demo() -> None:
    result = _run(
        sys.executable,
        "scripts/check_large_files.py",
        "docs/_static/show4dstem-serin-gold.gif",
    )

    assert result.returncode == 0, result.stdout
    assert "Approved size exception" in result.stdout
