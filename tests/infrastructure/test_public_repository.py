"""Guard what this public repository may track.

Datasets stay out except the small synthetic fixtures under tests/data/, no
tracked text may name a private computer, person, partner, dataset or home
path, and tracked notebooks carry neither saved widget state nor image outputs.
See AGENTS.md.
"""

import json
import subprocess
from pathlib import Path

from private_names import mentions_private_name

REPO = Path(__file__).resolve().parents[2]
MAX_BYTES = 5 * 1024 * 1024
GOLD_DEMO_GIF = "docs/_static/show4dstem-serin-gold.gif"
DATA_SUFFIXES = (
    ".jsonl",
    ".log",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".h5",
    ".hdf5",
    ".emd",
    ".dm3",
    ".dm4",
    ".mrc",
    ".ser",
    ".npy",
    ".npz",
    ".qem",
)
FIXTURE_DIRECTORIES = ("tests/data/",)
# Third-party package names in the npm lock file are not ours to rename.
LOCK_FILES = ("package-lock.json",)
# h5py's global lock object shares its name with a private host; only these exact lines may use it.
H5PY_LOCK = "ph" + "il"  # built from parts so this guard does not trip on itself
H5PY_LOCK_LINES = {f"from h5py._objects import {H5PY_LOCK}", f"with {H5PY_LOCK}:"}


def _tracked_files() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True
    ).stdout.decode()
    return [path for path in listing.split("\0") if path and (REPO / path).is_file()]


def _notebook_text(notebook: dict) -> str:
    """Cell sources and text outputs; image outputs are random letters that match short names by chance."""
    parts = []
    for cell in notebook["cells"]:
        parts.append("".join(cell["source"]))
        for output in cell.get("outputs", []):
            parts.append("".join(output.get("text", "")))
            for mime, value in output.get("data", {}).items():
                if not mime.startswith("image/"):
                    parts.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


def _public_text(path: str) -> str | None:
    """Tracked text without binary files, lock files, notebook images or the h5py lock lines."""
    if Path(path).name in LOCK_FILES:
        return None
    data = (REPO / path).read_bytes()
    if b"\0" in data[:8192]:
        return None
    text = data.decode("utf-8", errors="ignore")
    if path.endswith(".ipynb"):
        text = _notebook_text(json.loads(text))
    return "\n".join(line for line in text.splitlines() if line.strip() not in H5PY_LOCK_LINES)


def _image_outputs(notebook: dict) -> int:
    return sum(
        1
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        for mime in output.get("data", {})
        if mime.startswith("image/")
    )


def test_no_large_files_except_the_gold_demo_gif() -> None:
    large = [
        path
        for path in _tracked_files()
        if path != GOLD_DEMO_GIF and (REPO / path).stat().st_size > MAX_BYTES
    ]

    assert large == []


def test_data_and_log_files_stay_in_fixture_directories() -> None:
    misplaced = [
        path
        for path in _tracked_files()
        if path.lower().endswith(DATA_SUFFIXES) and not path.startswith(FIXTURE_DIRECTORIES)
    ]

    assert misplaced == []


def test_no_macos_metadata_files() -> None:
    junk = [
        path
        for path in _tracked_files()
        if Path(path).name.startswith("._") or Path(path).name == ".DS_Store"
    ]

    assert junk == []


def test_notebooks_carry_no_widget_state_or_image_outputs() -> None:
    offending = []
    for path in _tracked_files():
        if not path.endswith(".ipynb"):
            continue
        notebook = json.loads((REPO / path).read_text(encoding="utf-8"))
        if "widgets" in notebook.get("metadata", {}):
            offending.append(f"{path}: metadata.widgets")
        if images := _image_outputs(notebook):
            offending.append(f"{path}: {images} image outputs")

    assert offending == []


def test_tracked_text_names_no_private_computer_person_or_path() -> None:
    leaked = []
    for path in _tracked_files():
        text = _public_text(path)
        if text is None or not mentions_private_name(text):
            continue
        lines = [number for number, line in enumerate(text.splitlines(), 1) if mentions_private_name(line)]
        leaked.append(f"{path}:{lines or 'adjacent lines'}")

    assert leaked == []
