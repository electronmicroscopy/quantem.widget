"""Tests for the Browse file/folder browser widget."""

import json
import pathlib

import numpy as np
import pytest

from quantem.widget.browse import Browse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dir(tmp_path):
    (tmp_path / "folder_a").mkdir()
    (tmp_path / "folder_b").mkdir()
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / "data.h5").write_bytes(b"\x00" * 1024)
    (tmp_path / "image.tif").write_bytes(b"\x00" * 2048)
    (tmp_path / "stack.npy").write_bytes(b"")
    np.save(str(tmp_path / "stack.npy"), np.random.rand(32, 32).astype(np.float32))
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / ".hidden_file").write_text("secret")
    return tmp_path


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_root(tmp_path):
    b = Browse(root=str(tmp_path))
    assert b.root == str(tmp_path)
    assert b.current_path == str(tmp_path)
    assert b.selection_mode == "files"
    assert b.title == "Browse"


def test_selection_modes(tmp_path):
    for mode in ("file", "files", "folder", "folders"):
        b = Browse(root=str(tmp_path), selection_mode=mode)
        assert b.selection_mode == mode


def test_title(tmp_path):
    b = Browse(root=str(tmp_path), title="My Browser")
    assert b.title == "My Browser"


def test_repr(sample_dir):
    b = Browse(root=str(sample_dir))
    r = repr(b)
    assert "Browse(" in r
    assert "mode=files" in r
    assert "entries" in r


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def test_entries_populated(sample_dir):
    b = Browse(root=str(sample_dir))
    entries = json.loads(b.entries_json)
    names = [e["name"] for e in entries]
    assert "folder_a" in names
    assert "data.h5" in names
    assert "image.tif" in names


def test_hidden_files_excluded_by_default(sample_dir):
    b = Browse(root=str(sample_dir))
    entries = json.loads(b.entries_json)
    names = [e["name"] for e in entries]
    assert ".hidden_file" not in names
    assert ".hidden_dir" not in names


def test_hidden_files_shown(sample_dir):
    b = Browse(root=str(sample_dir), show_hidden=True)
    entries = json.loads(b.entries_json)
    names = [e["name"] for e in entries]
    assert ".hidden_file" in names
    assert ".hidden_dir" in names


def test_extension_filter(sample_dir):
    b = Browse(root=str(sample_dir), filter_exts=[".h5"])
    entries = json.loads(b.entries_json)
    files = [e for e in entries if e["kind"] == "file"]
    assert all(e["ext"] == ".h5" for e in files)
    folders = [e for e in entries if e["kind"] == "folder"]
    assert len(folders) > 0  # folders always shown


def test_folders_first(sample_dir):
    b = Browse(root=str(sample_dir))
    entries = json.loads(b.entries_json)
    folder_indices = [i for i, e in enumerate(entries) if e["kind"] == "folder"]
    file_indices = [i for i, e in enumerate(entries) if e["kind"] == "file"]
    if folder_indices and file_indices:
        assert max(folder_indices) < min(file_indices)


def test_previewable_flag(sample_dir):
    b = Browse(root=str(sample_dir))
    entries = json.loads(b.entries_json)
    by_name = {e["name"]: e for e in entries}
    assert by_name["stack.npy"]["is_previewable"] is True
    assert by_name["image.tif"]["is_previewable"] is True
    assert by_name["notes.txt"]["is_previewable"] is False
    assert by_name["data.h5"]["is_previewable"] is False  # H5 not previewable (too complex)


def test_entry_kinds(sample_dir):
    b = Browse(root=str(sample_dir))
    entries = json.loads(b.entries_json)
    by_name = {e["name"]: e for e in entries}
    assert by_name["folder_a"]["kind"] == "folder"
    assert by_name["data.h5"]["kind"] == "file"


def test_sort_by_name_desc(sample_dir):
    b = Browse(root=str(sample_dir), sort_asc=False)
    entries = json.loads(b.entries_json)
    file_names = [e["name"].lower() for e in entries if e["kind"] == "file"]
    assert file_names == sorted(file_names, reverse=True)


def test_sort_by_size(sample_dir):
    b = Browse(root=str(sample_dir), sort_key="size")
    entries = json.loads(b.entries_json)
    file_sizes = [e["size"] for e in entries if e["kind"] == "file"]
    assert file_sizes == sorted(file_sizes)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def test_cd_into_subdir(sample_dir):
    b = Browse(root=str(sample_dir))
    sub = sample_dir / "folder_a"
    b.cd(sub)
    assert b.current_path == str(sub)


def test_cd_blocked_outside_root(sample_dir):
    b = Browse(root=str(sample_dir / "folder_a"))
    b.cd(sample_dir)  # try to go above root
    assert b.current_path == str((sample_dir / "folder_a").resolve())


def test_cd_returns_self(sample_dir):
    b = Browse(root=str(sample_dir))
    result = b.cd(sample_dir / "folder_a")
    assert result is b


def test_home(sample_dir):
    b = Browse(root=str(sample_dir))
    b.cd(sample_dir / "folder_a")
    b.home()
    assert b.current_path == b.root


def test_navigate_via_current_path_trait(sample_dir):
    b = Browse(root=str(sample_dir))
    sub = sample_dir / "folder_a"
    b.current_path = str(sub)
    entries = json.loads(b.entries_json)
    # folder_a is empty
    assert len(entries) == 0


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def test_preview_npy(sample_dir):
    b = Browse(root=str(sample_dir))
    npy_path = str(sample_dir / "stack.npy")
    b.preview_request = npy_path
    assert b.preview_width > 0
    assert b.preview_height > 0
    assert len(b.preview_bytes) > 0
    assert b.preview_error == ""
    assert b.preview_title == "stack.npy"


def test_preview_downsamples_large_image(tmp_path):
    large = np.random.rand(512, 512).astype(np.float32)
    np.save(str(tmp_path / "large.npy"), large)
    b = Browse(root=str(tmp_path))
    b.preview_request = str(tmp_path / "large.npy")
    assert b.preview_width <= 256
    assert b.preview_height <= 256


def test_preview_blocked_outside_root(sample_dir):
    b = Browse(root=str(sample_dir / "folder_a"))
    b.preview_request = str(sample_dir / "stack.npy")
    assert b.preview_error == "Path outside root"
    assert b.preview_bytes == b""


def test_preview_error_on_unsupported(sample_dir):
    b = Browse(root=str(sample_dir))
    b.preview_request = str(sample_dir / "notes.txt")
    assert "Unsupported" in b.preview_error


def test_preview_clear(sample_dir):
    b = Browse(root=str(sample_dir))
    b.preview_request = str(sample_dir / "stack.npy")
    assert b.preview_width > 0
    b.preview_request = ""
    assert b.preview_bytes == b""
    assert b.preview_width == 0


# ---------------------------------------------------------------------------
# State persistence (3 required tests)
# ---------------------------------------------------------------------------

def test_state_dict_roundtrip(sample_dir):
    b = Browse(root=str(sample_dir), selection_mode="files", filter_exts=[".h5"], show_hidden=True, sort_key="size")
    state = b.state_dict()
    b2 = Browse(root=str(sample_dir), state=state)
    assert b2.selection_mode == "files"
    assert b2.filter_exts == [".h5"]
    assert b2.show_hidden is True
    assert b2.sort_key == "size"


def test_save_load_file(sample_dir, tmp_path):
    b = Browse(root=str(sample_dir), selection_mode="folder", title="Test")
    save_path = str(tmp_path / "browse_state.json")
    b.save(save_path)
    data = json.loads(pathlib.Path(save_path).read_text())
    assert data["metadata_version"] == "1.0"
    assert data["widget_name"] == "Browse"
    assert "widget_version" in data
    assert "state" in data
    assert data["state"]["selection_mode"] == "folder"
    b2 = Browse(root=str(sample_dir), state=save_path)
    assert b2.selection_mode == "folder"
    assert b2.title == "Test"


def test_summary(sample_dir, capsys):
    b = Browse(root=str(sample_dir))
    b.summary()
    out = capsys.readouterr().out
    assert "Browse:" in out
    assert "Current:" in out
    assert "Mode:" in out
    assert "folders" in out
    assert "files" in out


# ---------------------------------------------------------------------------
# Tool parity
# ---------------------------------------------------------------------------

def test_disabled_tools(sample_dir):
    b = Browse(root=str(sample_dir))
    b.disabled_tools = ["navigation"]
    assert "navigation" in b.disabled_tools


def test_hidden_tools(sample_dir):
    b = Browse(root=str(sample_dir))
    b.hidden_tools = ["preview"]
    assert "preview" in b.hidden_tools


def test_unknown_tool_raises(sample_dir):
    b = Browse(root=str(sample_dir))
    with pytest.raises(ValueError, match="Unknown tool group"):
        b.set_disabled_tools(["nonexistent"])


def test_runtime_api(sample_dir):
    b = Browse(root=str(sample_dir))
    result = b.lock_tool("navigation")
    assert result is b
    assert "navigation" in b.disabled_tools
    b.unlock_tool("navigation")
    assert "navigation" not in b.disabled_tools


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def test_filter_method(sample_dir):
    b = Browse(root=str(sample_dir))
    result = b.filter(".h5", ".tif")
    assert result is b
    assert b.filter_exts == [".h5", ".tif"]
    entries = json.loads(b.entries_json)
    files = [e for e in entries if e["kind"] == "file"]
    assert all(e["ext"] in {".h5", ".tif"} for e in files)


def test_clear_selection(sample_dir):
    b = Browse(root=str(sample_dir))
    b.selected = [str(sample_dir / "data.h5")]
    result = b.clear_selection()
    assert result is b
    assert b.selected == []


def test_path_property(sample_dir):
    b = Browse(root=str(sample_dir))
    assert b.path is None
    b.selected = [str(sample_dir / "data.h5")]
    assert b.path == str(sample_dir / "data.h5")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_large_directory_capped(tmp_path):
    for i in range(50):
        (tmp_path / f"file_{i:04d}.txt").write_text("x")
    b = Browse(root=str(tmp_path))
    entries = json.loads(b.entries_json)
    assert len(entries) == 50  # well under cap, but verifies scanning works


def test_permission_error_handled(tmp_path):
    b = Browse(root=str(tmp_path))
    # Point to a non-existent dir — should not crash
    b.current_path = str(tmp_path / "nonexistent")
    # Should reset to root since nonexistent isn't a dir
    entries = json.loads(b.entries_json)
    assert isinstance(entries, list)


def test_cd_to_file_blocked(sample_dir):
    b = Browse(root=str(sample_dir))
    old = b.current_path
    b.current_path = str(sample_dir / "data.h5")
    assert b.current_path == old  # should revert


def test_empty_root(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    b = Browse(root=str(empty))
    entries = json.loads(b.entries_json)
    assert entries == []


def test_available_exts(sample_dir):
    b = Browse(root=str(sample_dir))
    exts = b.available_exts
    assert ".h5" in exts
    assert ".tif" in exts
    assert ".npy" in exts
    assert ".txt" in exts
    assert exts == sorted(exts)


def test_available_exts_unaffected_by_filter(sample_dir):
    b = Browse(root=str(sample_dir), filter_exts=[".h5"])
    exts = b.available_exts
    assert ".tif" in exts
    assert ".npy" in exts
    assert ".txt" in exts


def test_preview_h5(tmp_path):
    import h5py
    arr = np.random.rand(32, 32).astype(np.float32)
    with h5py.File(str(tmp_path / "test.h5"), "w") as f:
        f.create_dataset("data/images", data=arr)
    b = Browse(root=str(tmp_path))
    b.preview_request = str(tmp_path / "test.h5")
    assert b.preview_width == 32
    assert b.preview_height == 32
    assert len(b.preview_bytes) > 0
    assert b.preview_error == ""


def test_preview_h5_4d(tmp_path):
    import h5py
    arr = np.random.rand(2, 3, 24, 24).astype(np.float32)
    with h5py.File(str(tmp_path / "stem.h5"), "w") as f:
        f.create_dataset("entry/data/data", data=arr)
    b = Browse(root=str(tmp_path))
    b.preview_request = str(tmp_path / "stem.h5")
    assert b.preview_width == 24
    assert b.preview_height == 24
    assert b.preview_error == ""


def test_preview_3d_npy(tmp_path):
    arr = np.random.rand(5, 32, 32).astype(np.float32)
    np.save(str(tmp_path / "stack3d.npy"), arr)
    b = Browse(root=str(tmp_path))
    b.preview_request = str(tmp_path / "stack3d.npy")
    assert b.preview_width == 32
    assert b.preview_height == 32
    assert b.preview_error == ""


# ---------------------------------------------------------------------------
# Preview colormap
# ---------------------------------------------------------------------------

def test_preview_cmap_default(tmp_path):
    b = Browse(root=str(tmp_path))
    assert b.preview_cmap == "inferno"


def test_preview_cmap_in_state_dict(sample_dir):
    b = Browse(root=str(sample_dir))
    b.preview_cmap = "viridis"
    state = b.state_dict()
    assert state["preview_cmap"] == "viridis"
    b2 = Browse(root=str(sample_dir), state=state)
    assert b2.preview_cmap == "viridis"


# ---------------------------------------------------------------------------
# Scanning indicator
# ---------------------------------------------------------------------------

def test_is_scanning_trait(tmp_path):
    b = Browse(root=str(tmp_path))
    assert b.is_scanning is False


# ---------------------------------------------------------------------------
# Select all via entries
# ---------------------------------------------------------------------------

def test_select_all_status(sample_dir):
    b = Browse(root=str(sample_dir))
    entries = json.loads(b.entries_json)
    all_paths = [e["path"] for e in entries]
    b.selected = all_paths
    assert len(b.selected) == len(entries)
    total_size = sum(e["size"] for e in entries if e["kind"] == "file")
    assert total_size > 0
