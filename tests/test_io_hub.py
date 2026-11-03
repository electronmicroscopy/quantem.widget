"""Download only the selected tutorial family at its recorded revision."""

import fnmatch
import sys
from pathlib import PureWindowsPath
from types import ModuleType

import pytest
import huggingface_hub

from quantem.widget.io import hub


def test_qualified_download_preserves_revision_and_file_metadata(tmp_path, monkeypatch):
    files = ["haadf/gold.npy", "haadf/gold.json", "4dstem/gold/data.npy", "widget/ducky/show1d_monitor.jsonl", "widget/ducky/snapshots/reference.npy", "denoise/README.md", "denoise/empty/README.md"]
    calls = []
    monkeypatch.setattr(hub, "_shared_download", lambda *args, **kwargs: None)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda **kwargs: calls.append(("list", kwargs)) or files)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: calls.append(("download", kwargs)) or str(tmp_path))
    assert hub.download("haadf/gold", revision="recorded", verbose=False) == tmp_path / "haadf/gold.npy"
    assert calls[-1][1]["allow_patterns"] == ["haadf/gold.npy", "haadf/gold.json"]
    assert all(options["revision"] == "recorded" for _, options in calls)
    with pytest.raises(ValueError, match="qualified"):
        hub.download("gold", verbose=False)
    with pytest.raises(FileNotFoundError):
        hub.download("denoise/empty", verbose=False)
    assert hub.download("widget/ducky", revision="recorded", verbose=False) == tmp_path / "widget/ducky"
    assert calls[-1][1]["allow_patterns"] == "widget/ducky/*"
    # a file inside a folder dataset is not a dataset name, as in quantem.data
    with pytest.raises(FileNotFoundError):
        hub.download("4dstem/gold/data.npy", verbose=False)


# One listing for the fallback and for quantem.data: a file and a folder sharing a name, sidecars,
# placeholders, a folder outside the data buckets, a widget monitor and files that are not data.
LISTING = [
    "haadf/gold.tif", "haadf/gold.yaml", "haadf/gold.json", "haadf/lone.npy", "haadf/gold/frame.npy",
    "4dstem/gold/data_master.h5", "4dstem/gold/meta.yaml", "4dstem/placeholder_next/data.npy", "haadf/placeholder_image.png",
    "survey/overview/image.png", "survey/flat.npy", "widget/ducky/show1d_monitor.jsonl", "widget/monitor.jsonl",
    "denoise/README.md", "denoise/empty/README.md", "notebooks/demo.ipynb", "README.md",
]


def _resolved(download, name, monkeypatch, root):
    """What ``download(name)`` asks the Hub for on ``LISTING``: the dataset path and the files of its patterns, or the error type."""
    asked = []

    def snapshot(**options):
        patterns = options["allow_patterns"]
        asked.append(sorted(path for path in LISTING if any(fnmatch.fnmatchcase(path, pattern) for pattern in ([patterns] if isinstance(patterns, str) else patterns))))
        for path in asked[-1]:  # quantem.data checks that every selected file arrived
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).touch()
        return str(root)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda **options: LISTING)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    try:
        return download(name, verbose=False).relative_to(root).as_posix(), asked[-1]
    except (FileNotFoundError, ValueError) as error:
        return type(error).__name__


def test_fallback_resolves_the_names_quantem_data_resolves(tmp_path, monkeypatch):
    # the fallback took a "survey" bucket quantem.data does not have and listed placeholder_ entries,
    # so one call succeeded without quantem.data and failed with it
    monkeypatch.setattr(hub, "_shared_download", lambda *args, **kwargs: None)
    for name in ("survey/overview", "overview", "flat", "survey/flat.npy", "placeholder_next", "4dstem/placeholder_next", "placeholder_image"):
        assert _resolved(hub.download, name, monkeypatch, tmp_path) == "FileNotFoundError", name
    # a folder and a file share the name haadf/gold: the exact path names one, the short name is ambiguous
    assert _resolved(hub.download, "haadf/gold", monkeypatch, tmp_path) == ("haadf/gold", ["haadf/gold/frame.npy"])
    assert _resolved(hub.download, "haadf/gold.tif", monkeypatch, tmp_path) == ("haadf/gold.tif", ["haadf/gold.json", "haadf/gold.tif", "haadf/gold.yaml"])
    assert _resolved(hub.download, "gold", monkeypatch, tmp_path) == "ValueError"
    # repository paths keep "/" where Path answers haadf\gold.yaml (Windows), which left the sidecars behind
    monkeypatch.setattr(hub, "Path", PureWindowsPath)
    patterns = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **options: patterns.append(options["allow_patterns"]) or str(tmp_path))
    hub.download("haadf/gold.tif", verbose=False)
    assert patterns == [["haadf/gold.tif", "haadf/gold.yaml", "haadf/gold.json"]]


def test_fallback_and_quantem_data_agree_on_one_listing(tmp_path, monkeypatch):
    # runs where quantem.data is importable; the widget itself does not depend on it
    data_hub = pytest.importorskip("quantem.data.huggingface")
    monkeypatch.setattr(hub, "_shared_download", lambda *args, **kwargs: None)
    names = ["haadf/gold", "haadf/gold.tif", "gold", "lone", "haadf/lone", "haadf/lone.npy", "4dstem/gold", "4dstem/gold/data_master.h5",
             "placeholder_next", "4dstem/placeholder_next", "placeholder_image", "survey/overview", "overview", "flat", "survey/flat.npy",
             "ducky", "widget/ducky", "widget/monitor", "monitor", "widget/monitor.jsonl", "denoise/empty", "demo", "missing"]
    for name in names:
        assert _resolved(hub.download, name, monkeypatch, tmp_path / "widget") == _resolved(data_hub.download, name, monkeypatch, tmp_path / "data"), name


def test_download_delegates_all_options_to_compatible_shared_api(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(hub, "_shared_download", lambda name, **kwargs: calls.append((name, kwargs)) or tmp_path)
    assert hub.download("widget/ducky", revision="recorded", cache_dir=tmp_path, force_download=True, verbose=False) == tmp_path
    assert calls[0][0] == "widget/ducky"
    assert calls[0][1]["revision"] == "recorded"
    assert calls[0][1]["cache_dir"] == tmp_path
    assert calls[0][1]["force_download"] is True


def test_old_optional_shared_package_uses_widget_compatibility_resolver(monkeypatch):
    module = ModuleType("quantem.data.huggingface")

    def old_download(name, *, repo=None, out=None, verbose=True):
        raise AssertionError("Old resolver must not receive new revision options")

    module.download = old_download
    monkeypatch.setitem(sys.modules, "quantem.data.huggingface", module)
    assert hub._shared_download("widget/ducky", revision="recorded") is None


def test_existing_session_name_downloads_canonical_folder(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(hub, "_shared_download", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files",
        lambda **kwargs: ["widget/gold_session/gold.emd"],
    )
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda **kwargs: calls.append(kwargs) or str(tmp_path),
    )
    assert hub.download("gold_haadf_session", verbose=False) == tmp_path / "widget/gold_session"
    assert calls[0]["allow_patterns"] == "widget/gold_session/*"
