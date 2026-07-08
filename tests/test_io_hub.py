import importlib
import json
import sys
import types
from pathlib import Path


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _fresh_hub(monkeypatch, modules: dict[str, types.ModuleType]):
    for name in ("quantem.widget.io.hub", "quantem.data.hub", "quantem.data.huggingface", "quantem.data"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return importlib.import_module("quantem.widget.io.hub")


class FakeHuggingFaceHub:
    """Records file uploads/downloads without network access."""

    def __init__(self, files: dict[str, str] | None = None):
        self.files = dict(files or {})
        self._tmp: Path | None = None

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
        self.files[path_in_repo] = Path(path_or_fileobj).read_text()
        return type("CommitInfo", (), {"commit_url": "http://commit/file"})()

    def upload_folder(self, *, folder_path, path_in_repo, repo_id, repo_type):
        self.files[f"{path_in_repo}/_folder"] = "<folder>"
        return type("CommitInfo", (), {"commit_url": "http://commit/folder"})()

    def list_repo_files(self, *, repo_id, repo_type):
        return list(self.files)

    def hf_hub_download(self, *, repo_id, repo_type, filename):
        if self._tmp is None:
            raise RuntimeError("test must set FakeHuggingFaceHub._tmp")
        out = self._tmp / Path(filename).name
        out.write_text(self.files[filename])
        return str(out)


def test_hub_adapter_prefers_original_quantem_data_hub(monkeypatch):
    parent = _module("quantem.data")
    parent.__path__ = []
    source = _module(
        "quantem.data.hub",
        download=lambda name: f"hub:{name}",
        list_datasets=lambda: ["4dstem/gold"],
        read_meta=lambda name: {"name": name},
        status=lambda: {"source": "hub"},
    )

    hub = _fresh_hub(monkeypatch, {"quantem.data": parent, "quantem.data.hub": source})

    assert hub._SOURCE_NAME == "quantem.data.hub"
    assert hub.download("gold") == "hub:gold"
    assert hub.list_datasets() == ["4dstem/gold"]
    assert hub.read_meta("gold") == {"name": "gold"}
    assert hub.status() == {"source": "hub"}


def test_hub_adapter_uses_huggingface_module_when_hub_module_is_absent(monkeypatch):
    parent = _module("quantem.data")
    parent.__path__ = []
    source = _module(
        "quantem.data.huggingface",
        download=lambda name, verbose=True: Path("/cache") / name,
        list_datasets=lambda: ["haadf/gold_haadf"],
        read_meta=lambda name: {"flat": name},
        status=lambda: {"source": "huggingface"},
    )

    hub = _fresh_hub(monkeypatch, {"quantem.data": parent, "quantem.data.huggingface": source})

    assert hub._SOURCE_NAME == "quantem.data.huggingface"
    assert hub.download("gold_haadf") == Path("/cache/gold_haadf")
    assert hub.list_datasets() == ["haadf/gold_haadf"]
    assert hub.read_meta("gold_haadf") == {"flat": "gold_haadf"}


def test_hub_adapter_maps_registry_facade_names(monkeypatch):
    source = _module(
        "quantem.data",
        available=lambda technique=None: ["eds/gold"] if technique == "eds" else ["gold"],
        info=lambda name: {"name": name, "technique": "eds"},
        load_raw=lambda name: f"/cache/raw/{name}.emd",
        list_files=lambda technique=None: [
            {"path": "eds/gold.npy", "size_mb": 2048.0, "type": "data"},
            {"path": "eds/gold.json", "size_mb": 0.02, "type": "metadata"},
        ],
    )
    source.__path__ = []

    hub = _fresh_hub(monkeypatch, {"quantem.data": source})

    assert hub._SOURCE_NAME == "quantem.data"
    assert hub.list_datasets(technique="eds") == ["eds/gold"]
    assert hub.download("gold") == Path("/cache/raw/gold.emd")
    assert hub.read_meta("gold") == {"name": "gold", "technique": "eds"}
    assert hub.status()["total_mb"] == 2048.02


def test_hub_adapter_downloads_folder_style_registry_dataset(monkeypatch, tmp_path):
    source = _module(
        "quantem.data",
        list_files=lambda technique=None: [
            {"path": "4dstem/gold_128_npy_bin8/data.npy", "size_mb": 18.0, "type": "data"},
            {"path": "4dstem/gold_128_npy_bin8/meta.json", "size_mb": 0.01, "type": "metadata"},
        ],
    )
    source.__path__ = []

    calls = {}
    def snapshot_download(**kwargs):
        calls["snapshot"] = kwargs
        return str(tmp_path)

    hf = _module("huggingface_hub", snapshot_download=snapshot_download)

    hub = _fresh_hub(monkeypatch, {"quantem.data": source, "huggingface_hub": hf})

    result = hub.download("gold_128_npy_bin8", verbose=False)

    assert result == tmp_path / "4dstem/gold_128_npy_bin8"
    assert calls["snapshot"]["repo_id"] == "bobleesj/quantem-data"
    assert calls["snapshot"]["repo_type"] == "dataset"
    assert calls["snapshot"]["allow_patterns"] == "4dstem/gold_128_npy_bin8/*"


def test_hub_upload_keeps_legacy_file_sidecar(monkeypatch, tmp_path):
    source = _module("quantem.data")
    source.__path__ = []
    hub = _fresh_hub(monkeypatch, {"quantem.data": source})
    fake = FakeHuggingFaceHub()
    monkeypatch.setattr(hub, "_hub", lambda: fake)

    img = tmp_path / "gold.tif"
    img.write_text("pixels")
    hub.upload(img, name="gold_haadf", meta={"voltage_kV": 300, "fov_nm": 76.2})

    assert "haadf/gold_haadf.tif" in fake.files
    assert json.loads(fake.files["haadf/gold_haadf.json"]) == {
        "voltage_kV": 300,
        "fov_nm": 76.2,
    }


def test_hub_read_meta_keeps_legacy_folder_sidecar(monkeypatch, tmp_path):
    source = _module("quantem.data")
    source.__path__ = []
    hub = _fresh_hub(monkeypatch, {"quantem.data": source})
    fake = FakeHuggingFaceHub({
        "4dstem/gold_512/quantem_meta.json": json.dumps({"scan_shape": [512, 512]})
    })
    fake._tmp = tmp_path
    monkeypatch.setattr(hub, "_hub", lambda: fake)

    assert hub.read_meta("gold_512") == {"scan_shape": [512, 512]}
