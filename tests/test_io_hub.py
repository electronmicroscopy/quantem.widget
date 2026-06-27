import importlib
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
