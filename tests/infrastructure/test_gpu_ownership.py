import ast
import json
import re
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager


def test_widget_has_no_duplicate_gpu_or_io_public_api() -> None:
    import quantem.widget.io as widget_io
    from quantem import widget

    repo = Path(__file__).resolve().parents[2]
    widget_package = repo / "src" / "quantem" / "widget"
    stale_modules = [
        "backend.py",
        "detector.py",
        "dpc.py",
        "io/backends",
        "io/bitshuffle.py",
        "io/constants.py",
        "io/hdf5.py",
        "io/save.py",
        "kernels/compute",
        "kernels/io",
    ]

    stale_files = []
    for relative in stale_modules:
        path = widget_package / relative
        if path.is_file() or (path.is_dir() and any(path.rglob("*.py"))):
            stale_files.append(relative)
    assert stale_files == []
    assert not hasattr(widget, "load")
    assert "load" not in widget.__all__
    for name in (
        "Dataset4dstem",
        "MasterReadiness",
        "bin",
        "detect_backend",
        "discover_masters",
        "inspect_master_readiness",
        "is_master_ready",
        "load",
        "resolve_backend",
        "save",
    ):
        assert not hasattr(widget_io, name)


def test_live_gpu_status_hook_remains_available() -> None:
    from quantem.widget.gpu import vram_status

    assert callable(vram_status)


def test_public_docs_use_the_canonical_gpu_api() -> None:
    """Tutorials must not revive widget-owned IO or retired SSB fit calls."""

    repo = Path(__file__).resolve().parents[2]
    docs = repo / "docs"
    documents = list(docs.rglob("*.md"))
    retired_widget_load = re.compile(
        r"from\s+quantem\.widget\s+import\s+(?:\([^)]*\)|[^\n]*)"
    )
    retired_ssb_member = re.compile(
        r"\b(?:ssb|workflow)\.(?:optimize|refine|result|explore)\b"
    )
    offenders: list[str] = []
    for path in documents:
        source = path.read_text(encoding="utf-8")
        if any(
            re.search(r"\bload\b", match.group(0))
            for match in retired_widget_load.finditer(source)
        ):
            offenders.append(path.relative_to(repo).as_posix())
        if "quantem.widget.load" in source:
            offenders.append(path.relative_to(repo).as_posix())
        if retired_ssb_member.search(source):
            offenders.append(path.relative_to(repo).as_posix())

    showptycho_docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((docs / "tutorials").glob("showptycho*.md"))
    )
    assert re.search(r"\bSSB\.open\s*\(", showptycho_docs)
    assert offenders == []


def test_tutorial_notebook_code_is_valid_and_uses_gpu_owned_io() -> None:
    """Every committed tutorial code cell must parse and avoid widget load."""

    repo = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    retired_widget_load = re.compile(
        r"from\s+quantem\.widget\s+import\s+(?:\([^)]*\)|[^\n]*)"
    )
    retired_ssb_member = re.compile(
        r"\b(?:ssb|workflow)\.(?:optimize|refine|result|explore)\b"
    )
    for path in sorted((repo / "docs" / "tutorials").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            source = "".join(cell.get("source", []))
            if any(
                re.search(r"\bload\b", match.group(0))
                for match in retired_widget_load.finditer(source)
            ) or "quantem.widget.load" in source or retired_ssb_member.search(source):
                offenders.append(f"{path.name}:cell-{cell_index}")
            if cell.get("cell_type") != "code":
                continue
            python_source = TransformerManager().transform_cell(source)
            tree = ast.parse(python_source, filename=f"{path}:cell-{cell_index}")
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "quantem.widget"
                    and any(name.name == "load" for name in node.names)
                ):
                    offenders.append(f"{path.name}:cell-{cell_index}")
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "load"
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "widget"
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "quantem"
                ):
                    offenders.append(f"{path.name}:cell-{cell_index}")
    assert offenders == []


def test_widget_source_uses_public_gpu_domains() -> None:
    repo = Path(__file__).resolve().parents[2]
    widget_package = repo / "src" / "quantem" / "widget"
    stale_imports = (
        "quantem.widget.backend",
        "quantem.widget.detector",
        "quantem.widget.dpc",
        "quantem.widget.io.backends",
        "quantem.widget.io.bitshuffle",
        "quantem.widget.io.constants",
        "quantem.widget.io.hdf5",
        "quantem.widget.io.save",
        "quantem.widget.kernels.compute",
        "quantem.widget.kernels.io",
        "quantem.gpu.io.hdf5",
        "quantem.gpu.resident",
        "quantem.gpu.webgpu",
    )

    offenders = []
    for path in widget_package.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported_modules = {
            name.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for name in node.names
        }
        imported_modules.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        if any(
            module == stale_import or module.startswith(f"{stale_import}.")
            for module in imported_modules
            for stale_import in stale_imports
        ):
            offenders.append(path.relative_to(widget_package).as_posix())
    assert offenders == []


def test_widget_owns_display_kernels_and_syncs_only_science() -> None:
    """Display kernels are widget source; only science kernels come from quantem.gpu."""

    repo = Path(__file__).resolve().parents[2]
    sync_script = (repo / "scripts" / "sync-gpu-webgpu.mjs").read_text(
        encoding="utf-8"
    )
    build_script = (repo / "scripts" / "build.mjs").read_text(encoding="utf-8")
    show4dstem = (repo / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )
    showptycho = (repo / "js" / "showptycho" / "index.tsx").read_text(
        encoding="utf-8"
    )
    display = repo / "js" / "display"
    for name in ("colormaps.ts", "fft.ts", "stats.ts", "geometry.ts", "device.ts"):
        assert ".generated" not in (display / name).read_text(encoding="utf-8")
    assert 'targetDir = "js/.generated/engine"' in sync_script
    assert 'SCIENCE_DOMAINS = ["detector/", "dpc/", "formats/", "io/", "ssb/"]' in sync_script
    assert 'export * from "../../../display/device"' in sync_script
    assert '{ name: "show4dstem", science: true }' in build_script
    assert '{ name: "showptycho", science: true }' in build_script
    offenders = [
        path.relative_to(repo).as_posix()
        for path in (repo / "js").rglob("*.ts*")
        if ".generated" not in path.parts
        and path.parent.name not in ("show4dstem", "showptycho")
        and ".generated/" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
    assert "../.generated/engine/io/hdf5/webgpu/bslz4" in show4dstem
    assert "../.generated/engine/io/hdf5/webgpu/local-h5" in show4dstem
    assert "../.generated/engine/ssb/webgpu/backend" in showptycho
