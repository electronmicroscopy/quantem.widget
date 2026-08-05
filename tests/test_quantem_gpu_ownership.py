from __future__ import annotations

import ast
import json
import re
from pathlib import Path


def test_widget_has_no_duplicate_gpu_or_io_public_api() -> None:
    import quantem.widget as widget
    import quantem.widget.io as widget_io

    repo = Path(__file__).resolve().parents[1]
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
        "LoadResult",
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

    repo = Path(__file__).resolve().parents[1]
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

    repo = Path(__file__).resolve().parents[1]
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
            tree = ast.parse(source, filename=f"{path}:cell-{cell_index}")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module == "quantem.widget" and any(
                        name.name == "load" for name in node.names
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


def test_public_docs_do_not_contain_private_deployment_identifiers() -> None:
    """Published guidance uses placeholders, not lab hostnames or dated mounts."""

    repo = Path(__file__).resolve().parents[1]
    paths = [
        *repo.joinpath("docs").rglob("*.md"),
        *repo.joinpath("docs", "tutorials").glob("*.ipynb"),
        repo / "src/quantem/widget/paths.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert re.search(r"\btail[0-9a-f]+\.ts\.net\b", source) is None
    assert re.search(r"/data/(?:shared/)?arina/\d{8}[_-]", source) is None


def test_widget_source_uses_public_gpu_domains() -> None:
    repo = Path(__file__).resolve().parents[1]
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
        "quantem.gpu.compute",
        "quantem.gpu.io.hdf5",
        "quantem.gpu.io.backends",
        "quantem.gpu.io.mps_multi",
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


def test_widget_webgpu_sources_are_generated_from_quantem_gpu() -> None:
    repo = Path(__file__).resolve().parents[1]

    tracked_engine_sources = sorted(
        path.name for path in (repo / "js" / "engine").glob("*.ts")
    )
    assert tracked_engine_sources == []

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
    web_store = (repo / "web" / "src" / "local" / "store.ts").read_text(
        encoding="utf-8"
    )
    web_app = (repo / "web" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert 'targetDir = "js/.generated/engine"' in sync_script
    assert '"device/webgpu.ts"' in sync_script
    assert '"io/backends/webgpu/bslz4.ts"' in sync_script
    assert '"detector/compute/webgpu/backend.ts"' in sync_script
    assert '"dpc/compute/webgpu/fft.ts"' in sync_script
    assert "syncGpuWebgpuSources()" in build_script
    assert "../.generated/engine/io/backends/webgpu/bslz4" in show4dstem
    assert "../.generated/engine/io/backends/webgpu/local-h5" in show4dstem
    assert 'from "./lazy"' in show4dstem
    assert "Show4DSTEMCpuCompute" not in show4dstem
    assert "no CPU fallback is used" in show4dstem
    assert "../.generated/engine/ssb/compute/webgpu/backend" in showptycho
    assert "../../../js/.generated/engine/io/backends/webgpu/h5reader" in web_store
    assert "../../js/.generated/engine/detector/compute/webgpu/backend" in web_app
