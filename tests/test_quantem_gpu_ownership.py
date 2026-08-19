from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager


def test_widget_has_no_duplicate_gpu_or_io_public_api() -> None:
    import quantem.widget.io as widget_io
    from quantem import widget

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


def test_colab_tutorials_use_one_short_latest_rc_setup() -> None:
    """Every Colab notebook presents the same compact latest-RC setup step."""

    repo = Path(__file__).resolve().parents[1]
    colab_installers: list[tuple[str, str, str | None]] = []
    colab_notebooks: set[str] = set()
    for path in sorted((repo / "docs" / "tutorials").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        notebook_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        if "colab.research.google.com" in notebook_source:
            colab_notebooks.add(path.name)
        for cell in notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            if "scripts/install_colab.py" in source:
                colab_installers.append((path.name, source, cell.get("id")))

    installer_notebooks = [name for name, _source, _id in colab_installers]
    assert set(installer_notebooks) == colab_notebooks
    assert len(installer_notebooks) == len(colab_notebooks)
    for notebook_name, source, cell_id in colab_installers:
        assert cell_id
        assert "https://test.pypi.org/simple/" not in source, notebook_name
        assert source.startswith(
            '# @title Install QuantEM { display-mode: "form" }'
        ), notebook_name
        assert "scripts/install_colab.py" in source, notebook_name
        setup_lines = [line for line in source.splitlines() if line.strip()]
        expected_lines = 5 if notebook_name == "showdiffraction.ipynb" else 4
        assert len(setup_lines) == expected_lines, notebook_name
        assert "from urllib.request import urlopen" not in source, notebook_name
        assert "__import__" not in source, notebook_name
        assert "exec(" not in source, notebook_name
        assert "%run install_quantem.py" in source, notebook_name


def test_shared_colab_installer_resolves_only_hashed_quantem_wheels() -> None:
    """The shared installer keeps TestPyPI out of dependency resolution."""

    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "install_colab.py").read_text(encoding="utf-8")

    assert "https://test.pypi.org/pypi/{project}/json" in source
    assert '_latest_testpypi_wheel_url("quantem.widget")' in source
    assert '_latest_testpypi_wheel_url("quantem.gpu")' in source
    assert 'f"numpy=={np.__version__}"' in source
    assert 'f"numba=={numba_version}"' in source
    assert 'f"quantem.gpu[movie] @ {gpu_wheel}"' in source
    assert "from quantem.widget import profile" in source
    assert "profile()" in source
    assert "version('quantem.widget')" not in source
    assert "version('quantem.gpu')" not in source
    assert "#sha256={digest}" in source
    assert "--extra-index-url" not in source
    assert "--index-url" not in source


def test_show4dstem_colab_uses_kernel_compute_without_webgpu() -> None:
    """Colab uses Python compute through the same simple notebook API."""

    repo = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repo / "docs" / "tutorials" / "show4dstem.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    assert 'offline="google.colab" not in modules' in source
    assert "IPython.display" not in source
    assert "display(viewer)" not in source
    assert "asyncio.sleep" not in source
    assert "viewer.send_state" not in source
    last_code_cell = next(
        cell
        for cell in reversed(notebook["cells"])
        if cell.get("cell_type") == "code"
    )
    assert "".join(last_code_cell["source"]).rstrip().endswith("viewer")


def test_show4dstem_colab_guides_a_three_minute_widget_experiment() -> None:
    """The Colab tutorial leads directly from Run all to three UI actions."""

    repo = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repo / "docs" / "tutorials" / "show4dstem.ipynb").read_text(
            encoding="utf-8"
        )
    )
    cells = notebook["cells"]
    introduction = "".join(cells[0]["source"])
    experiment = "".join(cells[3]["source"])
    conclusion = "".join(cells[5]["source"])
    code_cells = [
        "".join(cell["source"])
        if isinstance(cell["source"], list)
        else cell["source"]
        for cell in cells
        if cell["cell_type"] == "code"
    ]

    assert introduction.startswith("# Explore 4D-STEM in three minutes")
    assert "Runtime → Run all" in introduction
    assert len(introduction.split()) < 60
    assert [source.splitlines()[0] for source in code_cells] == [
        '# @title Install QuantEM { display-mode: "form" }',
        '# @title 2. Load a small real dataset { display-mode: "form" }',
        '# @title 3. Open the explorer { display-mode: "form" }',
    ]
    for action in ("Choose a scan point", "Compare angles", "Measure a line"):
        assert action in experiment
    assert "Profile" in experiment
    assert "Reset" in experiment
    assert "No Python call is required" in conclusion


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
    display_reexports = {
        name: (repo / "js" / name).read_text(encoding="utf-8").strip()
        for name in (
            "colormaps.ts",
            "displayFilter.ts",
            "fft.ts",
            "frequencyFilter.ts",
            "geometry.ts",
            "stats.ts",
        )
    }

    assert 'targetDir = "js/.generated/engine"' in sync_script
    assert '"device/webgpu.ts"' in sync_script
    assert '"display/webgpu/colormaps.ts"' in sync_script
    assert '"display/webgpu/fft.ts"' in sync_script
    assert '"display/webgpu/filter.ts"' in sync_script
    assert '"display/webgpu/frequencyFilter.ts"' in sync_script
    assert '"display/webgpu/geometry.ts"' in sync_script
    assert '"display/webgpu/stats.ts"' in sync_script
    assert '"swift/Sources/MetalDisplayKernels/Resources/colormaps.json"' in sync_script
    assert '"io/backends/webgpu/bslz4.ts"' in sync_script
    assert '"detector/compute/webgpu/backend.ts"' in sync_script
    assert '"dpc/compute/webgpu/fft.ts"' in sync_script
    assert "syncGpuWebgpuSources()" in build_script
    expected_reexports = {
        "colormaps.ts": 'export * from "./.generated/engine/display/webgpu/colormaps";',
        "displayFilter.ts": 'export * from "./.generated/engine/display/webgpu/filter";',
        "fft.ts": 'export * from "./.generated/engine/display/webgpu/fft";',
        "frequencyFilter.ts": 'export * from "./.generated/engine/display/webgpu/frequencyFilter";',
        "geometry.ts": 'export * from "./.generated/engine/display/webgpu/geometry";',
        "stats.ts": 'export * from "./.generated/engine/display/webgpu/stats";',
    }
    for name, expected in expected_reexports.items():
        assert display_reexports[name].endswith(expected)
        assert display_reexports[name].count("export *") == 1
    assert "../.generated/engine/io/backends/webgpu/bslz4" in show4dstem
    assert "../.generated/engine/io/backends/webgpu/local-h5" in show4dstem
    assert 'from "./lazy"' in show4dstem
    assert "Show4DSTEMCpuCompute" not in show4dstem
    assert "no CPU fallback is used" in show4dstem
    assert "../.generated/engine/ssb/compute/webgpu/backend" in showptycho
    assert "../../../js/.generated/engine/io/backends/webgpu/h5reader" in web_store
    assert "../../js/.generated/engine/detector/compute/webgpu/backend" in web_app
