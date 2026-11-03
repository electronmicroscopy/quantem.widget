import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest
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
    assert "display/webgpu/borrowed-image" not in sync_script
    # one borrowed-image contract: quantem.gpu's, which every bundle drawing through js/display needs
    assert "\ntry {\n  syncGpuWebgpuSources();\n}" in build_script
    assert (display / "borrowedImage.ts").read_text(encoding="utf-8").endswith(
        'export { validateUint32ImageView, type Uint32ImageView } from "../.generated/engine/detector/webgpu/borrowed-image";\n'
    )
    offenders = [
        path.relative_to(repo).as_posix()
        for path in (repo / "js").rglob("*.ts*")
        if ".generated" not in path.parts
        and path.parent.name not in ("show4dstem", "showptycho")
        and path != display / "borrowedImage.ts"
        and ".generated/" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
    assert "../.generated/engine/io/hdf5/webgpu/bslz4" in show4dstem
    assert "../.generated/engine/io/hdf5/webgpu/local-h5" in show4dstem
    assert "../.generated/engine/ssb/webgpu/backend" in showptycho


def test_every_build_bundles_the_quantem_gpu_ref_in_gpu_ref_txt() -> None:
    """Every build uses one immutable GPU commit or the runtime floor's tag."""
    repo = Path(__file__).resolve().parents[2]
    workflows = {path.name: path.read_text(encoding="utf-8") for path in (repo / ".github" / "workflows").glob("*.yml")}
    builds = sorted(name for name, text in workflows.items() if "npm run build" in text)
    assert builds == ["widget-ci.yml", "widget-docs.yml", "widget-release.yml"]
    for name in builds:
        text = workflows[name]
        assert re.findall(r"repository: bobleesj/quantem\.gpu\n\s+ref: (.+)\n", text) == ["${{ steps.gpu.outputs.ref }}"], name
        assert 'echo "ref=$(cat gpu-ref.txt)"' in text, name
        assert "QUANTEM_GPU_SRC: ${{ github.workspace }}/quantem-gpu-src/src" in text, name
    pinned = (repo / "gpu-ref.txt").read_text(encoding="utf-8").strip()
    bounds = re.findall(r'"quantem\.gpu\[(?:cuda|mps)\]>=([^,;"]+),<([^;"]+)', (repo / "pyproject.toml").read_text(encoding="utf-8"))
    assert len(bounds) == 2
    assert bounds[0] == bounds[1]
    # Development builds can pin a qualified commit before the next release.
    # Reject movable branches and abbreviated hashes; release tags still match
    # the runtime dependency floor. Source qualification does not prove that an
    # older published runtime provides every API in the development checkout.
    if re.fullmatch(r"[0-9a-f]{40}", pinned) is None:
        assert [f"gpu-v{floor}" for floor, _ in bounds] == [pinned, pinned]
    # the next quantem.gpu rc may change the kernels and API this release bundles
    for floor, ceiling in bounds:
        prefix, number = floor.split("rc")
        assert ceiling == f"{prefix}rc{int(number) + 1}"

def test_sync_writes_a_manifest_of_the_pinned_ref_and_every_file(tmp_path) -> None:
    """MANIFEST.json names the ref in gpu-ref.txt and the commit synced, and the sha256 of every file written."""
    repo = Path(__file__).resolve().parents[2]
    target = tmp_path / "engine"
    script = (repo / "scripts" / "sync-gpu-webgpu.mjs").as_uri()
    result = subprocess.run(
        ["node", "--input-type=module", "-e",
         f"import {{ syncGpuWebgpuSources }} from {json.dumps(script)}; "
         f"syncGpuWebgpuSources({{ targetDir: {json.dumps(str(target))} }});"],
        capture_output=True, text=True, cwd=repo, timeout=120,
    )
    if "no quantem.gpu source was found" in result.stderr:
        pytest.skip("no quantem.gpu source: set QUANTEM_GPU_SRC (CI checks out gpu-ref.txt)")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((target / "MANIFEST.json").read_text(encoding="utf-8"))
    engine = manifest["quantem_gpu"]
    assert engine["pinned_ref"] == (repo / "gpu-ref.txt").read_text(encoding="utf-8").strip()
    on_disk = {path.relative_to(target).as_posix(): path for path in target.rglob("*") if path.is_file()}
    on_disk.pop("MANIFEST.json")
    recorded = {**manifest["files"], **manifest["generated"]}
    assert sorted(recorded) == sorted(on_disk)
    assert {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in on_disk.items()} == recorded
    assert all(name.startswith(("detector/", "dpc/", "formats/", "io/", "ssb/")) for name in manifest["files"])
    assert sorted(manifest["generated"]) == ["device/webgpu.ts"]
    assert "detector/webgpu/borrowed-image.ts" in manifest["files"]
    # one line names the source and the commit it synced from; a checkout found beside this one is named first
    *found, line = result.stdout.strip().splitlines()
    assert len(found) <= 1 and all(note.startswith("using the quantem.gpu checkout beside this one, ") for note in found)
    assert line.startswith(f"synced {len(manifest['files'])} quantem.gpu science kernel files from ")
    assert engine["git_sha"] is None or f"at {engine['ref']} ({engine['git_sha'][:12]})" in line


def test_profile_prints_the_quantem_gpu_ref_the_bundles_carry(tmp_path, monkeypatch, capsys) -> None:
    from quantem.widget import info

    manifest = tmp_path / "engine-manifest.json"
    monkeypatch.setattr(info, "ENGINE_MANIFEST", manifest)
    info.profile()
    assert "webgpu kernels  (frontend not built)" in capsys.readouterr().out
    engine = {"ref": "gpu-v0.0.1rc15", "git_sha": "0123456789abcdef", "version": None, "pinned_ref": "gpu-v0.0.1rc15"}
    manifest.write_text(json.dumps({"quantem_gpu": engine, "files": {}, "generated": {}}), encoding="utf-8")
    info.profile()
    assert "webgpu kernels  gpu-v0.0.1rc15 (0123456789ab)\n" in capsys.readouterr().out
    engine.update(ref=None, git_sha=None, version="0.0.1rc14")
    manifest.write_text(json.dumps({"quantem_gpu": engine, "files": {}, "generated": {}}), encoding="utf-8")
    info.profile()
    assert "webgpu kernels  quantem.gpu 0.0.1rc14, pinned gpu-v0.0.1rc15\n" in capsys.readouterr().out
