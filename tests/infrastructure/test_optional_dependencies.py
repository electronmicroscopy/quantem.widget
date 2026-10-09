"""The base install works without quantem core, without quantem.gpu and outside Jupyter.

quantem.gpu is optional (PLATFORM rule 1), and quantem core is a dependency
everywhere except an Intel Mac and Windows on ARM, which cannot install it; there
the readers return the widget's stand-in. A subprocess hides both behind an import
blocker, imports every widget module and constructs every widget on small
arrays, so a stray top-level ``import quantem.gpu`` or ``quantem.core`` anywhere
fails here even on a machine that has both installed. The blocker also hides
the Jupyter server-side packages that ``pip install quantem.widget`` does not
declare, as a plain Python without Jupyter lacks them, so a script that imports
a widget does not depend on a notebook kernel being installed.

hdf5plugin is not installed on Windows on ARM (no wheel). A second subprocess
hides it: every module still imports, Show4DSTEM reads .npy and HDF5 files
without its filters, and a bitshuffle-LZ4 master raises an ImportError that
names hdf5plugin and the x64 Python route instead of h5py's own read error.
Where it is installed, ``read_image`` registers its filters itself, without a
widget or the caller importing hdf5plugin first.
"""

import ast
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import h5py
import numpy as np
import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

import quantem.widget

WIDGET_ROOT = Path(quantem.widget.__file__).resolve().parents[1]
DEPENDENCIES = tomllib.loads((WIDGET_ROOT.parents[1] / "pyproject.toml").read_text())["project"]["dependencies"]
DECLARED = {re.match(r"[A-Za-z0-9_.-]+", requirement).group().lower().replace("-", "_") for requirement in DEPENDENCIES}
# Installed with Jupyter or ipykernel, not with anywidget: present in a notebook, absent from a plain Python.
JUPYTER_ONLY = {"ipykernel", "jupyter_client", "jupyter_server", "jupyterlab", "notebook", "tornado", "zmq"}
HIDDEN = sorted(JUPYTER_ONLY - DECLARED)

BLOCKER = f"""
import importlib.abc
import sys
import types

# A ``quantem`` package that holds only the widget: quantem core's __init__ and
# every other quantem distribution on the path stay invisible.
package = types.ModuleType("quantem")
package.__path__ = [{str(WIDGET_ROOT)!r}]
sys.modules["quantem"] = package


class HideOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        optional = name.startswith("quantem.") and not name.startswith("quantem.widget")
        if optional or name.split(".")[0] in {HIDDEN!r}:
            raise ModuleNotFoundError(f"No module named {{name!r}} (hidden by the test)", name=name)
        return None


sys.meta_path.insert(0, HideOptional())
"""

WIDGETS = """
import importlib
import pkgutil
import tempfile
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from ase import Atoms

import quantem.widget as qw
from quantem.widget import device
from quantem.widget.adapters import core as core_adapter
from quantem.widget.adapters import gpu as gpu_adapter

for module in pkgutil.walk_packages(qw.__path__, "quantem.widget."):
    importlib.import_module(module.name)
assert not gpu_adapter.available() and not core_adapter.available()

rng = np.random.default_rng(0)
image = rng.random((32, 32)).astype(np.float32)
counts = rng.integers(0, 50, (4, 4, 16, 16)).astype(np.uint16)
folder = Path(tempfile.mkdtemp())
np.save(folder / "image.npy", image)
with h5py.File(folder / "scan_data_000001.h5", "w") as handle:
    handle.create_dataset("entry/data/data", data=counts.reshape(16, 16, 16), chunks=(1, 16, 16),
                          **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
with h5py.File(folder / "scan_master.h5", "w") as handle:
    handle.require_group("entry/data")["data_000001"] = h5py.ExternalLink("scan_data_000001.h5", "entry/data/data")

dataset = qw.read_image(folder / "image.npy")
scan = qw.read_4dstem(folder / "scan_master.h5")
assert isinstance(dataset, core_adapter.ArrayDataset) and isinstance(scan, core_adapter.ArrayDataset)
assert scan.name == "scan" and scan.array.dtype == np.uint16 and np.array_equal(scan.array, counts)
widgets = [
    qw.Show1D(rng.random(64).astype(np.float32)),
    qw.Show2D(dataset),
    qw.Show3D(rng.random((3, 16, 16)).astype(np.float32)),
    qw.Show3DSlices(rng.random((8, 8, 8)).astype(np.float32)),
    qw.Show4DSTEM(counts, device="cpu"),
    qw.Show4DSTEM(scan, device="cpu"),
    qw.ShowDiffraction(rng.random((2, 32, 32)).astype(np.float32), device="cpu"),
    qw.ShowCIF(Atoms("Si2", scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25)], cell=[5.43] * 3, pbc=True)),
    qw.Plot2D(image[:4, :6], x=np.arange(6), y=np.arange(4)),
    qw.ChooseLattice(image),
    qw.ShowPtycho(np.exp(1j * image).astype(np.complex64), sampling=0.2),
]
widgets[6].denoise = "nlm"  # scikit-image, as the ShowDiffraction tutorial uses it
dense = widgets[5]
assert dense._session(0).masked_sum(np.ones((16, 16), bool)).tolist() == counts.sum(axis=(2, 3)).astype(np.float32).tolist()
ptycho = widgets[-1]
assert not ptycho.ssb_session and np.frombuffer(ptycho.phase_bytes, np.float32).reshape(32, 32).tolist() == np.angle(np.exp(1j * image).astype(np.complex64)).tolist()
route = device.gpu_path_hint() or device.no_gpu_path()  # this machine's way to quantem.gpu, or why it has none
assert route in ptycho.session_note
try:
    qw.ShowPtycho(object())
except TypeError as exc:
    assert route in str(exc)
else:
    raise AssertionError("ShowPtycho must name what it takes and that a session needs quantem.gpu")
for widget in widgets:
    widget.close()
hidden = sorted(name for name in sys.modules if name.startswith("quantem.") and not name.startswith("quantem.widget"))
assert hidden == [], hidden
print(f"constructed {len(widgets)} widgets without quantem core or quantem.gpu")
"""


def test_every_widget_imports_and_constructs_without_quantem_or_quantem_gpu(tmp_path) -> None:
    script = tmp_path / "base_install.py"
    script.write_text(BLOCKER + textwrap.dedent(WIDGETS), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300,
                            env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                                 "MPLBACKEND": "Agg"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "without quantem core or quantem.gpu" in result.stdout


# Distribution names whose import name differs.
DISTRIBUTION = {"PIL": "pillow", "skimage": "scikit_image", "yaml": "pyyaml", "IPython": "ipython"}
# Imported only inside the function that needs them, with an install message or a fallback.
OPTIONAL = {
    "abtem": "ShowCIF potential preview, the [crystal] extra",
    "cupy": "free_gpu and gpu_info on CUDA, brought by quantem.gpu[cuda]",
    "quantem": "quantem.gpu, and quantem core where it installs, guarded at the top of their adapter modules",
    "ipyfilechooser": "FolderPicker, used by quantem.live",
    "playwright": "quantem github widget screenshots",
    "yaml": "dataset.yaml, read by quantem.live",
    "tqdm": "progress bars when it is installed",
}
# Declared without a Python import: a tifffile codec plugin and the JupyterLab widget manager.
NOT_IMPORTED = {"imagecodecs", "jupyterlab_widgets"}


def test_every_import_is_declared_or_optional_and_every_dependency_is_imported() -> None:
    """A module imported outside a function must be a declared dependency, an optional one only
    inside the function that needs it (quantem core and quantem.gpu: in ``adapters/``). Every
    declared dependency has an importer."""
    undeclared, eager, imported = [], [], set()
    for path in sorted(WIDGET_ROOT.joinpath("widget").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        in_function = {id(inner) for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       for inner in ast.walk(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or name.startswith("quantem.widget"):
                    continue
                distribution = DISTRIBUTION.get(top, top).lower()
                imported.add(distribution)
                where = f"{path.relative_to(WIDGET_ROOT)}:{node.lineno} {name}"
                if top not in OPTIONAL and distribution not in DECLARED:
                    undeclared.append(where)
                adapter = top == "quantem" and path.parent.name == "adapters"
                if top in OPTIONAL and id(node) not in in_function and not adapter:
                    eager.append(where)
    assert undeclared == [], "imported but not declared: " + ", ".join(undeclared)
    assert eager == [], "optional modules imported outside a function: " + ", ".join(eager)
    assert DECLARED - imported == NOT_IMPORTED, sorted(DECLARED - imported)


def test_windows_on_arm_skips_the_dependencies_without_an_arm64_wheel() -> None:
    requirements = [Requirement(text) for text in DEPENDENCIES]  # an invalid PEP 508 marker raises here

    def selected(system: str, machine: str) -> dict[str, SpecifierSet]:
        environment = {"sys_platform": system, "platform_machine": machine}
        return {req.name: req.specifier for req in requirements if req.marker is None or req.marker.evaluate(environment)}

    # x64 Python 3.12+ on Windows on ARM also reports ARM64, so it takes this branch too
    windows_arm = selected("win32", "ARM64")
    assert "hdf5plugin" not in windows_arm
    assert windows_arm["ncempy"] == SpecifierSet(">=1.11,<1.15")  # 1.15+ requires hdf5plugin
    # quantem core needs hdf5plugin 6 (no Windows ARM64 wheel) and torch 2.7 (no Intel Mac wheel)
    assert "quantem" not in windows_arm and "quantem" not in selected("darwin", "x86_64")
    for system, machine in (("linux", "x86_64"), ("win32", "AMD64"), ("darwin", "arm64"), ("darwin", "x86_64")):
        others = selected(system, machine)
        assert others["hdf5plugin"] == SpecifierSet(">=4.1") and others["ncempy"] == SpecifierSet(">=1.11")
        if (system, machine) != ("darwin", "x86_64"):
            assert others["quantem"] == SpecifierSet(">=0.1.9")


WITHOUT_HDF5PLUGIN = """
import importlib
import pkgutil
from pathlib import Path

import numpy as np

sys.modules["hdf5plugin"] = None  # not installed, as on Windows on ARM
import quantem.widget as qw

for module in pkgutil.walk_packages(qw.__path__, "quantem.widget."):
    importlib.import_module(module.name)
folder = Path(sys.argv[1])
counts = np.load(folder / "scan.npy")
for name in ("scan.npy", "plain_master.h5", "gzip.h5"):
    widget = qw.Show4DSTEM(folder / name, device="cpu")
    assert widget._session(0).masked_sum(np.ones((16, 16), bool)).tolist() == counts.sum(axis=(2, 3)).astype(np.float32).tolist()
    widget.close()
try:
    qw.Show4DSTEM(folder / "compressed_master.h5", device="cpu")
except ImportError as exc:
    print(exc)
else:
    raise AssertionError("a bitshuffle-LZ4 master must name hdf5plugin")
"""


def test_show4dstem_without_hdf5plugin_reads_npy_and_names_it_for_a_compressed_master(tmp_path) -> None:
    hdf5plugin = pytest.importorskip("hdf5plugin")
    counts = np.random.default_rng(0).integers(0, 50, (4, 4, 16, 16)).astype(np.uint16)
    np.save(tmp_path / "scan.npy", counts)
    frames = counts.reshape(16, 16, 16)
    with h5py.File(tmp_path / "plain_data_000001.h5", "w") as handle:
        handle.create_dataset("entry/data/data", data=frames)
    with h5py.File(tmp_path / "gzip.h5", "w") as handle:
        handle.create_dataset("entry/data/data", data=frames, chunks=(1, 16, 16), compression="gzip", shuffle=True)
    with h5py.File(tmp_path / "compressed_data_000001.h5", "w") as handle:
        handle.create_dataset("entry/data/data", data=frames, chunks=(1, 16, 16),
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    for kind in ("plain", "compressed"):
        with h5py.File(tmp_path / f"{kind}_master.h5", "w") as handle:
            handle.require_group("entry/data")["data_000001"] = h5py.ExternalLink(f"{kind}_data_000001.h5", "entry/data/data")
    script = tmp_path / "without_hdf5plugin.py"
    script.write_text(BLOCKER + textwrap.dedent(WITHOUT_HDF5PLUGIN), encoding="utf-8")
    # an empty plugin directory: a system bitshuffle plugin must not stand in for hdf5plugin
    result = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, timeout=300,
                            env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                                 "MPLBACKEND": "Agg", "HDF5_PLUGIN_PATH": str(tmp_path / "no-plugins")})
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "compressed_master.h5: the detector data is compressed with the bitshuffle filter (HDF5 filter 32008), "
        "which h5py reads through hdf5plugin, and hdf5plugin is not installed or did not import. Install it with "
        "pip install hdf5plugin. Windows on ARM has no hdf5plugin wheel for ARM64 Python: use x64 Python, "
        "which Windows runs under emulation, and pip install hdf5plugin there."
    ) in result.stdout


READ_IMAGE = """
import numpy as np

import quantem.widget as qw

np.save(sys.argv[2], qw.read_image(sys.argv[1]).array)
"""


def test_read_image_decodes_a_bitshuffle_emd_without_a_prior_hdf5plugin_import(tmp_path) -> None:
    hdf5plugin = pytest.importorskip("hdf5plugin")
    image = np.random.default_rng(1).integers(0, 1000, (32, 48)).astype(np.uint16)
    with h5py.File(tmp_path / "frame.emd", "w") as handle:
        handle.create_dataset("data/frame/data", data=image, chunks=(8, 48), **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    script = tmp_path / "read_image.py"
    script.write_text(BLOCKER + textwrap.dedent(READ_IMAGE), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), str(tmp_path / "frame.emd"), str(tmp_path / "read.npy")],
                            capture_output=True, text=True, timeout=300,
                            env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                                 "MPLBACKEND": "Agg", "HDF5_PLUGIN_PATH": str(tmp_path / "no-plugins")})
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(np.load(tmp_path / "read.npy"), image)
