"""Opt-in real-data browser E2E for folder-backed Show4DSTEM.

This is a workstation signoff, not a normal CI test.  It opens a supplied
real-data folder through :meth:`Show4DSTEM.from_folder`, drives the mounted
widget in JupyterLab, closes the first kernel cleanly, and then opens a fresh
kernel over the same persistent preview cache.

The source folder is never modified.  Initial masters and the watched arrival
are represented by symlinks inside the supplied report directory.  Reports,
screenshots, the generated notebook, Jupyter logs, browser state, and cache
files likewise stay under explicit caller-supplied roots.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
import traceback
from typing import Any
import urllib.request

import pytest


RUN_ENV = "QT_RUN_SHOW4DSTEM_FOLDER_E2E"
SOURCE_ENV = "QT_SHOW4DSTEM_FOLDER_E2E_SOURCE"
REPORT_ENV = "QT_SHOW4DSTEM_FOLDER_E2E_REPORT_DIR"
CACHE_ENV = "QT_SHOW4DSTEM_FOLDER_E2E_CACHE_DIR"
DEVICES_ENV = "QT_SHOW4DSTEM_FOLDER_E2E_CUDA_DEVICES"
RUN_ID_ENV = "QT_SHOW4DSTEM_FOLDER_E2E_RUN_ID"

pytestmark = pytest.mark.skipif(
    os.environ.get(RUN_ENV) != "1",
    reason=(
        f"set {RUN_ENV}=1 plus {SOURCE_ENV}, {REPORT_ENV}, {CACHE_ENV}, and "
        f"{DEVICES_ENV} to run the real-data folder/Jupyter/browser E2E"
    ),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Config:
    source: Path
    report_root: Path
    cache_root: Path
    run_id: str
    artifact_dir: Path
    cache_dir: Path
    staging_dir: Path
    devices: tuple[int, ...]
    python: str
    chrome: str
    pattern: str
    page_size: int
    max_masters: int
    timeout_ms: int
    watch_timeout_s: float
    max_cached_first_ms: float
    max_cached_visible_ms: float
    base_port: int
    headed: bool


@dataclass
class _JupyterServer:
    process: subprocess.Popen[str]
    log_stream: Any
    log_path: Path
    port: int
    token: str

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log_stream.close()


def _fail(message: str) -> None:
    pytest.fail(message, pytrace=False)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        _fail(f"{name} is required when {RUN_ENV}=1; provide an explicit path/value.")
    return value


def _integer_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        _fail(f"{name} must be an integer, got {raw!r}.")
    if value < minimum:
        _fail(f"{name} must be >= {minimum}, got {value}.")
    return value


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        _fail(f"{name} must be numeric, got {raw!r}.")
    if not value >= minimum:
        _fail(f"{name} must be >= {minimum}, got {value}.")
    return value


def _parse_devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item.strip()) for item in value.split(","))
    except ValueError:
        _fail(
            f"{DEVICES_ENV} must be comma-separated physical CUDA indices, "
            f"for example '1' or '0,1'; got {value!r}."
        )
    if not devices or any(device < 0 for device in devices):
        _fail(f"{DEVICES_ENV} must contain one or more non-negative indices.")
    if len(set(devices)) != len(devices):
        _fail(f"{DEVICES_ENV} must not repeat devices, got {devices!r}.")
    return devices


def _chrome_executable() -> str | None:
    candidates = [
        os.environ.get("CHROME_EXECUTABLE"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/opt/google/chrome/chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_config() -> _Config:
    source = Path(_required_env(SOURCE_ENV)).expanduser().resolve()
    report_root = Path(_required_env(REPORT_ENV)).expanduser().resolve()
    cache_root = Path(_required_env(CACHE_ENV)).expanduser().resolve()
    devices = _parse_devices(_required_env(DEVICES_ENV))
    run_id = os.environ.get(
        RUN_ID_ENV,
        time.strftime("show4dstem-folder-e2e-%Y%m%d-%H%M%S"),
    ).strip()
    if not run_id or run_id in {".", ".."} or "/" in run_id:
        _fail(f"{RUN_ID_ENV} must be one safe path component, got {run_id!r}.")
    if not source.is_dir():
        _fail(f"{SOURCE_ENV} is not a readable directory: {source}")
    if _inside(report_root, source) or _inside(cache_root, source):
        _fail(
            "Report/cache roots must be outside the real-data source so the E2E "
            f"cannot create artifacts in {source}."
        )
    if report_root == cache_root:
        _fail(f"{REPORT_ENV} and {CACHE_ENV} must be distinct directories.")
    for path, name in ((report_root, REPORT_ENV), (cache_root, CACHE_ENV)):
        if path.exists() and not path.is_dir():
            _fail(f"{name} points to a file, not a directory: {path}")

    artifact_dir = report_root / run_id
    cache_dir = cache_root / run_id
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        _fail(
            f"Artifact directory already contains files: {artifact_dir}. "
            f"Choose a new {RUN_ID_ENV}; the test will not overwrite evidence."
        )
    if cache_dir.exists() and any(cache_dir.iterdir()):
        _fail(
            f"Cache namespace already contains files: {cache_dir}. "
            f"Choose a new {RUN_ID_ENV}; the cold phase will not delete another run."
        )

    chrome = _chrome_executable()
    if chrome is None:
        _fail(
            "Chrome/Chromium is required for the opted-in E2E. Set "
            "CHROME_EXECUTABLE to a headed/headless-capable browser binary."
        )
    python = os.environ.get(
        "QT_SHOW4DSTEM_FOLDER_E2E_PYTHON",
        sys.executable,
    ).strip()
    if not python or not Path(python).is_file():
        _fail(
            "QT_SHOW4DSTEM_FOLDER_E2E_PYTHON must identify the CUDA environment's "
            f"Python executable, got {python!r}."
        )
    if Path(python).resolve() != Path(sys.executable).resolve():
        _fail(
            "Run pytest with the CUDA environment's Python instead of pointing "
            "the test at a second interpreter. For example: "
            f"{python} -m pytest tests/show4dstem/test_folder_live_jupyter_e2e.py. "
            f"Current interpreter: {sys.executable}."
        )

    page_size = _integer_env(
        "QT_SHOW4DSTEM_FOLDER_E2E_PAGE_SIZE",
        8,
        minimum=2,
    )
    max_masters = _integer_env(
        "QT_SHOW4DSTEM_FOLDER_E2E_MAX_MASTERS",
        0,
        minimum=0,
    )
    return _Config(
        source=source,
        report_root=report_root,
        cache_root=cache_root,
        run_id=run_id,
        artifact_dir=artifact_dir,
        cache_dir=cache_dir,
        staging_dir=artifact_dir / "staging",
        devices=devices,
        python=python,
        chrome=chrome,
        pattern=os.environ.get(
            "QT_SHOW4DSTEM_FOLDER_E2E_PATTERN",
            "*_master.h5",
        ),
        page_size=page_size,
        max_masters=max_masters,
        timeout_ms=_integer_env(
            "QT_SHOW4DSTEM_FOLDER_E2E_TIMEOUT_MS",
            900_000,
            minimum=10_000,
        ),
        watch_timeout_s=_float_env(
            "QT_SHOW4DSTEM_FOLDER_E2E_WATCH_TIMEOUT_S",
            300.0,
            minimum=10.0,
        ),
        max_cached_first_ms=_float_env(
            "QT_SHOW4DSTEM_FOLDER_E2E_MAX_CACHED_FIRST_MS",
            500.0,
            minimum=1.0,
        ),
        max_cached_visible_ms=_float_env(
            "QT_SHOW4DSTEM_FOLDER_E2E_MAX_CACHED_VISIBLE_MS",
            2_000.0,
            minimum=1.0,
        ),
        base_port=_integer_env(
            "QT_SHOW4DSTEM_FOLDER_E2E_PORT",
            0,
            minimum=0,
        ),
        headed=os.environ.get("QT_SHOW4DSTEM_FOLDER_E2E_HEADED") == "1",
    )


def _probe_runtime(config: _Config) -> dict[str, Any]:
    imports = subprocess.run(
        [
            config.python,
            "-c",
            (
                "import h5py, hdf5plugin, ipywidgets, jupyterlab, nbformat, "
                "playwright.sync_api, torch; "
                "print(torch.__version__)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if imports.returncode != 0:
        _fail(
            "The selected CUDA Python cannot import the live-E2E dependencies. "
            "Install/activate jupyterlab, nbformat, ipywidgets, playwright, torch, "
            "h5py, and hdf5plugin in that environment.\n"
            f"stdout:\n{imports.stdout}\nstderr:\n{imports.stderr}"
        )
    if shutil.which("nvidia-smi") is None:
        _fail("nvidia-smi is required to map and audit the selected CUDA devices.")
    snapshot = _nvidia_snapshot()
    visible = {int(row["index"]) for row in snapshot["gpus"]}
    missing = [device for device in config.devices if device not in visible]
    if missing:
        _fail(
            f"Selected CUDA devices {missing} are not visible to nvidia-smi; "
            f"available indices are {sorted(visible)}."
        )
    return {
        "python": config.python,
        "python_import_probe_stdout": imports.stdout.strip(),
        "chrome": config.chrome,
        "nvidia": snapshot,
    }


def _nvidia_snapshot() -> dict[str, Any]:
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,"
            "memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if gpu_query.returncode != 0:
        return {
            "error": gpu_query.stderr.strip() or gpu_query.stdout.strip(),
            "gpus": [],
            "processes": [],
        }
    gpus = []
    for line in gpu_query.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 7)]
        if len(values) != 8:
            continue
        gpus.append(
            {
                "index": values[0],
                "uuid": values[1],
                "pci_bus_id": values[2],
                "name": values[3],
                "memory_total_mib": values[4],
                "memory_used_mib": values[5],
                "memory_free_mib": values[6],
                "utilization_percent": values[7],
            }
        )
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "gpus": gpus,
        "processes": [
            line.strip()
            for line in process_query.stdout.splitlines()
            if line.strip()
        ],
        "process_query_error": (
            process_query.stderr.strip() if process_query.returncode else ""
        ),
    }


def _select_real_masters(config: _Config) -> tuple[list[Path], Path, dict[str, Any]]:
    try:
        from quantem.gpu.io import discover, inspect
    except ImportError as exc:
        _fail(
            "quantem.gpu I/O could not be imported from this checkout. Run with "
            f"PYTHONPATH={_REPO_ROOT / 'src'}: {exc}"
        )
    try:
        discovered = discover(
            str(config.source),
            pattern=config.pattern,
            recursive=True,
            verbose=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        _fail(
            f"Could not discover {config.pattern!r} below {config.source}: {exc}. "
            "Supply a source/pattern containing completed real masters."
        )
    ready: list[Path] = []
    rejected: list[dict[str, str]] = []
    for value in discovered:
        readiness = inspect(value)
        if readiness.ready:
            ready.append(Path(value).absolute())
        elif len(rejected) < 10:
            rejected.append(
                {
                    "path": str(value),
                    "reason": readiness.reason,
                    "action": readiness.action,
                }
            )
    available = ready[: config.max_masters or None]
    # Three display groups plus one held arrival are required.  Keep the initial
    # final group partial so appending the held master updates the page already
    # mounted in the browser instead of creating a new unseen group.
    total = len(available)
    minimum_initial = 2 * config.page_size + 1
    while total > minimum_initial + 1 and (total - 1) % config.page_size == 0:
        total -= 1
    if total - 1 < minimum_initial:
        _fail(
            "The real-data E2E needs at least three groups plus one watched "
            f"arrival: found {len(ready)} ready masters, but page_size="
            f"{config.page_size} requires at least {minimum_initial + 1}. "
            "Use a broader source/pattern, a smaller page size, or a larger "
            "QT_SHOW4DSTEM_FOLDER_E2E_MAX_MASTERS."
        )
    selected = available[:total]
    initial = selected[:-1]
    held = selected[-1]
    return initial, held, {
        "discovered_count": len(discovered),
        "ready_count": len(ready),
        "selected_count": len(selected),
        "initial_count": len(initial),
        "held_master": str(held),
        "rejected_examples": rejected,
    }


def _relative_master(master: Path, source: Path) -> Path:
    try:
        return master.relative_to(source)
    except ValueError:
        _fail(
            f"Discovered master {master} is outside source root {source}. "
            "Use a source without escaping directory symlinks."
        )


def _stage_link(master: Path, config: _Config) -> Path:
    relative = _relative_master(master, config.source)
    destination = config.staging_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        _fail(f"Refusing to replace existing staged path: {destination}")
    destination.symlink_to(master.resolve())
    return destination


def _stage_initial_masters(
    initial: list[Path],
    held: Path,
    config: _Config,
) -> Path:
    config.staging_dir.mkdir(parents=True, exist_ok=False)
    for master in initial:
        _stage_link(master, config)
    held_destination = config.staging_dir / _relative_master(held, config.source)
    if held_destination.exists() or held_destination.is_symlink():
        _fail(f"Held arrival was unexpectedly staged already: {held_destination}")
    return held_destination


def _source_signature_digests(masters: list[Path]) -> dict[str, str]:
    from quantem.widget.show4dstem_preview_cache import Show4DSTEMPreviewCache

    digests: dict[str, str] = {}
    for master in masters:
        signature = Show4DSTEMPreviewCache.source_signature(master)
        encoded = json.dumps(
            signature,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digests[str(master)] = hashlib.sha256(encoded).hexdigest()
    return digests


_NOTEBOOK_CELL = r'''
import html
import json
import os
import time
from pathlib import Path

import ipywidgets as widgets
import torch

import quantem.widget as quantem_widget
from quantem.widget import Show4DSTEM

folder = Path(os.environ["QT_S4D_E2E_STAGING"]).resolve()
cache_dir = Path(os.environ["QT_S4D_E2E_CACHE"]).resolve()
page_size = int(os.environ["QT_S4D_E2E_PAGE_SIZE"])
expected_gpu_count = int(os.environ["QT_S4D_E2E_GPU_COUNT"])
logical_gpus = list(range(torch.cuda.device_count()))
if len(logical_gpus) != expected_gpu_count:
    raise RuntimeError(
        "CUDA visibility mismatch: expected "
        f"{expected_gpu_count} logical devices, got {logical_gpus}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
    )

started = time.perf_counter()
w = Show4DSTEM.from_folder(
    folder,
    pattern=os.environ["QT_S4D_E2E_PATTERN"],
    recursive=True,
    gpus=logical_gpus,
    backend="cuda",
    page_budget="auto",
    det_bin=int(os.environ.get("QT_S4D_E2E_DET_BIN", "4")),
    dtype=os.environ.get("QT_S4D_E2E_DTYPE", "u16"),
    columns=int(os.environ.get("QT_S4D_E2E_COLUMNS", "4")),
    page_size=page_size,
    preload_all_if_fits=False,
    compare_dp_mode="selected",
    preview_cache=True,
    preview_cache_dir=cache_dir,
    preview_cache_max_bytes=int(
        os.environ.get("QT_S4D_E2E_CACHE_MAX_BYTES", str(4 << 30))
    ),
    rebuild_preview_cache=os.environ.get("QT_S4D_E2E_REBUILD") == "1",
    warm_cache=False,
    watch=True,
    watch_interval=float(os.environ.get("QT_S4D_E2E_WATCH_INTERVAL", "1.0")),
    debug=True,
    title="Show4DSTEM folder E2E",
    verbose=True,
)
QT_S4D_E2E_WIDGET = w
build_seconds = time.perf_counter() - started

def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return safe(value.item())
        except Exception:
            pass
    return str(value)

snapshot_sequence = 0
state_output = widgets.HTML()
control_status = widgets.HTML(
    '<span id="qt-s4d-e2e-closed" data-closed="false">open</span>'
)

def alive(name):
    thread = getattr(w, name, None)
    return bool(thread is not None and thread.is_alive())

def snapshot():
    data = w._data
    return safe({
        "sequence": snapshot_sequence,
        "model_id": w.model_id,
        "module": quantem_widget.__file__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu_count": torch.cuda.device_count(),
        "logical_gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "build_seconds": build_seconds,
        "n_frames": w.n_frames,
        "folder_watch_state": w.folder_watch_state,
        "folder_watch_detail": w.folder_watch_detail,
        "compare_page_idx": w.compare_page_idx,
        "compare_page_count": w.compare_page_count,
        "compare_page_generation": w.compare_page_generation,
        "compare_page_loading": w.compare_page_loading,
        "compare_page_expected_indices": w.compare_page_expected_indices,
        "compare_page_cached_indices": w.compare_page_cached_indices,
        "compare_page_cache_state": w.compare_page_cache_state,
        "compare_page_loaded_count": w.compare_page_loaded_count,
        "compare_page_first_panel_ms": w.compare_page_first_panel_ms,
        "compare_page_first_fresh_ms": w.compare_page_first_fresh_ms,
        "compare_page_total_ms": w.compare_page_total_ms,
        "compare_panel_indices": w.compare_panel_indices,
        "compare_status": w.compare_status,
        "compare_dp_mode": w.compare_dp_mode,
        "frame_idx": w.frame_idx,
        "gpu_memory_label": w.gpu_memory_label,
        "memory_warning": w.memory_warning,
        "raw_preload_status": getattr(w, "_raw_preload_status", None),
        "raw_residency_plan": getattr(w, "_raw_residency_plan", {}),
        "loaded_indices": data.loaded_indices(),
        "vram_resident": data.vram_resident(),
        "resident_nbytes": data.resident_nbytes,
        "target_devices": data.devices,
        "page_devices": getattr(data, "_page_devices", []),
        "preview_cache": w.preview_cache_info,
        "paint_ack_enabled": getattr(w, "_compare_page_paint_ack_enabled", False),
        "threads": {
            "folder_watch": alive("_folder_watch_thread"),
            "compare_page": alive("_compare_page_thread"),
            "cache_warm": alive("_compare_cache_warm_thread"),
            "dataset_preload": alive("_dataset_preload_thread"),
        },
    })

def refresh_state(_=None):
    global snapshot_sequence
    snapshot_sequence += 1
    payload = snapshot()
    payload["sequence"] = snapshot_sequence
    state_output.value = (
        '<pre id="qt-s4d-python-state">'
        + html.escape(json.dumps(payload, indent=2))
        + '</pre>'
    )

def close_widget(_=None):
    try:
        cache = getattr(w, "_compare_preview_cache", None)
        if cache is not None:
            cache.flush()
        w.close()
        control_status.value = (
            '<span id="qt-s4d-e2e-closed" data-closed="true">closed</span>'
        )
    except Exception as exc:
        control_status.value = (
            '<span id="qt-s4d-e2e-closed" data-closed="error">'
            + html.escape(f"{type(exc).__name__}: {exc}")
            + '</span>'
        )
        raise

refresh_button = widgets.Button(description="Refresh E2E Python state")
refresh_button.on_click(refresh_state)
close_button = widgets.Button(description="Close E2E widget")
close_button.on_click(close_widget)
refresh_state()
print(f"QT_S4D_E2E_MODEL_ID={w.model_id}")
widgets.VBox([
    w,
    widgets.HBox([refresh_button, close_button]),
    control_status,
    state_output,
])
'''


def _write_notebook(config: _Config) -> Path:
    import nbformat

    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(_NOTEBOOK_CELL)],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
    )
    path = config.artifact_dir / "show4dstem_from_folder_e2e.ipynb"
    nbformat.write(notebook, path)
    return path


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _jupyter_environment(config: _Config, *, rebuild: bool) -> dict[str, str]:
    environment = os.environ.copy()
    inherited = [
        str(Path(value).expanduser().resolve())
        for value in environment.get("PYTHONPATH", "").split(os.pathsep)
        if value
    ]
    pythonpath = [str(_REPO_ROOT / "src")]
    pythonpath.extend(value for value in inherited if value not in pythonpath)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in config.devices),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "XDG_CACHE_HOME": str(config.cache_dir / "xdg"),
            "CUPY_CACHE_DIR": str(config.cache_dir / "cupy"),
            "TMPDIR": str(config.artifact_dir / "tmp"),
            "JUPYTER_CONFIG_DIR": str(config.artifact_dir / "jupyter" / "config"),
            "JUPYTER_DATA_DIR": str(config.artifact_dir / "jupyter" / "data"),
            "JUPYTER_RUNTIME_DIR": str(config.artifact_dir / "jupyter" / "runtime"),
            "QT_S4D_E2E_STAGING": str(config.staging_dir),
            "QT_S4D_E2E_CACHE": str(config.cache_dir / "previews"),
            "QT_S4D_E2E_PAGE_SIZE": str(config.page_size),
            "QT_S4D_E2E_GPU_COUNT": str(len(config.devices)),
            "QT_S4D_E2E_PATTERN": config.pattern,
            "QT_S4D_E2E_REBUILD": "1" if rebuild else "0",
            "QT_S4D_E2E_DET_BIN": os.environ.get(
                "QT_SHOW4DSTEM_FOLDER_E2E_DET_BIN",
                "4",
            ),
            "QT_S4D_E2E_DTYPE": os.environ.get(
                "QT_SHOW4DSTEM_FOLDER_E2E_DTYPE",
                "u16",
            ),
            "QT_S4D_E2E_COLUMNS": os.environ.get(
                "QT_SHOW4DSTEM_FOLDER_E2E_COLUMNS",
                "4",
            ),
            "QT_S4D_E2E_CACHE_MAX_BYTES": os.environ.get(
                "QT_SHOW4DSTEM_FOLDER_E2E_CACHE_MAX_BYTES",
                str(4 << 30),
            ),
            "QT_S4D_E2E_WATCH_INTERVAL": os.environ.get(
                "QT_SHOW4DSTEM_FOLDER_E2E_WATCH_INTERVAL",
                "1.0",
            ),
        }
    )
    for path in (
        config.cache_dir / "xdg",
        config.cache_dir / "cupy",
        config.artifact_dir / "tmp",
        config.artifact_dir / "jupyter" / "config",
        config.artifact_dir / "jupyter" / "data",
        config.artifact_dir / "jupyter" / "runtime",
    ):
        path.mkdir(parents=True, exist_ok=True)
    return environment


def _start_jupyter(
    config: _Config,
    *,
    phase: str,
    rebuild: bool,
    port: int,
) -> _JupyterServer:
    token = f"show4dstem-folder-e2e-{secrets.token_hex(18)}"
    log_path = config.artifact_dir / f"jupyter-{phase}.log"
    log_stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            config.python,
            "-m",
            "jupyter",
            "lab",
            "--no-browser",
            f"--ServerApp.port={port}",
            "--ServerApp.ip=127.0.0.1",
            "--ServerApp.port_retries=0",
            f"--IdentityProvider.token={token}",
            f"--ServerApp.root_dir={config.artifact_dir}",
            "--ServerApp.open_browser=False",
            "--ServerApp.terminals_enabled=False",
            "--ServerApp.allow_remote_access=False",
        ],
        cwd=str(config.artifact_dir),
        env=_jupyter_environment(config, rebuild=rebuild),
        stdin=subprocess.DEVNULL,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    server = _JupyterServer(process, log_stream, log_path, port, token)
    try:
        _wait_for_jupyter(server, config.timeout_ms)
    except BaseException:
        server.stop()
        raise
    return server


def _wait_for_jupyter(server: _JupyterServer, timeout_ms: int) -> None:
    deadline = time.monotonic() + min(timeout_ms / 1000.0, 180.0)
    url = f"http://127.0.0.1:{server.port}/api/status?token={server.token}"
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            server.log_stream.flush()
            log = server.log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"JupyterLab exited early with code {server.process.returncode}.\n{log}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    server.log_stream.flush()
    log = server.log_path.read_text(encoding="utf-8", errors="replace")
    raise RuntimeError(f"JupyterLab did not become ready at {url}.\n{log}")


def _notebook_url(server: _JupyterServer, notebook: Path) -> str:
    return (
        f"http://127.0.0.1:{server.port}/lab/tree/{notebook.name}"
        f"?token={server.token}"
    )


def _browser_logs(page: Any) -> dict[str, list[str]]:
    logs: dict[str, list[str]] = {
        "console_errors": [],
        "console_warnings": [],
        "page_errors": [],
        "http_errors": [],
    }

    def console(message: Any) -> None:
        text = str(message.text)
        if message.type == "error":
            if text.endswith(".map") or "favicon.ico" in text:
                return
            logs["console_errors"].append(text)
        elif message.type == "warning":
            logs["console_warnings"].append(text)

    page.on("console", console)
    page.on("pageerror", lambda error: logs["page_errors"].append(str(error)))
    page.on(
        "response",
        lambda response: logs["http_errors"].append(
            f"{response.status} {response.url}"
        )
        if response.status >= 400
        and not response.url.endswith("/favicon.ico")
        and not response.url.endswith(".map")
        else None,
    )
    return logs


def _open_widget(page: Any, url: str, config: _Config) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=config.timeout_ms)
    page.wait_for_selector(".jp-Notebook", timeout=config.timeout_ms)
    page.locator(".jp-Cell").first.click()
    page.keyboard.press("Shift+Enter")
    try:
        page.wait_for_selector(".show4dstem-root", timeout=config.timeout_ms)
        page.wait_for_selector(
            '[data-folder-watch-state="watching"]',
            timeout=config.timeout_ms,
        )
        page.wait_for_function(
            """() => {
              const panels = [...document.querySelectorAll(
                '[aria-label^="Show4DSTEM multiple panel"]'
              )];
              return panels.length > 0 && panels.every(panel =>
                panel.getAttribute('data-show4dstem-panel-cache') !== 'empty'
              );
            }""",
            timeout=config.timeout_ms,
        )
    except BaseException as exc:
        errors = _notebook_errors(page)
        raise AssertionError(
            f"Live Show4DSTEM did not mount successfully: {exc}; "
            f"notebook errors={errors!r}"
        ) from exc
    page.locator(".show4dstem-root").scroll_into_view_if_needed()


def _notebook_errors(page: Any) -> list[str]:
    tracebacks = page.locator(".jp-RenderedTraceback").all_inner_texts()
    stderr_errors = [
        text
        for text in _notebook_stderr(page)
        if any(
            marker in text
            for marker in (
                "Traceback (most recent call last)",
                "CUDA error",
                "cudaError",
                "MemoryError",
                "Exception:",
            )
        )
    ]
    return [*tracebacks, *stderr_errors]


def _notebook_stderr(page: Any) -> list[str]:
    return page.locator(
        '[data-mime-type="application/vnd.jupyter.stderr"]'
    ).all_inner_texts()


def _page_count(page: Any) -> int:
    count = int(
        page.evaluate(
            """() => {
              const prefix = 'Show Show4DSTEM multiple group ';
              const values = [...document.querySelectorAll('[aria-label]')]
                .map(node => node.getAttribute('aria-label') || '')
                .filter(value => value.startsWith(prefix))
                .map(value => Number(value.slice(prefix.length)))
                .filter(Number.isFinite);
              return values.length ? Math.max(...values) : 0;
            }"""
        )
    )
    assert count >= 3, f"Expected at least three page groups, found {count}."
    return count


def _wait_page_complete(page: Any, page_index: int, timeout_ms: int) -> None:
    page.wait_for_function(
        """target => {
          const perf = window.__quantemShow4DSTEMPerf?.comparePage;
          return Boolean(
            perf && perf.page === target && perf.completeAtMs !== null
            && perf.loadedCount > 0
          );
        }""",
        page_index,
        timeout=timeout_ms,
    )
    page.wait_for_function(
        """() => [...document.querySelectorAll(
          '[aria-label^="Show4DSTEM multiple panel"]'
        )].every(panel => panel.getAttribute('aria-busy') !== 'true')""",
        timeout=timeout_ms,
    )


def _click_group(page: Any, group: int, timeout_ms: int) -> None:
    button = page.get_by_role(
        "button",
        name=f"Show Show4DSTEM multiple group {group}",
        exact=True,
    )
    button.wait_for(state="visible", timeout=timeout_ms)
    button.click()
    _wait_page_complete(page, group - 1, timeout_ms)


def _rapid_groups_1_2_3(page: Any, timeout_ms: int) -> dict[str, Any]:
    if page.get_by_role(
        "button",
        name="Show Show4DSTEM multiple group 1",
        exact=True,
    ).get_attribute("aria-pressed") != "true":
        _click_group(page, 1, timeout_ms)
    started = time.perf_counter()
    page.get_by_role(
        "button",
        name="Show Show4DSTEM multiple group 2",
        exact=True,
    ).click()
    third = page.get_by_role(
        "button",
        name="Show Show4DSTEM multiple group 3",
        exact=True,
    )
    third.wait_for(state="visible", timeout=min(timeout_ms, 5_000))
    third.click()
    dispatch_ms = (time.perf_counter() - started) * 1000.0
    _wait_page_complete(page, 2, timeout_ms)
    assert third.get_attribute("aria-pressed") == "true"
    return {
        "dispatch_ms": dispatch_ms,
        "final": _browser_snapshot(page),
    }


_BROWSER_SNAPSHOT = r'''
async () => {
  const root = document.querySelector('.show4dstem-root');
  const digest = async canvas => {
    if (!canvas) return null;
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    if (!blob) return null;
    const hash = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
    return [...new Uint8Array(hash)]
      .map(value => value.toString(16).padStart(2, '0')).join('');
  };
  const panels = [...root.querySelectorAll(
    '[aria-label^="Show4DSTEM multiple panel"]'
  )];
  const watch = root.querySelector('[data-folder-watch-state]');
  const watchDot = watch?.querySelector('[data-folder-watch-dot="true"]');
  const cacheBadge = root.querySelector(
    '[data-testid="show4dstem-compare-cache-status"]'
  );
  return {
    perf: JSON.parse(JSON.stringify(
      window.__quantemShow4DSTEMPerf?.comparePage ?? null
    )),
    watch: watch ? {
      state: watch.getAttribute('data-folder-watch-state'),
      label: watch.getAttribute('aria-label'),
      dot: watchDot ? getComputedStyle(watchDot).backgroundColor : null,
    } : null,
    cache_badge: cacheBadge ? {
      text: cacheBadge.textContent,
      tone: cacheBadge.getAttribute('data-show4dstem-cache-tone'),
    } : null,
    dp_hash: await digest(root.querySelector('canvas')),
    panels: await Promise.all(panels.map(async panel => ({
      label: (panel.getAttribute('aria-label') || '').split(',')[0],
      busy: panel.getAttribute('aria-busy'),
      cache: panel.getAttribute('data-show4dstem-panel-cache'),
      hash: await digest(panel.querySelector('canvas')),
    }))),
  };
}
'''


def _browser_snapshot(page: Any) -> dict[str, Any]:
    return dict(page.evaluate(_BROWSER_SNAPSHOT))


def _perf(page: Any) -> dict[str, Any]:
    value = page.evaluate(
        "() => JSON.parse(JSON.stringify("
        "window.__quantemShow4DSTEMPerf?.comparePage ?? null))"
    )
    assert isinstance(value, dict), "Show4DSTEM compare-page perf object is absent."
    return value


def _refresh_python_state(page: Any, timeout_ms: int) -> dict[str, Any]:
    before = page.evaluate(
        """() => {
          const node = document.querySelector('#qt-s4d-python-state');
          if (!node) return -1;
          try { return JSON.parse(node.textContent).sequence; }
          catch { return -1; }
        }"""
    )
    page.get_by_role(
        "button",
        name="Refresh E2E Python state",
        exact=True,
    ).click()
    page.wait_for_function(
        """previous => {
          const node = document.querySelector('#qt-s4d-python-state');
          if (!node) return false;
          try { return JSON.parse(node.textContent).sequence > previous; }
          catch { return false; }
        }""",
        before,
        timeout=timeout_ms,
    )
    return dict(
        page.evaluate(
            "() => JSON.parse(document.querySelector("
            "'#qt-s4d-python-state').textContent)"
        )
    )


def _close_widget(page: Any, timeout_ms: int) -> None:
    button = page.get_by_role("button", name="Close E2E widget", exact=True)
    if button.count() == 0:
        return
    button.click(timeout=timeout_ms)
    page.wait_for_selector(
        '#qt-s4d-e2e-closed[data-closed="true"]',
        timeout=timeout_ms,
    )


def _best_effort_close_widget(
    page: Any,
    config: _Config,
    report: dict[str, Any],
    phase: str,
) -> None:
    try:
        if page.locator(".show4dstem-root").count() > 0:
            _close_widget(page, config.timeout_ms)
    except Exception as exc:
        report.setdefault("cleanup_errors", []).append(
            f"{phase} widget close failed: {type(exc).__name__}: {exc}"
        )


def _screenshot(page: Any, path: Path) -> str:
    page.locator(".show4dstem-root").scroll_into_view_if_needed()
    page.screenshot(path=str(path), full_page=False)
    return str(path)


def _assert_python_gpu_mapping(state: dict[str, Any], config: _Config) -> None:
    assert state["logical_gpu_count"] == len(config.devices)
    assert state["cuda_visible_devices"] == ",".join(
        str(value) for value in config.devices
    )
    planned = set(state["target_devices"])
    expected = {f"cuda:{index}" for index in range(len(config.devices))}
    assert expected <= planned, (
        "Every selected physical GPU must appear in the logical capacity plan: "
        f"expected {expected}, got {planned}."
    )


def _cold_drive(page: Any, config: _Config) -> dict[str, Any]:
    page_count = _page_count(page)
    initial = _browser_snapshot(page)
    assert initial["watch"]["state"] == "watching"
    assert initial["watch"]["dot"] == "rgb(46, 125, 50)"
    rapid = _rapid_groups_1_2_3(page, config.timeout_ms)
    _click_group(page, 1, config.timeout_ms)
    _click_group(page, 2, config.timeout_ms)
    cold_second = _browser_snapshot(page)
    assert cold_second["perf"]["firstFreshPanelPaintAtMs"] is not None
    assert cold_second["perf"]["freshVisiblePaintAtMs"] is not None
    assert all(panel["cache"] == "fresh" for panel in cold_second["panels"])
    _click_group(page, page_count, config.timeout_ms)
    last = _browser_snapshot(page)
    _click_group(page, 1, config.timeout_ms)
    final = _browser_snapshot(page)
    python_state = _refresh_python_state(page, config.timeout_ms)
    _assert_python_gpu_mapping(python_state, config)
    screenshot = _screenshot(
        page,
        config.artifact_dir / "cold-page-sequence.png",
    )
    return {
        "page_count": page_count,
        "sequence": [1, 2, page_count, 1],
        "initial": initial,
        "rapid_1_2_3": rapid,
        "cold_page_2": cold_second,
        "last_page": last,
        "final_page_1": final,
        "python_state": python_state,
        "screenshot": screenshot,
    }


def _wait_cached_then_fresh(page: Any, config: _Config) -> dict[str, Any]:
    page.get_by_role(
        "button",
        name="Show Show4DSTEM multiple group 2",
        exact=True,
    ).click()
    page.wait_for_function(
        """() => {
          const perf = window.__quantemShow4DSTEMPerf?.comparePage;
          return Boolean(
            perf && perf.page === 1 && perf.cachedVisiblePaintAtMs !== null
          );
        }""",
        timeout=config.timeout_ms,
    )
    cached = _browser_snapshot(page)
    cached_screenshot = _screenshot(
        page,
        config.artifact_dir / "warm-cached-preview.png",
    )
    page.wait_for_function(
        """() => {
          const perf = window.__quantemShow4DSTEMPerf?.comparePage;
          return Boolean(
            perf && perf.page === 1 && perf.freshVisiblePaintAtMs !== null
            && perf.completeAtMs !== null
          );
        }""",
        timeout=config.timeout_ms,
    )
    fresh = _browser_snapshot(page)
    fresh_screenshot = _screenshot(
        page,
        config.artifact_dir / "warm-fresh-refresh.png",
    )
    perf = fresh["perf"]
    assert perf["clickToFirstCachedPanelPaintMs"] is not None
    assert perf["clickToCachedVisiblePaintMs"] is not None
    assert perf["clickToFirstFreshPanelPaintMs"] is not None
    assert perf["clickToFreshVisiblePaintMs"] is not None
    assert perf["clickToFirstCachedPanelPaintMs"] <= config.max_cached_first_ms
    assert perf["clickToCachedVisiblePaintMs"] <= config.max_cached_visible_ms
    assert perf["cachedVisiblePaintAtMs"] <= perf["freshVisiblePaintAtMs"]
    cached_hashes = {
        panel["label"]: panel["hash"]
        for panel in cached["panels"]
    }
    fresh_hashes = {
        panel["label"]: panel["hash"]
        for panel in fresh["panels"]
    }
    assert cached_hashes
    assert cached_hashes == fresh_hashes, (
        "Persistent cached panels must match the authoritative fresh redraw."
    )
    assert any(panel["cache"] == "cached" for panel in cached["panels"])
    assert all(panel["cache"] == "fresh" for panel in fresh["panels"])
    return {
        "cached": cached,
        "fresh": fresh,
        "cached_screenshot": cached_screenshot,
        "fresh_screenshot": fresh_screenshot,
    }


def _mui_select(page: Any, aria_label: str, option: str) -> str:
    selected = page.evaluate(
        """({ariaLabel, option}) => {
          const control = [...document.querySelectorAll('[aria-label]')]
            .find(node => node.getAttribute('aria-label') === ariaLabel);
          if (!control) return 'missing-control';
          const root = control.closest('.MuiInputBase-root') || control.parentElement;
          const target = root?.querySelector('[role="combobox"]') || root || control;
          target.dispatchEvent(new MouseEvent('mousedown', {
            bubbles: true, cancelable: true, view: window
          }));
          target.click();
          return 'opened';
        }""",
        {"ariaLabel": aria_label, "option": option},
    )
    assert selected == "opened", f"Could not open MUI select {aria_label!r}."
    page.get_by_role("option", name=option, exact=True).click()
    page.wait_for_timeout(250)
    value = page.locator(f'[aria-label="{aria_label}"]').get_attribute("value")
    return str(value or "")


def _main_dp_data_url(page: Any) -> str:
    value = page.evaluate(
        "() => document.querySelector('.show4dstem-root canvas')?.toDataURL() || ''"
    )
    assert value, "Main diffraction canvas was not available for hashing."
    return str(value)


def _data_url_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _exercise_dp_modes(page: Any, timeout_ms: int) -> dict[str, Any]:
    panels = page.locator('[aria-label^="Show4DSTEM multiple panel"]')
    assert panels.count() >= 2
    first = panels.nth(0)
    second = panels.nth(1)
    first.click(force=True)
    page.wait_for_timeout(250)
    first_selected_url = _main_dp_data_url(page)
    second.click(force=True)
    page.wait_for_function(
        """before => {
          const canvas = document.querySelector('.show4dstem-root canvas');
          return Boolean(canvas && canvas.toDataURL() !== before);
        }""",
        first_selected_url,
        timeout=timeout_ms,
    )
    selected_url = _main_dp_data_url(page)
    average_value = _mui_select(
        page,
        "Show4DSTEM multiple DP source",
        "Average",
    )
    page.wait_for_function(
        """before => {
          const canvas = document.querySelector('.show4dstem-root canvas');
          return Boolean(canvas && canvas.toDataURL() !== before);
        }""",
        selected_url,
        timeout=timeout_ms,
    )
    average_url = _main_dp_data_url(page)
    selected_value = _mui_select(
        page,
        "Show4DSTEM multiple DP source",
        "Selected",
    )
    first_hash = _data_url_hash(first_selected_url)
    second_hash = _data_url_hash(selected_url)
    average_hash = _data_url_hash(average_url)
    assert first_hash != second_hash, (
        "Selecting a different real dataset must update the diffraction canvas."
    )
    assert second_hash != average_hash, (
        "Selected and visible-page average diffraction sources must render "
        "different real-data canvases."
    )
    assert average_value == "average"
    assert selected_value == "selected"
    return {
        "first_selected_hash": first_hash,
        "second_selected_hash": second_hash,
        "average_hash": average_hash,
        "average_value": average_value,
        "selected_value": selected_value,
    }


def _exercise_star_hide_restore(page: Any, timeout_ms: int) -> dict[str, Any]:
    panels = page.locator('[aria-label^="Show4DSTEM multiple panel"]')
    before_count = panels.count()
    assert before_count >= 2
    star = page.locator(".show4dstem-compare-star-button").first
    starred_frame = int(star.get_attribute("data-frame"))
    star.click(force=True)
    page.wait_for_function(
        """frame => document.querySelector(
          `.show4dstem-compare-star-button[data-frame="${frame}"]`
        )?.getAttribute('aria-label')?.startsWith('Unstar')""",
        starred_frame,
        timeout=timeout_ms,
    )

    hide = page.locator(".show4dstem-compare-hide-button").nth(1)
    hidden_frame = int(hide.get_attribute("data-frame"))
    hide.click(force=True)
    page.wait_for_function(
        """expected => document.querySelectorAll(
          '[aria-label^="Show4DSTEM multiple panel"]'
        ).length === expected""",
        before_count - 1,
        timeout=timeout_ms,
    )
    page.locator(".show4dstem-compare-hidden-menu").click(force=True)
    page.locator(
        f'[aria-label="Show Show4DSTEM multiple panel {hidden_frame + 1}"]'
    ).click()
    page.wait_for_function(
        """expected => document.querySelectorAll(
          '[aria-label^="Show4DSTEM multiple panel"]'
        ).length === expected""",
        before_count,
        timeout=timeout_ms,
    )
    page.locator(".show4dstem-compare-reset").click()
    page.wait_for_timeout(250)
    return {
        "before_count": before_count,
        "starred_frame": starred_frame,
        "hidden_frame": hidden_frame,
        "restored_count": panels.count(),
    }


def _watch_sample(page: Any) -> dict[str, Any]:
    return dict(
        page.evaluate(
            """() => {
              const badge = document.querySelector('[data-folder-watch-state]');
              const perf = window.__quantemShow4DSTEMPerf?.comparePage;
              return {
                at_ms: performance.now(),
                state: badge?.getAttribute('data-folder-watch-state') || null,
                detail: badge?.getAttribute('aria-label') || null,
                generation: perf?.generation ?? null,
                page: perf?.page ?? null,
                fresh_visible_paint_at_ms: perf?.freshVisiblePaintAtMs ?? null,
                complete_at_ms: perf?.completeAtMs ?? null,
                panel_count: document.querySelectorAll(
                  '[aria-label^="Show4DSTEM multiple panel"]'
                ).length,
              };
            }"""
        )
    )


def _exercise_watched_arrival(
    page: Any,
    held: Path,
    held_destination: Path,
    config: _Config,
) -> dict[str, Any]:
    page_count = _page_count(page)
    _click_group(page, page_count, config.timeout_ms)
    before = _watch_sample(page)
    assert before["state"] == "watching"
    before_generation = before["generation"]
    held_destination.parent.mkdir(parents=True, exist_ok=True)
    held_destination.symlink_to(held.resolve())

    timeline: list[dict[str, Any]] = []
    seen_updating = False
    updating_screenshot: str | None = None
    deadline = time.monotonic() + config.watch_timeout_s
    last_key: tuple[Any, ...] | None = None
    while time.monotonic() < deadline:
        sample = _watch_sample(page)
        key = (
            sample["state"],
            sample["generation"],
            sample["fresh_visible_paint_at_ms"],
            sample["complete_at_ms"],
            sample["panel_count"],
        )
        if key != last_key:
            timeline.append(sample)
            last_key = key
        if sample["state"] == "error":
            raise AssertionError(f"Folder watcher entered error: {sample}")
        if sample["state"] == "updating":
            seen_updating = True
            if updating_screenshot is None:
                updating_screenshot = _screenshot(
                    page,
                    config.artifact_dir / "watch-updating.png",
                )
        generation_changed = str(sample["generation"]) != str(before_generation)
        if (
            seen_updating
            and sample["state"] == "watching"
            and generation_changed
            and sample["fresh_visible_paint_at_ms"] is not None
            and sample["complete_at_ms"] is not None
        ):
            break
        page.wait_for_timeout(100)
    else:
        raise AssertionError(
            "Watched arrival did not complete Updating -> fresh browser paint -> "
            f"Watching within {config.watch_timeout_s}s; timeline={timeline!r}"
        )
    after = _watch_sample(page)
    assert seen_updating
    assert after["panel_count"] == before["panel_count"] + 1
    assert after["state"] == "watching"
    assert after["fresh_visible_paint_at_ms"] is not None
    python_state = _refresh_python_state(page, config.timeout_ms)
    assert python_state["paint_ack_enabled"] is True
    complete_screenshot = _screenshot(
        page,
        config.artifact_dir / "watch-fresh-ack-complete.png",
    )
    return {
        "held_master": str(held),
        "staged_arrival": str(held_destination),
        "before": before,
        "timeline": timeline,
        "after": after,
        "python_state": python_state,
        "updating_screenshot": updating_screenshot,
        "complete_screenshot": complete_screenshot,
    }


def _assert_browser_clean(page: Any, logs: dict[str, list[str]]) -> None:
    notebook_errors = _notebook_errors(page)
    assert not notebook_errors, f"Jupyter notebook emitted errors: {notebook_errors}"
    assert not logs["page_errors"], logs["page_errors"]
    assert not logs["console_errors"], logs["console_errors"]
    assert not logs["http_errors"], logs["http_errors"]


def _write_reports(report: dict[str, Any], artifact_dir: Path) -> None:
    report_path = artifact_dir / "show4dstem-folder-e2e-report.json"
    perf_path = artifact_dir / "show4dstem-folder-e2e-perf.json"
    encoded = json.dumps(report, indent=2, default=str)
    report_path.write_text(encoded, encoding="utf-8")
    perf = {
        phase: payload
        for phase, payload in report.get("phases", {}).items()
        if isinstance(payload, dict)
    }
    perf_path.write_text(json.dumps(perf, indent=2, default=str), encoding="utf-8")
    status = "Passed" if report.get("passed") else "Failed or incomplete"
    status_color = "#16794b" if report.get("passed") else "#a33a2b"
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Show4DSTEM folder E2E report</title>
  <style>
    body {{ font: 16px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 72rem; padding: 0 1rem; }}
    .status {{ color: {status_color}; font-weight: 700; }}
    pre {{ background: #f4f6f8; border-radius: .5rem; overflow: auto; padding: 1rem; }}
  </style>
</head>
<body>
  <h1>Show4DSTEM folder E2E report</h1>
  <p class="status">{status}</p>
  <p><a href="show4dstem-folder-e2e-report.json">Full JSON report</a> ·
     <a href="show4dstem-folder-e2e-perf.json">Performance JSON</a></p>
  <pre>{html.escape(encoded)}</pre>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(index, encoding="utf-8")


def test_show4dstem_from_folder_live_paging_cache_watch_e2e() -> None:
    """Drive the real one-/multi-GPU folder workflow in live JupyterLab."""
    config = _load_config()
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "passed": False,
        "run_id": config.run_id,
        "source": str(config.source),
        "report_dir": str(config.artifact_dir),
        "cache_dir": str(config.cache_dir),
        "staging_dir": str(config.staging_dir),
        "pattern": config.pattern,
        "page_size": config.page_size,
        "physical_cuda_devices": list(config.devices),
        "phases": {},
    }
    cold_server: _JupyterServer | None = None
    warm_server: _JupyterServer | None = None
    browser = None
    source_before: dict[str, str] = {}
    selected: list[Path] = []
    try:
        report["runtime"] = _probe_runtime(config)
        initial, held, selection = _select_real_masters(config)
        selected = [*initial, held]
        report["selection"] = selection
        source_before = _source_signature_digests(selected)
        report["source_signature_before"] = source_before
        held_destination = _stage_initial_masters(initial, held, config)
        notebook = _write_notebook(config)
        report["notebook"] = str(notebook)
        report["nvidia_before"] = _nvidia_snapshot()

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            _fail(
                "playwright.sync_api is unavailable despite the runtime probe: "
                f"{exc}"
            )

        cold_port = config.base_port or _free_port()
        warm_port = config.base_port + 1 if config.base_port else _free_port()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=config.chrome,
                headless=not config.headed,
                args=[
                    "--enable-unsafe-webgpu",
                    "--enable-features=Vulkan,WebGPU",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
            )

            cold_server = _start_jupyter(
                config,
                phase="cold",
                rebuild=True,
                port=cold_port,
            )
            cold_context = browser.new_context(viewport={"width": 1440, "height": 1100})
            cold_page = cold_context.new_page()
            cold_logs = _browser_logs(cold_page)
            cold_closed = False
            try:
                _open_widget(
                    cold_page,
                    _notebook_url(cold_server, notebook),
                    config,
                )
                report["phases"]["cold"] = _cold_drive(cold_page, config)
                report["phases"]["cold"]["browser_logs"] = cold_logs
                report["phases"]["cold"]["notebook_stderr"] = (
                    _notebook_stderr(cold_page)
                )
                report["phases"]["cold"]["jupyter_url"] = _notebook_url(
                    cold_server,
                    notebook,
                )
                report["nvidia_after_cold"] = _nvidia_snapshot()
                _assert_browser_clean(cold_page, cold_logs)
                _close_widget(cold_page, config.timeout_ms)
                cold_closed = True
            finally:
                if not cold_closed:
                    _best_effort_close_widget(cold_page, config, report, "cold")
                cold_context.close()
                cold_server.stop()
                cold_server = None

            # A new Jupyter server and ipykernel establish a genuine fresh
            # process.  Only the explicit persistent preview cache is shared.
            warm_server = _start_jupyter(
                config,
                phase="warm",
                rebuild=False,
                port=warm_port,
            )
            warm_context = browser.new_context(viewport={"width": 1440, "height": 1100})
            warm_page = warm_context.new_page()
            warm_logs = _browser_logs(warm_page)
            warm_closed = False
            try:
                _open_widget(
                    warm_page,
                    _notebook_url(warm_server, notebook),
                    config,
                )
                warm_cache = _wait_cached_then_fresh(warm_page, config)
                dp_modes = _exercise_dp_modes(warm_page, config.timeout_ms)
                curation = _exercise_star_hide_restore(
                    warm_page,
                    config.timeout_ms,
                )
                watched = _exercise_watched_arrival(
                    warm_page,
                    held,
                    held_destination,
                    config,
                )
                warm_python_state = _refresh_python_state(
                    warm_page,
                    config.timeout_ms,
                )
                _assert_python_gpu_mapping(warm_python_state, config)
                report["phases"]["warm"] = {
                    "cache": warm_cache,
                    "dp_modes": dp_modes,
                    "curation": curation,
                    "watched_arrival": watched,
                    "final": _browser_snapshot(warm_page),
                    "python_state": warm_python_state,
                    "browser_logs": warm_logs,
                    "notebook_stderr": _notebook_stderr(warm_page),
                    "jupyter_url": _notebook_url(warm_server, notebook),
                }
                report["nvidia_after_warm"] = _nvidia_snapshot()
                _assert_browser_clean(warm_page, warm_logs)
                _close_widget(warm_page, config.timeout_ms)
                warm_closed = True
            finally:
                if not warm_closed:
                    _best_effort_close_widget(warm_page, config, report, "warm")
                warm_context.close()
                warm_server.stop()
                warm_server = None
            browser.close()
            browser = None

        source_after = _source_signature_digests(selected)
        report["source_signature_after"] = source_after
        assert source_after == source_before, (
            "Real-data source signatures changed during an E2E that must be "
            "strictly read-only."
        )
        report["nvidia_after_cleanup"] = _nvidia_snapshot()
        report["passed"] = True
    except BaseException as exc:
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        for server in (cold_server, warm_server):
            if server is not None:
                try:
                    server.stop()
                except Exception as exc:
                    report.setdefault("cleanup_errors", []).append(str(exc))
        if selected and source_before:
            try:
                final_signatures = _source_signature_digests(selected)
                report.setdefault("source_signature_after", final_signatures)
                if final_signatures != source_before:
                    report.setdefault("cleanup_errors", []).append(
                        "real-data source signatures changed"
                    )
            except Exception as exc:
                report.setdefault("cleanup_errors", []).append(
                    f"source signature recheck failed: {exc}"
                )
        report.setdefault("nvidia_after_cleanup", _nvidia_snapshot())
        _write_reports(report, config.artifact_dir)
