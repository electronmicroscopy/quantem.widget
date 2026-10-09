"""Platform rules: one device resolver, loud GPU notices, adapters that tolerate missing packages.

The resolver and notice are checked with availability monkeypatched, so every
platform's branch runs on any machine. Parity tests against quantem.gpu (the
colormap LUTs, detector geometry and dense reductions) run when it is installed
and skip otherwise.
"""

import importlib.util
import sys

import numpy as np
import pytest
import torch

from quantem.widget import device
from quantem.widget.adapters import core as core_adapter
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.colormap import colormap_lut, colormap_names
from quantem.widget.counts import count_bound, detector_bin_mean, exact_max, exact_sum, int32_block_max, int32_block_sum
from quantem.widget.show4dstem.dense import DenseSession
from quantem.widget.show4dstem.detector import detector_mask, fit_probe


@pytest.fixture
def fresh(monkeypatch):
    """A process that has printed nothing yet, on a machine without a GPU unless a test says otherwise."""
    monkeypatch.setattr(device, "_printed", set())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(device, "mps_available", lambda: False)
    monkeypatch.setattr(device, "gpu_notice_text", lambda: None)
    return monkeypatch


def _load_without(module_file: str, hidden: str, monkeypatch):
    """A private copy of an adapter module imported while ``hidden`` cannot be imported."""
    monkeypatch.setitem(sys.modules, hidden, None)
    spec = importlib.util.spec_from_file_location(f"adapter_copy_{hidden.replace('.', '_')}", module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Device resolver
# ---------------------------------------------------------------------------


def test_auto_picks_cuda_then_mps_then_cpu_and_says_so_once(fresh, capsys):
    assert device.resolve_device("auto") == "cpu"
    assert device.resolve_device(None) == "cpu"
    assert capsys.readouterr().out == 'quantem.widget: device="auto" selected cpu.\n'
    fresh.setattr(device, "mps_available", lambda: True)
    assert device.resolve_device() == "mps"
    fresh.setattr(torch.cuda, "is_available", lambda: True)
    assert device.resolve_device() == "cuda:0"
    assert capsys.readouterr().out.splitlines() == ['quantem.widget: device="auto" selected mps.',
                                                    'quantem.widget: device="auto" selected cuda:0.']


def test_a_tensor_stays_where_it_is_unless_a_device_is_named(fresh, capsys):
    data_t = torch.zeros(2)
    assert device.resolve_device("auto", data_t) == "cpu"
    assert capsys.readouterr().out == ""
    fresh.setattr(torch.cuda, "is_available", lambda: True)
    fresh.setattr(torch.cuda, "device_count", lambda: 2)
    for _ in range(2):
        assert device.resolve_device("cuda:1", data_t) == "cuda:1"
    assert capsys.readouterr().out == 'quantem.widget: device="cuda:1" moves the data from cpu to cuda:1.\n'
    assert device.resolve_device("cpu", data_t) == "cpu"
    assert capsys.readouterr().out == ""


def test_explicit_devices_are_validated(fresh):
    with pytest.raises(RuntimeError, match="MPS device is unavailable"):
        device.resolve_device("mps")
    with pytest.raises(RuntimeError, match="CUDA device is unavailable"):
        device.resolve_device("cuda")
    fresh.setattr(torch.cuda, "is_available", lambda: True)
    fresh.setattr(torch.cuda, "device_count", lambda: 1)
    assert device.resolve_device("CUDA") == "cuda:0"
    with pytest.raises(ValueError, match="1 CUDA device"):
        device.resolve_device("cuda:3")
    with pytest.raises(ValueError, match="Unknown device"):
        device.resolve_device("tpu")


# ---------------------------------------------------------------------------
# GPU notices
# ---------------------------------------------------------------------------


def _machine(monkeypatch, *, system: str, machine: str, nvidia: bool = False, installed=(), cuda_torch: bool = True,
             mps: bool = False):
    monkeypatch.setattr(device, "host_platform", lambda: (system, machine))
    monkeypatch.setattr(device, "nvidia_present", lambda: nvidia)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: nvidia and cuda_torch)
    monkeypatch.setattr(device, "module_installed", lambda name: name in installed)
    monkeypatch.setattr(device.torch.version, "cuda", "12.8" if cuda_torch else None)
    monkeypatch.setattr(device, "mps_available", lambda: mps)


def test_notice_names_the_missing_gpu_runtime_and_the_pip_command(monkeypatch):
    _machine(monkeypatch, system="linux", machine="x86_64", nvidia=True)
    assert device.gpu_notice_text() == (
        "quantem.widget: NVIDIA GPU found but quantem.gpu[cuda] is not installed (quantem.gpu, cupy missing); "
        'Show4DSTEM file loading runs on CPU. For the GPU path: pip install "quantem.widget[cuda]"'
    )
    _machine(monkeypatch, system="linux", machine="x86_64", nvidia=True, installed=("quantem.gpu", "cupy"))
    monkeypatch.setattr(gpu_adapter, "gpu_runtime_notice", None)
    assert device.gpu_notice_text() is None
    monkeypatch.setattr(gpu_adapter, "gpu_runtime_notice", lambda: "quantem.gpu: an NVIDIA GPU is present but CuPy is not installed")
    assert device.gpu_notice_text().startswith("quantem.gpu:")  # quantem.gpu names its own runtime when installed
    _machine(monkeypatch, system="linux", machine="x86_64", nvidia=False)
    assert device.gpu_notice_text() is None
    _machine(monkeypatch, system="linux", machine="x86_64", nvidia=True, cuda_torch=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert "this torch build has no CUDA" in device.gpu_notice_text()
    _machine(monkeypatch, system="darwin", machine="arm64", mps=True)
    assert device.gpu_notice_text().startswith(
        "quantem.widget: Apple GPU found but quantem.gpu[mps] is not installed (quantem.gpu, Metal, mlx missing)")
    assert device.gpu_notice_text().endswith('pip install "quantem.widget[mps]"')
    _machine(monkeypatch, system="darwin", machine="x86_64", mps=True)
    assert device.gpu_notice_text() == (
        "quantem.widget: on this Intel Mac the widgets compute with torch MPS on its GPU, and Show4DSTEM reads files "
        "on the CPU and holds every count in memory. quantem.gpu, which keeps files encoded on a GPU, does not run "
        "on Intel Macs: it needs torch 2.3 or newer, and PyTorch publishes torch 2.2 at most for them."
    )
    _machine(monkeypatch, system="darwin", machine="x86_64", mps=False)
    assert "compute with the CPU" in device.gpu_notice_text()


def test_the_pip_route_is_offered_only_for_a_gpu_quantem_gpu_runs_on(monkeypatch):
    for system, machine in (("linux", "x86_64"), ("win32", "AMD64")):
        _machine(monkeypatch, system=system, machine=machine, nvidia=True)
        assert device.gpu_path_hint() == 'pip install "quantem.widget[cuda]"'
    _machine(monkeypatch, system="linux", machine="x86_64")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # WSL2: CUDA works, no nvidia-smi on PATH
    assert device.gpu_path_hint() == 'pip install "quantem.widget[cuda]"'
    _machine(monkeypatch, system="darwin", machine="arm64")
    assert device.gpu_path_hint() == 'pip install "quantem.widget[mps]"'
    # Windows on ARM and a Linux box without an NVIDIA driver: [cuda] would install nothing useful
    for system, machine in (("win32", "ARM64"), ("linux", "x86_64")):
        _machine(monkeypatch, system=system, machine=machine, nvidia=False)
        assert device.gpu_path_hint() is None
        assert device.no_gpu_path() == device.NO_GPU_PATH
    _machine(monkeypatch, system="darwin", machine="x86_64")
    assert device.gpu_path_hint() is None
    assert device.no_gpu_path() == device.NO_GPU_PATH_INTEL_MAC


def test_dense_reads_without_a_gpu_route_name_what_works_and_no_pip_command(monkeypatch, tmp_path, capsys):
    from quantem.widget.show4dstem import reader

    np.save(tmp_path / "scan.npy", np.ones((4, 4, 8, 8), np.uint16))
    monkeypatch.setattr(gpu_adapter, "available", lambda: False)
    for system, machine, reason in (("darwin", "x86_64", "does not run on Intel Macs"),
                                    ("win32", "ARM64", "runs with CUDA on an NVIDIA GPU or with Metal")):
        _machine(monkeypatch, system=system, machine=machine)
        with pytest.raises(MemoryError) as refusal:
            reader.read_4dstem(tmp_path / "scan.npy", "cpu", max_bytes=100)
        message = str(refusal.value)
        assert "Nothing was binned or cropped. The file is read on the CPU and every count is held in memory" in message
        assert reason in message and "pip install" not in message
        assert message.endswith("open a smaller dataset or part of the scan as an array, or use a machine with more memory.")
        reader.read_4dstem(tmp_path / "scan.npy", "cpu")
        assert "pip install" not in capsys.readouterr().out
    _machine(monkeypatch, system="linux", machine="x86_64", nvidia=True)
    with pytest.raises(MemoryError, match=r'keep it encoded on a GPU: pip install "quantem.widget\[cuda\]"$'):
        reader.read_4dstem(tmp_path / "scan.npy", "cpu", max_bytes=100)
    reader.read_4dstem(tmp_path / "scan.npy", "cpu")
    assert capsys.readouterr().out.endswith('loads it faster: pip install "quantem.widget[cuda]".\n')


def test_notice_prints_once_per_process(monkeypatch, capsys):
    monkeypatch.setattr(device, "_printed", set())
    _machine(monkeypatch, system="linux", machine="x86_64", nvidia=True)
    for _ in range(3):
        device.gpu_notice()
    assert len(capsys.readouterr().out.splitlines()) == 1


# ---------------------------------------------------------------------------
# Adapters without the optional packages
# ---------------------------------------------------------------------------


def test_gpu_adapter_without_quantem_gpu_explains_the_install(monkeypatch):
    adapter = _load_without(gpu_adapter.__file__, "quantem.gpu", monkeypatch)
    assert not adapter.available() and not adapter.accelerator_ready()
    assert not adapter.is_acquisition(np.zeros(3)) and not adapter.is_ssb(object())
    _machine(monkeypatch, system="linux", machine="x86_64", nvidia=True)
    for call in (lambda: adapter.load_acquisition("scan_master.h5"), lambda: adapter.prepare_session(None),
                 lambda: adapter.ssb_session(None), lambda: adapter.discover_masters(".")):
        with pytest.raises(ImportError, match=r'pip install "quantem.widget\[cuda\]"'):
            call()
    _machine(monkeypatch, system="win32", machine="ARM64")
    with pytest.raises(ImportError, match="Apple silicon Mac, and neither is available to this Python") as error:
        adapter.prepare_session(None)
    assert "pip install" not in str(error.value)


def test_stand_in_carries_the_quantem_core_surface_the_widgets_use(monkeypatch):
    """Where quantem core cannot install, readers return ``ArrayDataset``: every attribute it has is a core
    dataset attribute with the same value and type, so user code reads it the same on every machine."""
    pytest.importorskip("quantem.core.datastructures", reason="parity needs quantem core")
    adapter = _load_without(core_adapter.__file__, "quantem.core.datastructures", monkeypatch)
    assert not adapter.available() and core_adapter.available()
    surface = ("array", "name", "sampling", "units", "metadata", "shape", "ndim", "dtype")
    stand_in_names = {name for name in dir(adapter.ArrayDataset) if not name.startswith("_")}
    stand_in_names |= set(adapter.ArrayDataset.__dataclass_fields__)
    assert stand_in_names == set(surface)
    cases = [
        (np.arange(12, dtype=np.uint16).reshape(3, 4), {"sampling": (0.5, 0.25), "units": ("nm", "nm"), "name": "survey"}),
        (np.arange(24, dtype=np.float32).reshape(2, 3, 4), {}),
        (np.arange(16, dtype=np.uint8).reshape(2, 2, 2, 2), {"sampling": (2.0, 2.0, 3.68, 3.68), "units": ("A", "A", "mrad", "mrad")}),
    ]
    for values, calibration in cases:
        stand_in = adapter.make_dataset(values, **calibration)
        core = core_adapter.make_dataset(values, **calibration)
        assert type(core).__name__ == {2: "Dataset2d", 3: "Dataset3d", 4: "Dataset4dstem"}[values.ndim]
        assert isinstance(stand_in, adapter.ArrayDataset) and not isinstance(stand_in, type(core))
        for name in surface:
            mine, theirs = getattr(stand_in, name), getattr(core, name)
            assert type(mine) is type(theirs), (name, type(mine), type(theirs))
            if isinstance(mine, np.ndarray):
                assert mine.dtype == theirs.dtype, name
                np.testing.assert_array_equal(mine, theirs)
            else:
                assert mine == theirs, name
        assert stand_in.array is values and core.array is values  # kept, not copied
        assert adapter.is_dataset(stand_in, ndim=values.ndim) and core_adapter.is_dataset(core, ndim=values.ndim)
        assert adapter.as_array(stand_in) is values and core_adapter.as_array(core) is values
    assert not adapter.is_dataset(np.ones((3, 4))) and not hasattr(adapter.ArrayDataset, "__array__")


# ---------------------------------------------------------------------------
# One definition of the math: parity with quantem.gpu when it is installed
# ---------------------------------------------------------------------------


def test_colormap_luts_are_the_quantem_gpu_bytes():
    assert colormap_lut("gray")[[0, 255], :3].tolist() == [[0, 0, 0], [1, 1, 1]]
    display = pytest.importorskip("quantem.gpu.display", reason="parity needs quantem.gpu")
    assert colormap_names() == display.colormap_names()
    for name in colormap_names():
        assert colormap_lut(name).tobytes() == display.colormap_lut(name).tobytes(), name


def test_detector_geometry_matches_quantem_gpu():
    gpu_detector = pytest.importorskip("quantem.gpu.detector", reason="parity needs quantem.gpu")
    rows, cols = np.indices((48, 40))
    pattern = np.where((rows - 20.3) ** 2 + (cols - 17.8) ** 2 <= 9.5**2, 50.0, 1.0).astype(np.float32)
    assert fit_probe(pattern) == gpu_detector.fit_probe(pattern)
    for center, inner, outer in (((20.3, 17.8), 0.0, 9.5), ((23.5, 19.5), 6.0, 15.25), ((0.0, 0.0), 1.0, 60.0)):
        np.testing.assert_array_equal(detector_mask(center, inner, outer, (48, 40)),
                                      gpu_detector.detector_mask(center, inner, outer, (48, 40)))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.float32])
def test_dense_reductions_match_quantem_gpu_sessions(dtype):
    gpu_detector = pytest.importorskip("quantem.gpu.detector", reason="parity needs quantem.gpu")
    values = np.random.default_rng(1).integers(0, 200, (6, 5, 12, 12))
    data_t = torch.as_tensor(values).to(dtype)
    if dtype.is_floating_point:
        data_t = data_t / 7
    dense = DenseSession(data_t)
    reference = gpu_detector.prepare(data_t.numpy())  # the float64 host reference
    mask = detector_mask((5.5, 6.0), 2.0, 5.0, (12, 12))
    np.testing.assert_array_equal(dense.masked_sum(mask), reference.masked_sum(mask))
    np.testing.assert_array_equal(dense.mean_dp(), reference.mean_dp())
    indices = np.array([0, 4, 7, 29])
    for mode in ("sum", "max"):
        np.testing.assert_array_equal(dense.reduce_frames(indices, mode), reference.reduce_frames(indices, mode))
    np.testing.assert_allclose(dense.reduce_frames(indices, "mean"), reference.reduce_frames(indices, "mean"), rtol=1e-6)
    np.testing.assert_array_equal(dense.frame(13), data_t.reshape(30, 12, 12)[13].numpy())


# ---------------------------------------------------------------------------
# Exact integer sums where int64 reductions fail (Apple MPS on an Intel Mac)
# ---------------------------------------------------------------------------


def test_int32_block_sums_stay_exact_past_int32_and_for_wide_values():
    # 40000 frames at 65535 counts pass 2**31 per pixel: one int32 sum would wrap
    counts = np.full((40000, 6), 65535, dtype=np.int64)
    counts[::3, 1] = 7
    counts_t = torch.as_tensor(counts.astype(np.int32))
    assert count_bound(counts_t) == 65535
    np.testing.assert_array_equal(int32_block_sum(counts_t, 0, 65535).numpy(), counts.sum(axis=0))
    np.testing.assert_array_equal(int32_block_sum(counts_t.T, 1, 65535).numpy(), counts.sum(axis=0))
    # values beyond int32, signed, go through 16-bit limbs
    wide = np.random.default_rng(2).integers(-(2**40), 2**40, (5000, 7))
    wide[0, 0], wide[1, 1] = -(2**62), 2**62
    wide_t = torch.as_tensor(wide)
    bound = count_bound(wide_t)
    assert bound == 2**62
    np.testing.assert_array_equal(int32_block_sum(wide_t, 0, bound).numpy(), wide.sum(axis=0))
    np.testing.assert_array_equal(int32_block_sum(wide_t, 1, bound).numpy(), wide.sum(axis=1))
    np.testing.assert_array_equal(int32_block_max(wide_t, bound).numpy(), wide.max(axis=0))
    assert count_bound(torch.zeros(3, dtype=torch.uint8)) == 255
    assert count_bound(torch.tensor([-(2**31), 5], dtype=torch.int32)) == 2**31


def test_dense_session_on_the_int32_block_path_equals_the_int64_path():
    data = np.random.default_rng(3).integers(0, 65536, (200, 200, 4, 5)).astype(np.int32)
    data[:, :, 1, 2] = 65535  # this pixel's total over 40000 frames needs more than int32
    data_t = torch.as_tensor(data)
    reference = DenseSession(data_t)
    blocks = DenseSession(data_t)
    blocks.count_bound = count_bound(data_t)  # what DenseSession sets for an integer tensor on MPS
    mask = detector_mask((1.5, 2.0), 0.0, 1.6, (4, 5))
    np.testing.assert_array_equal(blocks.mean_dp(), reference.mean_dp())
    np.testing.assert_array_equal(blocks.masked_sum(mask), reference.masked_sum(mask))
    indices = np.random.default_rng(4).choice(40000, 33000, replace=False)
    for mode in ("sum", "max", "mean"):
        np.testing.assert_array_equal(blocks.reduce_frames(indices, mode), reference.reduce_frames(indices, mode))
    assert int(blocks.reduce_frames(np.arange(40000), "sum")[1, 2]) == 40000 * 65535


def test_exact_sums_over_several_axes_agree_on_both_paths():
    """``exact_sum`` with a bound (the MPS path) equals the int64 path, also over a tuple of axes."""
    counts = np.random.default_rng(6).integers(0, 2**22, (3, 2, 8, 4, 6, 4)).astype(np.int32)
    counts_t = torch.as_tensor(counts)
    expected = counts.astype(np.int64).sum(axis=(3, 5))
    np.testing.assert_array_equal(exact_sum(counts_t, (3, 5)).numpy(), expected)
    np.testing.assert_array_equal(exact_sum(counts_t, (3, 5), count_bound(counts_t)).numpy(), expected)
    # a bound near int32 forces one-value blocks and the 16-bit limb split after the first axis
    np.testing.assert_array_equal(exact_sum(counts_t, (3, 5), 2**31 - 2).numpy(), expected)
    np.testing.assert_array_equal(exact_max(counts_t.reshape(3, -1), count_bound(counts_t)).numpy(),
                                  counts.reshape(3, -1).max(axis=0))
    mask_t = torch.as_tensor(counts > 2**21)
    assert int(exact_sum(mask_t.flatten(), 0, count_bound(mask_t))) == int((counts > 2**21).sum())


def test_detector_bin_mean_divides_the_exact_count_total_once():
    """Counts above 2**24 per block: a float32 sum would round, the integer total does not."""
    counts = np.random.default_rng(7).integers(2**21, 2**22, (2, 3, 8, 8)).astype(np.int32)
    total = counts.reshape(2, 3, 2, 4, 2, 4).astype(np.int64).sum(axis=(3, 5))
    expected = (total / 16).astype(np.float32)
    binned = detector_bin_mean(torch.as_tensor(counts), 4)
    assert binned.dtype == np.float32
    np.testing.assert_array_equal(binned, expected)
    floats = counts.astype(np.float32) / 7
    np.testing.assert_allclose(detector_bin_mean(torch.as_tensor(floats), 4),
                               floats.reshape(2, 3, 2, 4, 2, 4).mean(axis=(3, 5)), rtol=1e-6)
    np.testing.assert_array_equal(detector_bin_mean(torch.as_tensor(counts), 1), counts.astype(np.float32))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_widgets_reduce_counts_exactly_on_mps():
    """Show4DSTEM and ShowDiffraction on MPS give the CPU numbers, past int32 totals per pixel."""
    from quantem.widget import Show4DSTEM, ShowDiffraction

    counts = np.random.default_rng(8).integers(0, 65536, (200, 200, 8, 8)).astype(np.int32)
    counts[:, :, 3, 4] = 65535  # 40000 x 65535 needs more than int32
    widget = Show4DSTEM(torch.as_tensor(counts).to("mps"), precompute_virtual_images=False)
    try:
        session = widget._session(0)
        mask = detector_mask((3.5, 3.5), 0.0, 2.5, (8, 8))
        np.testing.assert_array_equal(session.reduce_frames(np.arange(40000), "sum"), counts.astype(np.uint64).sum(axis=(0, 1)))
        np.testing.assert_array_equal(session.reduce_frames(np.arange(40000), "max"), counts.max(axis=(0, 1)).astype(np.uint64))
        np.testing.assert_array_equal(session.masked_sum(mask), counts[:, :, mask].astype(np.int64).sum(axis=-1).astype(np.float32))
        np.testing.assert_array_equal(detector_bin_mean(torch.as_tensor(counts[:4]).to("mps"), 2),
                                      detector_bin_mean(torch.as_tensor(counts[:4]), 2))
    finally:
        widget.close()
    pattern = np.zeros((64, 64), np.uint16)
    pattern[20:30, 34:44] = 60000
    on_mps = ShowDiffraction(pattern, device="mps", verbose=False)
    on_cpu = ShowDiffraction(pattern, device="cpu", verbose=False)
    assert (on_mps.center_row, on_mps.center_col) == (on_cpu.center_row, on_cpu.center_col) == (24.5, 38.5)
