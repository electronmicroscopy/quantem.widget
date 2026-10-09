"""One dataset type per machine, and the same numbers from every widget for every input type.

The rule (Bob, 2026-10-08): a 4D-STEM file loads as quantem.gpu's ``Dataset4dstemGPU`` when
quantem.gpu is installed and a GPU works, else as quantem core's ``Dataset4dstem``; images and
stacks load as quantem core ``Dataset2d`` and ``Dataset3d``. Where quantem core cannot install
(an Intel Mac, Windows on ARM) the widget's stand-in carries the same surface. The machine state
is monkeypatched (no GPU route, no quantem core) so every branch runs here; the GPU branch runs
for real when quantem.gpu and a GPU are present.
"""

import json

import h5py
import numpy as np
import pytest
import torch

from quantem.widget import (
    ChooseLattice,
    Plot2D,
    Show1D,
    Show2D,
    Show3D,
    Show3DSlices,
    Show4DSTEM,
    ShowDiffraction,
    ShowPtycho,
    datasets,
    read_4dstem,
    read_image,
    read_image_stack,
)
from quantem.widget.adapters import core as core_adapter
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.show4dstem import reader

core = pytest.importorskip("quantem.core.datastructures", reason="the core branch needs quantem core")
CUDA = torch.cuda.is_available()


@pytest.fixture
def no_gpu_route(monkeypatch):
    """A machine where quantem.gpu cannot load files: not installed, or no CUDA or Metal GPU."""
    monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)


@pytest.fixture
def accelerator():
    if not gpu_adapter.accelerator_ready():
        pytest.skip("the GPU branch needs quantem.gpu and a CUDA or MPS GPU")


def _counts(scan=(6, 6), det=(16, 16)) -> np.ndarray:
    """A bright disk of varying intensity on a weak background, as uint16 counts."""
    rows, cols = np.indices(det)
    disk = ((rows - 7.5) ** 2 + (cols - 7.5) ** 2) <= 3.0**2
    scale = np.random.default_rng(0).integers(20, 60, scan, dtype=np.uint16)
    return (np.where(disk, 10, 1).astype(np.uint16)[None, None] * scale[..., None, None]).astype(np.uint16)


def _write_master(folder, stem: str, values: np.ndarray):
    rows, cols, det_rows, det_cols = values.shape
    with h5py.File(folder / f"{stem}_data_000001.h5", "w") as handle:
        handle.create_dataset("entry/data/data", data=values.reshape(rows * cols, det_rows, det_cols))
    master = folder / f"{stem}_master.h5"
    with h5py.File(master, "w") as handle:
        handle.require_group("entry/data")["data_000001"] = h5py.ExternalLink(f"{stem}_data_000001.h5", "entry/data/data")
        handle.require_group("entry/instrument/detector/detectorSpecific").create_dataset("ntrigger", data=rows * cols)
    return master


def _hide_quantem_core(patch) -> None:
    """A machine without quantem core (an Intel Mac, Windows on ARM): readers build the stand-in."""
    for name in ("Dataset", "Dataset2d", "Dataset3d", "Dataset4dstem"):
        patch.setattr(core_adapter, name, None)


def _stand_in(monkeypatch, values, **calibration):
    """The dataset ``make_dataset`` builds where quantem core is absent."""
    with monkeypatch.context() as patch:
        _hide_quantem_core(patch)
        return core_adapter.make_dataset(values, **calibration)


# ---------------------------------------------------------------------------
# Loaders return the dataset type of the machine
# ---------------------------------------------------------------------------


def test_a_4dstem_file_is_a_gpu_dataset_on_a_gpu_and_a_core_dataset_elsewhere(tmp_path, monkeypatch, accelerator):
    counts = _counts()
    masters = [_write_master(tmp_path, f"scan_{index}", counts + index) for index in range(2)]
    on_gpu = read_4dstem(masters[0])
    try:
        assert type(on_gpu).__name__ == "Dataset4dstemGPU"
        assert on_gpu.shape == counts.shape and on_gpu.dtype == np.uint16
        np.testing.assert_array_equal(on_gpu[:, :].cpu().numpy(), counts)
    finally:
        on_gpu.close()
    pair = read_4dstem(masters)
    assert [type(item).__name__ for item in pair] == ["Dataset4dstemGPU"] * 2
    for item in pair:
        item.close()
    dense = read_4dstem(masters[0], "cpu", verbose=False)  # device="cpu" reads densely where quantem.gpu runs
    assert isinstance(dense, core.Dataset4dstem) and dense.name == "scan_0"
    np.testing.assert_array_equal(dense.array, counts)
    monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    assert isinstance(read_4dstem(masters[0], verbose=False), core.Dataset4dstem)


@pytest.mark.parametrize("stand_in", [False, True])
def test_without_a_gpu_route_files_are_core_datasets_or_the_stand_in(tmp_path, monkeypatch, no_gpu_route, stand_in):
    counts = _counts()
    master = _write_master(tmp_path, "scan", counts)
    np.save(tmp_path / "survey.npy", counts[0, 0].astype(np.float32))
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(3):
        np.save(frames / f"frame_{index}.npy", counts[index, 0].astype(np.float32))
    expected = {"read_4dstem": (core.Dataset4dstem, counts), "read_image": (core.Dataset2d, counts[0, 0]),
                "read_image_stack": (core.Dataset3d, counts[:3, 0])}
    if stand_in:
        _hide_quantem_core(monkeypatch)
    loaded = {"read_4dstem": read_4dstem(master, verbose=False), "read_image": read_image(tmp_path / "survey.npy"),
              "read_image_stack": read_image_stack(frames, progress=False)}
    for function, dataset in loaded.items():
        kind, values = expected[function]
        assert isinstance(dataset, core_adapter.ArrayDataset if stand_in else kind), function
        np.testing.assert_array_equal(dataset.array, values)
        assert core_adapter.is_dataset(dataset, ndim=values.ndim)
    # uint16 counts stay uint16, as quantem.gpu stores them
    assert loaded["read_4dstem"].name == "scan" and loaded["read_4dstem"].array.dtype == np.uint16
    assert loaded["read_image"].name == "survey" and list(loaded["read_image_stack"].units) == ["frame", "pixels", "pixels"]


@pytest.mark.parametrize("gpu", [False, True])
def test_the_4dstem_tutorial_loader_follows_the_machine(tmp_path, monkeypatch, gpu):
    if gpu and not gpu_adapter.accelerator_ready():
        pytest.skip("the GPU branch needs quantem.gpu and a CUDA or MPS GPU")
    if not gpu:
        monkeypatch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
    folder = tmp_path / "widget-tutorials" / "show4dstem" / "gold-128-bin8" / "full"
    folder.mkdir(parents=True)
    counts = _counts(scan=(8, 8))
    np.save(folder / "data.npy", counts)
    (folder / "meta.json").write_text(json.dumps({"name": "gold", "sampling": [2.0, 2.0, 3.68, 3.68],
                                                  "units": ["A", "A", "mrad", "mrad"]}))
    monkeypatch.setattr(datasets, "snapshot_download", lambda **kwargs: tmp_path, raising=False)
    dataset = datasets.show4dstem_gold(size="small", verbose=False)
    assert type(dataset).__name__ == ("Dataset4dstemGPU" if gpu else "Dataset4dstem")
    values = dataset[:, :].cpu().numpy() if gpu else dataset.array
    np.testing.assert_array_equal(values, counts[::4, ::4])
    assert tuple(float(value) for value in dataset.sampling) == (8.0, 8.0, 3.68, 3.68)
    assert list(dataset.units) == ["A", "A", "mrad", "mrad"]
    viewer = Show4DSTEM(dataset, precompute_virtual_images=False, verbose=False)
    try:
        assert viewer.title == "Gold 4D-STEM bin8 preview" and viewer.pixel_size == 8.0 and viewer.k_pixel_size == 3.68
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# Every widget shows the same numbers for every input type
# ---------------------------------------------------------------------------


def _show4dstem_numbers(widget) -> list[bytes]:
    widget.pos_row, widget.pos_col = 4, 1
    mean = np.asarray(widget._session(0).mean_dp(), np.float32)
    return [bytes(widget.virtual_image_bytes), bytes(widget.frame_bytes), mean.tobytes(),
            np.float64([widget.center_row, widget.center_col, widget.bf_radius]).tobytes()]


def test_show4dstem_shows_the_same_numbers_for_every_input_type(tmp_path, monkeypatch):
    counts = _counts()
    calibration = {"sampling": (2.0, 2.0, 3.68, 3.68), "units": ("A", "A", "mrad", "mrad"), "name": "gold"}
    np.save(tmp_path / "gold.npy", counts)
    inputs = {
        "numpy": counts,
        "torch cpu": torch.from_numpy(counts),
        "core Dataset4dstem": core_adapter.make_dataset(counts, **calibration),
        "core Dataset4dstem from a cpu tensor": core.Dataset4dstem.from_tensor(torch.from_numpy(counts), **calibration),
        "stand-in": _stand_in(monkeypatch, counts, **calibration),
    }
    if CUDA:
        inputs["torch cuda"] = torch.from_numpy(counts).cuda()
        inputs["core Dataset4dstem from a cuda tensor"] = core.Dataset4dstem.from_tensor(torch.from_numpy(counts).cuda(), **calibration)
    if gpu_adapter.accelerator_ready():
        encoded = read_4dstem(tmp_path / "gold.npy")
        metadata = {"name": "gold", "sampling": list(calibration["sampling"]), "units": list(calibration["units"])}
        inputs["Dataset4dstemGPU, encoded from a file"] = encoded
        inputs["Dataset4dstemGPU over a tensor"] = gpu_adapter.acquisition_from_tensor(
            torch.from_numpy(counts).to(encoded.device), metadata)
    numbers = {}
    try:
        for label, data in inputs.items():
            widget = Show4DSTEM(data, precompute_virtual_images=False, verbose=False)
            try:
                numbers[label] = _show4dstem_numbers(widget)
                # every dataset names itself (a file by its name); an .npy file carries no calibration
                assert widget.title == ("" if label in ("numpy", "torch cpu", "torch cuda") else "gold"), label
                calibrated = label not in ("numpy", "torch cpu", "torch cuda", "Dataset4dstemGPU, encoded from a file")
                scales = (widget.pixel_size, widget.pixel_unit, widget.k_pixel_size, widget.k_pixel_unit)
                assert scales == ((2.0, "A", 3.68, "mrad") if calibrated else (1.0, "pixels", 1.0, "pixels")), label
            finally:
                widget.close()
    finally:
        for data in inputs.values():
            if gpu_adapter.is_acquisition(data):
                data.close()
    reference = numbers.pop("numpy")
    assert reference[0] != bytes(len(reference[0]))
    for label, shown in numbers.items():
        assert shown == reference, label


def test_a_list_of_datasets_opens_one_comparison_on_every_machine(tmp_path, monkeypatch):
    counts = [_counts(), _counts() + 3]
    masters = [_write_master(tmp_path, f"tilt_{index}", values) for index, values in enumerate(counts)]
    states = ["core"]
    if gpu_adapter.accelerator_ready():
        states.append("gpu")
    shown = {}
    for state in states:
        with monkeypatch.context() as patch:
            if state == "core":
                patch.setattr(gpu_adapter, "accelerator_ready", lambda: False)
            loaded = read_4dstem(masters, verbose=False)
            widget = Show4DSTEM(loaded, precompute_virtual_images=False, verbose=False)
        try:
            assert widget.view_mode == "multiple" and list(widget.frame_labels) == ["tilt_0", "tilt_1"]
            widget.frame_idx = 1
            shown[state] = _show4dstem_numbers(widget) + [bytes(widget.compare_virtual_image_bytes)]
        finally:
            widget.close()
            if state == "gpu":
                for acquisition in loaded:
                    acquisition.close()
    assert all(numbers == shown["core"] for numbers in shown.values())


def test_scan_region_mixed_lists_and_refused_npy_files_behave_alike(tmp_path, monkeypatch, capsys):
    counts = _counts()
    region = (1, 6, 0, 5)
    expected = Show4DSTEM(counts[1:6, 0:5], precompute_virtual_images=False, verbose=False)
    reference = _show4dstem_numbers(expected)
    expected.close()
    regions = [core_adapter.make_dataset(counts), _stand_in(monkeypatch, counts), torch.from_numpy(counts)]
    if gpu_adapter.accelerator_ready():
        np.save(tmp_path / "scan.npy", counts)
        regions.append(read_4dstem(tmp_path / "scan.npy"))
    for data in regions:
        widget = Show4DSTEM(data, scan_region=region, precompute_virtual_images=False, verbose=False)
        try:
            assert (widget.shape_rows, widget.shape_cols) == (5, 5) and _show4dstem_numbers(widget) == reference
        finally:
            widget.close()
    with pytest.raises(ValueError, match="must lie inside the scan"):
        Show4DSTEM(counts, scan_region=(0, 9, 0, 2))
    mixed = Show4DSTEM([core_adapter.make_dataset(counts), counts + 1], precompute_virtual_images=False, verbose=False)
    try:
        assert mixed.n_frames == 2
    finally:
        mixed.close()
    if not gpu_adapter.accelerator_ready():
        return
    regions[-1].close()
    # quantem.gpu keeps exact uint8, uint16 or float32 values; any other NumPy file is read densely, and says so
    np.save(tmp_path / "signed.npy", counts.astype(np.int16))
    capsys.readouterr()
    signed = read_4dstem(tmp_path / "signed.npy", verbose=False)
    assert "quantem.gpu cannot hold signed.npy on the GPU" in capsys.readouterr().out
    assert isinstance(signed, core.Dataset4dstem) and signed.array.dtype == np.uint16
    np.testing.assert_array_equal(signed.array, counts)


@pytest.mark.skipif(not CUDA, reason="the GPU memory ceiling is checked on CUDA")
def test_a_dense_scan_too_large_for_the_gpu_is_refused_not_reduced(monkeypatch):
    real_ceiling = reader.ceiling
    monkeypatch.setattr(reader, "ceiling", lambda device=None: 1000 if str(device).startswith("cuda") else real_ceiling(device))
    with pytest.raises(MemoryError, match="Nothing was binned or cropped"):
        Show4DSTEM(core_adapter.make_dataset(_counts()), device="cuda")
    Show4DSTEM(core_adapter.make_dataset(_counts()), device="cpu", precompute_virtual_images=False).close()


IMAGE = np.random.default_rng(3).random((24, 32)).astype(np.float32)
STACK = np.random.default_rng(4).random((4, 24, 32)).astype(np.float32)
SIGNAL = np.random.default_rng(5).random(40).astype(np.float32)


def _image_widgets():
    """(widget, values, construct, the bytes it shows, what a dataset named "gold" at 0.2 nm sets)."""
    return [
        ("Show1D", SIGNAL, lambda data: Show1D(data), lambda widget: widget.y_bytes, {"title": "gold"}),
        ("Show2D", IMAGE, lambda data: Show2D(data, verbose=False), lambda widget: widget.frame_bytes,
         {"title": "gold", "pixel_size": 0.2, "pixel_unit": "nm"}),
        ("Show3D", STACK, lambda data: Show3D(data, verbose=False), lambda widget: widget._offline_float_stack,
         {"title": "gold", "pixel_size": 2.0, "pixel_unit": "A"}),
        ("Show3DSlices", STACK, lambda data: Show3DSlices(data), lambda widget: widget.volume_bytes,
         {"title": "gold", "pixel_size": 2.0}),
        ("ShowDiffraction", IMAGE, lambda data: ShowDiffraction(data, device="cpu", verbose=False),
         lambda widget: widget.frame_bytes, {"title": "gold"}),
        ("ChooseLattice", IMAGE, lambda data: ChooseLattice(data), lambda widget: widget.frame_bytes, {"title": "gold"}),
        ("ShowPtycho", IMAGE, lambda data: ShowPtycho(data), lambda widget: widget.phase_bytes, {"pixel_size": 2.0}),
        # Plot2D's axes are the x and y it is given
        ("Plot2D", IMAGE, lambda data: Plot2D(data, x=np.arange(32.0), y=np.arange(24.0)), lambda widget: widget.data_bytes, {}),
    ]


@pytest.mark.parametrize("name", [entry[0] for entry in _image_widgets()])
def test_image_widgets_show_the_same_numbers_for_every_input_type(name, monkeypatch):
    _, values, construct, shown, from_dataset = next(entry for entry in _image_widgets() if entry[0] == name)
    sampling = (0.2,) * values.ndim
    units = ("nm",) * values.ndim
    inputs = {
        "numpy": values,
        "torch cpu": torch.from_numpy(values),
        "quantem core": core_adapter.make_dataset(values, sampling=sampling, units=units, name="gold"),
        "stand-in": _stand_in(monkeypatch, values, sampling=sampling, units=units, name="gold"),
    }
    if CUDA:
        inputs["torch cuda"] = torch.from_numpy(values).cuda()
    numbers = {}
    for label, data in inputs.items():
        widget = construct(data)
        try:
            numbers[label] = bytes(shown(widget))
            metadata = {attribute: getattr(widget, attribute) for attribute in from_dataset}
            if label in ("quantem core", "stand-in"):
                assert metadata == from_dataset, label
            else:  # an array carries no title or scale
                assert metadata.get("title", "") == "" and metadata.get("pixel_size", 0.0) == 0.0, label
        finally:
            widget.close()
    reference = numbers.pop("numpy")
    assert reference
    for label, data in numbers.items():
        assert data == reference, label
