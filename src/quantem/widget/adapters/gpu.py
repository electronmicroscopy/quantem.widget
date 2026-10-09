"""quantem.gpu acquisitions, detector sessions and SSB, whether or not quantem.gpu is installed.

quantem.gpu is optional (``quantem.widget[cuda]`` or ``quantem.widget[mps]``). This is the only
module that imports it: encoded acquisitions from ``quantem.gpu.io.load``, their
detector sessions, master discovery and inspection, SSB and centre of mass.
Without quantem.gpu every name below is None, ``available()`` is False and a
function that needs it raises ``ImportError`` with the install command; the
widgets then read files densely with ``show4dstem.reader`` and reduce with
their own torch code.
"""

import numpy as np
import torch

from quantem.widget.device import device_name, gpu_path_hint, no_gpu_path, print_once

try:
    from quantem.gpu import SSB
    from quantem.gpu import io as gpu_io
    from quantem.gpu.detector import prepare
    from quantem.gpu.dpc import center_of_mass as gpu_center_of_mass
    from quantem.gpu.io import Dataset4dstemGPU
except ImportError:
    SSB = gpu_io = prepare = gpu_center_of_mass = Dataset4dstemGPU = None
try:
    # quantem.gpu 0.0.1rc13 has no runtime notice; later releases name their missing runtime
    from quantem.gpu.device import runtime_notice as gpu_runtime_notice
except ImportError:
    gpu_runtime_notice = None


def available() -> bool:
    """Whether quantem.gpu imported."""
    return gpu_io is not None


def require(feature: str) -> None:
    """Raise ``ImportError`` naming ``feature`` and the install command when quantem.gpu is missing."""
    if not available():
        raise ImportError(f"{feature} needs quantem.gpu, which is not installed. {gpu_path_hint() or no_gpu_path()}.")


def accelerator_ready() -> bool:
    """Whether quantem.gpu can load files on this machine: installed, and a CUDA or Metal backend works."""
    if not available():
        return False
    from quantem.gpu.device import detect

    try:
        detect()
    except RuntimeError:
        return False
    return True


def is_acquisition(obj) -> bool:
    """Whether ``obj`` is an acquisition from ``quantem.gpu.io.load``."""
    return available() and isinstance(obj, Dataset4dstemGPU)


def is_ssb(obj) -> bool:
    """Whether ``obj`` is a ``quantem.gpu.SSB`` session."""
    return available() and isinstance(obj, SSB)


def load_acquisition(path, device: str | None = None, *, backend: str = "auto", scan_shape=None):
    """Load one master (or a list) into encoded GPU storage with ``quantem.gpu.io.load``.

    ``device`` is a ``resolve_device`` string passed through: ``"cuda:N"``
    selects that GPU, ``"mps"`` the Apple GPU, None (``device="auto"``) lets
    quantem.gpu choose, and the choice is printed once, as ``resolve_device``
    prints it for arrays.
    """
    require("Loading an acquisition on the GPU")
    loaded = gpu_io.load(path, backend=backend, device=device, scan_shape=scan_shape, verbose=False)
    if device is None:
        first = loaded[0] if isinstance(loaded, list) else loaded
        print_once(f'quantem.widget: device="auto" selected {device_name(torch.device(first.device))}.')
    return loaded


def acquisition_from_tensor(values_t: torch.Tensor, metadata: dict):
    """A ``Dataset4dstemGPU`` over a 4D tensor already on the GPU, with ``metadata`` as its calibration and name.

    quantem.gpu holds such a tensor as a dense acquisition: indexing, ``read``
    and its detector sessions work as on an encoded one, so a tensor computed
    on the GPU (a strided preview of a loaded file) keeps the GPU dataset type.
    """
    require("A GPU acquisition")
    return Dataset4dstemGPU(values_t, dict(metadata))


def discover_masters(folder, *, pattern: str = "*_master.h5", recursive: bool = True, scan_shape=None) -> list[str]:
    """Readable candidate masters under ``folder`` (``quantem.gpu.io.discover``)."""
    require("Master discovery")
    return list(gpu_io.discover(str(folder), pattern=pattern, recursive=recursive, scan_shape=scan_shape, verbose=False))


def inspect_master(path, *, scan_shape=None):
    """Readiness report of one master (``quantem.gpu.io.inspect``): shapes, dtype, reason, action."""
    require("Master inspection")
    return gpu_io.inspect(str(path), scan_shape=scan_shape)


def prepare_session(source):
    """A ``quantem.gpu`` detector session over an acquisition view."""
    require("Encoded detector reductions")
    return prepare(source)


def center_of_mass(source, scan_shape) -> tuple[np.ndarray, np.ndarray]:
    """Mean-subtracted detector centre of mass ``(com_row, com_col)`` of every scan position."""
    require("Centre of mass")
    return gpu_center_of_mass(source, scan_shape=scan_shape)


def ssb_session(source, **kwargs):
    """A ``quantem.gpu.SSB`` session over ``source``; use it as a context manager."""
    require("Compute SSB")
    return SSB(source, **kwargs)


class AcquisitionView:
    """One acquisition, or a scan region of it, read in bounded windows.

    ``source`` is a ``Dataset4dstemGPU`` from ``io.load`` or a 4D tensor on the
    GPU. ``quantem.gpu.detector.prepare`` recognises the marker attributes and
    sums virtual detectors on the encoded counts; reads return float32 with the
    detector pixels the source flagged set to zero.
    """

    _bounded_detector_source = True

    def __init__(self, source, region=None):
        self.source = source
        self.region = region or (0, source.shape[0], 0, source.shape[1])
        row_start, row_stop, col_start, col_stop = self.region
        if not (0 <= row_start < row_stop <= source.shape[0] and 0 <= col_start < col_stop <= source.shape[1]):
            raise ValueError(f"Scan region {self.region} is outside {tuple(source.shape[:2])}.")
        self.shape = (row_stop - row_start, col_stop - col_start, *source.shape[2:])
        device = torch.device(source.device)
        self.device = torch.device(device.type, device.index or 0)
        self._detector_source = source
        self._detector_region = self.region
        self.valid = None
        if is_acquisition(source):
            session = prepare(source)
            validity = session.detector_validity
            session.close()
            if validity is not None:
                self.valid = torch.as_tensor(validity, device=self.device).reshape(self.shape[-2:])

    def read(self, *, scan_region):
        """``scan_region`` relative to this view, as a float32 tensor on the source GPU."""
        row_start, row_stop, col_start, col_stop = scan_region
        row_offset, _, col_offset, _ = self.region
        region = (row_start + row_offset, row_stop + row_offset, col_start + col_offset, col_stop + col_offset)
        if torch.is_tensor(self.source):
            return self.source[region[0] : region[1], region[2] : region[3]]
        values_t = self.source.read(scan_region=region).float()
        return values_t if self.valid is None else values_t.masked_fill(~self.valid, 0)


class AcquisitionSeries:
    """Several acquisitions of one scan and detector geometry as a growing series."""

    def __init__(self, sources):
        self.frames = [AcquisitionView(source) for source in sources]
        first = self.frames[0]
        if any(frame.shape != first.shape or frame.device != first.device for frame in self.frames):
            raise ValueError("Comparison sources must share scan/detector shape and device.")
        self.device = first.device

    @property
    def shape(self) -> tuple[int, ...]:
        """``(n_acquisitions, scan_rows, scan_cols, det_rows, det_cols)``."""
        return (len(self.frames), *self.frames[0].shape)

    def append(self, source) -> None:
        """Add one acquisition that arrived while a folder is watched."""
        frame = AcquisitionView(source)
        if frame.shape != self.frames[0].shape or frame.device != self.device:
            raise ValueError(
                f"Acquisition shape {frame.shape} on {frame.device} does not match "
                f"{self.frames[0].shape} on {self.device}."
            )
        self.frames.append(frame)
