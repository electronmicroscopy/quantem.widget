"""Live comparisons without stacking complete resident measurements."""

import math

import torch

from quantem.widget.show4dstem import Show4DSTEM


class _View:
    _is_gpu_frames = True
    _bounded_detector_source = True
    ndim = 4
    dtype = torch.float32

    def __init__(self, source, region=None):
        self.source = source
        self.region = region or (0, source.shape[0], 0, source.shape[1])
        r0, r1, c0, c1 = self.region
        if not (0 <= r0 < r1 <= source.shape[0] and 0 <= c0 < c1 <= source.shape[1]):
            raise ValueError(f'Scan region {self.region} is outside {source.shape[:2]}.')
        self.shape = (r1-r0, c1-c0, *source.shape[2:])
        payload = getattr(source, 'data', source)
        device = getattr(payload, 'device', None)
        if device is None:
            device = getattr(source, 'device', None)
        if device is None:
            device = source.metadata['device']
        self.device = torch.device('cuda', device) if isinstance(device, int) else torch.device(device)
        self.device = torch.device(self.device.type, self.device.index or 0)
        self.nbytes = 0  # Borrowed; this wrapper allocates no measurement storage.
        # Preserve the native reduction owner through an optional scan-region view.
        owner = getattr(source, 'source', source)
        base = getattr(source, 'region', (0, source.shape[0], 0, source.shape[1]))
        self._detector_source = owner
        self._detector_region = (base[0]+r0, base[0]+r1, base[2]+c0, base[2]+c1)
        validity = getattr(getattr(source, 'data', None), 'valid_pixels', None)
        self.valid = None if validity is None else torch.as_tensor(
            validity, device=self.device, dtype=torch.bool
        ).reshape(self.shape[-2:])
        if getattr(source, 'valid', None) is not None:
            self.valid = source.valid

    def read(self, *, scan_region):
        r0, r1, c0, c1 = scan_region
        row, _, col, _ = self.region
        scan_region = (r0 + row, r1 + row, c0 + col, c1 + col)
        if torch.is_tensor(self.source):
            r0, r1, c0, c1 = scan_region
            return self.source[r0:r1, c0:c1]
        value_t = self.source.read(scan_region=scan_region)
        return value_t if self.valid is None else value_t.float().masked_fill(~self.valid, 0)

    def numel(self):
        return math.prod(self.shape)

    def nelement(self):
        return self.numel()

    def element_size(self):
        return 4  # Bounded detector calculations return float32 working arrays.

    def to(self, device):
        target = torch.device(device)
        if target.type == self.device.type and (target.index or 0) == (self.device.index or 0):
            return self
        raise ValueError("An encoded acquisition owns its device. Reload it on the selected GPU instead of copying a dense cube.")

    def __getitem__(self, row):
        if isinstance(row, tuple):
            row, col = row
            return self.read(scan_region=(row, row + 1, col, col + 1))[0, 0]
        # The base viewer uses one initial row only for display range estimation.
        return self.read(scan_region=(row, row + 1, 0, min(32, self.shape[1])))[0]


class _Views:
    _is_gpu_frames = True
    ndim = 5
    dtype = torch.float32
    nbytes = 0

    def __init__(self, sources):
        self.frames = [_View(source) for source in sources]
        first = self.frames[0]
        if any(frame.shape != first.shape or frame.device != first.device for frame in self.frames):
            raise ValueError('Comparison sources must share scan/detector shape and device.')
        self.shape = (len(self.frames), *first.shape)
        self.device = first.device

    def __getitem__(self, index):
        return self.frames[index]

    def numel(self):
        return math.prod(self.shape)


def show_bounded(sources, *, scan_region=None, **options):
    """Use the existing live UI with borrowed region-reading sources."""
    if options.get('offline') or options.get('data_url'):
        raise ValueError('Bounded resident comparisons require a live kernel; offline export is not supported.')
    options.setdefault('precompute_virtual_images', False)
    options.setdefault('verbose', False)
    options.setdefault('offline', False)
    options.setdefault('view_mode', 'multiple')
    options.setdefault('compare_dp_mode', 'selected')
    if scan_region is not None and len(sources) > 1:
        raise ValueError('Select the same source regions before multi-source comparison.')
    data = _View(sources[0], scan_region) if len(sources) == 1 else _Views(sources)
    return Show4DSTEM(data, **options)
