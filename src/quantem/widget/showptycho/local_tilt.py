"""Region request validation and image transport for the ShowPtycho workspace."""

import json
from collections.abc import Mapping

import numpy as np


def _regions(values: list[dict], shape: tuple[int, int]) -> list[dict]:
    """Validate native scan bounds before sending them to a scientific callback."""
    if len(values) > 64:
        raise ValueError('Choose at most 64 regions per fit.')
    regions = []
    for value in values:
        bounds = value['bounds']
        if len(bounds) != 4 or any(isinstance(v, bool) or int(v) != v for v in bounds):
            raise ValueError('Region bounds must be four integer scan coordinates.')
        r0, r1, c0, c1 = map(int, bounds)
        if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
            raise ValueError(f'Region {bounds} lies outside scan {shape}. Move it inside the scan.')
        if min(r1-r0, c1-c0) < 32:
            raise ValueError('Use at least 32 native scan samples along each region axis.')
        regions.append({'id': str(value['id']), 'label': str(value['label']),
                        'bounds': [r0, r1, c0, c1]})
    if len({r['id'] for r in regions}) != len(regions):
        raise ValueError('Give each region a distinct id.')
    if len({r['label'] for r in regions}) != len(regions):
        raise ValueError('Give each region a distinct label so saved images remain separate.')
    return regions


def _images(images: Mapping[str, np.ndarray], sampling: float | Mapping[str, float]) -> tuple[str, bytes]:
    """Transfer finite phase images, with one explicit float32 display boundary."""
    metadata, chunks, offset = [], [], 0
    for label, source in images.items():
        values = np.asarray(source)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f'{label}: supply a finite two-dimensional phase image.')
        spacing = float(sampling[label] if isinstance(sampling, Mapping) else sampling)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError(f'{label}: supply finite, positive image sampling in angstrom.')
        values = np.ascontiguousarray(values, dtype=np.float32)
        metadata.append({'label': label, 'shape': list(values.shape), 'offset': offset,
                         'sampling_A': spacing})
        chunks.append(values.tobytes())
        offset += values.size
    return json.dumps(metadata), b''.join(chunks)
