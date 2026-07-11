"""Show3D data-replacement invariants for kernel-less viewing surfaces."""

import numpy as np

from quantem.widget import Show3D


def test_set_image_repacks_offline_stack():
    # Docs pages, exported HTML, and saved widget state slice frames straight
    # out of _offline_stack. set_image() must repack it, or any appended frame
    # index points past the old stack's end and renders blank on reopen.
    widget = Show3D(np.random.rand(4, 32, 32).astype("float32"))
    assert widget.offline
    assert len(widget._offline_stack) == 4 * 32 * 32

    widget.set_image(np.random.rand(8, 32, 32).astype("float32"))

    assert widget.n_slices == 8
    assert len(widget._offline_stack) == 8 * 32 * 32
    lo, hi = widget._offline_min, widget._offline_max
    assert hi > lo
