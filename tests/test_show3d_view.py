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


def test_rgb_offline_stack_frames_decode_at_rgb_stride():
    # RGB stacks pack 3 bytes/px; frame k must live at k*3*H*W. A grayscale
    # stride here scrambles every frame after the first on kernel-less reopen
    # (the fig4 RGB export bug).
    n, h, w = 6, 12, 12
    src = np.zeros((n, h, w, 3), dtype=np.float32)
    for k in range(n):
        src[k] = [k / n, 0.5, 1.0 - k / n]
    widget = Show3D(src)
    assert widget.is_rgb and widget.offline
    stack = np.frombuffer(widget._offline_stack, dtype=np.uint8).reshape(n, h, w, 3)
    for k in range(n):
        expected = np.clip(src[k] * 255.0, 0, 255).astype(np.uint8)
        assert np.array_equal(stack[k], expected), f"RGB frame {k} scrambled"


def test_set_image_accepts_rgb_and_repacks():
    # Live color-stack growth: set_image must keep RGB mode and repack the
    # offline stack at the RGB stride, including gray<->RGB swaps.
    n, h, w = 6, 12, 12
    rgb = np.random.rand(n, h, w, 3).astype("float32")
    widget = Show3D(rgb)

    grown = np.random.rand(n + 4, h, w, 3).astype("float32")
    widget.set_image(grown)
    assert widget.is_rgb
    assert widget.n_slices == n + 4
    assert len(widget._offline_stack) == (n + 4) * h * w * 3
    stack = np.frombuffer(widget._offline_stack, dtype=np.uint8).reshape(n + 4, h, w, 3)
    assert np.array_equal(
        stack[-1], np.clip(grown[-1] * 255.0, 0, 255).astype(np.uint8)
    )

    widget.set_image(np.random.rand(3, h, w).astype("float32"))
    assert not widget.is_rgb
    assert len(widget._offline_stack) == 3 * h * w
