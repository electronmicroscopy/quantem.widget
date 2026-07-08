import numpy as np

from quantem.widget import Show2D, Show3D


def test_show2d_fft_metrics_default_and_state_roundtrip() -> None:
    data = np.zeros((32, 32), dtype=np.float32)
    widget = Show2D(data, show_fft=True, verbose=False)

    assert widget.fft_metrics is True
    state = widget.state_dict()
    assert state["fft_metrics"] is True

    restored = Show2D(data, show_fft=True, fft_metrics=False, verbose=False)
    assert restored.fft_metrics is False
    restored.load_state_dict(state)
    assert restored.fft_metrics is True


def test_show3d_fft_metrics_default_and_state_roundtrip() -> None:
    data = np.zeros((3, 32, 32), dtype=np.float32)
    widget = Show3D(data, show_fft=True, show_controls=False, verbose=False)

    assert widget.fft_metrics is True
    state = widget.state_dict()
    assert state["fft_metrics"] is True

    restored = Show3D(
        data,
        show_fft=True,
        fft_metrics=False,
        show_controls=False,
        verbose=False,
    )
    assert restored.fft_metrics is False
    restored.load_state_dict(state)
    assert restored.fft_metrics is True
