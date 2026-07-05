import numpy as np

from quantem.widget import Show2D, Show3D, profile_widget, widget_timing_report
from quantem.widget._timing import (
    format_bytes,
    format_ms,
    format_timing_table,
    format_widget_render_timing,
)


def test_format_timing_helpers_are_human_readable():
    assert format_ms(4.2) == "4.2 ms"
    assert format_ms(604) == "604 ms"
    assert format_ms(5473) == "5.47 s"
    assert format_bytes(40 * 4096 * 4096 * 2) == "1.25 GiB"

    table = format_timing_table(
        "Example timing",
        [("Load", "0.60 s"), ("Build", "5.47 s")],
    )

    assert table.splitlines() == [
        "Example timing",
        "--------------",
        "Load   0.60 s",
        "Build  5.47 s",
    ]


def test_format_widget_render_timing_uses_standard_rows():
    report = format_widget_render_timing(
        "Show3D",
        shape=(40, 4096, 4096),
        dtype="uint16",
        raw_bytes=40 * 4096 * 4096 * 2,
        total_ms=5473,
        python_ms=820,
        wire_js_ms=4653,
        extra_rows=[("Frame server", "on")],
    )

    assert "Show3D timing" in report
    assert "Data             40x4096x4096 | uint16 | 1.25 GiB" in report
    assert "Render total     5.47 s" in report
    assert "Python build     820 ms" in report
    assert "Wire + JS paint  4.65 s" in report
    assert "Frame server     on" in report


def test_profile_widget_prints_build_profile(capsys):
    data = np.zeros((2, 8, 8), dtype=np.float32)

    widget, profile = profile_widget(
        "Show2D build",
        lambda: Show2D(data, verbose=False),
        data=data,
        load_ms=12.3,
        pack_ms=4.5,
        backend="cpu",
    )

    out = capsys.readouterr().out
    assert widget is not None
    assert profile.build_ms >= 0
    assert "Show2D build profile" in out
    assert "Backend       cpu" in out
    assert "Data          2x8x8 | float32 | 512 B" in out
    assert "Load          12 ms" in out
    assert "Pack/prep     4.5 ms" in out
    assert "Widget build" in out


def test_show2d_first_render_uses_timing_table(capsys):
    widget = Show2D(np.zeros((8, 8), dtype=np.float32), verbose=True)
    widget._on_first_render({"new": True})

    out = capsys.readouterr().out
    assert "Show2D timing" in out
    assert "Render total" in out
    assert "Python build" in out
    assert "Wire + JS paint" in out
    assert "8x8 | float32 | 256 B" in out


def test_show3d_first_render_uses_timing_table_and_honors_verbose(capsys):
    quiet = Show3D(np.zeros((2, 8, 8), dtype=np.float32), verbose=False)
    quiet._on_first_render({"new": True})
    assert capsys.readouterr().out == ""

    noisy = Show3D(np.zeros((2, 8, 8), dtype=np.float32), verbose=True)
    noisy._on_first_render({"new": True})
    out = capsys.readouterr().out
    assert "Show3D timing" in out
    assert "2x8x8 | float32 | 512 B" in out
    assert "Frame server" in out
    assert "Offline stack" in out


def test_widget_timing_report_after_first_render():
    widget = Show2D(np.zeros((8, 8), dtype=np.float32), verbose=False)
    assert "pending" in widget_timing_report(widget)

    widget._on_first_render({"new": True})
    report = widget_timing_report(widget)

    assert "Show2D timing" in report
    assert "Render total" in report
    assert "8x8" in report
