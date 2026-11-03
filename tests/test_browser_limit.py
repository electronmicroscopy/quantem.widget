"""The display_bin hint printed when a payload is too large to reach the browser."""

from quantem.widget.state import BROWSER_MESSAGE_LIMIT_BYTES, announce_browser_limit


def test_suggests_the_smallest_bin_that_fits(capsys):
    announce_browser_limit("Show3D", 50 * 4096 * 4096 * 4)
    assert "Pass display_bin=2 (800 MB)" in capsys.readouterr().out


def test_suggests_a_bin_above_the_one_already_passed(capsys):
    # display_bin=2 of a 9 x limit native stack still leaves 2.25 x limit; 3x is the next that fits.
    announce_browser_limit("Show3D", 9 * BROWSER_MESSAGE_LIMIT_BYTES // 4, display_bin=2)
    out = capsys.readouterr().out
    assert "with display_bin=2;" in out and "Pass display_bin=3 (2048 MB)" in out


def test_silent_when_the_payload_fits(capsys):
    announce_browser_limit("Show2D", BROWSER_MESSAGE_LIMIT_BYTES)
    assert capsys.readouterr().out == ""
