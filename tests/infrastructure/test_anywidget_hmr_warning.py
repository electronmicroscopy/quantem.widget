import subprocess
import sys
import textwrap


def test_hmr_opt_in_without_watchfiles_prints_no_warning(tmp_path):
    """ANYWIDGET_HMR=1 without watchfiles must not print anywidget's live-reload nag."""
    blocker = tmp_path / "watchfiles.py"
    blocker.write_text("raise ImportError('watchfiles hidden for this test')\n")
    code = textwrap.dedent(
        """
        import warnings
        warnings.simplefilter("default")
        from quantem.widget.show2d import Show2D
        """
    )
    result = subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "ANYWIDGET_HMR": "1",
            "PYTHONPATH": f"{tmp_path}:src",
            "OMP_NUM_THREADS": "4",
        },
        check=True,
    )
    assert "Live-reloading" not in result.stderr
