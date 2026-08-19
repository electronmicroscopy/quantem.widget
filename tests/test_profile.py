from __future__ import annotations

from pathlib import Path


def test_profile_reports_the_installed_quantem_stack(capsys) -> None:
    """A notebook records every QuantEM package through one profile call."""
    import quantem.widget as qw

    qw.profile()

    output = capsys.readouterr().out
    assert "quantem.widget" in output
    assert "quantem.gpu" in output
    assert "quantem" in output
    assert "torch" in output
    assert "python" in output


def test_documented_environment_checks_use_widget_profile() -> None:
    """User and maintainer guidance share the same environment report."""

    repo = Path(__file__).resolve().parents[1]
    docs = [
        repo / "docs" / "install.md",
        repo / "docs" / "api" / "index.md",
        repo / "docs" / "maintainer" / "widget-release.md",
    ]
    for path in docs:
        source = path.read_text(encoding="utf-8")
        assert "qw.profile()" in source, path
        assert "print(qw.__version__)" not in source, path

    smoke = (repo / "scripts" / "e2e_fresh.py").read_text(encoding="utf-8")
    assert "w.profile()" in smoke
