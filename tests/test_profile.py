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
    assert "install" in output


def test_profile_checks_testpypi_only_when_requested(monkeypatch, capsys) -> None:
    """A notebook opts into release checks without changing the normal report."""
    import quantem.widget as qw

    calls = []

    def latest(distribution_name: str) -> str:
        calls.append(distribution_name)
        return "99.0rc1"

    monkeypatch.setattr("quantem.widget._profile._latest_testpypi_version", latest)

    qw.profile()
    assert calls == []

    qw.profile(check_updates=True)

    output = capsys.readouterr().out
    assert "TestPyPI      latest 99.0rc1" in output
    assert "WARNING" in output
    assert calls == ["quantem.widget", "quantem.gpu"]
    assert output.count("TestPyPI      latest 99.0rc1") == 2


def test_profile_does_not_print_editable_source_paths(monkeypatch, capsys) -> None:
    """A shared profile report labels an editable checkout without leaking paths."""
    from quantem.widget import _profile

    source = Path.cwd() / "private-source" / "quantem.widget"
    monkeypatch.setattr(_profile, "_editable_source", lambda name: source)
    monkeypatch.setattr(
        _profile,
        "_loaded_path",
        lambda name: source / "src/quantem/widget/__init__.py",
    )
    monkeypatch.setattr(_profile, "_print_checkout", lambda source: None)

    _profile.print_distribution_status(
        "quantem.widget",
        "0.0.1rc36",
        check_updates=False,
    )

    output = capsys.readouterr().out
    assert "editable checkout" in output
    assert str(source) not in output

    loaded = Path.cwd() / "another-checkout/src/quantem/widget/__init__.py"
    monkeypatch.setattr(_profile, "_loaded_path", lambda name: loaded)
    _profile.print_distribution_status(
        "quantem.widget",
        "0.0.1rc36",
        check_updates=False,
    )

    output = capsys.readouterr().out
    assert "source override" in output
    assert str(source) not in output
    assert str(loaded) not in output


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
