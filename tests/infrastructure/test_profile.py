"""Profile output identifies the code used in a scientist's bug report."""

import json
from types import SimpleNamespace

import pytest

import quantem.widget as qw
from quantem.widget import info


@pytest.fixture
def installation(monkeypatch, tmp_path):
    """Describe an installed Widget and GPU without contacting an index."""
    versions = {
        "quantem.widget": "0.0.1rc43+g261a0c69",
        "quantem.gpu": "0.0.1rc17+g5fba77a2",
    }
    metadata = {
        name: {"url": "file:///private/build/package.whl", "archive_info": {}}
        for name in versions
    }
    origins = {
        name: tmp_path / "site-packages" / name.replace(".", "/") / "__init__.py"
        for name in versions
    }

    def version(name):
        if name not in versions:
            raise info.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(info, "version", version)
    monkeypatch.setattr(qw, "__version__", versions["quantem.widget"])
    monkeypatch.setattr(
        info,
        "distribution",
        lambda name: SimpleNamespace(
            read_text=lambda filename: (
                json.dumps(metadata[name]) if metadata[name] is not None else None
            )
        ),
    )
    monkeypatch.setattr(
        info, "find_spec", lambda name: SimpleNamespace(origin=str(origins[name]))
    )
    monkeypatch.setattr(info, "ENGINE_MANIFEST", tmp_path / "no-manifest.json")
    monkeypatch.setattr(info.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(info.torch.backends.mps, "is_available", lambda: False)
    return versions, metadata, origins


def test_profile_identifies_local_widget_and_gpu_versions(installation, capsys):
    qw.profile()
    text = capsys.readouterr().out
    assert "quantem.widget  0.0.1rc43+g261a0c69" in text
    assert "quantem.gpu     0.0.1rc17+g5fba77a2" in text
    assert text.count("install       local build") == 2
    assert "commit        261a0c69 (version metadata)" in text
    assert "commit        5fba77a2 (version metadata)" in text
    assert "published" not in text and "/private" not in text


@pytest.mark.parametrize(
    "metadata,version,label,commit",
    [
        (
            {"url": "file:///private/plain.whl", "archive_info": {}},
            "1.0",
            "local build",
            "not recorded",
        ),
        (None, "1.0", "installed package", "not recorded"),
        (None, "1.0.dev2+g1234567", "development build", "1234567 (version metadata)"),
        (
            {"url": "https://private.example/package.whl", "archive_info": {}},
            "1.0",
            "direct install",
            "not recorded",
        ),
        (
            {
                "url": "https://private.example/repo",
                "vcs_info": {"vcs": "git", "commit_id": "abcdef1234567890"},
            },
            "1.0",
            "VCS install",
            "abcdef1234567890 (VCS metadata)",
        ),
    ],
)
def test_profile_uses_recorded_install_provenance(
    installation, monkeypatch, capsys, metadata, version, label, commit
):
    versions, records, _ = installation
    versions["quantem.widget"] = version
    records["quantem.widget"] = metadata
    monkeypatch.setattr(qw, "__version__", version)
    qw.profile()
    text = capsys.readouterr().out.split("quantem.gpu")[0]
    assert f"quantem.widget  {version}" in text
    assert f"install       {label}" in text
    assert f"commit        {commit}" in text
    assert "published" not in text and "private" not in text


@pytest.mark.parametrize("override", [False, True])
def test_profile_uses_the_imported_checkout_commit(
    installation, monkeypatch, tmp_path, capsys, override
):
    _, metadata, origins = installation
    checkout = tmp_path / "current-checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: another-worktree")
    origins["quantem.widget"] = checkout / "src/quantem/widget/__init__.py"
    installed = tmp_path / "previous-checkout" if override else checkout
    metadata["quantem.widget"] = {
        "url": installed.as_uri(),
        "dir_info": {"editable": True},
    }

    def git(command, **kwargs):
        assert command[:3] == ["git", "-C", str(checkout)]
        return {
            "ls-files": "src/quantem/widget/__init__.py",
            "rev-parse": "0123456789ab",
            "status": " M src/quantem/widget/info.py",
        }[command[3]]

    monkeypatch.setattr(info.subprocess, "check_output", git)
    qw.profile()
    text = capsys.readouterr().out.split("quantem.gpu")[0]
    label = (
        "source override (differs from installed metadata)"
        if override
        else "editable checkout"
    )
    assert f"install       {label}" in text
    assert "commit        0123456789ab (dirty checkout)" in text
    assert "261a0c69 (version metadata)" not in text
    assert str(tmp_path) not in text
