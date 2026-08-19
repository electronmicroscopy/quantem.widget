from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import ModuleType


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_latest_testpypi_wheel_ignores_yanked_files() -> None:
    """The Colab installer chooses the newest usable wheel and pins its hash."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "install_colab.py"
    namespace = {"__name__": "quantem_colab_installer_test"}
    exec(  # noqa: S102 - load the installer without triggering its main block
        compile(script.read_text(encoding="utf-8"), script, "exec"), namespace
    )
    payload = b"""{
      "releases": {
        "0.0.1rc1": [{
          "filename": "quantem_widget-0.0.1rc1-py3-none-any.whl",
          "url": "https://example.test/rc1.whl",
          "upload_time_iso_8601": "2026-08-17T00:00:00Z",
          "digests": {"sha256": "old"},
          "yanked": false
        }],
        "0.0.1rc2": [{
          "filename": "quantem_widget-0.0.1rc2-py3-none-any.whl",
          "url": "https://example.test/rc2.whl",
          "upload_time_iso_8601": "2026-08-18T00:00:00Z",
          "digests": {"sha256": "new"},
          "yanked": false
        }, {
          "filename": "quantem_widget-0.0.1rc2-yanked.whl",
          "url": "https://example.test/yanked.whl",
          "upload_time_iso_8601": "2026-08-19T00:00:00Z",
          "digests": {"sha256": "bad"},
          "yanked": true
        }]
      }
    }"""
    namespace["urlopen"] = lambda *_args, **_kwargs: _Response(payload)

    result = namespace["_latest_testpypi_wheel_url"]("quantem.widget")

    assert result == "https://example.test/rc2.whl#sha256=new"


def test_install_latest_rc_preserves_colab_numpy_and_numba(
    monkeypatch, capsys
) -> None:
    """The installer must not disturb Colab's compatible numerical stack."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "install_colab.py"
    namespace = {"__name__": "quantem_colab_installer_test"}
    exec(  # noqa: S102 - load the installer without triggering its main block
        compile(script.read_text(encoding="utf-8"), script, "exec"), namespace
    )

    numpy = ModuleType("numpy")
    numpy.__version__ = "2.0.2"
    google = ModuleType("google")
    colab = ModuleType("google.colab")
    output = ModuleType("google.colab.output")
    enabled = []
    output.enable_custom_widget_manager = lambda: enabled.append(True)
    colab.output = output
    google.colab = colab
    quantem = ModuleType("quantem")
    quantem.__path__ = []
    widget = ModuleType("quantem.widget")
    profile_calls = []
    widget.profile = lambda: profile_calls.append(True)
    quantem.widget = widget
    monkeypatch.setitem(__import__("sys").modules, "numpy", numpy)
    monkeypatch.setitem(__import__("sys").modules, "google", google)
    monkeypatch.setitem(__import__("sys").modules, "google.colab", colab)
    monkeypatch.setitem(__import__("sys").modules, "google.colab.output", output)
    monkeypatch.setitem(__import__("sys").modules, "quantem", quantem)
    monkeypatch.setitem(__import__("sys").modules, "quantem.widget", widget)

    wheels = {
        "quantem.widget": "https://example.test/widget.whl#sha256=widget",
        "quantem.gpu": "https://example.test/gpu.whl#sha256=gpu",
    }
    commands = []
    namespace["_latest_testpypi_wheel_url"] = wheels.__getitem__
    namespace["version"] = lambda project: {"numba": "0.60.0"}[project]
    monkeypatch.setattr(
        namespace["subprocess"],
        "run",
        lambda command, check: commands.append((command, check)),
    )

    namespace["install_latest_rc"]()

    assert enabled == [True]
    assert profile_calls == [True]
    assert capsys.readouterr().out == "QuantEM ready\n"
    assert commands == [
        (
            [
                namespace["sys"].executable,
                "-m",
                "pip",
                "install",
                "-q",
                "numpy==2.0.2",
                "numba==0.60.0",
                "quantem.gpu[movie] @ https://example.test/gpu.whl#sha256=gpu",
                "https://example.test/widget.whl#sha256=widget",
            ],
            True,
        )
    ]
