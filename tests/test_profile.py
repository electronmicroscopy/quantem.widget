from quantem.widget import profile


def test_profile_returns_dict():
    info = profile()
    assert isinstance(info, dict)
    expected_keys = {
        "quantem_widget_version",
        "quantem_version",
        "python_version",
        "numpy_version",
        "pytorch_version",
        "compute_device",
        "gpu_name",
        "system_ram_gb",
        "system_ram_available_gb",
        "vram_gb",
        "vram_available_gb",
        "disk_total_gb",
        "disk_free_gb",
        "jupyter_version",
        "anywidget_version",
        "platform_info",
    }
    assert expected_keys.issubset(info.keys())


def test_profile_prints_output(capsys):
    profile()
    captured = capsys.readouterr()
    assert "quantem.widget" in captured.out
    assert "Python" in captured.out
    assert "NumPy" in captured.out
    assert "Disk" in captured.out
    assert "Platform" in captured.out


def test_profile_versions_present():
    info = profile()
    assert info["quantem_widget_version"]
    assert info["python_version"]
    assert info["numpy_version"]


def test_profile_compute_device():
    info = profile()
    assert info["compute_device"] in ("mps", "cuda", "cpu")
