import builtins


def test_state_io_compatibility_shim():
    from quantem.widget import state
    from quantem.widget.utils import state_io

    assert state.resolve_widget_version is state_io.resolve_widget_version
    assert state.unwrap_state_payload is state_io.unwrap_state_payload
    assert state.save_state_file is state_io.save_state_file


def test_recon_config_compatibility_shim():
    from quantem.widget import config_utils
    from quantem.widget.utils import recon_config

    assert config_utils._load_quantem_config is recon_config._load_quantem_config
    assert config_utils._pixel_size_from_quantem_config is recon_config._pixel_size_from_quantem_config
    assert config_utils._rotate_stack_inplane is recon_config._rotate_stack_inplane


def test_array_utils_compatibility_shim():
    from quantem.widget import array_utils
    from quantem.widget.utils import array

    assert array_utils.to_numpy is array.to_numpy
    assert array_utils.bin2d is array.bin2d
    assert array_utils._resize_image is array._resize_image


def test_array_utils_tolerates_core_dataset_import_failure(monkeypatch):
    from quantem.widget.utils import array

    monkeypatch.setattr(array, "_DATASET4DSTEM_IMPORT_ATTEMPTED", False)
    monkeypatch.setattr(array, "_DATASET4DSTEM_TYPE", None)
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "quantem.core.datastructures":
            raise ImportError("simulated editable quantem.core import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    data = object()

    assert array.unwrap_core_4dstem(data) is data


def test_render_compatibility_shims():
    from quantem.widget import gif_utils
    from quantem.widget.render import gif

    assert gif_utils.write_gif is gif.write_gif


def test_dataset5dstem_compatibility_shim():
    from quantem.widget.data import Dataset5dstem as DataDataset5dstem

    assert DataDataset5dstem.__name__ == "Dataset5dstem"
