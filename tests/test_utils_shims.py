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


def test_render_compatibility_shims():
    from quantem.widget import _snapshot, gif_utils
    from quantem.widget.render import gif, snapshot

    assert _snapshot.render_image_png is snapshot.render_image_png
    assert _snapshot.render_panels_png is snapshot.render_panels_png
    assert gif_utils.write_gif is gif.write_gif


def test_dataset5dstem_compatibility_shim():
    from quantem.widget.data import Dataset5dstem as DataDataset5dstem

    assert DataDataset5dstem.__name__ == "Dataset5dstem"
