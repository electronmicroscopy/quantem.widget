from __future__ import annotations

import pathlib


def test_show4dstem_preset_clicks_sync_without_comm_guard() -> None:
    """C1: BF/ABF/ADF clicks must sync in JupyterLab models without ``comm``."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    source = (repo_root / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )
    save_start = source.index("const saveChangesIfLiveComm")
    save_end = source.index("const publishVirtualImageBytes", save_start)
    save_block = source[save_start:save_end]
    preset_start = source.index("const requestViPreset")
    preset_end = source.index("const setViSource", preset_start)
    preset_block = source[preset_start:preset_end]

    assert "liveModel.comm" not in save_block
    assert 'typeof liveModel.save_changes !== "function"' in save_block
    assert 'model.set("_preset_request", preset);' in preset_block
    assert "saveChangesIfLiveComm();" in preset_block


def test_show4dstem_frontend_announces_when_it_is_ready() -> None:
    """A mounted output asks Python for its initial scientific image buffers."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    source = (repo_root / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert 'type: "show4dstem_frontend_ready"' in source
    assert "version: 1" in source


def test_show4dstem_file_open_exports_keep_local_h5_folder_grant_visible() -> None:
    """C2: file-opened H5 exports must expose the no-server folder grant path."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    source = (repo_root / "js" / "show4dstem" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert "__QT_REQUIRE_LOCAL_H5_FILES" in source
    assert "showLocalH5GrantBanner" in source
    assert "data-show4dstem-open-folder" in source
    assert "Open data folder" in source
    assert "webkitdirectory" in source
    assert "No server needed" in source
