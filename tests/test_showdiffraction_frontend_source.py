from __future__ import annotations

import pathlib


def test_showdiffraction_denoise_controls_bind_synced_traits() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    source = (repo_root / "js" / "showdiffraction" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert 'useModelState<string>("denoise")' in source
    assert 'useModelState<string>("detect_denoise")' in source
    assert 'useModelState<boolean>("show_detection_view")' in source

    assert ">Denoise</Typography>" in source
    assert ">Detect</Typography>" in source
    assert '"Detect View"' in source


def test_slow_denova_modes_stay_out_of_the_dropdowns() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    source = (repo_root / "js" / "showdiffraction" / "index.tsx").read_text(
        encoding="utf-8"
    )

    assert '"denova_tv"' not in source
    assert '"denova_tv12"' not in source
