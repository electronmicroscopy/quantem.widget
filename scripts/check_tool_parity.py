#!/usr/bin/env python3
"""Validate tool lock/hide parity across Python and viewer frontends."""

from __future__ import annotations

import inspect
import pathlib
import sys

from quantem.widget import (
    Align2D,
    Edit2D,
    Mark2D,
    Show2D,
    Show3D,
    Show3DVolume,
    Show4D,
    Show4DSTEM,
    ShowComplex2D,
)


WIDGET_CLASSES = {
    "Show2D": Show2D,
    "Show3D": Show3D,
    "Show3DVolume": Show3DVolume,
    "Show4D": Show4D,
    "Show4DSTEM": Show4DSTEM,
    "ShowComplex2D": ShowComplex2D,
    "Mark2D": Mark2D,
    "Edit2D": Edit2D,
    "Align2D": Align2D,
}

VIEWER_FRONTENDS = {
    "Show2D": pathlib.Path("js/show2d/index.tsx"),
    "Show3D": pathlib.Path("js/show3d/index.tsx"),
    "Show3DVolume": pathlib.Path("js/show3dvolume/index.tsx"),
    "Show4D": pathlib.Path("js/show4d/index.tsx"),
    "Show4DSTEM": pathlib.Path("js/show4dstem/index.tsx"),
    "ShowComplex2D": pathlib.Path("js/showcomplex/index.tsx"),
}


def run_checks(repo_root: pathlib.Path | None = None) -> list[str]:
    repo_root = pathlib.Path.cwd() if repo_root is None else repo_root
    errors: list[str] = []

    for widget_name, cls in WIDGET_CLASSES.items():
        sig = inspect.signature(cls.__init__)
        params = set(sig.parameters.keys())
        traits = set(getattr(cls, "class_trait_names", lambda: [])())

        for trait_name in ("disabled_tools", "hidden_tools"):
            if trait_name not in traits:
                errors.append(f"{widget_name}: missing trait '{trait_name}'")

        for param_name in ("disabled_tools", "hidden_tools"):
            if param_name not in params:
                errors.append(f"{widget_name}: missing __init__ parameter '{param_name}'")

    for widget_name, rel_path in VIEWER_FRONTENDS.items():
        path = repo_root / rel_path
        if not path.exists():
            errors.append(f"{widget_name}: missing frontend file {rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        if 'useModelState<string[]>("disabled_tools")' not in text:
            errors.append(f'{widget_name}: frontend missing useModelState("disabled_tools")')
        if 'useModelState<string[]>("hidden_tools")' not in text:
            errors.append(f'{widget_name}: frontend missing useModelState("hidden_tools")')
        if f'computeToolVisibility("{widget_name}"' not in text:
            errors.append(f"{widget_name}: frontend missing computeToolVisibility('{widget_name}') usage")

    return errors


def main() -> int:
    errors = run_checks()
    if errors:
        print("Tool parity check failed:")
        for idx, error in enumerate(errors, start=1):
            print(f"{idx}. {error}")
        return 1
    print("Tool parity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
