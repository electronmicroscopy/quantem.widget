"""Shared ROI geometry helpers for QuantEM image widgets."""

from __future__ import annotations

from typing import Any

import numpy as np


def roi_geometries(
    roi_list: list[Any],
    *,
    height: int | float,
    width: int | float,
    visible_only: bool = True,
) -> list[dict[str, Any]]:
    """Return normalized ROI geometry in image ``(row, col)`` coordinates."""
    image_height = float(height)
    image_width = float(width)
    geometries: list[dict[str, Any]] = []

    def point(row: float, col: float) -> dict[str, float]:
        return {"row": float(row), "col": float(col)}

    def bounds(row_min: float, row_max: float, col_min: float, col_max: float) -> dict[str, float]:
        return {
            "row_min": float(row_min),
            "row_max": float(row_max),
            "col_min": float(col_min),
            "col_max": float(col_max),
        }

    def clipped_bounds(row_min: float, row_max: float, col_min: float, col_max: float) -> dict[str, float]:
        return bounds(
            max(0.0, min(image_height, row_min)),
            max(0.0, min(image_height, row_max)),
            max(0.0, min(image_width, col_min)),
            max(0.0, min(image_width, col_max)),
        )

    def box_corners(row_min: float, row_max: float, col_min: float, col_max: float) -> list[dict[str, float]]:
        return [
            point(row_min, col_min),
            point(row_min, col_max),
            point(row_max, col_max),
            point(row_max, col_min),
        ]

    for index, roi in enumerate(roi_list):
        if not isinstance(roi, dict):
            continue
        visible = bool(roi.get("visible", True))
        if visible_only and not visible:
            continue

        shape = str(roi.get("shape", "circle")).lower()
        row = float(roi.get("row", image_height / 2))
        col = float(roi.get("col", image_width / 2))
        geometry: dict[str, Any] = {
            "index": int(index),
            "shape": shape,
            "visible": visible,
            "center": point(row, col),
            "row": row,
            "col": col,
        }
        for key in ("color", "line_width", "highlight", "locked"):
            if key in roi:
                geometry[key] = roi[key]

        if shape == "rectangle":
            width_px = float(roi.get("width", 20))
            height_px = float(roi.get("height", 20))
            half_row = height_px / 2.0
            half_col = width_px / 2.0
            geometry.update({"width": width_px, "height": height_px})
        elif shape == "annular":
            radius_outer = float(roi.get("radius", 10))
            radius_inner = float(roi.get("radius_inner", 5))
            half_row = radius_outer
            half_col = radius_outer
            geometry.update({
                "radius": radius_outer,
                "radius_inner": radius_inner,
                "radius_outer": radius_outer,
            })
        else:
            radius = float(roi.get("radius", 10))
            half_row = radius
            half_col = radius
            if shape == "square":
                geometry["half_size"] = radius
                geometry["radius"] = radius
            else:
                geometry["radius"] = radius

        row_min = row - half_row
        row_max = row + half_row
        col_min = col - half_col
        col_max = col + half_col
        geometry["bounds"] = bounds(row_min, row_max, col_min, col_max)
        geometry["bounds_clipped"] = clipped_bounds(row_min, row_max, col_min, col_max)
        if shape in {"rectangle", "square"}:
            geometry["corners"] = box_corners(row_min, row_max, col_min, col_max)
            clipped = geometry["bounds_clipped"]
            geometry["corners_clipped"] = box_corners(
                clipped["row_min"],
                clipped["row_max"],
                clipped["col_min"],
                clipped["col_max"],
            )

        geometries.append(geometry)

    return geometries


def roi_masks(
    roi_list: list[Any],
    *,
    height: int,
    width: int,
    visible_only: bool = True,
) -> np.ndarray:
    """Rasterize ROIs as Boolean masks in image ``(row, col)`` coordinates.

    Parameters
    ----------
    roi_list : list
        Synced Show2D ROI dictionaries.
    height, width : int
        Output image dimensions in pixels.
    visible_only : bool, default=True
        If ``True``, omit ROIs whose synced state has ``visible=False``.

    Returns
    -------
    numpy.ndarray
        Boolean array with shape ``(region, row, col)``. An empty ROI list
        returns an empty leading dimension rather than an all-false region.
    """
    if height < 1 or width < 1:
        raise ValueError(
            f"height and width must be positive, got {(height, width)}."
        )
    rows, columns = np.ogrid[:height, :width]
    masks = []
    for geometry in roi_geometries(
        roi_list,
        height=height,
        width=width,
        visible_only=visible_only,
    ):
        center_row = float(geometry["row"])
        center_column = float(geometry["col"])
        shape = geometry["shape"]
        if shape == "circle":
            radius = float(geometry["radius"])
            mask = (
                (rows - center_row) ** 2 + (columns - center_column) ** 2
                <= radius**2
            )
        elif shape == "annular":
            radius_inner = float(geometry["radius_inner"])
            radius_outer = float(geometry["radius_outer"])
            distance_squared = (
                (rows - center_row) ** 2 + (columns - center_column) ** 2
            )
            mask = (
                (distance_squared >= radius_inner**2)
                & (distance_squared <= radius_outer**2)
            )
        else:
            bounds = geometry["bounds"]
            mask = (
                (rows >= bounds["row_min"])
                & (rows <= bounds["row_max"])
                & (columns >= bounds["col_min"])
                & (columns <= bounds["col_max"])
            )
        masks.append(np.asarray(mask, dtype=bool))
    if not masks:
        return np.zeros((0, height, width), dtype=bool)
    return np.stack(masks)
