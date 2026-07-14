"""Compatibility exports for STEM movie helpers.

The implementation now lives in :mod:`quantem.gpu.movie` so accelerated export
backends are owned by the backend package.  This module preserves the existing
``quantem.widget.movie`` public import path for one migration window.
"""

from __future__ import annotations

from quantem.gpu.movie import MovieData, save_gif, save_movie, save_mp4

__all__ = ["MovieData", "save_gif", "save_movie", "save_mp4"]
