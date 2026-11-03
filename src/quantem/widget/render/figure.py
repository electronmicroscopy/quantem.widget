"""Scale-bar label helpers: the one Python port of js/figure.ts.

Every static render (Show2D PNG fallback and save_image, Show3D GIF
frames, Show4DSTEM save_image) labels its scale bar through these
functions so the label is character-identical to the live widget's canvas
label. Keep the thresholds in step with js/figure.ts; the unit table is
``quantem.widget.units``.
"""

import math

from matplotlib import font_manager

from quantem.widget.units import LENGTH_LADDER, unit_entry


def round_to_nice(value: float) -> float:
    """Snap a physical length to 1, 2, 5 or 10 times a power of ten.

    Port of js/figure.ts roundToNiceValue. A scale bar whose length is a
    round number is what readers expect on a figure; the thresholds 1.5, 3.5
    and 7.5 are the geometric midpoints between the candidates.
    """
    if value <= 0:
        return 1.0
    base = 10 ** math.floor(math.log10(value))
    mantissa = value / base
    if mantissa < 1.5:
        return base
    if mantissa < 3.5:
        return 2 * base
    if mantissa < 7.5:
        return 5 * base
    return 10 * base


def format_scale_label(value: float, unit: str) -> str:
    """Scale bar label (port of js/figure.ts formatScaleLabel).

    Length values auto-pick the unit that reads as a clean integer:
    0.5 nm -> "5 Å", 0.005 nm -> "5 pm". Non-length units (mrad, ps, px)
    keep their unit. ``round_to_nice`` gives n*10^k and every ladder step
    is a power of 10, so the rescaled number is always exact. Rounding is
    JS ``Math.round`` (half away from zero); Python's banker's rounding
    would format 2.5 as "2" where the widget shows "3". ``unit`` is read
    through ``quantem.widget.units``: a string outside its table raises.
    """
    nice = round_to_nice(value)
    kind, size, symbol = unit_entry(unit)
    if kind != "length":
        return f"{math.floor(nice + 0.5)} {symbol}" if nice >= 1 else f"{nice:.2f} {symbol}"
    angstroms = nice * size
    # largest ladder unit where the value is >= 1 -> the fewest-digit integer
    for symbol, ladder_angstroms in LENGTH_LADDER:
        if angstroms / ladder_angstroms >= 1:
            return f"{math.floor(angstroms / ladder_angstroms + 0.5)} {symbol}"
    symbol, ladder_angstroms = LENGTH_LADDER[-1]
    return f"{math.floor(angstroms / ladder_angstroms + 0.5)} {symbol}"


OVERLAY_FONT: list[str] | None = None


def static_overlay_font() -> list[str]:
    """Closest installed match to the widget's ``-apple-system, BlinkMacSystemFont,
    'Segoe UI', sans-serif`` stack, resolved once. Filtering to installed fonts
    avoids matplotlib findfont warnings on machines without Helvetica/Arial."""
    global OVERLAY_FONT
    if OVERLAY_FONT is None:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        preferred = ("Helvetica Neue", "Segoe UI", "Arial", "Liberation Sans")
        OVERLAY_FONT = [name for name in preferred if name in installed] + ["DejaVu Sans"]
    return OVERLAY_FONT
