"""Show2D's standalone HTML viewer: the widget cloned into an offline model and embedded with the live JS bundle,
so the file opens without a kernel."""

import pathlib
from typing import Self

import numpy as np

from quantem.widget.export import export_slug
from quantem.widget.show2d.state import _rescale_view
from quantem.widget.utils.array import bin2d


class Show2DExport:
    """Mixin: standalone HTML export."""

    _HTML_ENCODINGS = ("full", "uint8")
    _HTML_IMAGE_BINS = (1, 2, 4, 8)

    def _estimate_html_export_mb(self, *, quantized: bool, image_bin: int) -> float:
        """Rough MB an embedded single-file Show2D HTML would occupy.

        ``export_html`` checks this before writing so an oversized page fails
        fast with the uint8 and image_bin alternatives, instead of producing
        a file Chrome cannot open from disk. The pixels are base64 (4/3 of the
        raw bytes) plus about 2 MB of bundle and page.
        """
        has_local_stacks = any(count > 1 for count in self.panel_frame_counts)
        if has_local_stacks:
            n_values = sum(stack.size for stack in self._display_panel_stacks)
        else:
            n_values = self._display_data.size
        # a color panel embeds its three channels next to the luminance plane
        n_values += 3 * sum(frame.size // 3 for frame, rgb in zip(self._display_rgb, self.is_rgb) if rgb and frame is not None)
        n_values //= image_bin**2
        bytes_per_value = 1 if quantized else 4
        payload_mb = n_values * bytes_per_value * (4.0 / 3.0) / (1024 * 1024)
        return payload_mb + 2.0

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        encoding: str = "full",
        image_bin: int | None = None,
        max_mb: float | None = 80.0,
    ) -> pathlib.Path:
        """Write a standalone HTML viewer for this widget.

        The exported file mounts the widget with its current state (data,
        labels, cmap, contrast, log scale, calibration, view) and opens in any
        browser without a kernel or a network.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination HTML path; default ``<title>_<shape>_<exact|quantized>.html``
            in the working directory.
        title : str, optional
            Browser page title. Defaults to widget ``title`` or "Show2D".
        encoding : {"full", "uint8"}, default "full"
            Exact float32 display values, or uint8 per panel over its own range
            (smaller, and the same after colormapping).
        image_bin : {None, 1, 2, 4, 8}, optional
            Mean-bin factor for the embedded pixels; above 1 needs
            ``encoding="uint8"``. The scale bar calibration follows.
        max_mb : float or None, default 80
            Refuse an estimated file size above this; ``None`` disables the check.

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        options = self._html_export_choice(encoding, image_bin)
        self._require_color_export_supported(options["quantized"])
        return self._write_export_and_report(path, title=title, max_mb=max_mb, **options)

    def _require_color_export_supported(self, quantized: bool) -> None:
        """Refuse the export packings a gallery with color panels cannot carry.

        Color panels travel as their true-color float32 frames; the uint8 pack
        quantizes only grayscale planes and would drop the color.
        """
        if quantized and any(self.is_rgb):
            raise ValueError("a gallery with color panels exports only with the full encoding (true color); "
                             "the uint8 encoding holds grayscale panels only")

    def _export_stem(self) -> str:
        """``<title>_<shape>`` file-name stem shared by the SVG and HTML exports, so one figure's files sort together."""
        shape = f"{self.n_images}x{self.height}x{self.width}" if self.n_images > 1 else f"{self.height}x{self.width}"
        return f"{export_slug(self.title, 'show2d')}_{shape}"

    def _default_html_export_path(self, quantized: bool, *, image_bin: int = 1) -> pathlib.Path:
        """``<stem>_<exact|quantized>[_image_binN].html`` in the working directory, naming how the pixels were packed."""
        encoding_label = "quantized" if quantized else "exact"
        suffix = f"_image_bin{image_bin}" if quantized and image_bin > 1 else ""
        return pathlib.Path.cwd() / f"{self._export_stem()}_{encoding_label}{suffix}.html"

    def _clone_for_html_export(self, *, quantized: bool, image_bin: int = 1) -> Self:
        """An export-only Show2D: the displayed pixels mean-binned by ``image_bin``, every view setting, no export menu.

        The standalone page has no kernel behind it, so the clone carries its
        own copy of what is on screen, paints on a light background, and with
        ``quantized`` packs the pixels as uint8 per panel.
        """
        self._require_color_export_supported(quantized)

        def binned_frames(stack: np.ndarray) -> np.ndarray:
            """A ``(F, H, W)`` stack mean-binned frame by frame by ``image_bin``; the pixel size below scales to match."""
            if image_bin <= 1:
                return stack
            return np.stack([bin2d(frame, factor=image_bin, mode="mean") for frame in stack])

        has_local_stacks = any(count > 1 for count in self.panel_frame_counts)
        if has_local_stacks:
            # A one-frame panel goes in as a 2D image so the clone keeps it static.
            export_data = [
                binned_frames(stack) if stack.shape[0] > 1 else binned_frames(stack)[0]
                for stack in self._display_panel_stacks
            ]
        elif any(self.is_rgb):
            # color panels go in as their (H, W, 3) frames so the page keeps true color
            from quantem.widget.show2d.widget import _binned_rgb

            export_data = [
                (_binned_rgb([self._display_rgb[i]], image_bin)[0] if image_bin > 1 else self._display_rgb[i])
                if self.is_rgb[i] else binned_frames(self._display_data[i:i + 1])[0]
                for i in range(self.n_images)
            ]
        else:
            export_data = binned_frames(self._display_data)
        clone = type(self)(export_data, labels=list(self.labels), panel_frame_indices=list(self.panel_frame_indices),
                           display_bin=1, verbose=False)
        # The clone's pixels are this view's display pixels (already binned by display_bin and
        # rotated) binned by image_bin, so its view is this one's in that coarser grid with no
        # rotation left to apply: the standalone page turns the canvas by image_rotations itself.
        state = _rescale_view(self.state_dict(), 1 / (self._display_bin * image_bin))
        state["image_rotations"] = [0] * self.n_images
        clone.load_state_dict(state)
        clone.offline = quantized
        clone._export_light = True
        clone._save_state = True
        clone.export_enabled = False
        clone.export_status = ""
        clone.export_payload = b""
        clone.export_payload_id = ""
        clone.export_filename = ""
        clone._update_all_frames()
        return clone

