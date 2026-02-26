# Mathematics

How quantem.widget processes and displays data. Each section explains why the operation exists, what the user sees, and where it runs.

| Operation | Runs on | Widgets |
|-----------|---------|---------|
| [Contrast clipping](#contrast-clipping) | JavaScript (browser CPU) | all viewers |
| [Colormap](#colormap) | JavaScript (browser CPU) | all viewers |
| [3D volume rendering](#3d-volume-rendering) | JavaScript (WebGPU) | Show3DVolume |
| [FFT display](#fft-display) | JavaScript (WebGPU, CPU fallback) | Show2D, Show3D, Show4D, Show4DSTEM, Mark2D, ShowComplex2D, Bin |
| [Windowing (Hann)](#edge-artifact-suppression-hann-window) | JavaScript (browser CPU) | all FFT-capable widgets |
| [Windowing (Tukey)](#why-tukey-instead-of-hann) | Python (NumPy / PyTorch GPU) | Align2D, Align2DBulk |
| [D-spacing measurement](#d-spacing-measurement) | JavaScript (browser CPU) | Show2D, Show3D, Show4D, Show4DSTEM, Mark2D, ShowComplex2D, Show3DVolume |
| [Image alignment](#image-alignment) | Python (NumPy / PyTorch GPU) | Align2D, Align2DBulk |
| [Alignment quality (NCC)](#alignment-quality-ncc) | JavaScript (live) + Python (PyTorch GPU) | Align2D, Align2DBulk |
| [Spatial binning](#spatial-binning) | Python (PyTorch GPU, NumPy fallback) | Align2DBulk, Bin |
| [ROI mean time series](#roi-mean-time-series) | Python (PyTorch GPU, NumPy fallback) | Show3D |
| [Virtual imaging](#virtual-imaging) | Python (PyTorch GPU) | Show4DSTEM |

## Rendering pipeline

Every 2D image widget follows the same pipeline from raw data to pixels on screen:

```
Python sends raw float32 array to JavaScript
  → (optional) log scale: x → ln(1 + x)
  → contrast clipping: pick vmin and vmax, clamp pixels to that range
  → normalize to 0-255
  → colormap lookup: 0-255 → RGB color
  → draw to canvas
```

This entire pipeline runs in the browser. Python only sends the raw data - all display processing happens in JavaScript for instant interactivity. When you drag a contrast slider, there is no round-trip to the Python kernel.

## Contrast clipping

**Runs on:** JavaScript (browser CPU) | **Code:** `js/stats.ts`

Some pixels are too bright or too dark. If the colormap stretches across the full range including these extreme values, most of the image looks flat because the interesting detail is squeezed into a tiny sliver of the color range.

Contrast clipping picks a $v_\text{min}$ and $v_\text{max}$ that ignore the extremes. Everything below $v_\text{min}$ becomes black, everything above $v_\text{max}$ becomes white, and the range in between is stretched across the full colormap.

### The histogram and the contrast sliders are two different things

The **histogram** is just a chart. It shows how many pixels have each intensity value. It does not control anything.

The **contrast sliders** are the two draggable handles below the histogram. These are what set $v_\text{min}$ and $v_\text{max}$. Dragging the left handle clips the dark end, dragging the right handle clips the bright end. The slider positions are percentages of the full data range:

$$v_\text{min} = x_\text{min} + \frac{p_\text{low\%}}{100}(x_\text{max} - x_\text{min}), \quad v_\text{max} = x_\text{min} + \frac{p_\text{high\%}}{100}(x_\text{max} - x_\text{min})$$

The histogram is there to help you see where the pixel values are concentrated, so you know where to place the sliders.

### Auto-contrast

When the **Auto** toggle is enabled, the widget automatically places the sliders at the $p_\text{low}$-th and $p_\text{high}$-th percentiles of the data. This overrides manual slider positions.

| Widget | $p_\text{low}$ | $p_\text{high}$ |
|--------|:-:|:-:|
| Show2D, Mark2D, Show1D | 2 | 98 |
| Show3D, ShowComplex2D | 1 | 99 |
| Show4D, Show4DSTEM | 0.5 | 99.5 |

These can be overridden via the `percentile_low` and `percentile_high` constructor parameters.

### How the clipping is applied

Once $v_\text{min}$ and $v_\text{max}$ are set (by either method), each pixel value $x$ is mapped to a 0-255 value for colormap lookup:

1. If $x < v_\text{min}$, force it to $v_\text{min}$ (clips dark outliers)
2. If $x > v_\text{max}$, force it to $v_\text{max}$ (clips bright outliers)
3. Normalize the result to 0-255:

$$v_\text{norm} = \left\lfloor \frac{x_\text{clipped} - v_\text{min}}{v_\text{max} - v_\text{min}} \times 255 \right\rfloor$$

So a pixel exactly at $v_\text{min}$ maps to 0 (black), a pixel at $v_\text{max}$ maps to 255 (white), and everything in between is linearly distributed.

### Example

An image with pixel values from 0 to 50,000. Dragging the contrast sliders to set $v_\text{min} = 100$, $v_\text{max} = 600$:

| Pixel value | After clipping | $v_\text{norm}$ | Display |
|:-:|:-:|:-:|---------|
| 350 | 350 | 127 | mid-gray |
| 600 | 600 | 255 | white |
| 50,000 | clamped to 600 | 255 | white |
| 50 | clamped to 100 | 0 | black |

Without clipping, pixel value 350 would map to $350/50000 \approx 0.007 \times 255 \approx 2$, which is nearly black. Clipping to $[100, 600]$ maps it to 127 (mid-gray), making the detail visible.

## Colormap

**Runs on:** JavaScript (browser CPU) | **Code:** `js/colormaps.ts`

A colormap converts a single intensity number into a color. Every pixel has been normalized to a value between 0 and 255 (by the contrast clipping step above). The colormap is a table of 256 colors. The pixel's normalized value picks a row from the table, and that row gives the RGB color to display.

### Example with the "inferno" colormap

| Normalized value | Color |
|:---:|-------|
| 0 | black |
| 64 | dark purple |
| 128 | orange-red |
| 192 | yellow |
| 255 | white-yellow |

### Example with the "gray" colormap

Value 0 = black, value 255 = white, everything in between is a shade of gray.

Available colormaps: gray, inferno, viridis, plasma, magma, hot, hsv, turbo, RdBu.

## 3D volume rendering

**Runs on:** JavaScript (WebGPU ray-casting) | **Code:** `js/webgpu-volume.ts`

Show3DVolume has two rendering paths: 2D orthogonal slices (same canvas pipeline as other widgets) and a 3D ray-cast volume. The 3D path is different because the data lives in a GPU texture, not a JavaScript array.

### How the volume gets to the GPU

`uploadVolume()` normalizes raw float32 data to a `r8unorm` 3D texture:

$$t = \frac{x - d_\text{min}}{d_\text{max} - d_\text{min}} \in [0, 1]$$

This normalization happens once when the data is loaded. The original data range is baked into the texture.

### How contrast sliders work on the 3D volume

Since the texture already stores normalized values in $[0, 1]$, the slider percentages map directly:

$$v_\text{min} = \frac{p_\text{low\%}}{100}, \quad v_\text{max} = \frac{p_\text{high\%}}{100}$$

The WGSL fragment shader remaps each sampled voxel:

$$I = \text{clamp}\!\left(\frac{t - v_\text{min}}{v_\text{max} - v_\text{min}},\; 0,\; 1\right)$$

This is mathematically equivalent to the 2D path. Sliders at 20%-80% produce the same visual result in both the 2D slices and the 3D volume.

### Why the histogram uses the full volume

Show3DVolume computes the histogram from the entire volume, not the current slice. This keeps the histogram stable as you scrub through slices. The 2D slice widgets (Show2D, Show3D) compute histograms per-frame because each frame may have different content.

## FFT display

**Runs on:** JavaScript (WebGPU when available, CPU fallback) | **Code:** `js/webgpu-fft.ts`

The FFT toggle shows the frequency content of an image. Periodic structures like crystal lattices produce bright spots at specific frequencies, which is useful for identifying lattice spacings, checking for drift, or verifying crystallographic orientation.

### Edge artifact suppression (Hann window)

When an ROI is active, cropping a region creates hard edges at the crop boundary. The FFT treats the data as if it tiles infinitely, so these hard edges create bright cross-shaped artifacts that obscure the real frequency content.

We apply a **Hann window** to the cropped ROI region before computing the FFT. The Hann window smoothly tapers the data to zero at all edges, removing the crop boundary discontinuity. The 1D formula:

$$w(n) = 0.5\left(1 - \cos\frac{2\pi n}{N-1}\right), \quad n = 0, 1, \ldots, N-1$$

The 2D window is the outer product of two 1D windows:

$$W(r, c) = w_\text{row}(r) \cdot w_\text{col}(c)$$

This creates a dome shape - maximum at the center, zero at all edges. After windowing, the cropped region is zero-padded to the next power of 2 for the FFT. Code: `js/webgpu-fft.ts:applyHannWindow2D`.

See the [interactive windowing demo](examples/math/windowing) to try toggling the window on an ROI FFT.

### Auto-enhancement

The raw FFT magnitude has a huge DC component (center pixel) that dominates the display. `autoEnhanceFFT` masks the DC pixel and clips to the 99.9th percentile of the remaining values so the interesting frequency peaks are visible. Code: `js/webgpu-fft.ts:autoEnhanceFFT`.

## D-spacing measurement

**Runs on:** JavaScript (browser CPU) | **Code:** per-widget `findFFTPeak()`

Click on a bright spot in the FFT panel to measure its lattice spacing. The widget converts the click position to a spatial frequency and computes the corresponding real-space distance.

### How it works

1. Click on the FFT panel
2. Snap to the nearest bright peak within 5 pixels (sub-pixel refinement via centroid of a 3x3 window)
3. Convert pixel position to frequency: $f_r = \frac{\text{row} - N/2}{N \cdot \text{pixelSize}}$, $f_c = \frac{\text{col} - N/2}{N \cdot \text{pixelSize}}$
4. Compute spatial frequency: $f = \sqrt{f_r^2 + f_c^2}$
5. D-spacing: $d = 1/f$

The result is displayed as a crosshair overlay with the d-spacing value (e.g., "2.35 A"). Requires `pixel_size` to be set for calibrated values.

## Image alignment

**Runs on:** Python (NumPy CPU for Align2D, PyTorch GPU for Align2DBulk) | **Code:** `align2d.py`, `align2d_bulk.py`

When aligning two images (e.g., consecutive frames in a TEM series), we need to find how much one image is shifted relative to the other. We use FFT phase correlation because it's fast and robust to intensity changes between frames.

### The alignment pipeline

1. **Taper edges** with a Tukey window ($\alpha = 0.2$)
2. **FFT both images** and compute the cross-power spectrum
3. **Find the peak** in the inverse FFT - this gives the integer-pixel shift
4. **Refine to sub-pixel** using the matrix DFT method (Guizar-Sicairos et al. 2008) - samples at 0.01 pixel intervals around the integer peak
5. **Apply the shift** via bilinear interpolation (`torch.nn.functional.grid_sample` on GPU, NumPy fallback on CPU)

In Align2DBulk, steps 2-3 are batched across all frames in a single GPU call. Step 3 is computed per-frame to avoid allocating $N \times H \times W$ complex intermediates.

### Why Tukey instead of Hann

Both windows remove edge artifacts, but they make different trade-offs:

- **Hann** tapers the entire image (dome shape). Good for FFT display, but it weakens the signal everywhere - the correlation peak becomes broader and less accurate.
- **Tukey** ($\alpha = 0.2$) is flat at 1.0 across 80% of the image and only tapers the outer 10% on each side. This preserves the full signal for correlation while still removing the edge discontinuity.

The 1D Tukey window with normalized position $t = n/(N-1) \in [0, 1]$:

$$w(t) = \begin{cases}
0.5\left(1 + \cos\frac{2\pi}{\alpha}\left(t - \frac{\alpha}{2}\right)\right) & \text{if } t < \frac{\alpha}{2} \\
1 & \text{if } \frac{\alpha}{2} \leq t \leq 1 - \frac{\alpha}{2} \\
0.5\left(1 + \cos\frac{2\pi}{\alpha}\left(t - 1 + \frac{\alpha}{2}\right)\right) & \text{if } t > 1 - \frac{\alpha}{2}
\end{cases}$$

With $\alpha = 0.2$: the left taper covers $t \in [0, 0.1]$, the flat region covers $t \in [0.1, 0.9]$, and the right taper covers $t \in [0.9, 1.0]$. The 2D window is the same outer product: $W(r, c) = w(r) \cdot w(c)$.

Code: `align2d.py:_tukey_2d` (NumPy), `align2d_bulk.py:_tukey_2d_torch` (PyTorch GPU).

See the [interactive windowing demo](examples/math/windowing) for a visual comparison of Hann vs Tukey.

## Alignment quality (NCC)

**Runs on:** JavaScript (browser CPU, live during drag) + Python (PyTorch GPU for Align2DBulk) | **Code:** `align2d/index.tsx`, `align2d_bulk.py`

After aligning two images, how do you know if the alignment is good? Normalized cross-correlation (NCC) gives a score from -1 to 1, where 1 means the images match perfectly. This is shown as a percentage in the widget so users can judge alignment quality at a glance.

In Align2D, NCC is computed live in the browser as the user drags the alignment pad. In Align2DBulk, NCC is computed on GPU after alignment, vectorized across all frames.

## Spatial binning

**Runs on:** Python (PyTorch GPU `avg_pool2d` when available, NumPy CPU fallback) | **Code:** `array_utils.py:bin2d`

Binning reduces image size by averaging groups of pixels together. A 2x bin turns every 2x2 block of pixels into one pixel, halving the image dimensions. This improves signal-to-noise ratio (averaging reduces noise) and makes large datasets more manageable. Pixels that don't fill a complete block are trimmed.

## ROI mean time series

**Runs on:** Python (PyTorch GPU when available, NumPy CPU fallback) | **Code:** `show3d.py:_compute_roi_plot`

When you're recording a TEM image series (e.g., in-situ heating, dose series), you want to track how intensity changes over time in a specific region. Draw an ROI on the image, and the widget plots the mean intensity within that ROI for every frame, letting you spot drift, beam damage, or phase transitions at a glance.

The mean is computed for all frames in one vectorized call on GPU: `data[:, mask].mean(dim=1)`.

## Virtual imaging

**Runs on:** Python (PyTorch GPU) | **Code:** `show4dstem.py:_fast_masked_sum`

In 4D-STEM, you record a diffraction pattern at every scan position, producing a 4D dataset $(N_y, N_x, D_y, D_x)$. To form a real-space image, you need to choose *which part* of each diffraction pattern to sum up. This is virtual imaging: place an ROI on the diffraction pattern (e.g., around the bright-field disk, or an annular ring for dark-field), and the widget sums the intensity within that ROI at every scan position to build a 2D image.

Note: this is a **sum**, not a mean. This preserves the total scattered intensity, which is physically meaningful (it corresponds to the number of electrons detected within the ROI).

### Common virtual imaging modes

| Mode | ROI shape | What it shows |
|------|-----------|---------------|
| **BF** (bright field) | Circle at beam center | Direct + low-angle scattered electrons |
| **ABF** (annular bright field) | Annular ring, inner half of BF disk | Light-element sensitive contrast |
| **ADF** (annular dark field) | Annular ring, outside BF disk | Z-contrast (heavier atoms appear brighter) |
| **Custom** | Any ROI shape | User-defined virtual detector |

### Performance optimization

The widget automatically picks the faster path based on how much of the detector the ROI covers:

- **Small ROI** (<20% of detector, e.g., BF disk): only loads the masked pixels via sparse indexing
- **Large ROI** (>=20% of detector, e.g., ADF annulus): uses `torch.tensordot` for full GPU utilization

The 20% threshold is empirically chosen.
