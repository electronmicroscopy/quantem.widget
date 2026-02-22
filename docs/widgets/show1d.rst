Show1D
======

Interactive 1D data viewer for spectra, line profiles, and time series.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import Show1D

   # Single trace
   spectrum = np.random.rand(512)
   Show1D(spectrum, x_label="Energy", x_unit="eV", y_label="Counts")

   # Multiple traces with calibrated X axis
   energy = np.linspace(0, 800, 512).astype(np.float32)
   traces = [np.random.rand(512) for _ in range(3)]
   Show1D(traces, x=energy, labels=["A", "B", "C"])

Features
--------

- **Multi-trace overlay** — Display multiple 1D signals with distinct colors and legend
- **Calibrated axes** — X/Y axis labels and units (e.g. "Energy (eV)", "Counts")
- **Log scale** — Logarithmic Y axis with ``log_scale=True``
- **Grid lines** — Toggle grid with ``show_grid=True``
- **Interactive zoom/pan** — Scroll to zoom, drag to pan, R to reset
- **Crosshair** — Snap-to-nearest cursor readout with trace label
- **Peak detection** — Find peaks with ``find_peaks()``, measure FWHM, export to CSV
- **Export** — Publication PNG (white background) or screenshot PNG
- **Tool lock/hide** — ``disable_*`` / ``hide_*`` API for shared read-only workflows

Methods
-------

.. code-block:: python

   w = Show1D(data, title="Live Plot")

   # Replace data (preserves display settings)
   w.set_data(new_data, x=new_x)

   # Add/remove individual traces
   w.add_trace(trace, label="New")
   w.remove_trace(0)

   # Clear all traces
   w.clear()

   # Peak detection
   w.find_peaks(trace_idx=0, height=0.5, distance=10)
   w.add_peak(x=42.0, trace_idx=0, label="Fe-K")
   w.remove_peak(0)
   w.clear_peaks()
   w.measure_fwhm(peak_idx=0)
   w.export_peaks("peaks.csv")

   # Properties
   w.peaks             # list of detected peaks
   w.selected_peak_data  # data for the selected peak

   # Export
   w.save_image("spectrum.png")       # publication figure
   w.save_image("spectrum.pdf")       # PDF format

Control Groups
--------------

.. code-block:: python

   # Lock groups (visible but non-interactive)
   w = Show1D(
       data,
       disable_display=True,
       disable_peaks=True,
       disable_export=True,
   )

   # Hide groups entirely
   w = Show1D(
       data,
       hide_stats=True,
       hide_peaks=True,
       hide_export=True,
   )

State Persistence
-----------------

.. code-block:: python

   w = Show1D(spectrum, title="EELS", log_scale=True)

   w.summary()          # Print human-readable state
   w.state_dict()       # Get all settings as a dict
   w.save("state.json") # Save versioned envelope JSON file

   # Restore from file or dict
   w2 = Show1D(spectrum, state="state.json")

Examples
--------

- :doc:`Simple demo </examples/show1d/show1d_simple>`
- :doc:`All features </examples/show1d/show1d_all_features>`

API Reference
-------------

See :class:`quantem.widget.Show1D` for full documentation.
