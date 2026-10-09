"""Fresh-install end-to-end check for quantem.widget Show4DSTEM.

Run inside a clean environment that has only ``pip install quantem_widget-*.whl``
(no editable source on the path). Proves a new user can install the wheel and
run the documented API on real 4D-STEM data:

    from quantem.gpu.io import load
    from quantem.widget import Show4DSTEM
    Show4DSTEM(load(master))            # one acquisition
    Show4DSTEM(load([m0, m1, m2]))      # comparison of several

Pass the data folder as argv[1] or set WIDGET_E2E_DATA. Prints ALL PASS on
success; any failure raises and exits non-zero.
"""
import glob
import os
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WIDGET_E2E_DATA", "")


def main():
    """Load real masters through the installed wheel and construct every widget; any failure raises."""
    # The install must not resolve to an editable source tree.
    import quantem.widget as widget_package
    package_dir = os.path.dirname(widget_package.__file__)
    widget_package.profile()
    print(f"widget loaded from {package_dir}")
    assert "site-packages" in package_dir, f"not a clean install: {package_dir}"

    from quantem.gpu.device import detect
    from quantem.gpu.io import inspect, load
    from quantem.widget import Show4DSTEM
    backend = detect()
    print(f"backend: {backend}")
    # A CUDA box must never decode on CPU: with an NVIDIA GPU present, a
    # non-CUDA pick means cupy is missing and the install is broken.
    if os.path.exists("/dev/nvidia0") and sys.platform.startswith("linux"):
        assert backend == "cuda", (
            f"NVIDIA GPU present but backend={backend!r}: cupy is missing, so the "
            f"CUDA decode path is not active. Install cupy from conda-forge."
        )

    # Acquisitions still being written (missing linked data files) cannot load.
    masters = [path for path in sorted(glob.glob(f"{DATA}/*master.h5")) if inspect(path).ready]
    assert len(masters) >= 2, f"need at least two ready masters under {DATA}"
    print(f"{len(masters)} ready masters")

    with load(masters[0], verbose=False) as acquisition:
        single = Show4DSTEM(acquisition)
        print(
            f"single: scan={single.shape_rows}x{single.shape_cols} "
            f"det={single.det_rows}x{single.det_cols} "
            f"resident={acquisition.resident_bytes / 2**20:.1f} MiB"
        )
        assert single.shape_rows > 0 and single.det_rows > 0
        single.close()

    acquisitions = load(masters[:3], verbose=False)
    comparison = Show4DSTEM(acquisitions)
    print(f"multi: n_frames={comparison.n_frames} labels={list(comparison.frame_labels)}")
    assert comparison.n_frames == len(acquisitions)
    comparison.close()
    for acquisition in acquisitions:
        acquisition.close()

    # The public surface is exactly the unified API (no legacy name).
    assert not hasattr(widget_package, "load_4dstem_macbook"), "legacy load_4dstem_macbook still exported"

    # Every other shipped widget constructs from the same clean install
    # (synthetic data: a packaging/import smoke, not a render check).
    import numpy as np
    from quantem.widget import Show2D, Show3D, Show3DSlices
    show2d = Show2D(np.random.rand(64, 64), verbose=False)
    show3d = Show3D(np.random.rand(8, 64, 64))
    show3dslices = Show3DSlices(np.random.rand(8, 64, 64))
    print(f"widgets: Show2D={type(show2d).__name__} Show3D={type(show3d).__name__} "
          f"Show3DSlices={type(show3dslices).__name__}")

    print("ALL PASS")


if __name__ == "__main__":
    main()
