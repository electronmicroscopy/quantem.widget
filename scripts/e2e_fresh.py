"""Fresh-install end-to-end check for quantem.widget Show4DSTEM.

Run inside a CLEAN conda env that has ONLY `pip install quantem_widget-*.whl` (no
editable source on the path). Proves a brand-new user can install the wheel and run
the documented API on real 4D-STEM data:

    from quantem.widget import load, Show4DSTEM
    Show4DSTEM(load(master, det_bin=4))            # single
    Show4DSTEM(load([m0, m1, m2], det_bin=4))      # many

Pass the data dir as argv[1] (default data path (set WIDGET_E2E_DATA)). Prints ALL PASS on success;
any failure raises and exits non-zero.
"""
import glob
import os
import os
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WIDGET_E2E_DATA", "")  # pass a path arg or set WIDGET_E2E_DATA


def main():
    # the install must NOT resolve to an editable source tree
    import quantem.widget as w
    src = os.path.dirname(w.__file__)
    print(f"quantem.widget {w.__version__} from {src}")
    assert "site-packages" in src, f"not a clean install: {src}"

    from quantem.widget import load, Show4DSTEM
    from quantem.widget.io import detect_backend
    backend = detect_backend()
    print(f"backend: {backend}")
    # A CUDA box must NEVER decode on CPU. If an NVIDIA GPU is present, the chosen
    # backend has to be cuda (cupy installed) — a cpu pick here is a broken install.
    if os.path.exists("/dev/nvidia0") and sys.platform.startswith("linux"):
        assert backend == "cuda", (
            f"NVIDIA GPU present but backend={backend!r} — cupy missing, the CUDA "
            f"decode path is not active. Install cupy from conda-forge."
        )

    masters = sorted(glob.glob(f"{DATA}/*master.h5"))
    assert masters, f"no masters under {DATA}"
    print(f"{len(masters)} masters")

    # single
    v1 = Show4DSTEM(load(masters[0], det_bin=4, verbose=False), verbose=False)
    print(f"single: {type(v1).__name__} scan={v1._scan_shape} det={v1._det_shape}")
    assert v1._scan_shape[0] > 0 and v1._det_shape[0] > 0

    # multi (>=2 datasets)
    sub = masters[:3]
    v2 = Show4DSTEM(load(sub, det_bin=4, verbose=False), verbose=False)
    print(f"multi: {type(v2).__name__} n_frames={v2.n_frames} frame_dim={v2.frame_dim_label}")
    assert v2.n_frames >= 1

    # the public surface is exactly the unified API (no legacy name)
    assert not hasattr(w, "load_4dstem_macbook"), "legacy load_4dstem_macbook still exported"

    # every other shipped widget constructs from the same clean install (synthetic
    # data — this is a packaging/import smoke, not a render check).
    import numpy as np
    from quantem.widget import Show2D, Show3D, Show3DSlices
    s2 = Show2D(np.random.rand(64, 64), verbose=False)
    s3 = Show3D(np.random.rand(8, 64, 64))
    s3s = Show3DSlices(np.random.rand(8, 64, 64))
    print(f"widgets: Show2D={type(s2).__name__} Show3D={type(s3).__name__} "
          f"Show3DSlices={type(s3s).__name__}")

    print("ALL PASS")


if __name__ == "__main__":
    main()
