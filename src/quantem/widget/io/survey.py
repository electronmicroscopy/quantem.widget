"""Look-before-you-load: survey a folder of 4D-STEM masters without decoding.

``survey(folder)`` reads only HDF5 headers (KB, no pixel data) and reports, per
master: scan shape, detector shape, frame count, raw/bin2/bin4 memory, and chunk
completeness. Then a folder summary: how much it costs to load everything at each
bin level vs the active backend's memory budget (MPS = unified RAM, CUDA = VRAM).

Purpose: on a 24 GB MacBook a user with a folder of 4D-STEM files needs to know
"can I load these, and at what bin?" BEFORE committing seconds + gigabytes — and
to spot incomplete/truncated files before a load fails mid-decode. Backend-blind
arithmetic; only the budget line differs by backend.

Usage::

    from quantem.widget.io import survey
    survey("~/data/session")            # prints a table
    info = survey("~/data/session", show=False)   # returns the dict
"""
from __future__ import annotations

import glob
import os


def _bytes_human(n: int) -> str:
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n}B"


def _backend_budget_gb() -> tuple[str, float]:
    """(backend_name, usable_memory_GB) for the active GPU."""
    try:
        from quantem.widget.kernels import detect
        be = detect()
    except Exception:
        be = "cpu"
    if be == "cuda":
        try:
            import cupy as cp
            free, total = cp.cuda.runtime.memGetInfo()
            return "cuda", total / 1e9
        except Exception:
            return "cuda", 0.0
    if be == "mps":
        # Apple unified memory — total RAM is the shared pool.
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return "mps", int(out.strip()) / 1e9
        except Exception:
            return "mps", 0.0
    return be, 0.0


def _master_chunks(master_path: str) -> list[str]:
    base = master_path[:-len("_master.h5")] if master_path.endswith("_master.h5") else master_path
    return sorted(glob.glob(f"{base}_data_*.h5"))


def _expected_chunk_count(master_path: str) -> int | None:
    """Chunk count the master's virtual dataset references (header-only)."""
    try:
        import h5py
        with h5py.File(master_path, "r") as f:
            return len(f["entry/data"].keys())
    except Exception:
        return None


def _frame_count(chunk_files: list[str]) -> int:
    """Total frames across chunk files (reads each chunk's shape[0], header-only)."""
    import h5py
    total = 0
    for cf in chunk_files:
        try:
            with h5py.File(cf, "r") as f:
                total += int(f["entry/data/data"].shape[0])
        except Exception:
            pass  # truncated/unreadable chunk — completeness flag will catch it
    return total


def survey(folder: str, *, scan_size: int | None = None,
           show: bool = True) -> "SurveyResult":
    """Header-only survey of every ``*_master.h5`` in ``folder``.

    ``scan_size`` (e.g. 512 or 256) keeps only masters whose scan is that NxN
    size, matching the same filter on ``load_4dstem_macbook`` so a folder holding
    both 512 and 256 acquisitions reports just the one you will load.

    Returns a ``SurveyResult`` that renders as a clean table (HTML in Jupyter,
    text in a console) and carries ``.datasets`` / ``.summary`` / ``.df`` for
    programmatic use. Reads no pixel data, costs milliseconds.
    """
    from quantem.widget.io.hdf5 import get_metadata

    folder = os.path.expanduser(folder)
    masters = sorted(glob.glob(os.path.join(folder, "*_master.h5")))
    datasets = []
    for m in masters:
        name = os.path.basename(m)[:-len("_master.h5")]
        row: dict = {"name": name, "path": m}
        try:
            meta = get_metadata(m)
            det = meta.get("detector_shape")
            scan = meta.get("scan_shape")
            if scan_size is not None and tuple(scan or ()) != (int(scan_size), int(scan_size)):
                continue  # mixed folder: skip masters that aren't the requested scan size
            chunk_files = _master_chunks(m)
            expected = _expected_chunk_count(m)
            present = len(chunk_files)
            frames = _frame_count(chunk_files)
            row["scan_shape"] = scan
            row["detector_shape"] = det
            row["frames"] = frames
            row["chunks_present"] = present
            row["chunks_expected"] = expected
            row["complete"] = (expected is not None and present == expected and frames > 0)
            if det and frames:
                det_px = det[0] * det[1]
                row["raw_bytes"] = frames * det_px * 2
                row["bin2_bytes"] = frames * (det[0] // 2) * (det[1] // 2) * 2
                row["bin4_bytes"] = frames * (det[0] // 4) * (det[1] // 4) * 2
        except Exception as exc:
            row["error"] = str(exc)[:80]
            row["complete"] = False
        datasets.append(row)

    backend, budget_gb = _backend_budget_gb()
    complete = [d for d in datasets if d.get("complete")]
    sums = {k: sum(d.get(f"{k}_bytes", 0) for d in complete)
            for k in ("raw", "bin2", "bin4")}
    summary = {
        "folder": folder,
        "n_masters": len(masters),
        "n_complete": len(complete),
        "backend": backend,
        "budget_gb": budget_gb,
        "load_all_raw_gb": sums["raw"] / 1e9,
        "load_all_bin2_gb": sums["bin2"] / 1e9,
        "load_all_bin4_gb": sums["bin4"] / 1e9,
    }
    return SurveyResult(datasets, summary)


def _recommendation(summary: dict) -> str:
    """One-line load recommendation: finest bin level whose all-datasets load
    fits in 70% of the backend budget, else how many fit at bin4."""
    budget = summary["budget_gb"]
    n_ok = summary["n_complete"]
    if not budget or not n_ok:
        return "no complete datasets to load"
    usable = budget * 0.7
    totals = (("no-bin", summary["load_all_raw_gb"]),
              ("bin2", summary["load_all_bin2_gb"]),
              ("bin4", summary["load_all_bin4_gb"]))
    for label, total in totals:
        if total <= usable:
            return (f"load all {n_ok} @ {label} "
                    f"({total:.1f} GB, {total / n_ok:.1f} GB each)")
    per_bin4 = summary["load_all_bin4_gb"] / n_ok
    n_fit = int(usable // per_bin4) if per_bin4 else 0
    return (f"all {n_ok} won't fit even at bin4 — "
            f"load {n_fit} @ bin4 ({n_fit * per_bin4:.1f} GB) or fewer at finer bins")


class SurveyResult:
    """Folder survey that renders as a clean table.

    Jupyter shows a styled HTML table (via pandas) plus a recommendation banner;
    a console shows an aligned text table. Carries ``.datasets`` (per-file dicts),
    ``.summary`` (folder totals + budget), and ``.df`` (the pandas DataFrame) for
    programmatic use, so it works whether printed, returned, or scripted against.
    """

    _COLS = ["file", "scan", "detector", "frames", "no-bin", "bin2", "bin4",
             "chunks", "status"]

    def __init__(self, datasets: list[dict], summary: dict):
        self.datasets = datasets
        self.summary = summary

    def _rows(self) -> list[dict]:
        rows = []
        for d in self.datasets:
            if d.get("error"):
                rows.append({"file": d["name"], "status": f"ERROR: {d['error']}"})
                continue
            scan = f"{d['scan_shape'][0]}×{d['scan_shape'][1]}" if d.get("scan_shape") else "?"
            det = (f"{d['detector_shape'][0]}×{d['detector_shape'][1]}"
                   if d.get("detector_shape") else "?")
            rows.append({
                "file": d["name"],
                "scan": scan,
                "detector": det,
                "frames": f"{d.get('frames', 0):,}",
                "no-bin": _bytes_human(d.get("raw_bytes", 0)),
                "bin2": _bytes_human(d.get("bin2_bytes", 0)),
                "bin4": _bytes_human(d.get("bin4_bytes", 0)),
                "chunks": f"{d.get('chunks_present', 0)}/{d.get('chunks_expected', '?')}",
                "status": "complete" if d.get("complete") else "INCOMPLETE",
            })
        return rows

    @property
    def df(self):
        """pandas DataFrame of the per-file table (optional convenience; imports
        pandas lazily so the survey itself never depends on it)."""
        import pandas as pd
        return pd.DataFrame(self._rows(), columns=self._COLS)

    def _summary_lines(self) -> list[str]:
        s = self.summary
        lines = [
            f"{s['n_complete']}/{s['n_masters']} complete · backend {s['backend']} "
            f"· budget {s['budget_gb']:.0f} GB",
        ]
        for label, gb in (("no-bin", s["load_all_raw_gb"]),
                          ("bin2", s["load_all_bin2_gb"]),
                          ("bin4", s["load_all_bin4_gb"])):
            lines.append(f"  load all @ {label:<7}: {gb:6.1f} GB")
        lines.append(f"RECOMMEND: {_recommendation(s)}")
        return lines

    def __repr__(self) -> str:
        rows = self._rows()
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) if rows
                  else len(c) for c in self._COLS}
        head = "  ".join(c.ljust(widths[c]) for c in self._COLS)
        sep = "  ".join("-" * widths[c] for c in self._COLS)
        body = "\n".join("  ".join(str(r.get(c, "")).ljust(widths[c])
                                   for c in self._COLS) for r in rows)
        return "\n".join([head, sep, body, "", *self._summary_lines()])

    def _repr_html_(self) -> str:
        s = self.summary
        th = "".join(
            f'<th style="text-align:left;padding:4px 10px;'
            f'border-bottom:2px solid #444">{c}</th>' for c in self._COLS)
        trs = []
        for r in self._rows():
            ok = r.get("status") == "complete"
            color = "#2e7d32" if ok else "#c62828"
            tds = []
            for c in self._COLS:
                val = r.get(c, "")
                style = "padding:3px 10px;border-bottom:1px solid #eee"
                if c == "status":
                    style += f";color:{color};font-weight:600"
                tds.append(f'<td style="{style}">{val}</td>')
            trs.append(f"<tr>{''.join(tds)}</tr>")
        table = (f'<table style="border-collapse:collapse;font-size:13px">'
                 f"<thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>")
        rec = _recommendation(s)
        cost = " &nbsp;·&nbsp; ".join(
            f"{label} {gb:.1f} GB" for label, gb in
            (("no-bin", s["load_all_raw_gb"]), ("bin2", s["load_all_bin2_gb"]),
             ("bin4", s["load_all_bin4_gb"])))
        return f"""
<div style="font-family:-apple-system,sans-serif">
  <div style="font-weight:600;margin-bottom:4px">
    {os.path.basename(s['folder']) or s['folder']} —
    {s['n_complete']}/{s['n_masters']} complete ·
    backend {s['backend']} · budget {s['budget_gb']:.0f} GB
  </div>
  {table}
  <div style="margin-top:6px;color:#555">load all: {cost}</div>
  <div style="margin-top:4px;padding:6px 10px;background:#e8f4ea;
              border-left:3px solid #2e7d32;font-weight:600">
    RECOMMEND: {rec}
  </div>
</div>"""

    # Back-compat: old callers did result["datasets"] / result["summary"].
    def __getitem__(self, key):
        return {"datasets": self.datasets, "summary": self.summary}[key]
