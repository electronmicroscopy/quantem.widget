"""Notebook review surface for microscopy data-transfer plans."""
from __future__ import annotations

from pathlib import Path
import html
import time
from typing import Any

from quantem.widget.io import (
    DataTransferPlan,
    copy_data_transfer,
    inspect_data_transfer,
    plan_data_transfer,
    read_data_transfer_manifest,
    summarize_data_transfer,
    write_data_transfer_manifest,
)


def _format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if abs(size) < 1000 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1000.0
    return f"{size:.1f} TB"


class DataTransfer:
    """Plan, inspect, and execute a file transfer from a notebook.

    This widget is intentionally a thin review/control layer. The durable state
    is the :class:`~quantem.widget.io.data_transfer.DataTransferPlan` manifest;
    the widget only renders that plan and calls the shared core utilities.
    """

    def __init__(
        self,
        source: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
        targets: list[str | Path] | tuple[str | Path, ...] | None = None,
        *,
        plan: DataTransferPlan | None = None,
        manifest: str | Path | None = None,
        pattern: str = "*_master.h5",
        recursive: bool = True,
        require_ready: bool = False,
        strategy: str = "balance-by-size",
        hash_algorithm: str | None = None,
        title: str | None = None,
    ) -> None:
        if plan is not None:
            self.plan = plan
        elif manifest is not None:
            self.plan = read_data_transfer_manifest(manifest)
        elif source is not None and targets:
            self.plan = plan_data_transfer(
                source,
                targets,
                pattern=pattern,
                recursive=recursive,
                require_ready=require_ready,
                strategy=strategy,
                hash_algorithm=hash_algorithm,
            )
        else:
            raise ValueError("Provide plan=, manifest=, or source= plus targets=.")
        self.title = title or f"DataTransfer: {self.plan.logical_name}"
        self._last_results = []
        self.widget = self._build_widget()
        self.refresh()

    @property
    def target_roots(self) -> tuple[Path, ...]:
        """Target roots from the current transfer plan."""
        return tuple(Path(path) for path in self.plan.target_roots)

    def target_master_paths(self, *, existing_only: bool = True) -> list[Path]:
        """Return planned target master paths for ShowFolder/Show4DSTEM handoff."""
        masters = [Path(entry.target_master) for entry in self.plan.entries]
        if existing_only:
            masters = [path for path in masters if path.exists()]
        return masters

    def inspect(self, *, verify: str = "size"):
        """Inspect current source/target state."""
        return inspect_data_transfer(self.plan, verify=verify)

    def summary(self, *, verify: str = "size"):
        """Return aggregate transfer state."""
        return summarize_data_transfer(self.inspect(verify=verify))

    def write_manifest(self, path: str | Path) -> Path:
        """Write the current plan manifest."""
        return write_data_transfer_manifest(self.plan, path)

    def copy(
        self,
        *,
        dry_run: bool = False,
        verify: str = "size",
        overwrite: bool = False,
    ):
        """Copy files from the current reviewed plan and refresh the display."""
        started = time.perf_counter()
        self._last_results = copy_data_transfer(
            self.plan,
            dry_run=dry_run,
            verify=verify,
            overwrite=overwrite,
        )
        self._set_message(
            f"{'planned' if dry_run else 'copied'} {len(self._last_results)} files "
            f"in {time.perf_counter() - started:.2f}s"
        )
        self.refresh(verify=verify)
        return self._last_results

    def open_showfolder(self):
        """Open transferred target folder(s) with ShowFolder."""
        from quantem.widget import ShowFolder

        roots = self.target_roots
        if len(roots) == 1:
            return ShowFolder(roots[0])
        return [ShowFolder(root) for root in roots]

    def open_show4dstem(
        self,
        *,
        gpus=None,
        page_budget="auto",
        det_bin=4,
        dtype="u8",
        page_max_vram_fraction=0.75,
        page_reserve_vram_bytes=None,
        page_max_vram_bytes=None,
    ):
        """Open transferred masters as one lazy, paged Show4DSTEM viewer."""
        import torch
        from quantem.widget import Show4DSTEM, load
        from quantem.widget.data import Dataset5dstem

        masters = self.target_master_paths(existing_only=True)
        if not masters:
            raise FileNotFoundError("No transferred target masters exist yet.")
        if isinstance(gpus, int):
            gpus = [gpus]
        elif gpus is not None:
            gpus = list(gpus)
            if not gpus:
                raise ValueError("gpus must be None, an int, or a non-empty sequence.")

        def load_master(master: Path, idx: int) -> torch.Tensor:
            result = load(str(master), det_bin=det_bin, dtype=dtype, verbose=False)
            data = result.data
            tensor = data if isinstance(data, torch.Tensor) else torch.from_dlpack(data)
            if gpus is not None:
                tensor = tensor.to(f"cuda:{gpus[idx % len(gpus)]}")
            return tensor

        first_tensor = load_master(masters[0], 0)
        loaders = [
            (lambda master=master, idx=idx: load_master(master, idx))
            for idx, master in enumerate(masters)
        ]
        dataset = Dataset5dstem.from_lazy_loaders(
            loaders,
            shape=(len(loaders), *tuple(first_tensor.shape)),
            dtype=first_tensor.dtype,
            initial_frames={0: first_tensor},
            name="DataTransfer lazy 4D-STEM masters",
        )
        labels = [
            path.name[: -len("_master.h5")] if path.name.endswith("_master.h5") else path.stem
            for path in masters
        ]
        return Show4DSTEM(
            dataset,
            page_budget=page_budget,
            page_device=gpus if gpus is not None else None,
            page_max_vram_fraction=page_max_vram_fraction,
            page_reserve_vram_bytes=page_reserve_vram_bytes,
            page_max_vram_bytes=page_max_vram_bytes,
            frame_dim_label="Dataset",
            frame_labels=labels,
            verbose=False,
        )

    def refresh(self, *, verify: str = "size") -> "DataTransfer":
        """Refresh the rendered state table."""
        states = self.inspect(verify=verify)
        summary = summarize_data_transfer(states)
        self._summary.value = (
            "<div style='font-size:13px;margin:4px 0 8px 0'>"
            f"<b>{html.escape(self.title)}</b> · "
            f"{len(self.plan.entries)} groups · "
            f"{summary.total_files} files · "
            f"{_format_bytes(summary.complete_bytes)} complete / "
            f"{_format_bytes(summary.total_bytes)} total · "
            f"{summary.problem_files} problem files"
            "</div>"
        )
        self._table.value = self._render_table(states)
        return self

    def _build_widget(self):
        from ipywidgets import Button, HBox, HTML, Output, VBox

        self._summary = HTML()
        self._table = HTML()
        self._message = HTML()
        self._output = Output()
        refresh = Button(description="Refresh", tooltip="Inspect source/target state")
        dry_run = Button(description="Dry run", tooltip="Show planned copy rows")
        copy = Button(description="Copy", button_style="warning", tooltip="Copy missing files")

        def _refresh(_=None):
            self.refresh()

        def _dry_run(_=None):
            self.copy(dry_run=True)

        def _copy(_=None):
            try:
                self.copy(dry_run=False)
            except Exception as exc:
                self._set_message(str(exc), error=True)

        refresh.on_click(_refresh)
        dry_run.on_click(_dry_run)
        copy.on_click(_copy)
        return VBox([self._summary, HBox([refresh, dry_run, copy]), self._message, self._table, self._output])

    def _render_table(self, states) -> str:
        rows = []
        max_rows = 200
        for state in states[:max_rows]:
            color = {
                "exists": "#0f766e",
                "not-started": "#555",
                "partial": "#b45309",
                "mismatch": "#b91c1c",
                "missing-source": "#b91c1c",
            }.get(state.status, "#555")
            rows.append(
                "<tr>"
                f"<td>{html.escape(state.logical_id)}</td>"
                f"<td><code>{html.escape(Path(state.source).name)}</code></td>"
                f"<td><code>{html.escape(str(Path(state.target)))}</code></td>"
                f"<td style='text-align:right'>{html.escape(_format_bytes(state.size_bytes))}</td>"
                f"<td style='color:{color};font-weight:600'>{html.escape(state.status)}</td>"
                "</tr>"
            )
        more = ""
        if len(states) > max_rows:
            more = f"<div style='color:#666;margin-top:4px'>Showing first {max_rows} of {len(states)} files.</div>"
        return (
            "<table style='border-collapse:collapse;font-size:12px'>"
            "<thead><tr>"
            "<th style='text-align:left'>Logical ID</th>"
            "<th style='text-align:left'>Source</th>"
            "<th style='text-align:left'>Target</th>"
            "<th style='text-align:right'>Size</th>"
            "<th style='text-align:left'>Status</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            + more
        )

    def _set_message(self, message: str, *, error: bool = False) -> None:
        color = "#b91c1c" if error else "#555"
        self._message.value = (
            f"<div style='font-size:12px;color:{color};margin:4px 0'>"
            f"{html.escape(message)}</div>"
        )

    def _repr_mimebundle_(self, **kwargs):
        return self.widget._repr_mimebundle_(**kwargs)

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.widget)
