"""Notebook review surface for microscopy data-transfer plans."""
from __future__ import annotations

import asyncio
from pathlib import Path
import html
import shutil
import time
from typing import Any

from quantem.widget.io import (
    DataTransferPlan,
    copy_data_transfer,
    filter_data_transfer_plan,
    inspect_data_transfer,
    plan_data_transfer,
    read_data_transfer_manifest,
    summarize_data_transfer,
    update_data_transfer_plan,
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
        det_bin: int = 4,
        dtype: str = "u8",
        gpus: int | list[int] | tuple[int, ...] | None = None,
        page_budget: str | int = "auto",
        watch_interval: float = 5.0,
    ) -> None:
        self._source_arg = source
        self._targets_arg = targets
        self._pattern = pattern
        self._recursive = recursive
        self._require_ready = require_ready
        self._strategy = strategy
        self._hash_algorithm = hash_algorithm
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
        self.det_bin = int(det_bin)
        self.dtype = dtype
        self.gpus = self._normalize_gpus(gpus)
        self.page_budget = page_budget
        self.watch_interval = float(watch_interval)
        self._watch_task = None
        self._watch_enabled = False
        self._last_scan_time = 0.0
        self._last_scan_added = 0
        self._last_opened = None
        self._perf: dict[str, Any] = {
            "action": "initialized",
            "seconds": 0.0,
            "files": 0,
            "bytes": 0,
        }
        self._last_results = []
        self.widget = self._build_widget()
        self.refresh()

    @property
    def target_roots(self) -> tuple[Path, ...]:
        """Target roots from the current transfer plan."""
        return tuple(Path(path) for path in self.plan.target_roots)

    @staticmethod
    def _normalize_gpus(gpus) -> list[int] | None:
        if gpus is None or gpus == "":
            return None
        if isinstance(gpus, int):
            return [gpus]
        if isinstance(gpus, str):
            return [int(item.strip()) for item in gpus.split(",") if item.strip()]
        values = [int(item) for item in gpus]
        return values or None

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

    def rescan(self, *, verify: str = "size") -> int:
        """Append newly discovered source groups while preserving assignments."""
        before = {entry.logical_id for entry in self.plan.entries}
        started = time.perf_counter()
        self.plan = update_data_transfer_plan(
            self.plan,
            source=self._source_arg if self._source_arg is not None else Path(self.plan.source_root),
            pattern=self._pattern,
            recursive=self._recursive,
            require_ready=self._require_ready,
            hash_algorithm=self._hash_algorithm,
        )
        after = {entry.logical_id for entry in self.plan.entries}
        added = len(after - before)
        self._last_scan_added = added
        self._last_scan_time = time.time()
        self._record_perf(
            "rescan",
            time.perf_counter() - started,
            files=sum(len(entry.files) for entry in self.plan.entries),
            bytes_=self.plan.total_bytes,
        )
        self._set_message(f"scan found {added} new group{'s' if added != 1 else ''}")
        self.refresh(verify=verify)
        return added

    def start_watch(self, *, interval: float | None = None) -> None:
        """Start a lightweight notebook-side watcher for newly saved masters."""
        self.watch_interval = float(interval or self.watch_interval)
        self._watch_enabled = True
        if self._watch_task is None or self._watch_task.done():
            self._watch_task = asyncio.create_task(self._watch_loop())
        self._set_message(f"watching every {self.watch_interval:.1f}s")
        self.refresh()

    def stop_watch(self) -> None:
        """Stop the notebook-side watcher."""
        self._watch_enabled = False
        if self._watch_task is not None:
            self._watch_task.cancel()
            self._watch_task = None
        self._set_message("watch stopped")
        self.refresh()

    async def _watch_loop(self) -> None:
        while self._watch_enabled:
            await asyncio.sleep(self.watch_interval)
            try:
                self.rescan()
            except Exception as exc:
                self._set_message(str(exc), error=True)
                self._watch_enabled = False
                break

    def copy(
        self,
        *,
        dry_run: bool = False,
        verify: str = "size",
        overwrite: bool = False,
        pending_only: bool = False,
    ):
        """Copy files from the current reviewed plan and refresh the display."""
        started = time.perf_counter()
        plan = (
            filter_data_transfer_plan(
                self.plan,
                statuses=("not-started", "partial"),
                verify=verify,
            )
            if pending_only
            else self.plan
        )
        self._last_results = copy_data_transfer(
            plan,
            dry_run=dry_run,
            verify=verify,
            overwrite=overwrite,
        )
        elapsed = time.perf_counter() - started
        self._record_perf(
            "dry run" if dry_run else "copy",
            elapsed,
            files=len(self._last_results),
            bytes_=sum(result.size_bytes for result in self._last_results),
        )
        scope = "pending " if pending_only else ""
        self._set_message(
            f"{'planned' if dry_run else 'copied'} {len(self._last_results)} {scope}files "
            f"in {elapsed:.2f}s"
        )
        self.refresh(verify=verify)
        return self._last_results

    def copy_pending(
        self,
        *,
        dry_run: bool = False,
        verify: str = "size",
        overwrite: bool = False,
    ):
        """Copy only missing or partial files from the current plan."""
        return self.copy(
            dry_run=dry_run,
            verify=verify,
            overwrite=overwrite,
            pending_only=True,
        )

    def open_showfolder(self):
        """Open transferred target folder(s) with ShowFolder."""
        from quantem.widget import ShowFolder

        started = time.perf_counter()
        roots = self.target_roots
        if len(roots) == 1:
            viewer = ShowFolder(roots[0])
        else:
            viewer = [ShowFolder(root) for root in roots]
        self._last_opened = "ShowFolder"
        self._record_perf("open ShowFolder", time.perf_counter() - started)
        self.refresh()
        return viewer

    def open_show4dstem(
        self,
        *,
        gpus=None,
        page_budget=None,
        det_bin=None,
        dtype=None,
        page_max_vram_fraction=0.75,
        page_reserve_vram_bytes=None,
        page_max_vram_bytes=None,
    ):
        """Open transferred masters as one lazy, paged Show4DSTEM viewer."""
        import torch
        from quantem.widget import Show4DSTEM, load
        from quantem.widget.data import Dataset5dstem

        started = time.perf_counter()
        det_bin = int(self.det_bin if det_bin is None else det_bin)
        dtype = self.dtype if dtype is None else dtype
        page_budget = self.page_budget if page_budget is None else page_budget
        masters = self.target_master_paths(existing_only=True)
        if not masters:
            raise FileNotFoundError("No transferred target masters exist yet.")
        if gpus is None:
            gpus = self.gpus
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
        viewer = Show4DSTEM(
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
        self._last_opened = "Show4DSTEM"
        self._record_perf(
            "open Show4DSTEM",
            time.perf_counter() - started,
            files=len(masters),
            bytes_=sum(path.stat().st_size for path in masters if path.exists()),
        )
        self.refresh()
        return viewer

    def _record_perf(self, action: str, seconds: float, *, files: int = 0, bytes_: int = 0) -> None:
        self._perf = {
            "action": action,
            "seconds": float(seconds),
            "files": int(files),
            "bytes": int(bytes_),
        }

    def refresh(self, *, verify: str = "size") -> "DataTransfer":
        """Refresh the rendered state table."""
        states = self.inspect(verify=verify)
        summary = summarize_data_transfer(states)
        self._summary.value = self._render_summary(states, summary)
        self._backend.value = self._render_backend_panel()
        self._perf_panel.value = self._render_perf_panel()
        self._datasets.value = self._render_dataset_table(states)
        self._table.value = self._render_table(states)
        return self

    def _build_widget(self):
        from ipywidgets import Button, Dropdown, FloatText, HBox, HTML, Output, Text, VBox

        self._summary = HTML()
        self._backend = HTML()
        self._perf_panel = HTML()
        self._datasets = HTML()
        self._table = HTML()
        self._message = HTML()
        self._output = Output()
        self._det_bin_control = Dropdown(
            description="Det bin",
            options=[("1 - native", 1), ("2", 2), ("4", 4), ("8", 8), ("16", 16)],
            value=self.det_bin,
            layout={"width": "150px"},
        )
        self._dtype_control = Dropdown(
            description="Dtype",
            options=[("U8 fast", "u8"), ("U16", "u16"), ("float32", "float32")],
            value=self.dtype,
            layout={"width": "150px"},
        )
        self._gpus_control = Text(
            description="GPUs",
            value="" if self.gpus is None else ",".join(str(gpu) for gpu in self.gpus),
            placeholder="auto or 0,1",
            layout={"width": "180px"},
        )
        self._page_budget_control = Text(
            description="Pages",
            value=str(self.page_budget),
            placeholder="auto or number",
            layout={"width": "170px"},
        )
        self._watch_interval_control = FloatText(
            description="Watch s",
            value=self.watch_interval,
            layout={"width": "150px"},
        )
        refresh = Button(description="Refresh", tooltip="Inspect source/target state")
        rescan = Button(description="Scan new", tooltip="Append newly arrived master groups")
        watch = Button(description="Start watch", tooltip="Poll for newly arrived master groups")
        stop_watch = Button(description="Stop watch", tooltip="Stop folder polling")
        dry_run = Button(description="Dry run pending", tooltip="Show pending copy rows")
        copy = Button(description="Copy pending", button_style="warning", tooltip="Copy missing or partial files")
        dry_run_all = Button(description="Dry run all", tooltip="Show every planned copy row")
        open_folder = Button(description="Open ShowFolder", tooltip="Open transferred targets")
        open_4d = Button(description="Open Show4DSTEM", button_style="info", tooltip="Open lazy/paged 4D-STEM viewer")

        def _refresh(_=None):
            self._sync_loader_controls()
            self.refresh()

        def _rescan(_=None):
            self._sync_loader_controls()
            try:
                self.rescan()
            except Exception as exc:
                self._set_message(str(exc), error=True)

        def _watch(_=None):
            self._sync_loader_controls()
            try:
                self.start_watch(interval=self._watch_interval_control.value)
            except Exception as exc:
                self._set_message(str(exc), error=True)

        def _stop_watch(_=None):
            self.stop_watch()

        def _dry_run(_=None):
            self._sync_loader_controls()
            self.copy_pending(dry_run=True)

        def _copy(_=None):
            self._sync_loader_controls()
            try:
                self.copy_pending(dry_run=False)
            except Exception as exc:
                self._set_message(str(exc), error=True)

        def _dry_run_all(_=None):
            self._sync_loader_controls()
            self.copy(dry_run=True)

        def _open_folder(_=None):
            try:
                with self._output:
                    from IPython.display import display

                    display(self.open_showfolder())
            except Exception as exc:
                self._set_message(str(exc), error=True)

        def _open_4d(_=None):
            self._sync_loader_controls()
            try:
                with self._output:
                    from IPython.display import display

                    display(self.open_show4dstem())
            except Exception as exc:
                self._set_message(str(exc), error=True)

        refresh.on_click(_refresh)
        rescan.on_click(_rescan)
        watch.on_click(_watch)
        stop_watch.on_click(_stop_watch)
        dry_run.on_click(_dry_run)
        copy.on_click(_copy)
        dry_run_all.on_click(_dry_run_all)
        open_folder.on_click(_open_folder)
        open_4d.on_click(_open_4d)
        return VBox([
            self._summary,
            HBox([
                self._det_bin_control,
                self._dtype_control,
                self._gpus_control,
                self._page_budget_control,
                self._watch_interval_control,
            ]),
            HBox([refresh, rescan, watch, stop_watch, dry_run, copy, dry_run_all]),
            HBox([open_folder, open_4d]),
            self._message,
            self._backend,
            self._perf_panel,
            self._datasets,
            self._table,
            self._output,
        ])

    def _sync_loader_controls(self) -> None:
        self.det_bin = int(self._det_bin_control.value)
        self.dtype = str(self._dtype_control.value)
        self.gpus = self._normalize_gpus(self._gpus_control.value)
        page_budget = self._page_budget_control.value.strip()
        self.page_budget = int(page_budget) if page_budget.isdigit() else page_budget or "auto"
        self.watch_interval = float(self._watch_interval_control.value)

    def _render_summary(self, states, summary) -> str:
        complete_pct = (
            100.0 * summary.complete_bytes / summary.total_bytes
            if summary.total_bytes else 100.0
        )
        target_rows = self._render_target_cards(states)
        problem_text = (
            f"{summary.problem_files} problem files"
            if summary.problem_files else "no problems"
        )
        return (
            "<div style='font-family:system-ui,-apple-system,Segoe UI,sans-serif;"
            "font-size:13px;line-height:1.35;margin:4px 0 10px 0;max-width:980px'>"
            "<div style='display:flex;gap:10px;align-items:center;flex-wrap:wrap'>"
            f"<b>{html.escape(self.title)}</b>"
            f"<span>{len(self.plan.entries)} groups</span>"
            f"<span>{summary.total_files} files</span>"
            f"<span>{_format_bytes(summary.complete_bytes)} / "
            f"{_format_bytes(summary.total_bytes)}</span>"
            f"<span>{html.escape(problem_text)}</span>"
            "</div>"
            "<div style='height:8px;background:#e5e7eb;border-radius:999px;"
            "overflow:hidden;margin:7px 0 8px 0'>"
            f"<div style='height:100%;width:{complete_pct:.1f}%;background:#2563eb'></div>"
            "</div>"
            f"{target_rows}"
            "<div style='color:#475569;margin-top:6px'>"
            "Review target balance, dry-run pending files, copy pending files, then "
            "open transferred data with <code>open_showfolder()</code> or "
            "<code>open_show4dstem(gpus=[...])</code>. The manifest is the durable state."
            "</div>"
            "</div>"
        )

    def _render_target_cards(self, states) -> str:
        target_map: dict[str, dict[str, int]] = {
            str(root): {"total": 0, "exists": 0, "pending": 0, "problem": 0}
            for root in self.plan.target_roots
        }
        for state in states:
            target = Path(state.target)
            matching_roots = [
                root
                for root in target_map
                if target == Path(root) or Path(root) in target.parents
            ]
            root_key = max(
                matching_roots,
                key=lambda root: len(Path(root).parts),
                default=None,
            )
            if root_key is None:
                continue
            row = target_map[root_key]
            row["total"] += state.size_bytes
            if state.status == "exists":
                row["exists"] += state.size_bytes
            elif state.status in ("mismatch", "missing-source"):
                row["problem"] += state.size_bytes
            else:
                row["pending"] += state.size_bytes
        cards = []
        for idx, (root, values) in enumerate(target_map.items(), start=1):
            total = values["total"]
            pct = 100.0 * values["exists"] / total if total else 100.0
            cards.append(
                "<div style='border:1px solid #cbd5e1;border-radius:6px;"
                "padding:6px 8px;min-width:210px;background:#f8fafc'>"
                f"<div><b>Target {idx}</b> <code>{html.escape(Path(root).name or root)}</code></div>"
                f"<div>{_format_bytes(values['exists'])} complete / {_format_bytes(total)}</div>"
                f"<div>{_format_bytes(values['pending'])} pending"
                + (
                    f" · {_format_bytes(values['problem'])} problem"
                    if values["problem"] else ""
                )
                + "</div>"
                "<div style='height:5px;background:#e5e7eb;border-radius:999px;"
                "overflow:hidden;margin-top:4px'>"
                f"<div style='height:100%;width:{pct:.1f}%;background:#0f766e'></div>"
                "</div></div>"
            )
        return "<div style='display:flex;gap:8px;flex-wrap:wrap'>" + "".join(cards) + "</div>"

    def _render_backend_panel(self) -> str:
        disk_cards = []
        for idx, root in enumerate(self.target_roots, start=1):
            probe = root
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            try:
                usage = shutil.disk_usage(probe)
                free = _format_bytes(usage.free)
                total = _format_bytes(usage.total)
            except OSError:
                free = "unknown"
                total = "unknown"
            disk_cards.append(
                "<div style='border:1px solid #dbe3ef;border-radius:6px;padding:6px 8px;"
                "background:#fff;min-width:230px'>"
                f"<b>Disk {idx}</b> <code>{html.escape(root.name or str(root))}</code><br>"
                f"<span>free {free} / {total}</span>"
                "</div>"
            )

        gpu_cards = []
        try:
            import torch

            if torch.cuda.is_available():
                selected = set(self.gpus or range(torch.cuda.device_count()))
                for idx in range(torch.cuda.device_count()):
                    try:
                        free, total = torch.cuda.mem_get_info(idx)
                        memory = f"{_format_bytes(free)} free / {_format_bytes(total)}"
                    except Exception:
                        memory = "memory unknown"
                    marker = "selected" if idx in selected else "available"
                    gpu_cards.append(
                        "<div style='border:1px solid #dbe3ef;border-radius:6px;padding:6px 8px;"
                        "background:#fff;min-width:260px'>"
                        f"<b>GPU {idx}</b> <span>{html.escape(marker)}</span><br>"
                        f"<span>{html.escape(torch.cuda.get_device_name(idx))}</span><br>"
                        f"<span>{memory}</span>"
                        "</div>"
                    )
            else:
                gpu_cards.append("<div style='color:#64748b'>CUDA is not available in this Python process.</div>")
        except Exception as exc:
            gpu_cards.append(f"<div style='color:#64748b'>GPU status unavailable: {html.escape(str(exc))}</div>")

        return (
            "<div style='font-family:system-ui,-apple-system,Segoe UI,sans-serif;"
            "font-size:12px;margin:8px 0;max-width:980px'>"
            "<div style='font-weight:700;margin-bottom:4px'>Backend</div>"
            "<div style='display:flex;gap:8px;flex-wrap:wrap'>"
            + "".join(disk_cards + gpu_cards)
            + "</div></div>"
        )

    def _render_perf_panel(self) -> str:
        seconds = float(self._perf.get("seconds", 0.0))
        bytes_ = int(self._perf.get("bytes", 0))
        rate = _format_bytes(int(bytes_ / seconds)) + "/s" if seconds > 0 and bytes_ else "n/a"
        watch = "on" if self._watch_enabled else "off"
        last_scan = (
            time.strftime("%H:%M:%S", time.localtime(self._last_scan_time))
            if self._last_scan_time else "not yet"
        )
        return (
            "<div style='font-family:system-ui,-apple-system,Segoe UI,sans-serif;"
            "font-size:12px;margin:8px 0;max-width:980px'>"
            "<div style='display:flex;gap:10px;flex-wrap:wrap;background:#f8fafc;"
            "border:1px solid #e2e8f0;border-radius:6px;padding:6px 8px'>"
            f"<span><b>Last action</b> {html.escape(str(self._perf.get('action', 'n/a')))}</span>"
            f"<span><b>Time</b> {seconds:.3f}s</span>"
            f"<span><b>Files</b> {int(self._perf.get('files', 0))}</span>"
            f"<span><b>Bytes</b> {_format_bytes(bytes_)}</span>"
            f"<span><b>Rate</b> {rate}</span>"
            f"<span><b>Watch</b> {watch}</span>"
            f"<span><b>Last scan</b> {last_scan}</span>"
            f"<span><b>New groups</b> {self._last_scan_added}</span>"
            f"<span><b>Loader</b> det_bin={self.det_bin}, dtype={html.escape(self.dtype)}, "
            f"gpus={html.escape('auto' if self.gpus is None else ','.join(map(str, self.gpus)))}</span>"
            "</div></div>"
        )

    def _render_dataset_table(self, states) -> str:
        states_by_target = {state.target: state for state in states}
        rows = []
        for entry in self.plan.entries[:120]:
            file_states = [
                states_by_target[target].status
                for target in entry.target_files
                if target in states_by_target
            ]
            if any(status in ("mismatch", "missing-source") for status in file_states):
                status = "problem"
                color = "#b91c1c"
            elif any(status == "partial" for status in file_states):
                status = "partial"
                color = "#b45309"
            elif file_states and all(status == "exists" for status in file_states):
                status = "ready"
                color = "#0f766e"
            else:
                status = "pending"
                color = "#555"
            ready = "yes" if Path(entry.target_master).exists() else "no"
            rows.append(
                "<tr>"
                f"<td style='padding:3px 8px;border-bottom:1px solid #e5e7eb'>{html.escape(entry.logical_id)}</td>"
                f"<td style='padding:3px 8px;border-bottom:1px solid #e5e7eb'>{len(entry.files)}</td>"
                f"<td style='padding:3px 8px;border-bottom:1px solid #e5e7eb;text-align:right'>{_format_bytes(entry.size_bytes)}</td>"
                f"<td style='padding:3px 8px;border-bottom:1px solid #e5e7eb'><code>{html.escape(Path(entry.target_root).name or entry.target_root)}</code></td>"
                f"<td style='padding:3px 8px;border-bottom:1px solid #e5e7eb;color:{color};font-weight:600'>{status}</td>"
                f"<td style='padding:3px 8px;border-bottom:1px solid #e5e7eb'>{ready}</td>"
                "</tr>"
            )
        more = ""
        if len(self.plan.entries) > 120:
            more = f"<div style='color:#64748b;margin-top:4px'>Showing first 120 of {len(self.plan.entries)} datasets.</div>"
        return (
            "<div style='font-family:system-ui,-apple-system,Segoe UI,sans-serif;"
            "font-size:12px;margin:8px 0;max-width:980px'>"
            "<div style='font-weight:700;margin-bottom:4px'>Datasets</div>"
            "<table style='border-collapse:collapse;font-size:12px'>"
            "<thead><tr>"
            "<th style='text-align:left;padding:3px 8px'>Dataset</th>"
            "<th style='text-align:left;padding:3px 8px'>Files</th>"
            "<th style='text-align:right;padding:3px 8px'>Size</th>"
            "<th style='text-align:left;padding:3px 8px'>Target</th>"
            "<th style='text-align:left;padding:3px 8px'>Status</th>"
            "<th style='text-align:left;padding:3px 8px'>Show4DSTEM</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            + more
            + "</div>"
        )

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
                f"<td style='padding:3px 8px;border-bottom:1px solid #e5e7eb'>{html.escape(state.logical_id)}</td>"
                f"<td style='padding:3px 8px;border-bottom:1px solid #e5e7eb'><code>{html.escape(Path(state.source).name)}</code></td>"
                f"<td style='padding:3px 8px;border-bottom:1px solid #e5e7eb;"
                f"max-width:560px;word-break:break-all'><code>{html.escape(str(Path(state.target)))}</code></td>"
                f"<td style='padding:3px 8px;border-bottom:1px solid #e5e7eb;text-align:right'>{html.escape(_format_bytes(state.size_bytes))}</td>"
                f"<td style='padding:3px 8px;border-bottom:1px solid #e5e7eb;color:{color};font-weight:600'>{html.escape(state.status)}</td>"
                "</tr>"
            )
        more = ""
        if len(states) > max_rows:
            more = f"<div style='color:#666;margin-top:4px'>Showing first {max_rows} of {len(states)} files.</div>"
        return (
            "<table style='border-collapse:collapse;font-size:12px;max-width:100%;"
            "font-family:ui-monospace,SFMono-Regular,Menlo,monospace'>"
            "<thead><tr>"
            "<th style='text-align:left;padding:3px 8px'>Logical ID</th>"
            "<th style='text-align:left;padding:3px 8px'>Source</th>"
            "<th style='text-align:left;padding:3px 8px'>Target</th>"
            "<th style='text-align:right;padding:3px 8px'>Size</th>"
            "<th style='text-align:left;padding:3px 8px'>Status</th>"
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
