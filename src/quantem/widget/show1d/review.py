"""Trial review behind the Show1D Review panel: ranking rows, alerts, best trial.

A reconstruction sweep is one trace per trial (``lambda 1``, ``lambda 10``,
...). The browser's Review panel ranks those trials and flags problems; the
rows and alerts are computed here from the traces, the snapshot images and
the per-trial metrics a monitor file carries, then synced as plain JSON.
"""

import math
import pathlib
from collections.abc import Mapping, Sequence

import numpy as np

from quantem.widget.utils.array import to_numpy

REVIEW_MODES = ("trace", "optimization")
TRIAL_SORT_KEYS = (
    "default",
    "label",
    "lambda",
    "final_loss",
    "min_loss",
    "rmse",
    "flicker",
    "object_quality",
    "probe_quality",
    "alert_count",
)
_RMSE_KEYS = ("rmse", "rmse_per_frame_mask", "rmse_time_average_mask", "reference_rmse")
_FLICKER_KEYS = ("flicker", "temporal_flicker", "temporal_flicker_mask", "mean_phase_std_mask")


def as_float(value) -> float:
    """One float from a Python, NumPy or torch scalar; NaN for missing or non-numeric values."""
    if value is None:
        return math.nan
    try:
        return float(np.asarray(to_numpy(value), dtype=np.float64).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return math.nan


def json_safe(value):
    """Nested value as strict JSON: NumPy scalars unwrapped, non-finite floats as null, paths as text."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def trial_label_key(label: str) -> str:
    """Case- and punctuation-insensitive key so ``lambda_10``, ``lambda 10`` and ``Lambda-10`` are one trial."""
    return "".join(ch.lower() for ch in str(label) if ch.isalnum())


def normalise_trial_labels(labels: Sequence[str]) -> list[str]:
    """Stripped labels, first spelling kept per trial key, blanks dropped."""
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        text = str(label).strip()
        key = trial_label_key(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def label_in_collection(label: str, values: Sequence[str]) -> bool:
    """Whether ``label`` names the same trial as any of ``values`` (trial-key match)."""
    key = trial_label_key(label)
    return any(trial_label_key(value) == key for value in values)


def lookup_by_trial_key(mapping: Mapping[str, object], label: str):
    """The value stored under any spelling of ``label``'s trial, or None, so notes and metrics follow a renamed spelling."""
    key = trial_label_key(label)
    for raw_label, value in mapping.items():
        if trial_label_key(str(raw_label)) == key:
            return value
    return None


def normalise_trial_notes(notes: Mapping[str, str]) -> dict[str, str]:
    """One stripped note per trial key; empty notes delete the entry."""
    out: dict[str, str] = {}
    seen: set[str] = set()
    for label, note in notes.items():
        clean_label = str(label).strip()
        key = trial_label_key(clean_label)
        text = str(note).strip()
        if key and key not in seen and text:
            seen.add(key)
            out[clean_label] = text
    return out


def normalise_trial_tags(tags: Mapping[str, Sequence[str] | str]) -> dict[str, list[str]]:
    """Unique stripped tags per trial key; trials without tags are dropped."""
    out: dict[str, list[str]] = {}
    seen_labels: set[str] = set()
    for label, values in tags.items():
        clean_label = str(label).strip()
        key = trial_label_key(clean_label)
        if not key or key in seen_labels:
            continue
        seen_labels.add(key)
        raw_values = [values] if isinstance(values, str) else values
        clean_values = list(dict.fromkeys(str(tag).strip() for tag in raw_values if str(tag).strip()))
        if clean_values:
            out[clean_label] = clean_values
    return out


def parse_lambda_from_label(label: str) -> float:
    """The regularisation strength named in a trial label (``lambda 0.3`` -> 0.3), NaN when absent."""
    text = str(label).replace("_", " ").lower()
    if "lambda" not in text:
        return math.nan
    tail = text.split("lambda", 1)[1].split()
    try:
        return float(tail[0]) if tail else math.nan
    except ValueError:
        return math.nan


def first_metric(metrics: Mapping[str, object], keys: Sequence[str]) -> float:
    """The first present metric among ``keys``; a per-frame sequence reports its last finite value."""
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                finite = np.asarray([as_float(item) for item in value], dtype=np.float32)
                finite = finite[np.isfinite(finite)]
                return float(finite[-1]) if finite.size else math.nan
            return as_float(value)
    return math.nan


def best_optimization_row(rankings: Sequence[Mapping[str, object]]) -> dict | None:
    """The visible row with the lowest score; ties break on label."""
    candidates = [dict(row) for row in rankings if not row.get("hidden") and math.isfinite(as_float(row.get("score")))]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (as_float(row.get("score")), str(row.get("label") or "").lower()))


class TrialReview:
    """Mixin: ``trial_rankings`` / ``trial_alerts`` / ``best_trial_label`` from the widget state.

    Expects the host to hold ``_data`` (traces), ``_snapshots`` with their
    group traits, ``_trial_metrics`` (label -> metric dict from monitor events)
    and ``_monitor_warnings``.
    """

    def _update_trial_analysis(self) -> None:
        """Recompute and sync the ranking rows, alerts and best trial after traces, snapshots or review marks change."""
        rows = self._compute_trial_rankings() if self.labels else []
        alerts = self._compute_trial_alerts(rows)
        self.trial_rankings = json_safe(rows)
        self.trial_alerts = json_safe(alerts)
        best = best_optimization_row(rows) if self.review_mode == "optimization" else None
        self.best_trial_label = str(best["label"]) if best else ""

    def _compute_trial_rankings(self) -> list[dict]:
        """One row per trace: loss statistics, monitor metrics, image quality, review marks, rank.

        In ``optimization`` review the final trace value is a loss and the rows
        sort by ``trial_sort_key``; in ``trace`` review the values are
        scientific series and rows only sort by label.
        """
        image_quality = self._snapshot_quality_by_label()
        optimization = self.review_mode == "optimization"
        rows: list[dict] = []
        for trace_index, label in enumerate(self.labels):
            values = self._data[trace_index] if trace_index < self._data.shape[0] else np.empty(0, dtype=np.float32)
            finite = values[np.isfinite(values)]
            first = float(finite[0]) if finite.size else math.nan
            final = float(finite[-1]) if finite.size else math.nan
            lowest = float(np.min(finite)) if finite.size else math.nan
            mean = float(np.mean(finite)) if finite.size else math.nan
            std = float(np.std(finite)) if finite.size else math.nan
            metrics = lookup_by_trial_key(self._trial_metrics, label) or {}
            quality = image_quality.get(trial_label_key(label), {})
            image_std = as_float(quality.get("image_std"))
            flicker = first_metric(metrics, _FLICKER_KEYS)
            rows.append({
                "label": str(label),
                "trace_index": trace_index,
                "lambda": parse_lambda_from_label(label),
                "first_value": first,
                "final_value": final,
                "min_value": lowest,
                "mean_value": mean,
                "std_value": std,
                "first_loss": first if optimization else math.nan,
                "final_loss": final if optimization else math.nan,
                "min_loss": lowest if optimization else math.nan,
                "mean_loss": mean if optimization else math.nan,
                "std_loss": std if optimization else math.nan,
                "rmse": first_metric(metrics, _RMSE_KEYS),
                "flicker": flicker if math.isfinite(flicker) else as_float(quality.get("image_flicker")),
                "object_quality": as_float(metrics.get("object_quality", image_std)),
                "probe_quality": as_float(metrics.get("probe_quality", image_std)),
                "image_std": image_std,
                "image_collapsed": bool(quality.get("collapsed", False)),
                "nan_count": int(np.count_nonzero(~np.isfinite(values))),
                "starred": label_in_collection(label, self.starred_snapshot_image_labels),
                "hidden": label_in_collection(label, self.hidden_snapshot_image_labels),
                "note": str(lookup_by_trial_key(self.trial_notes, label) or ""),
                "tags": list(lookup_by_trial_key(self.trial_tags, label) or []),
            })
        alert_counts: dict[str, int] = {}
        for alert in self._compute_trial_alerts(rows):
            if alert["label"]:
                alert_counts[alert["label"]] = alert_counts.get(alert["label"], 0) + 1
        sort_key = self.trial_sort_key if optimization else "label"
        if sort_key == "default":
            sort_key = "final_loss"
        for row in rows:
            row["alert_count"] = alert_counts.get(row["label"], 0)
            row["score"] = _ranking_score(row, sort_key)
        if sort_key == "label":
            rows.sort(key=lambda row: row["label"].lower(), reverse=self.trial_sort_descending)
        else:
            # finite scores first, ascending (lower loss is better), then by label
            rows.sort(
                key=lambda row: (
                    not math.isfinite(row["score"]),
                    row["score"] if math.isfinite(row["score"]) else math.inf,
                    row["label"].lower(),
                ),
                reverse=self.trial_sort_descending,
            )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def _compute_trial_alerts(self, rows: Sequence[Mapping[str, object]]) -> list[dict]:
        """Problems a reviewer should see first: NaNs, a loss that got worse, spikes, flat loss, collapsed or flickering images, monitor warnings."""
        alerts: list[dict] = []
        optimization = self.review_mode == "optimization"
        for row in rows:
            label = str(row["label"])
            values = self._data[int(row["trace_index"])]
            finite = values[np.isfinite(values)]
            if optimization and int(row["nan_count"]) > 0:
                alerts.append(_alert(label, "nonfinite", "error", "contains NaN/inf values"))
            if optimization and finite.size >= 2:
                first = float(finite[0])
                final = float(finite[-1])
                if final > first and (final - first) / max(abs(first), 1e-12) > 0.25:
                    alerts.append(_alert(label, "worse_final", "warning", "final loss is worse than initial loss"))
                if np.max(np.abs(finite)) > 10 * max(float(np.median(np.abs(finite))), 1e-12):
                    alerts.append(_alert(label, "spike", "warning", "large loss spike detected"))
            if optimization and finite.size >= 8:
                quarter = max(2, finite.size // 4)
                start = float(np.median(finite[:quarter]))
                end = float(np.median(finite[-quarter:]))
                if abs((start - end) / max(abs(start), 1e-12)) < 1e-3:
                    alerts.append(_alert(label, "flat_loss", "info", "loss is nearly flat"))
            if row["image_collapsed"]:
                alerts.append(_alert(label, "image_collapse", "error", "snapshot image appears collapsed"))
            flicker = as_float(row["flicker"])
            if math.isfinite(flicker) and flicker > 0.75:
                alerts.append(_alert(label, "flicker", "warning", "large frame-to-frame flicker"))
        alerts.extend(_alert("", "monitor_warning", "warning", message) for message in self._monitor_warnings)
        return alerts

    def _snapshot_quality_by_label(self) -> dict[str, dict[str, float | bool]]:
        """Per trial: median image std (contrast), median mean |value|, median relative frame-to-frame change, collapse flag."""
        grouped: dict[str, list[tuple[float, np.ndarray]]] = {}
        for image_index, image in enumerate(self._snapshots):
            iteration = float(self.snapshot_group_iterations[self.snapshot_group_indices[image_index]])
            grouped.setdefault(trial_label_key(self.snapshot_image_labels[image_index]), []).append((iteration, image))
        out: dict[str, dict[str, float | bool]] = {}
        for key, frames in grouped.items():
            stats = []
            for _, image in sorted(frames, key=lambda item: item[0]):
                finite = image[np.isfinite(image)]
                if finite.size:
                    stats.append((float(np.std(finite)), float(np.mean(np.abs(finite))), image))
            if not stats:
                continue
            stds = np.asarray([item[0] for item in stats], dtype=np.float32)
            means = np.asarray([item[1] for item in stats], dtype=np.float32)
            diffs = [
                float(np.nanmean(np.abs(current - previous)) / max(mean_abs, 1e-6))
                for (_, mean_abs, previous), (_, _, current) in zip(stats[:-1], stats[1:], strict=False)
                if previous.shape == current.shape
            ]
            out[key] = {
                "image_std": float(np.median(stds)),
                "image_mean_abs": float(np.median(means)),
                "image_flicker": float(np.median(diffs)) if diffs else math.nan,
                "collapsed": bool(np.max(stds) < 1e-7 or np.max(means) < 1e-9),
            }
        return out


def _ranking_score(row: Mapping[str, object], key: str) -> float:
    """Lower is better for every key; quality metrics are negated so a sharper image ranks first."""
    if key == "label":
        return math.nan
    if key in {"object_quality", "probe_quality"}:
        value = as_float(row[key])
        return -value if math.isfinite(value) else math.nan
    return as_float(row[key])


def _alert(label: str, kind: str, severity: str, message: str) -> dict:
    """One alert row as the Review panel reads it; ``label`` is empty for run-wide monitor warnings."""
    return {"label": label, "kind": kind, "severity": severity, "message": message}
