#!/usr/bin/env python3
"""Drive generated widget HTML exports in a real browser.

This script consumes the artifact directory created by ``widget_html_smoke.py``.
It opens every exported HTML page, verifies visible canvases are nonblank,
performs basic pointer/control interactions, captures screenshots, and writes a
browser smoke report next to the export matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import http.server
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


STORY_IDS_BY_VARIANT = {
    "show2d-single": ["S2D-01", "S2D-04", "S2D-05", "S2D-09"],
    "show2d-gallery-3": ["S2D-02", "S2D-05", "S2D-06"],
    "show2d-gallery-6-fft": ["S2D-02", "S2D-07", "S2D-12"],
    "show2d-hidden-starred": ["S2D-03", "S2D-12"],
    "show2d-compact-no-titles": ["S2D-11", "S2D-13"],
    "show3d-single-stack": ["S3D-01", "S3D-02", "S3D-05"],
    "show3d-single-fft-bottom": ["S3D-07", "S3D-09"],
    "show3d-single-fft-overlay": ["S3D-08", "S3D-09"],
    "show3d-three-panels": ["S3D-03", "S3D-05", "S3D-06"],
    "show3d-hidden-panel": ["S3D-04", "S3D-14"],
    "show3d-four-panel-downsample": ["S3D-14", "S3D-15", "S3D-16"],
    "show4dstem": ["S4D-01", "S4D-02", "S4D-03", "S4D-06", "S4D-09"],
    "showfolder": ["SF-2", "SF-5", "SF-8"],
}


def _chrome_executable() -> str | None:
    candidates = [
        os.environ.get("CHROME_EXECUTABLE"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/opt/google/chrome/chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class _StaticServer:
    root: Path
    port: int
    httpd: http.server.ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None

    def __enter__(self) -> str:
        root = self.root

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root), **kwargs)

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                return

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _image_nonblank(png_bytes: bytes, *, min_unique: int = 4, min_span: int = 8) -> tuple[bool, dict[str, Any]]:
    image = Image.open(BytesIO(png_bytes)).convert("RGB")
    # Downsample for cheap uniqueness checks while preserving blank/flat failures.
    image.thumbnail((160, 160))
    colors = image.getcolors(maxcolors=160 * 160 + 1) or []
    extrema = image.getextrema()
    span = max(hi - lo for lo, hi in extrema)
    return len(colors) >= min_unique and span >= min_span, {
        "width": image.width,
        "height": image.height,
        "unique_colors": len(colors),
        "max_channel_span": span,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in name)


def _visible_canvas_boxes(page) -> list[dict[str, float]]:
    return page.evaluate(
        """() => [...document.querySelectorAll('canvas')].map((canvas, index) => {
          const rect = canvas.getBoundingClientRect();
          return {
            index,
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            backingWidth: canvas.width,
            backingHeight: canvas.height
          };
        }).filter(item =>
          item.width >= 24 && item.height >= 24 &&
          item.backingWidth > 0 && item.backingHeight > 0
        )"""
    )


def _visible_text_present(page, text: str) -> bool:
    return bool(
        page.evaluate(
            """(text) => {
              const normalize = (value) => value.trim().replace(/:$/, '').toLowerCase();
              const wanted = normalize(text);
              for (const node of [...document.querySelectorAll('button,[role="button"],label,p,span,div')]) {
                const value = normalize(node.textContent || '');
                if (value !== wanted) continue;
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                if (rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none') {
                  return true;
                }
              }
              return false;
            }""",
            text,
        )
    )


def _toggle_labeled_switch(page, label: str) -> dict[str, Any]:
    return page.evaluate(
        """(label) => {
          const normalize = (value) => value.trim().replace(/:$/, '').toLowerCase();
          const wanted = normalize(label);
          const visible = (node) => {
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
          };
          const textNodes = [...document.querySelectorAll('button,[role="button"],label,p,span,div')]
            .filter(node => normalize(node.textContent || '') === wanted && visible(node));
          for (const node of textNodes) {
            const labelRect = node.getBoundingClientRect();
            const labelCy = labelRect.y + labelRect.height / 2;
            const candidates = [...document.querySelectorAll('input[type="checkbox"]')]
              .map((input) => {
                const host = input.closest('.MuiSwitch-root') || input.closest('label') || input.parentElement || input;
                const rect = host.getBoundingClientRect();
                return {input, host, rect, score: Math.abs((rect.y + rect.height / 2) - labelCy) + Math.max(0, labelRect.right - rect.left)};
              })
              .filter((item) =>
                visible(item.host) &&
                Math.abs((item.rect.y + item.rect.height / 2) - labelCy) <= Math.max(24, labelRect.height * 1.8) &&
                item.rect.left >= labelRect.left - 4
              )
              .sort((a, b) => a.score - b.score);
            for (const {input} of candidates) {
              const before = Boolean(input.checked);
              input.click();
              return {found: true, before, after: Boolean(input.checked)};
            }
          }
          return {found: false, before: null, after: null};
        }""",
        label,
    )


def _canvas_layout_summary(page) -> dict[str, Any]:
    return page.evaluate(
        """() => {
          const boxes = [...document.querySelectorAll('canvas')].map((canvas, index) => {
            const rect = canvas.getBoundingClientRect();
            const style = getComputedStyle(canvas);
            return {
              index,
              x: rect.x,
              y: rect.y,
              width: rect.width,
              height: rect.height,
              area: rect.width * rect.height,
              backingWidth: canvas.width,
              backingHeight: canvas.height,
              visible: rect.width > 24 && rect.height > 24 && canvas.width > 0 && canvas.height > 0 &&
                style.visibility !== 'hidden' && style.display !== 'none' && Number(style.opacity || '1') > 0.05
            };
          }).filter((item) => item.visible);
          const maxArea = boxes.reduce((value, item) => Math.max(value, item.area), 0);
          const large = boxes.filter((item) => item.area >= maxArea * 0.45 && item.width >= 48 && item.height >= 48);
          const rows = [];
          for (const item of [...large].sort((a, b) => a.y - b.y || a.x - b.x)) {
            const row = rows.find((candidate) => Math.abs(candidate.y - item.y) <= Math.max(12, item.height * 0.18));
            if (row) {
              row.y = (row.y * row.items.length + item.y) / (row.items.length + 1);
              row.items.push(item);
            } else {
              rows.push({y: item.y, items: [item]});
            }
          }
          const rowCounts = rows.map((row) => row.items.length);
          const primary = [...boxes].sort((a, b) => b.area - a.area)[0] || null;
          const signature = large
            .map((item) => [Math.round(item.x), Math.round(item.y), Math.round(item.width), Math.round(item.height)].join(','))
            .join('|');
          const allSignature = boxes
            .map((item) => [Math.round(item.x), Math.round(item.y), Math.round(item.width), Math.round(item.height)].join(','))
            .join('|');
          return {
            canvas_count: boxes.length,
            large_canvas_count: large.length,
            row_counts: rowCounts,
            max_row_count: rowCounts.length ? Math.max(...rowCounts) : 0,
            primary,
            primary_aspect: primary ? primary.width / Math.max(1, primary.height) : null,
            signature,
            all_signature: allSignature,
          };
        }"""
    )


def _mui_select_value(page, aria_label: str) -> str | None:
    return page.evaluate(
        """(ariaLabel) => {
          const control = [...document.querySelectorAll('[aria-label]')]
            .find((node) => node.getAttribute('aria-label') === ariaLabel);
          if (!control) return null;
          return control.value ?? (control.textContent || '').trim();
        }""",
        aria_label,
    )


def _select_mui_option(page, aria_label: str, value: int | str) -> dict[str, Any]:
    before = _mui_select_value(page, aria_label)
    opened = page.evaluate(
        """(ariaLabel) => {
          const visible = (node) => {
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
          };
          const control = [...document.querySelectorAll('[aria-label]')]
            .find((node) => node.getAttribute('aria-label') === ariaLabel);
          if (!control) return false;
          const root = control.closest('.MuiInputBase-root') || control.parentElement;
          const target = root?.querySelector('[role="combobox"]') || root || control;
          if (!target || !visible(target)) return false;
          target.dispatchEvent(new MouseEvent('mousedown', {bubbles: true, cancelable: true, view: window}));
          target.click();
          return true;
        }""",
        aria_label,
    )
    if not opened:
        return {"found": False, "before": before, "after": before, "selected": False}
    page.wait_for_timeout(120)
    selected = page.evaluate(
        """(value) => {
          const wanted = String(value).trim();
          const visible = (node) => {
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
          };
          const options = [...document.querySelectorAll('[role="option"], .MuiMenuItem-root')]
            .filter(visible);
          const option = options.find((node) => (node.textContent || '').trim() === wanted);
          if (!option) return false;
          option.click();
          return true;
        }""",
        str(value),
    )
    page.wait_for_timeout(260)
    after = _mui_select_value(page, aria_label)
    return {"found": True, "before": before, "after": after, "selected": bool(selected)}


def _exercise_column_select(page, aria_label: str, target: int) -> dict[str, Any]:
    before = _canvas_layout_summary(page)
    before_value = _mui_select_value(page, aria_label)
    to_target = _select_mui_option(page, aria_label, target)
    target_layout = _canvas_layout_summary(page)
    restored: dict[str, Any] | None = None
    restored_layout: dict[str, Any] | None = None
    if before_value is not None and before_value != str(target):
        restored = _select_mui_option(page, aria_label, before_value)
        restored_layout = _canvas_layout_summary(page)
    changed = (
        before.get("signature") != target_layout.get("signature") or
        before.get("primary_aspect") != target_layout.get("primary_aspect")
    )
    return {
        "aria_label": aria_label,
        "target": target,
        "before_value": before_value,
        "to_target": to_target,
        "before_layout": before,
        "target_layout": target_layout,
        "changed": bool(to_target.get("selected") and changed),
        "restored": restored,
        "restored_layout": restored_layout,
    }


def _exercise_fft_toggle(page) -> dict[str, Any]:
    before_layout = _canvas_layout_summary(page)
    before_hash = _sha256(page.screenshot(full_page=False))
    off = _toggle_labeled_switch(page, "FFT")
    page.wait_for_timeout(260)
    off_layout = _canvas_layout_summary(page)
    off_hash = _sha256(page.screenshot(full_page=False))
    on = _toggle_labeled_switch(page, "FFT") if off.get("found") else {"found": False, "before": None, "after": None}
    page.wait_for_timeout(420)
    on_layout = _canvas_layout_summary(page)
    on_hash = _sha256(page.screenshot(full_page=False))
    return {
        "off": off,
        "on": on,
        "before_layout": before_layout,
        "off_layout": off_layout,
        "on_layout": on_layout,
        "layout_changed_when_off": before_layout.get("all_signature") != off_layout.get("all_signature"),
        "visual_changed_when_off": before_hash != off_hash,
        "visual_changed_after_restore": off_hash != on_hash,
        "rendered_after_on": on_layout.get("canvas_count", 0) >= 2,
    }


def _measure_fps(page, duration_ms: int) -> float:
    return float(
        page.evaluate(
            """async (durationMs) => {
              const start = performance.now();
              let frames = 0;
              return await new Promise(resolve => {
                function step(now) {
                  frames += 1;
                  if (now - start >= durationMs) {
                    resolve(frames * 1000 / Math.max(1, now - start));
                  } else {
                    requestAnimationFrame(step);
                  }
                }
                requestAnimationFrame(step);
              });
            }""",
            duration_ms,
        )
    )


def _click_text_controls(page, labels: list[str]) -> list[str]:
    clicked: list[str] = []
    for label in labels:
        did_click = page.evaluate(
            """(label) => {
              const wanted = label.toLowerCase();
              const nodes = [...document.querySelectorAll('button,[role="button"],label,p,span,div')]
                .filter(node => (node.textContent || '').trim().toLowerCase() === wanted);
              for (const node of nodes) {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                if (rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none') {
                  node.click();
                  return true;
                }
              }
              return false;
            }""",
            label,
        )
        if did_click:
            clicked.append(label)
            page.wait_for_timeout(120)
    return clicked


def _click_switches(page, limit: int) -> int:
    return int(
        page.evaluate(
            """(limit) => {
              let clicked = 0;
              for (const input of [...document.querySelectorAll('input[type="checkbox"]')]) {
                const host = input.closest('label') || input.parentElement || input;
                const rect = host.getBoundingClientRect();
                const style = getComputedStyle(host);
                if (rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none') {
                  input.click();
                  clicked++;
                  if (clicked >= limit) break;
                }
              }
              return clicked;
            }""",
            limit,
        )
    )


def _drag_first_slider(page) -> bool:
    slider = page.evaluate(
        """() => {
          const roots = [...document.querySelectorAll('.MuiSlider-root')];
          for (const root of roots) {
            const rect = root.getBoundingClientRect();
            if (rect.width > 40 && rect.height > 8) {
              return {x: rect.x, y: rect.y, width: rect.width, height: rect.height};
            }
          }
          return null;
        }"""
    )
    if not slider:
        return False
    y = slider["y"] + slider["height"] / 2
    page.mouse.move(slider["x"] + slider["width"] * 0.25, y)
    page.mouse.down()
    page.mouse.move(slider["x"] + slider["width"] * 0.72, y, steps=10)
    page.mouse.up()
    page.wait_for_timeout(180)
    return True


def _drive_canvas(page, box: dict[str, float]) -> None:
    x = box["x"] + box["width"] * 0.52
    y = box["y"] + box["height"] * 0.52
    page.mouse.move(x, y)
    page.mouse.wheel(0, -450)
    page.wait_for_timeout(140)
    page.mouse.down()
    page.mouse.move(x + min(40, box["width"] * 0.18), y + min(30, box["height"] * 0.18), steps=10)
    page.mouse.up()
    page.wait_for_timeout(180)


def _show3d_reorder_labels(page) -> list[str]:
    return page.evaluate(
        """() => {
          return [...document.querySelectorAll('[data-show3d-reorder-panel]')]
            .map((el) => {
              const rect = el.getBoundingClientRect();
              const label = (el.getAttribute('aria-label') || '').replace(/^Move\\s+/, '');
              return { label, x: rect.x, y: rect.y, width: rect.width, height: rect.height };
            })
            .filter((item) => item.label && item.width > 40 && item.height > 40)
            .sort((a, b) => a.y - b.y || a.x - b.x)
            .map((item) => item.label);
        }"""
    )


def _show3d_reorder_ghost(page) -> dict[str, Any] | None:
    return page.evaluate(
        """() => {
          const el = document.querySelector('[data-show3d-reorder-ghost]');
          if (!el) return null;
          const rect = el.getBoundingClientRect();
          return {
            panel: el.getAttribute('data-show3d-reorder-ghost'),
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
          };
        }"""
    )


def _exercise_show3d_reorder(page) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": False,
        "before": [],
        "during": [],
        "after": [],
        "changed": False,
        "dynamic_changed_before_release": False,
        "ghost_visible": False,
        "ghost_moved": False,
    }
    opened = page.evaluate(
        """() => {
          const button = [...document.querySelectorAll('button')]
            .find((el) => (el.textContent || '').trim() === 'Reorder');
          if (!button) return false;
          button.click();
          return true;
        }"""
    )
    if not opened:
        return result
    page.wait_for_timeout(180)
    before = _show3d_reorder_labels(page)
    result["attempted"] = True
    result["before"] = before
    boxes = page.evaluate(
        """() => {
          return [...document.querySelectorAll('[data-show3d-reorder-panel]')]
            .map((el) => {
              const rect = el.getBoundingClientRect();
              const label = (el.getAttribute('aria-label') || '').replace(/^Move\\s+/, '');
              return { label, x: rect.x, y: rect.y, width: rect.width, height: rect.height };
            })
            .filter((item) => item.label && item.width > 40 && item.height > 40)
            .sort((a, b) => a.y - b.y || a.x - b.x);
        }"""
    )
    if len(boxes) < 2:
        return result
    source = boxes[0]
    target = boxes[-1]
    start_x = source["x"] + source["width"] * 0.5
    start_y = source["y"] + source["height"] * 0.5
    end_x = target["x"] + target["width"] * 0.78
    end_y = target["y"] + target["height"] * 0.5
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.wait_for_timeout(80)
    ghost_start = _show3d_reorder_ghost(page)
    page.mouse.move(start_x + (end_x - start_x) * 0.35, start_y + (end_y - start_y) * 0.25, steps=6)
    page.mouse.move(start_x + (end_x - start_x) * 0.85, start_y + (end_y - start_y) * 0.1, steps=8)
    page.wait_for_timeout(220)
    ghost_mid = _show3d_reorder_ghost(page)
    during = _show3d_reorder_labels(page)
    result["during"] = during
    result["dynamic_changed_before_release"] = bool(before and during and before != during)
    result["ghost_visible"] = bool(ghost_start and ghost_mid)
    result["ghost_moved"] = bool(
        ghost_start
        and ghost_mid
        and abs(float(ghost_mid["x"]) - float(ghost_start["x"])) > 12
    )
    page.mouse.up()
    page.wait_for_timeout(250)
    after = _show3d_reorder_labels(page)
    result["after"] = after
    result["changed"] = bool(before and after and before != after and before[0] == after[-1])
    page.evaluate(
        """() => {
          const button = [...document.querySelectorAll('button')]
            .find((el) => (el.textContent || '').trim() === 'Reorder' && el.getAttribute('aria-pressed') === 'true');
          if (button) button.click();
        }"""
    )
    page.wait_for_timeout(120)
    return result


def _story_ids_for(row: dict[str, Any]) -> list[str]:
    variant = str(row["variant"])
    if variant.startswith("show4dstem"):
        return STORY_IDS_BY_VARIANT["show4dstem"]
    return STORY_IDS_BY_VARIANT.get(variant, [])


def _semantic_checks(page, row: dict[str, Any], canvas_count: int) -> dict[str, Any]:
    variant = str(row["variant"])
    checks: dict[str, Any] = {
        "story_ids": _story_ids_for(row),
        "export_action_visible": _visible_text_present(page, "Export"),
    }
    errors: list[str] = []

    if "fft" in variant:
        fft_present = _visible_text_present(page, "FFT")
        body_has_fft = bool(page.evaluate("document.body.innerText.toLowerCase().includes('fft')"))
        fft_rendered = canvas_count >= 2
        fft_toggle = _exercise_fft_toggle(page)
        checks["fft_label_visible"] = fft_present
        checks["fft_text_present"] = body_has_fft
        checks["fft_rendered"] = fft_rendered
        checks["fft_toggle"] = fft_toggle
        if not (fft_present or body_has_fft or fft_rendered):
            errors.append("FFT state is not visible for FFT variant")
        if not fft_toggle["off"]["found"]:
            errors.append("FFT toggle could not be found")
        elif not (fft_toggle["off"]["before"] is True and fft_toggle["off"]["after"] is False):
            errors.append(f"FFT toggle did not turn off cleanly: {fft_toggle['off']}")
        elif not (fft_toggle["on"]["found"] and fft_toggle["on"]["after"] is True):
            errors.append(f"FFT toggle did not restore on cleanly: {fft_toggle['on']}")
        if (
            fft_toggle["off"]["found"]
            and not fft_toggle["layout_changed_when_off"]
            and not fft_toggle["visual_changed_when_off"]
        ):
            errors.append("FFT toggle did not change visible canvas layout or viewport pixels")
        if fft_toggle["off"]["found"] and not fft_toggle["visual_changed_after_restore"]:
            errors.append("FFT toggle restore did not change viewport pixels")
        if not fft_toggle["rendered_after_on"]:
            errors.append("FFT did not render visible canvases after being toggled back on")
        page.wait_for_timeout(120)

    narrow_viewport = bool(page.evaluate("window.innerWidth < 700"))
    show2d_column_targets = {
        "show2d-gallery-3": 1,
        "show2d-gallery-6-fft": 2,
        "show2d-hidden-starred": 2,
        "show2d-compact-no-titles": 2,
    }
    show3d_column_targets = {
        "show3d-hidden-panel": 1,
        "show3d-four-panel-downsample": 2,
    }
    if variant in show2d_column_targets:
        column_check = _exercise_column_select(page, "Gallery columns", show2d_column_targets[variant])
        checks["column_select"] = column_check
        if not column_check["to_target"]["found"]:
            errors.append("Show2D column select could not be found")
        elif not column_check["to_target"]["selected"]:
            errors.append(f"Show2D column select could not choose {column_check['target']}")
        elif not column_check["changed"] and not narrow_viewport:
            errors.append(
                "Show2D column select did not change gallery layout "
                f"({column_check['before_layout']} -> {column_check['target_layout']})"
            )
        elif not column_check["changed"]:
            column_check["layout_change_skipped_reason"] = "narrow responsive viewport"
        if column_check["restored"] is not None and not column_check["restored"].get("selected"):
            errors.append(f"Show2D column select did not restore {column_check['before_value']}")

    if variant in show3d_column_targets:
        column_check = _exercise_column_select(page, "Show3D panel columns", show3d_column_targets[variant])
        checks["column_select"] = column_check
        if not column_check["to_target"]["found"]:
            errors.append("Show3D column select could not be found")
        elif not column_check["to_target"]["selected"]:
            errors.append(f"Show3D column select could not choose {column_check['target']}")
        elif not column_check["changed"] and not narrow_viewport:
            errors.append(
                "Show3D column select did not change panel layout "
                f"({column_check['before_layout']} -> {column_check['target_layout']})"
            )
        elif not column_check["changed"]:
            column_check["layout_change_skipped_reason"] = "narrow responsive viewport"
        if column_check["restored"] is not None and not column_check["restored"].get("selected"):
            errors.append(f"Show3D column select did not restore {column_check['before_value']}")

    if variant in {"show2d-gallery-3", "show3d-three-panels"}:
        reorder_visible = _visible_text_present(page, "Reorder")
        panels_visible = _visible_text_present(page, "Panels")
        state_has_panel_order = bool(page.evaluate("document.documentElement.innerHTML.includes('panel_order')"))
        checks["reorder_visible"] = reorder_visible
        checks["panels_menu_visible"] = panels_visible
        checks["state_has_panel_order"] = state_has_panel_order
        if not reorder_visible:
            errors.append("Reorder control is not visible for reordered multi-panel variant")
        if not panels_visible:
            errors.append("Panels menu is not visible for reordered multi-panel variant")
        if not state_has_panel_order:
            errors.append("export state does not include panel_order")

    if variant == "show3d-three-panels":
        canvas_geometry = page.evaluate(
            """() => {
              const canvases = [...document.querySelectorAll('canvas')].map((canvas) => {
                const rect = canvas.getBoundingClientRect();
                return {
                  width: canvas.width,
                  height: canvas.height,
                  cssWidth: rect.width,
                  cssHeight: rect.height,
                  area: rect.width * rect.height
                };
              }).filter(item => item.width > 0 && item.height > 0 && item.cssWidth > 0 && item.cssHeight > 0);
              if (!canvases.length) return null;
              canvases.sort((a, b) => b.area - a.area);
              return canvases[0];
            }"""
        )
        checks["primary_canvas_geometry"] = canvas_geometry
        if not canvas_geometry or canvas_geometry["width"] <= canvas_geometry["height"]:
            errors.append("Show3D Cols=3 rendered as a vertical panel stack")

        reorder_drag = _exercise_show3d_reorder(page)
        checks["reorder_drag"] = reorder_drag
        if not reorder_drag["attempted"]:
            errors.append("Show3D reorder drag could not be started")
        elif not reorder_drag["changed"]:
            errors.append(
                "Show3D reorder drag did not move the first panel to the final slot "
                f"({reorder_drag['before']} -> {reorder_drag['after']})"
            )
        if not reorder_drag["ghost_visible"]:
            errors.append("Show3D reorder drag ghost did not appear during drag")
        elif not reorder_drag["ghost_moved"]:
            errors.append("Show3D reorder drag ghost did not follow the pointer")
        if not reorder_drag["dynamic_changed_before_release"]:
            errors.append(
                "Show3D reorder order did not update before mouse release "
                f"({reorder_drag['before']} -> {reorder_drag['during']})"
            )

        column_check = _exercise_column_select(page, "Show3D panel columns", 1)
        checks["column_select"] = column_check
        if not column_check["to_target"]["found"]:
            errors.append("Show3D column select could not be found")
        elif not column_check["to_target"]["selected"]:
            errors.append(f"Show3D column select could not choose {column_check['target']}")
        elif not column_check["changed"] and not narrow_viewport:
            errors.append(
                "Show3D column select did not change panel layout "
                f"({column_check['before_layout']} -> {column_check['target_layout']})"
            )
        elif not column_check["changed"]:
            column_check["layout_change_skipped_reason"] = "narrow responsive viewport"
        if column_check["restored"] is not None and not column_check["restored"].get("selected"):
            errors.append(f"Show3D column select did not restore {column_check['before_value']}")

    if variant == "show4dstem-compare":
        compare = page.evaluate(
            """() => {
              const panels = [...document.querySelectorAll('[aria-label^="Show4DSTEM compare panel"]')];
              const rowTops = [];
              const rowCounts = [];
              const rowFor = (top) => {
                const idx = rowTops.findIndex((rowTop) => Math.abs(rowTop - top) < 3);
                if (idx >= 0) return idx;
                rowTops.push(top);
                rowCounts.push(0);
                return rowTops.length - 1;
              };
              const before = panels.map((panel) => {
                const rect = panel.getBoundingClientRect();
                const style = getComputedStyle(panel);
                rowCounts[rowFor(rect.top)] += 1;
                return {
                  label: panel.getAttribute('aria-label'),
                  width: rect.width,
                  height: rect.height,
                  top: rect.top,
                  left: rect.left,
                  border: style.borderColor,
                };
              });
              if (panels.length > 1) panels[1].click();
              return {
                count: panels.length,
                before,
                row_counts: rowCounts,
                max_row_count: rowCounts.length ? Math.max(...rowCounts) : 0,
                viewport_width: window.innerWidth,
                text: document.body.innerText,
              };
            }"""
        )
        page.wait_for_timeout(280)
        after_frame_text = bool(
            page.evaluate(
                "document.body.innerText.includes('scan-1') || document.body.innerText.includes('Dataset 2')"
            )
        )
        compare_actions: dict[str, Any] = {}
        try:
            star = page.locator('.show4dstem-compare-star[data-frame="1"]').nth(0)
            compare_actions["star_before"] = star.get_attribute("aria-label")
            star.click(timeout=2000)
            page.wait_for_timeout(120)
            compare_actions["star_after"] = page.locator('.show4dstem-compare-star[data-frame="1"]').nth(0).get_attribute("aria-label")

            page.locator(".show4dstem-compare-reorder").nth(0).click(timeout=2000)
            page.locator('[aria-label="Show4DSTEM compare panel 1"]').nth(0).click(timeout=2000)
            page.locator('[aria-label="Show4DSTEM compare panel 3"]').nth(0).click(timeout=2000)
            page.wait_for_function(
                """() => {
                    const panel = document.querySelector('[aria-label^="Show4DSTEM compare panel"]');
                    return panel && panel.textContent.includes('scan-1');
                }""",
                timeout=4000,
            )
            compare_actions["after_reorder"] = page.evaluate(
                """() => [...document.querySelectorAll('[aria-label^="Show4DSTEM compare panel"]')]
                    .slice(0, 4)
                    .map((panel) => panel.textContent.trim())"""
            )

            page.locator('.show4dstem-compare-hide[data-frame="1"]').nth(0).click(timeout=2000)
            page.wait_for_function(
                "() => document.querySelectorAll('[aria-label^=\"Show4DSTEM compare panel\"]').length === 13",
                timeout=4000,
            )
            compare_actions["count_after_hide"] = page.locator('[aria-label^="Show4DSTEM compare panel"]').count()
            page.locator(".show4dstem-compare-show-all").nth(0).click(timeout=2000)
            page.wait_for_function(
                "() => document.querySelectorAll('[aria-label^=\"Show4DSTEM compare panel\"]').length === 14",
                timeout=4000,
            )
            compare_actions["count_after_show_all"] = page.locator('[aria-label^="Show4DSTEM compare panel"]').count()
            page.locator(".show4dstem-compare-reset").nth(0).click(timeout=2000)
            page.wait_for_function(
                """() => {
                    const panel = document.querySelector('[aria-label^="Show4DSTEM compare panel"]');
                    return panel && panel.textContent.includes('scan-0');
                }""",
                timeout=4000,
            )
            compare_actions["after_reset"] = page.evaluate(
                """() => [...document.querySelectorAll('[aria-label^="Show4DSTEM compare panel"]')]
                    .slice(0, 4)
                    .map((panel) => panel.textContent.trim())"""
            )
        except Exception as exc:  # pragma: no cover - captured in smoke report
            compare_actions["error"] = str(exc)

        checks["compare_grid"] = {
            **compare,
            "selected_second_panel": after_frame_text,
            "actions": compare_actions,
        }
        expected_count = 14
        expected_max_cols = 2 if narrow_viewport else 4
        if compare.get("count", 0) != expected_count:
            errors.append(
                f"Show4DSTEM compare grid rendered {compare.get('count')} panels, "
                f"expected {expected_count}: {compare}"
            )
        if compare.get("max_row_count", 0) > expected_max_cols:
            errors.append(
                f"Show4DSTEM compare grid exceeded {expected_max_cols} columns "
                f"in this viewport: {compare}"
            )
        if not narrow_viewport and compare.get("max_row_count", 0) != expected_max_cols:
            errors.append(f"Show4DSTEM desktop compare grid did not use 4 columns: {compare}")
        if narrow_viewport and compare.get("max_row_count", 0) != expected_max_cols:
            errors.append(f"Show4DSTEM mobile compare grid did not use 2 columns: {compare}")
        if any(
            item.get("width", 0) <= 40 or item.get("height", 0) <= 40
            for item in compare.get("before", [])
        ):
            errors.append(f"Show4DSTEM compare grid has tiny/invalid panels: {compare}")
        if "Compare grid" not in str(compare.get("text", "")):
            errors.append("Show4DSTEM compare grid label is not visible")
        if not after_frame_text:
            errors.append("Show4DSTEM compare panel click did not select the second dataset")
        if compare_actions.get("error"):
            errors.append(f"Show4DSTEM compare panel controls failed: {compare_actions['error']}")
        if "Unstar" not in str(compare_actions.get("star_after", "")):
            errors.append(f"Show4DSTEM compare star button did not toggle: {compare_actions}")
        if not compare_actions.get("after_reorder", [""])[0].startswith("scan-1"):
            errors.append(f"Show4DSTEM compare reorder did not move scan-1 first: {compare_actions}")
        if compare_actions.get("count_after_hide") != 13:
            errors.append(f"Show4DSTEM compare hide did not remove one panel: {compare_actions}")
        if compare_actions.get("count_after_show_all") != 14:
            errors.append(f"Show4DSTEM compare show-all did not restore panels: {compare_actions}")
        if not compare_actions.get("after_reset", [""])[0].startswith("scan-0"):
            errors.append(f"Show4DSTEM compare reset did not restore natural order: {compare_actions}")

    checks["errors"] = errors
    return checks


def _write_html_report(artifact_dir: Path, report: dict[str, Any]) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['viewport'])}</td>"
        f"<td>{html.escape(row['widget'])}</td>"
        f"<td>{html.escape(row['variant'])}</td>"
        f"<td>{'pass' if row['passed'] else 'fail'}</td>"
        f"<td>{html.escape(str(row['canvas_count']))}</td>"
        f"<td>{html.escape(format(row.get('fps', 0), '.1f'))}</td>"
        f"<td>{html.escape(', '.join(row.get('story_ids', [])))}</td>"
        f"<td>{html.escape(str(row['switches_clicked']))}</td>"
        f"<td>{html.escape(str(row['slider_dragged']))}</td>"
        f"<td>{html.escape(str(row['canvas_changed']))}</td>"
        f"<td><a href='{html.escape(row['screenshot'])}'>{html.escape(row['screenshot'])}</a></td>"
        f"<td>{html.escape('; '.join(row['errors']))}</td>"
        f"<td>{html.escape('; '.join(row.get('console_warnings', [])))}</td>"
        "</tr>"
        for row in report["pages"]
    )
    report_json = html.escape(json.dumps(report, indent=2))
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>quantem.widget browser smoke</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin-top: 12px; min-width: 960px; }}
    th, td {{ border: 1px solid #ccd3db; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f5f7; }}
    code, pre {{ background: #f5f7f9; border-radius: 4px; }}
    code {{ padding: 2px 4px; }}
    pre {{ overflow: auto; padding: 12px; max-width: 1180px; }}
  </style>
</head>
<body>
  <h1>quantem.widget browser smoke</h1>
  <p>This report opens exported HTML in Chromium, checks nonblank rendering, and
  drives basic widget interactions.</p>
  <p>Passed: <strong>{report['passed']}</strong> / {len(report['pages'])}</p>
  <table>
    <thead><tr><th>Viewport</th><th>Widget</th><th>Variant</th><th>Status</th><th>Canvases</th><th>FPS</th><th>Stories</th><th>Switches</th><th>Slider</th><th>Canvas changed</th><th>Screenshot</th><th>Errors</th><th>Warnings</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Machine-readable report</h2>
  <p><a href="browser-smoke-report.json">browser-smoke-report.json</a></p>
  <pre>{report_json}</pre>
</body>
</html>
"""
    (artifact_dir / "browser-smoke.html").write_text(page, encoding="utf-8")


def _check_page(
    context,
    base_url: str,
    artifact_dir: Path,
    row: dict[str, Any],
    timeout_ms: int,
    viewport_label: str,
    fps_sample_ms: int,
    min_fps: float,
) -> dict[str, Any]:
    page = context.new_page()
    browser_errors: list[str] = []
    console_errors: list[str] = []
    console_warnings: list[str] = []
    http_errors: list[str] = []
    page.on("pageerror", lambda exc: browser_errors.append(str(exc)))

    def _handle_console(msg) -> None:
        if msg.type != "error" or "Failed to load resource:" in msg.text:
            return
        if "Unable to preventDefault inside passive event listener invocation." in msg.text:
            console_warnings.append(msg.text)
            return
        console_errors.append(msg.text)

    page.on("console", _handle_console)
    page.on(
        "response",
        lambda response: http_errors.append(f"{response.status} {response.url}")
        if response.status >= 400
        and not response.url.endswith("/favicon.ico")
        and not response.url.endswith("/anywidget.js")
        and not response.url.endswith(".map")
        else None,
    )
    variant = str(row["variant"])
    widget = str(row["widget"])
    screenshot_name = f"screenshots/{_safe_name(viewport_label)}-{_safe_name(variant)}.png"
    canvas_name = f"screenshots/{_safe_name(viewport_label)}-{_safe_name(variant)}-canvas.png"
    result: dict[str, Any] = {
        "widget": widget,
        "variant": variant,
        "viewport": viewport_label,
        "url": f"{base_url}/{Path(str(row['path'])).name}",
        "screenshot": screenshot_name,
        "canvas_screenshot": canvas_name,
        "story_ids": _story_ids_for(row),
        "canvas_count": 0,
        "canvas_nonblank": False,
        "canvas_changed": False,
        "fps": 0.0,
        "min_fps": min_fps,
        "fps_passed": False,
        "semantic_checks": {},
        "switches_clicked": 0,
        "text_controls_clicked": [],
        "slider_dragged": False,
        "errors": [],
        "passed": False,
    }
    try:
        page.goto(result["url"], wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_function("document.body && document.body.innerText.length > 0", timeout=timeout_ms)
        page.wait_for_timeout(700)

        boxes = _visible_canvas_boxes(page)
        result["canvas_count"] = len(boxes)
        if widget != "showfolder" and not boxes:
            result["errors"].append("no visible canvas")

        if boxes:
            # Use the largest visible canvas as the primary render target.
            box = max(boxes, key=lambda item: item["width"] * item["height"])
            locator = page.locator("canvas").nth(int(box["index"]))
            before = locator.screenshot(timeout=timeout_ms)
            (artifact_dir / canvas_name).write_bytes(before)
            nonblank, image_stats = _image_nonblank(before)
            result["canvas_nonblank"] = nonblank
            result["canvas_stats"] = image_stats
            if not nonblank:
                result["errors"].append("primary canvas is blank or flat")

            _drive_canvas(page, box)
            after = locator.screenshot(timeout=timeout_ms)
            result["canvas_changed"] = _sha256(before) != _sha256(after)

        semantic = _semantic_checks(page, row, int(result["canvas_count"]))
        result["semantic_checks"] = semantic
        result["story_ids"] = semantic["story_ids"]
        result["errors"].extend(semantic["errors"])

        labels = ["Profile", "FFT", "ROI", "Lens", "Panels", "Stats", "Export"]
        result["text_controls_clicked"] = _click_text_controls(page, labels)
        result["switches_clicked"] = _click_switches(page, 3)
        result["slider_dragged"] = _drag_first_slider(page)
        page.wait_for_timeout(300)
        result["fps"] = _measure_fps(page, fps_sample_ms)
        result["fps_passed"] = result["fps"] >= min_fps
        if not result["fps_passed"]:
            result["errors"].append(f"FPS {result['fps']:.1f} below minimum {min_fps:.1f}")

        page.screenshot(path=str(artifact_dir / screenshot_name), full_page=True, timeout=timeout_ms)
        result["browser_errors"] = browser_errors
        result["console_errors"] = console_errors
        result["console_warnings"] = console_warnings
        result["http_errors"] = http_errors
        result["errors"].extend(browser_errors)
        result["errors"].extend(console_errors)
        result["errors"].extend(http_errors)

        if widget == "showfolder":
            has_folder_marker = page.evaluate("document.body.innerText.includes('0010')")
            if not has_folder_marker:
                result["errors"].append("showfolder marker 0010 not visible")
        elif boxes and not result["canvas_nonblank"]:
            result["errors"].append("render check failed")

        result["passed"] = not result["errors"]
        return result
    finally:
        page.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Directory created by widget_html_smoke.py.")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--headed", action="store_true", help="Show the Chromium window while driving.")
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--max-pages", type=int, default=0, help="Limit pages for debugging; 0 means all.")
    parser.add_argument("--min-fps", type=float, default=30.0, help="Minimum requestAnimationFrame FPS for each page.")
    parser.add_argument("--fps-sample-ms", type=int, default=1000, help="Milliseconds to sample requestAnimationFrame FPS.")
    parser.add_argument("--mobile", action="store_true", help="Also run the smoke in a 390x844 mobile Chromium viewport.")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("playwright is required for browser smoke testing") from exc

    artifact_dir = args.artifact_dir.resolve()
    report_path = artifact_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"missing {report_path}; run scripts/widget_html_smoke.py first")
    (artifact_dir / "screenshots").mkdir(parents=True, exist_ok=True)
    export_report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = list(export_report["exports"])
    if args.max_pages:
        rows = rows[: args.max_pages]

    chrome = _chrome_executable()
    launch_kwargs: dict[str, Any] = {
        "headless": not args.headed,
        "args": [
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-search-engine-choice-screen",
        ],
    }
    if chrome is not None:
        launch_kwargs["executable_path"] = chrome

    port = args.port or _free_port()
    started_at = time.time()
    with _StaticServer(artifact_dir, port) as base_url:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs)
            try:
                pages: list[dict[str, Any]] = []
                viewports = [("desktop", {"viewport": {"width": 1280, "height": 950}})]
                if args.mobile:
                    viewports.append(
                        (
                            "mobile-390x844",
                            {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True},
                        )
                    )
                for viewport_label, viewport_options in viewports:
                    context = browser.new_context(**viewport_options)
                    try:
                        pages.extend(
                            _check_page(
                                context,
                                base_url,
                                artifact_dir,
                                row,
                                args.timeout_ms,
                                viewport_label,
                                args.fps_sample_ms,
                                args.min_fps,
                            )
                            for row in rows
                        )
                    finally:
                        context.close()
            finally:
                browser.close()

    report = {
        "artifact_dir": str(artifact_dir),
        "created_at_unix": started_at,
        "base_url": f"http://127.0.0.1:{port}",
        "headed": bool(args.headed),
        "mobile": bool(args.mobile),
        "min_fps": float(args.min_fps),
        "fps_sample_ms": int(args.fps_sample_ms),
        "passed": sum(1 for page in pages if page["passed"]),
        "pages": pages,
    }
    (artifact_dir / "browser-smoke-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_html_report(artifact_dir, report)
    print(json.dumps(report, indent=2))
    print(f"Browser smoke report: {artifact_dir / 'browser-smoke.html'}")
    if report["passed"] != len(pages):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
