#!/usr/bin/env python3
"""Stress-test current single-file Show3D HTML in Chromium."""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from playwright.sync_api import sync_playwright

from widget_browser_smoke import (
    _canvas_layout_summary,
    _chrome_executable,
    _exercise_column_select,
    _exercise_fft_toggle,
    _image_nonblank,
    _measure_fps,
    _start_canvas_update_probe,
    _stop_canvas_update_probe,
    _visible_canvas_boxes,
)


STATE_SCRIPT_RE = re.compile(
    r'<script[^>]+type=["\']application/vnd\.jupyter\.widget-state\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL,
)

VIEWPORTS = {
    "desktop": {"width": 1500, "height": 1000},
    "wide": {"width": 2200, "height": 1200},
    "narrow": {"width": 900, "height": 900},
}


@dataclass
class TargetSpec:
    name: str
    mode: str
    source: str
    url: str | None = None
    metadata: dict[str, Any] | None = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_artifact_dir() -> Path:
    return Path("/tmp") / "quantem-widget-show3d-stress" / _timestamp()


def _safe_name(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value)
    clean = re.sub(r"-+", "-", clean).strip("-")
    return clean or "show3d"


def _escape(value: object) -> str:
    return html.escape(str(value))


def _extract_widget_state(html_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    text = html_path.read_text(encoding="utf-8")
    match = STATE_SCRIPT_RE.search(text)
    if match is None:
        raise ValueError(f"{html_path} does not contain embedded ipywidgets state")
    state = json.loads(match.group(1))
    for model in state.get("state", {}).values():
        traits = model.get("state", {})
        if traits.get("_view_name") == "AnyView" and "n_slices" in traits and "n_panels" in traits:
            return traits, model
        if "_esm" in traits and "n_slices" in traits and "n_panels" in traits:
            return traits, model
    raise ValueError(f"{html_path} does not look like a Show3D export")


def _buffer_by_trait(model: dict[str, Any], trait: str) -> bytes:
    for item in model.get("buffers", []) or []:
        if item.get("path") == [trait]:
            data = item.get("data") or ""
            return base64.b64decode(data) if data else b""
    value = model.get("state", {}).get(trait)
    if isinstance(value, str) and value:
        try:
            return base64.b64decode(value)
        except Exception:
            return b""
    return b""


def _show3d_metadata_from_html(html_path: Path) -> dict[str, Any]:
    traits, model = _extract_widget_state(html_path)
    buffer_sizes = {
        "_offline_float_stack": len(_buffer_by_trait(model, "_offline_float_stack")),
        "_offline_stack": len(_buffer_by_trait(model, "_offline_stack")),
    }
    return {
        "path": str(html_path),
        "bytes": html_path.stat().st_size,
        "title": traits.get("title", ""),
        "width": int(traits.get("width", 0) or 0),
        "height": int(traits.get("height", 0) or 0),
        "n_slices": int(traits.get("n_slices", 0) or 0),
        "n_panels": int(traits.get("n_panels", 0) or 0),
        "panel_width_px": int(traits.get("panel_width_px", 0) or 0),
        "buffer_sizes": buffer_sizes,
    }




def _read_debug(page) -> dict[str, Any]:
    return page.evaluate(
        """() => {
          const raw = window.__quantemShow3DPerf ||
            document.documentElement.__quantemShow3DPerf || {};
          const out = {};
          for (const key of Object.keys(raw)) {
            const value = raw[key];
            if (value == null || ["string", "number", "boolean"].includes(typeof value)) {
              out[key] = value;
            } else if (Array.isArray(value)) {
              out[key] = value.slice(-12);
            } else if (typeof value === "object") {
              try { out[key] = JSON.parse(JSON.stringify(value)); } catch (_) {}
            }
          }
          return out;
        }"""
    )


def _click_button(page, labels: list[str]) -> bool:
    return bool(
        page.evaluate(
            """(labels) => {
              const wanted = labels.map((label) => String(label).toLowerCase());
              const visible = (node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return rect.width > 0 && rect.height > 0 &&
                  style.display !== "none" && style.visibility !== "hidden" &&
                  Number(style.opacity || "1") > 0.05;
              };
              for (const node of [...document.querySelectorAll("button,[role='button']")]) {
                if (!visible(node)) continue;
                const text = (node.textContent || "").trim().toLowerCase();
                const aria = (node.getAttribute("aria-label") || "").trim().toLowerCase();
                const title = (node.getAttribute("title") || "").trim().toLowerCase();
                if (wanted.some((label) => text === label || aria === label || title === label ||
                    aria.includes(label) || title.includes(label))) {
                  node.click();
                  return true;
                }
              }
              return false;
            }""",
            labels,
        )
    )


def _set_labeled_switch(page, label: str, checked: bool) -> dict[str, Any]:
    """Set a compact MUI switch by its nearby text label."""
    return page.evaluate(
        """({label, checked}) => {
          const wanted = String(label).trim().toLowerCase();
          const visible = (node) => {
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return rect.width > 0 && rect.height > 0 &&
              style.display !== "none" && style.visibility !== "hidden";
          };
          const labels = [...document.querySelectorAll("span,div,label,p")]
            .filter((node) => (node.textContent || "").trim().toLowerCase() === wanted && visible(node));
          for (const labelNode of labels) {
            const lr = labelNode.getBoundingClientRect();
            const lc = {x: lr.x + lr.width / 2, y: lr.y + lr.height / 2};
            const candidates = [...document.querySelectorAll('input[type="checkbox"]')]
              .map((input) => {
                const host = input.closest(".MuiSwitch-root") || input.closest("label") || input.parentElement || input;
                const r = host.getBoundingClientRect();
                const cx = r.x + r.width / 2;
                const cy = r.y + r.height / 2;
                const dx = Math.abs(cx - lc.x);
                const dy = Math.abs(cy - lc.y);
                return {
                  input,
                  host,
                  score: dy * 8 + dx,
                  dx,
                  dy,
                };
              })
              .filter((item) => visible(item.host) && item.dy <= 32 && item.dx <= 220)
              .sort((a, b) => a.score - b.score);
            if (!candidates.length) continue;
            const input = candidates[0].input;
            const before = Boolean(input.checked);
            if (before !== checked) input.click();
            return {found: true, before, after: Boolean(input.checked)};
          }
          return {found: false, before: null, after: null};
        }""",
        {"label": label, "checked": checked},
    )


def _set_labeled_switch_with_retry(
    page,
    label: str,
    checked: bool,
    *,
    attempts: int = 20,
    interval_ms: int = 200,
) -> dict[str, Any]:
    """Set a compact MUI switch after the exported widget has mounted."""
    last: dict[str, Any] = {"found": False, "before": None, "after": None}
    for attempt in range(1, attempts + 1):
        last = _set_labeled_switch(page, label, checked)
        last["attempts"] = attempt
        if last.get("found") and last.get("after") == checked:
            return last
        page.wait_for_timeout(interval_ms)
    return last


def _first_paint(page, *, timeout_ms: int) -> dict[str, Any]:
    start = time.perf_counter()
    last: dict[str, Any] = {}
    while (time.perf_counter() - start) * 1000 < timeout_ms:
        boxes = _visible_canvas_boxes(page)
        if boxes:
            primary = sorted(boxes, key=lambda item: item["width"] * item["height"], reverse=True)[0]
            png = page.locator("canvas").nth(int(primary["index"])).screenshot()
            nonblank, stats = _image_nonblank(png, min_unique=8, min_span=8)
            last = {"box": primary, "nonblank": bool(nonblank), "stats": stats}
            if nonblank:
                last["first_paint_ms"] = round((time.perf_counter() - start) * 1000, 1)
                return last
        page.wait_for_timeout(120)
    last["first_paint_ms"] = round((time.perf_counter() - start) * 1000, 1)
    last["timeout"] = True
    return last


def _save_screenshot(page, path: Path) -> dict[str, Any]:
    png = page.screenshot(path=str(path), full_page=False)
    nonblank, stats = _image_nonblank(png, min_unique=8, min_span=8)
    return {
        "path": str(path),
        "rel": f"screenshots/{path.name}",
        "nonblank": bool(nonblank),
        "stats": stats,
    }


def _primary_canvas_nonblank(page) -> dict[str, Any]:
    boxes = _visible_canvas_boxes(page)
    if not boxes:
        return {"nonblank": False, "error": "no visible canvas"}
    primary = sorted(boxes, key=lambda item: item["width"] * item["height"], reverse=True)[0]
    png = page.locator("canvas").nth(int(primary["index"])).screenshot()
    nonblank, stats = _image_nonblank(png, min_unique=8, min_span=8)
    return {"box": primary, "nonblank": bool(nonblank), "stats": stats}


def _primary_canvas_content_signature(page) -> dict[str, Any]:
    """Return a small content hash for the largest visible canvas."""

    return page.evaluate(
        """() => {
          const visible = (canvas) => {
            const rect = canvas.getBoundingClientRect();
            const style = getComputedStyle(canvas);
            return rect.width > 24 && rect.height > 24 && canvas.width > 0 && canvas.height > 0 &&
              style.display !== "none" && style.visibility !== "hidden" &&
              Number(style.opacity || "1") > 0.05;
          };
          const canvases = [...document.querySelectorAll("canvas")]
            .map((canvas, index) => ({canvas, index, rect: canvas.getBoundingClientRect()}))
            .filter((item) => visible(item.canvas))
            .sort((a, b) => (b.rect.width * b.rect.height) - (a.rect.width * a.rect.height));
          if (!canvases.length) return {ok: false, error: "no visible canvas"};
          const {canvas, index, rect} = canvases[0];
          const sample = document.createElement("canvas");
          sample.width = 32;
          sample.height = 32;
          const ctx = sample.getContext("2d", {willReadFrequently: true});
          if (!ctx) return {ok: false, error: "no 2d context", index};
          try {
            ctx.drawImage(canvas, 0, 0, sample.width, sample.height);
            const data = ctx.getImageData(0, 0, sample.width, sample.height).data;
            let hash = 2166136261 >>> 0;
            let sum = 0;
            let nonzero = 0;
            for (let i = 0; i < data.length; i += 4) {
              const value = (data[i] + data[i + 1] * 3 + data[i + 2] * 7 + data[i + 3] * 11) & 255;
              sum += value;
              if (value) nonzero += 1;
              hash ^= value;
              hash = Math.imul(hash, 16777619) >>> 0;
            }
            return {
              ok: true,
              index,
              hash: hash.toString(16).padStart(8, "0"),
              sum,
              nonzero,
              width: Math.round(rect.width),
              height: Math.round(rect.height),
            };
          } catch (error) {
            return {ok: false, error: String(error), index, width: Math.round(rect.width), height: Math.round(rect.height)};
          }
        }"""
    )


def _drive_playback(page, *, wait_ms: int) -> dict[str, Any]:
    before = _read_debug(page)
    before_layout = _canvas_layout_summary(page)
    before_text = page.evaluate("document.body.innerText")
    clicked = _click_button(page, ["play", "start playback", "play animation"])
    page.wait_for_timeout(wait_ms)
    after = _read_debug(page)
    after_layout = _canvas_layout_summary(page)
    after_text = page.evaluate("document.body.innerText")
    paused = _click_button(page, ["pause", "stop playback", "pause animation", "stop"])
    page.wait_for_timeout(250)
    return {
        "clicked": clicked,
        "paused": paused,
        "debug_before": before,
        "debug_after": after,
        "text_changed": before_text != after_text,
        "layout_changed": before_layout.get("all_signature") != after_layout.get("all_signature"),
    }


def _playback_slider_box(page) -> dict[str, Any]:
    """Find the Show3D frame playback slider by its accessible label."""

    return page.evaluate(
        """() => {
          const visible = (node) => {
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return rect.width > 0 && rect.height > 0 &&
              style.display !== "none" && style.visibility !== "hidden";
          };
          const nodes = [...document.querySelectorAll(".MuiSlider-root [aria-label]")];
          const match = nodes.find((node) => {
            const label = String(node.getAttribute("aria-label") || "").toLowerCase();
            return (
              label.includes("current frame") ||
              label.includes("current slice") ||
              label.includes("current roi") ||
              label.includes("loop range and current")
            );
          });
          if (!match) return {found: false};
          const root = match.closest(".MuiSlider-root") || match.closest('[role="slider"]') || match;
          if (!root || !visible(root)) return {found: false, label: match.getAttribute("aria-label")};
          const rect = root.getBoundingClientRect();
          const thumb = root.querySelector(".MuiSlider-thumb[data-index='1']") ||
            root.querySelector(".MuiSlider-thumb");
          const thumbRect = thumb ? thumb.getBoundingClientRect() : null;
          return {
            found: true,
            label: match.getAttribute("aria-label"),
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            thumb_x: thumbRect ? thumbRect.x : null,
            thumb_y: thumbRect ? thumbRect.y : null,
            thumb_width: thumbRect ? thumbRect.width : null,
            thumb_height: thumbRect ? thumbRect.height : null,
          };
        }"""
    )


def _playback_count_text(page) -> str:
    return str(
        page.evaluate(
            """() => {
              const node = document.querySelector("[data-show3d-playback-count='true']");
              return node ? (node.textContent || "").trim() : "";
            }"""
        )
        or ""
    )


def _drive_slider_scrub(page, *, fps_ms: int) -> dict[str, Any]:
    """Drag the frame slider and verify real frame updates during the drag."""

    box = _playback_slider_box(page)
    if not box.get("found"):
        return {"found": False, "errors": ["playback slider was not found"]}
    page.evaluate(
        """() => {
          const nodes = [...document.querySelectorAll(".MuiSlider-root [aria-label]")];
          const match = nodes.find((node) => {
            const label = String(node.getAttribute("aria-label") || "").toLowerCase();
            return (
              label.includes("current frame") ||
              label.includes("current slice") ||
              label.includes("current roi") ||
              label.includes("loop range and current")
            );
          });
          const root = match?.closest(".MuiSlider-root") || match;
          root?.scrollIntoView({block: "center", inline: "nearest"});
        }"""
    )
    page.wait_for_timeout(120)
    box = _playback_slider_box(page)
    if not box.get("found"):
        return {"found": False, "errors": ["playback slider disappeared after scroll"]}
    root_x = float(box["x"])
    root_w = float(box["width"])
    thumb_x = box.get("thumb_x")
    thumb_w = box.get("thumb_width")
    thumb_y = box.get("thumb_y")
    thumb_h = box.get("thumb_height")
    if thumb_x is not None and thumb_w is not None and thumb_y is not None and thumb_h is not None:
        x0 = float(thumb_x) + float(thumb_w) / 2
        y = float(thumb_y) + float(thumb_h) / 2
    else:
        x0 = root_x + max(4, root_w * 0.50)
        y = float(box["y"]) + float(box["height"]) / 2
    midpoint = root_x + root_w / 2
    target_pct = 0.15 if x0 >= midpoint else 0.85
    x1 = root_x + max(8, root_w * target_pct)

    before_debug = _read_debug(page)
    before_text = _playback_count_text(page)
    before_sig = _primary_canvas_content_signature(page)
    update_probe_start = _start_canvas_update_probe(
        page,
        selector="canvas",
        label="show3d frame slider drag",
    )
    started = time.perf_counter()
    page.mouse.move(x0, y)
    page.mouse.down()
    for step in range(1, 18):
        x = x0 + (x1 - x0) * step / 17
        page.mouse.move(x, y, steps=2)
        page.wait_for_timeout(18)
    page.wait_for_timeout(120)
    mid_debug = _read_debug(page)
    mid_text = _playback_count_text(page)
    mid_sig = _primary_canvas_content_signature(page)
    page.mouse.up()
    page.wait_for_timeout(450)
    final_debug = _read_debug(page)
    final_text = _playback_count_text(page)
    final_sig = _primary_canvas_content_signature(page)
    update_probe = _stop_canvas_update_probe(page)
    settle_fps = round(float(_measure_fps(page, fps_ms)), 1)
    changed_debug = (
        before_debug.get("lastFrame") != mid_debug.get("lastFrame") or
        before_debug.get("lastFrame") != final_debug.get("lastFrame")
    )
    changed_text = before_text != mid_text or before_text != final_text
    changed_pixels = (
        before_sig.get("hash") != mid_sig.get("hash") or
        before_sig.get("hash") != final_sig.get("hash")
    )
    return {
        "found": True,
        "slider": box,
        "duration_s": round(time.perf_counter() - started, 3),
        "update_probe_start": update_probe_start,
        "update_probe": update_probe,
        "drag_visual_update_hz": update_probe.get("visual_update_hz"),
        "drag_browser_raf_fps": update_probe.get("browser_raf_fps"),
        "settle_fps": settle_fps,
        "before_text": before_text,
        "mid_text": mid_text,
        "final_text": final_text,
        "before_debug": before_debug,
        "mid_debug": mid_debug,
        "final_debug": final_debug,
        "before_signature": before_sig,
        "mid_signature": mid_sig,
        "final_signature": final_sig,
        "changed_debug": bool(changed_debug),
        "changed_text": bool(changed_text),
        "changed_pixels": bool(changed_pixels),
    }


def _parse_frame_count(text: str) -> int | None:
    match = re.search(r"\b\d+\s*/\s*(\d+)\b", str(text or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _drive_keyboard_scrub(page, *, fps_ms: int) -> dict[str, Any]:
    """Press frame navigation keys after clicking the rendered widget surface."""

    boxes = _visible_canvas_boxes(page)
    if not boxes:
        return {"found": False, "errors": ["no visible canvas for keyboard focus"]}
    primary = sorted(boxes, key=lambda item: item["width"] * item["height"], reverse=True)[0]
    viewport = page.viewport_size or {"width": 1200, "height": 900}
    click_x = min(
        max(float(primary["x"]) + float(primary["width"]) * 0.5, 16),
        float(viewport["width"]) - 16,
    )
    click_y = min(
        max(float(primary["y"]) + float(primary["height"]) * 0.5, 72),
        float(viewport["height"]) - 20,
    )
    page.mouse.click(click_x, click_y)
    page.wait_for_timeout(120)
    focus_after_click = page.evaluate(
        """() => {
          const active = document.activeElement;
          return {
            tag: active?.tagName || "",
            className: String(active?.className || ""),
            isRoot: Boolean(active?.classList?.contains("show3d-root")),
          };
        }"""
    )

    page.keyboard.press("Home")
    page.wait_for_timeout(250)
    baseline_text = _playback_count_text(page)
    baseline_debug = _read_debug(page)
    baseline_sig = _primary_canvas_content_signature(page)

    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(350)
    right_text = _playback_count_text(page)
    right_debug = _read_debug(page)
    right_sig = _primary_canvas_content_signature(page)

    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(350)
    left_text = _playback_count_text(page)
    left_debug = _read_debug(page)
    left_sig = _primary_canvas_content_signature(page)
    settle_fps = round(float(_measure_fps(page, fps_ms)), 1)

    total_frames = (
        _parse_frame_count(baseline_text)
        or _parse_frame_count(right_text)
        or _parse_frame_count(left_text)
        or None
    )
    changed_debug = baseline_debug.get("lastFrame") != right_debug.get("lastFrame")
    changed_text = baseline_text != right_text
    changed_pixels = baseline_sig.get("hash") != right_sig.get("hash")
    returned_debug = right_debug.get("lastFrame") != left_debug.get("lastFrame")
    returned_text = right_text != left_text
    returned_pixels = right_sig.get("hash") != left_sig.get("hash")
    return {
        "found": True,
        "click": {"x": round(click_x, 1), "y": round(click_y, 1), "canvas": primary},
        "focus_after_click": focus_after_click,
        "total_frames": total_frames,
        "baseline_text": baseline_text,
        "right_text": right_text,
        "left_text": left_text,
        "baseline_debug": baseline_debug,
        "right_debug": right_debug,
        "left_debug": left_debug,
        "baseline_signature": baseline_sig,
        "right_signature": right_sig,
        "left_signature": left_sig,
        "changed_debug": bool(changed_debug),
        "changed_text": bool(changed_text),
        "changed_pixels": bool(changed_pixels),
        "returned_debug": bool(returned_debug),
        "returned_text": bool(returned_text),
        "returned_pixels": bool(returned_pixels),
        "settle_fps": settle_fps,
    }


def _histogram_clip_sliders(page) -> list[dict[str, Any]]:
    """Return unique Show3D histogram clip slider roots and thumb values."""

    return list(
        page.evaluate(
            """() => {
              const visible = (node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return rect.width > 0 && rect.height > 0 &&
                  style.display !== "none" && style.visibility !== "hidden";
              };
              const roots = [];
              const seen = new Set();
              for (const input of document.querySelectorAll('[aria-label="Histogram intensity clip range"]')) {
                const root = input.closest(".MuiSlider-root") || input;
                if (!root || !visible(root)) continue;
                const rect = root.getBoundingClientRect();
                const key = `${Math.round(rect.x)}:${Math.round(rect.y)}:${Math.round(rect.width)}:${Math.round(rect.height)}`;
                if (seen.has(key)) continue;
                seen.add(key);
                const values = [...root.querySelectorAll('[aria-label="Histogram intensity clip range"]')]
                  .map((node) => Number(node.getAttribute("aria-valuenow") || node.value || 0));
                roots.push({
                  index: roots.length,
                  x: rect.x,
                  y: rect.y,
                  width: rect.width,
                  height: rect.height,
                  values,
                });
              }
              return roots;
            }"""
        )
        or []
    )


def _show3d_switch_state(page) -> dict[str, bool]:
    """Read compact Show3D switch state by accessible label."""

    return dict(
        page.evaluate(
            """() => {
              const out = {};
              for (const input of document.querySelectorAll('input[type="checkbox"]')) {
                const label = input.getAttribute("aria-label") || "";
                if (label) out[label] = Boolean(input.checked);
              }
              return out;
            }"""
        )
        or {}
    )


def _drive_histogram_contrast(page, *, fps_ms: int) -> dict[str, Any]:
    """Drag the first histogram and catch Auto/off contrast reset regressions."""

    sliders = _histogram_clip_sliders(page)
    if not sliders:
        return {"found": False, "errors": ["histogram clip slider was not found"]}
    page.evaluate(
        """() => {
          const input = document.querySelector('[aria-label="Histogram intensity clip range"]');
          const root = input?.closest(".MuiSlider-root") || input;
          root?.scrollIntoView({block: "center", inline: "nearest"});
        }"""
    )
    page.wait_for_timeout(120)
    sliders = _histogram_clip_sliders(page)
    if not sliders:
        return {"found": False, "errors": ["histogram clip slider disappeared after scroll"]}

    before_switches = _show3d_switch_state(page)
    before_values = [item.get("values", []) for item in sliders]
    before_sig = _primary_canvas_content_signature(page)
    item = sliders[0]
    x0 = float(item["x"]) + max(4, float(item["width"]) * 0.20)
    x1 = float(item["x"]) + max(8, float(item["width"]) * 0.75)
    y = float(item["y"]) + float(item["height"]) / 2

    update_probe_start = _start_canvas_update_probe(
        page,
        selector="canvas",
        label="show3d histogram contrast drag",
    )
    page.mouse.move(x0, y)
    page.mouse.down()
    for step in range(1, 20):
        x = x0 + (x1 - x0) * step / 19
        page.mouse.move(x, y, steps=2)
        page.wait_for_timeout(18)
    mid_sig = _primary_canvas_content_signature(page)
    mid_values = [item.get("values", []) for item in _histogram_clip_sliders(page)]
    page.mouse.up()
    page.wait_for_timeout(450)
    final_sig = _primary_canvas_content_signature(page)
    final_values = [item.get("values", []) for item in _histogram_clip_sliders(page)]
    final_switches = _show3d_switch_state(page)
    update_probe = _stop_canvas_update_probe(page)
    settle_fps = float(_measure_fps(page, fps_ms))

    def _is_full_range(values: list[Any]) -> bool:
        if len(values) < 2:
            return False
        lo = float(values[0])
        hi = float(values[1])
        return lo <= 0.5 and hi >= 99.5

    started_auto = bool(
        before_switches.get("Toggle stack-wide automatic contrast")
        or before_switches.get("Toggle automatic percentile-based contrast")
    )
    ended_auto = bool(
        final_switches.get("Toggle stack-wide automatic contrast")
        or final_switches.get("Toggle automatic percentile-based contrast")
    )
    independent = before_switches.get("Link contrast across panels") is False and len(before_values) > 1
    reset_other_panels = []
    if independent and started_auto:
        for idx, (before, after) in enumerate(zip(before_values[1:], final_values[1:]), start=1):
            if before and after and not _is_full_range(before) and _is_full_range(after):
                reset_other_panels.append(idx)

    return {
        "found": True,
        "before_switches": before_switches,
        "final_switches": final_switches,
        "before_values": before_values,
        "mid_values": mid_values,
        "final_values": final_values,
        "before_signature": before_sig,
        "mid_signature": mid_sig,
        "final_signature": final_sig,
        "changed_mid": before_sig.get("hash") != mid_sig.get("hash"),
        "changed_final": before_sig.get("hash") != final_sig.get("hash"),
        "started_auto": started_auto,
        "ended_auto": ended_auto,
        "independent": independent,
        "reset_other_panels": reset_other_panels,
        "update_probe_start": update_probe_start,
        "update_probe": update_probe,
        "drag_visual_update_hz": update_probe.get("visual_update_hz"),
        "drag_browser_raf_fps": update_probe.get("browser_raf_fps"),
        "settle_fps": round(settle_fps, 1),
    }


def _drive_visible_canvas_region(page, box: dict[str, float]) -> None:
    """Zoom/pan inside the visible viewport even when a grid canvas is tall."""
    viewport = page.viewport_size or {"width": 1200, "height": 900}
    max_x = max(24, float(viewport["width"]) - 24)
    max_y = max(64, float(viewport["height"]) - 36)
    x = min(max(float(box["x"]) + float(box["width"]) * 0.52, 24), max_x)
    visible_top = max(float(box["y"]), 72.0)
    preferred_y = float(box["y"]) + min(float(box["height"]) * 0.42, float(viewport["height"]) * 0.55)
    y = min(max(preferred_y, visible_top), max_y)
    page.mouse.move(x, y)
    page.mouse.wheel(0, -450)
    page.wait_for_timeout(140)
    page.mouse.down()
    page.mouse.move(
        min(x + min(44, float(box["width"]) * 0.18), max_x),
        min(y + min(34, float(box["height"]) * 0.08), max_y),
        steps=10,
    )
    page.mouse.up()
    page.wait_for_timeout(180)


def _drive_zoom_pan_stress(page, *, seconds: float) -> dict[str, Any]:
    end_time = time.perf_counter() + max(0.5, seconds)
    cycles = 0
    render_paths: list[str] = []
    samples: list[dict[str, Any]] = []
    last_sample = 0.0
    update_probe_start = _start_canvas_update_probe(page, selector="canvas", label="show3d zoom pan stress")
    try:
        while time.perf_counter() < end_time:
            boxes = _visible_canvas_boxes(page)
            if not boxes:
                break
            primary = sorted(boxes, key=lambda item: item["width"] * item["height"], reverse=True)[0]
            _drive_visible_canvas_region(page, primary)
            cycles += 1
            dbg = _read_debug(page)
            path = dbg.get("lastInteractionRenderPath")
            if path:
                render_paths.append(str(path))
            now = time.perf_counter()
            if now - last_sample >= 0.9:
                samples.append({
                    "browser_raf_fps": round(float(_measure_fps(page, 250)), 1),
                    "debug": dbg,
                    "layout": _canvas_layout_summary(page),
                })
                last_sample = now
    finally:
        update_probe = _stop_canvas_update_probe(page)
    return {
        "seconds": seconds,
        "cycles": cycles,
        "render_paths": sorted(set(render_paths)),
        "samples": samples,
        "update_probe_start": update_probe_start,
        "update_probe": update_probe,
        "visual_update_hz": update_probe.get("visual_update_hz"),
        "browser_raf_fps": update_probe.get("browser_raf_fps"),
        "final_canvas": _primary_canvas_nonblank(page),
        "debug": _read_debug(page),
    }


def _run_case(
    page,
    *,
    target: TargetSpec,
    url: str,
    viewport_name: str,
    viewport: dict[str, int],
    artifact_dir: Path,
    seconds: float,
    timeout_ms: int,
    min_fps: float,
    independent_contrast: bool,
) -> dict[str, Any]:
    case_name = _safe_name(f"{target.name}-{viewport_name}")
    screenshots_dir = artifact_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    console_messages: list[dict[str, str]] = []
    page_errors: list[str] = []
    responses: list[dict[str, Any]] = []
    page.on("console", lambda msg: console_messages.append({"type": msg.type, "text": msg.text}))
    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
    page.on(
        "response",
        lambda response: responses.append({
            "url": response.url,
            "status": response.status,
            "content_range": response.headers.get("content-range"),
        })
        if "offline_stack" in response.url or response.status >= 400
        else None,
    )

    started = time.perf_counter()
    status = None
    page.set_viewport_size(viewport)
    response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    if response is not None:
        status = response.status
    contrast_step: dict[str, Any] | None = None
    if independent_contrast:
        contrast_step = _set_labeled_switch_with_retry(page, "Contrast", False)
        page.wait_for_timeout(650)
    first = _first_paint(page, timeout_ms=timeout_ms)
    initial_layout = _canvas_layout_summary(page)
    initial_debug = _read_debug(page)
    initial_fps = round(float(_measure_fps(page, 700)), 1)
    initial_shot = _save_screenshot(page, screenshots_dir / f"{case_name}-initial.png")
    screenshots = [initial_shot]
    histogram_contrast = _drive_histogram_contrast(page, fps_ms=700)
    screenshots.append(_save_screenshot(page, screenshots_dir / f"{case_name}-histogram-contrast.png"))
    page.evaluate("() => window.scrollTo(0, 0)")
    page.wait_for_timeout(120)

    col_steps = []
    for cols in (1, 2, 3, 4, 6, 8, 12):
        current = str(initial_debug.get("layoutRequestedMaxCols") or "")
        if current and current == str(cols):
            continue
        step = _exercise_column_select(page, "Show3D maximum columns", cols)
        if step.get("found") or step.get("to_target", {}).get("found"):
            col_steps.append(step)
            break

    fft_step = _exercise_fft_toggle(page)
    screenshots.append(_save_screenshot(page, screenshots_dir / f"{case_name}-fft-toggle.png"))
    playback = _drive_playback(page, wait_ms=max(900, min(2200, int(seconds * 350))))
    screenshots.append(_save_screenshot(page, screenshots_dir / f"{case_name}-playback.png"))
    keyboard_scrub = _drive_keyboard_scrub(page, fps_ms=700)
    screenshots.append(_save_screenshot(page, screenshots_dir / f"{case_name}-keyboard-scrub.png"))
    slider_scrub = _drive_slider_scrub(page, fps_ms=700)
    screenshots.append(_save_screenshot(page, screenshots_dir / f"{case_name}-slider-scrub.png"))
    stress = _drive_zoom_pan_stress(page, seconds=seconds)
    screenshots.append(_save_screenshot(page, screenshots_dir / f"{case_name}-zoom-pan.png"))
    final_fps = round(float(_measure_fps(page, 900)), 1)
    final_layout = _canvas_layout_summary(page)
    final_debug = _read_debug(page)
    final_shot = _save_screenshot(page, screenshots_dir / f"{case_name}-final.png")
    screenshots.append(final_shot)
    wall_s = round(time.perf_counter() - started, 3)

    errors: list[str] = []
    warnings: list[str] = []
    if not first.get("nonblank"):
        errors.append("first visible canvas did not become nonblank")
    if not stress.get("final_canvas", {}).get("nonblank"):
        errors.append("canvas became blank after zoom/pan stress")
    if final_fps < min_fps:
        errors.append(f"final browser FPS {final_fps} is below --min-fps={min_fps}")
    if not slider_scrub.get("found"):
        errors.append("playback slider was not found for mouse-drag scrub")
    elif not (
        slider_scrub.get("changed_pixels")
        or slider_scrub.get("changed_debug")
        or slider_scrub.get("changed_text")
    ):
        errors.append("mouse-drag scrub did not change the rendered frame or frame counter")
    if slider_scrub.get("settle_fps", min_fps) < min_fps:
        errors.append(f"mouse-drag scrub settle FPS {slider_scrub.get('settle_fps')} is below --min-fps={min_fps}")
    if int((slider_scrub.get("update_probe") or {}).get("visual_changes") or 0) <= 0:
        errors.append("mouse-drag scrub produced no visible canvas pixel updates during interaction")
    if not keyboard_scrub.get("found"):
        errors.append("keyboard scrub could not focus the rendered widget")
    elif (keyboard_scrub.get("total_frames") or 0) > 1 and not (
        keyboard_scrub.get("changed_pixels")
        or keyboard_scrub.get("changed_debug")
        or keyboard_scrub.get("changed_text")
    ):
        errors.append("ArrowRight did not change the rendered frame or frame counter after clicking the widget")
    if keyboard_scrub.get("settle_fps", min_fps) < min_fps:
        errors.append(f"keyboard scrub settle FPS {keyboard_scrub.get('settle_fps')} is below --min-fps={min_fps}")
    if keyboard_scrub.get("found") and not keyboard_scrub.get("focus_after_click", {}).get("isRoot"):
        warnings.append("keyboard focus after canvas click did not land on the Show3D root")
    if not histogram_contrast.get("found"):
        warnings.append("histogram clip slider was not found for contrast drag")
    elif not (histogram_contrast.get("changed_mid") or histogram_contrast.get("changed_final")):
        errors.append("histogram drag did not change the rendered image")
    if histogram_contrast.get("started_auto") and histogram_contrast.get("ended_auto"):
        errors.append("manual histogram drag did not turn Auto contrast off")
    reset_other_panels = histogram_contrast.get("reset_other_panels") or []
    if reset_other_panels:
        errors.append(
            "manual independent histogram drag reset other panels to full range: "
            + ", ".join(str(idx) for idx in reset_other_panels)
        )
    if histogram_contrast.get("settle_fps", min_fps) < min_fps:
        errors.append(
            f"histogram drag settle FPS {histogram_contrast.get('settle_fps')} is below --min-fps={min_fps}"
        )
    if histogram_contrast.get("found") and int((histogram_contrast.get("update_probe") or {}).get("visual_changes") or 0) <= 0:
        errors.append("histogram drag produced no visible canvas pixel updates during interaction")
    if page_errors:
        errors.extend(f"page error: {message}" for message in page_errors)
    for message in console_messages:
        if message.get("type") == "error":
            text = message.get("text", "")
            if "favicon" not in text.lower():
                if text.startswith("Failed to load resource") and "404" in text:
                    warnings.append(f"ignored likely favicon 404 console noise: {text[:180]}")
                    continue
                errors.append(f"console error: {text[:300]}")
    for item in responses:
        if int(item.get("status", 0)) >= 400:
            errors.append(f"HTTP {item['status']} while loading {item['url']}")

    panels = int((target.metadata or {}).get("n_panels", final_debug.get("layoutRequestedMaxCols", 1)) or 1)
    aspect = final_layout.get("primary_aspect")
    if panels >= 4 and aspect is not None and float(aspect) > 4.0:
        errors.append(f"multi-panel primary canvas is still strip-like, aspect={aspect:.2f}")
    if not stress.get("render_paths"):
        warnings.append("zoom/pan stress did not report an interaction render path")
    if int((stress.get("update_probe") or {}).get("visual_changes") or 0) <= 0:
        errors.append("zoom/pan stress produced no visible canvas pixel updates during interaction")
    if independent_contrast and not (contrast_step or {}).get("found"):
        warnings.append("--independent-contrast requested but the Contrast switch was not found")

    return {
        "name": case_name,
        "target": target.name,
        "mode": target.mode,
        "source": target.source,
        "url": url,
        "viewport": {"name": viewport_name, **viewport},
        "http_status": status,
        "wall_s": wall_s,
        "first_paint": first,
        "initial_fps": initial_fps,
        "final_fps": final_fps,
        "initial_layout": initial_layout,
        "final_layout": final_layout,
        "initial_debug": initial_debug,
        "final_debug": final_debug,
        "column_reflow": col_steps,
        "contrast": contrast_step,
        "histogram_contrast": histogram_contrast,
        "fft_toggle": fft_step,
        "playback": playback,
        "keyboard_scrub": keyboard_scrub,
        "slider_scrub": slider_scrub,
        "zoom_pan_stress": stress,
        "screenshots": screenshots,
        "responses": responses,
        "console_messages": console_messages[-80:],
        "page_errors": page_errors,
        "warnings": warnings,
        "errors": errors,
        "passed": not errors,
    }


def _target_url(target: TargetSpec) -> str:
    if target.url:
        return target.url
    path = Path(target.source).expanduser().resolve()
    return path.as_uri()


def _write_report(artifact_dir: Path, report: dict[str, Any]) -> None:
    (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cases = report.get("cases", [])
    errors = [error for case in cases for error in case.get("errors", [])]
    warnings = [warning for case in cases for warning in case.get("warnings", [])]
    status = "PASS" if not errors else "FAIL"
    rows = "".join(
        "<tr>"
        f"<td>{_escape(case.get('name'))}</td>"
        f"<td>{_escape(case.get('mode'))}</td>"
        f"<td>{_escape(case.get('viewport', {}).get('name'))}</td>"
        f"<td>{_escape(case.get('first_paint', {}).get('first_paint_ms'))}</td>"
        f"<td>{_escape(case.get('final_fps'))}</td>"
        f"<td>{_escape(case.get('keyboard_scrub', {}).get('settle_fps'))}</td>"
        f"<td>{_escape((case.get('slider_scrub', {}).get('update_probe') or {}).get('visual_update_hz'))}</td>"
        f"<td>{_escape(case.get('slider_scrub', {}).get('settle_fps'))}</td>"
        f"<td>{_escape((case.get('histogram_contrast', {}).get('update_probe') or {}).get('visual_update_hz'))}</td>"
        f"<td>{_escape(case.get('histogram_contrast', {}).get('settle_fps'))}</td>"
        f"<td>{_escape(case.get('histogram_contrast', {}).get('reset_other_panels'))}</td>"
        f"<td>{_escape((case.get('zoom_pan_stress', {}).get('update_probe') or {}).get('visual_update_hz'))}</td>"
        f"<td>{_escape(case.get('zoom_pan_stress', {}).get('cycles'))}</td>"
        f"<td>{'PASS' if case.get('passed') else 'FAIL'}</td>"
        "</tr>"
        for case in cases
    )
    cards = []
    for case in cases:
        images = "".join(
            f'<figure><img src="{_escape(shot.get("rel"))}"><figcaption>{_escape(Path(shot.get("path", "")).name)}</figcaption></figure>'
            for shot in case.get("screenshots", [])
        )
        payload = {
            key: case.get(key)
            for key in (
                "source",
                "url",
                "viewport",
                "first_paint",
                "initial_fps",
                "final_fps",
                "contrast",
                "histogram_contrast",
                "keyboard_scrub",
                "slider_scrub",
                "final_layout",
                "final_debug",
                "responses",
                "warnings",
                "errors",
            )
        }
        cards.append(
            f'<section class="card"><h2>{_escape(case.get("name"))}</h2>'
            f'<div class="shots">{images}</div>'
            f"<pre>{_escape(json.dumps(payload, indent=2))}</pre></section>"
        )
    errors_html = "".join(f"<li>{_escape(error)}</li>" for error in errors) or "<li>None</li>"
    warnings_html = "".join(f"<li>{_escape(warning)}</li>" for warning in warnings) or "<li>None</li>"
    sources_html = "".join(
        f"<li><code>{_escape(target.get('mode'))}</code> {_escape(target.get('source'))}</li>"
        for target in report.get("targets", [])
    )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Show3D stress report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172033; }}
    table {{ border-collapse: collapse; margin: 14px 0 22px; }}
    th, td {{ text-align: left; padding: 7px 12px; border-bottom: 1px solid #d8dee9; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    code {{ background: #f0f2f5; padding: 1px 4px; border-radius: 4px; }}
    pre {{ white-space: pre-wrap; overflow: auto; max-height: 420px; background: #f6f8fa; padding: 12px; border-radius: 6px; }}
    .status {{ font-weight: 800; color: {"#087f23" if status == "PASS" else "#b00020"}; }}
    .card {{ border: 2px solid #ff2bbd; border-radius: 8px; padding: 14px; margin: 16px 0; background: white; box-shadow: 0 0 0 3px rgba(255, 43, 189, 0.10); }}
    .card h2 {{ display: inline-block; margin-top: 0; padding: 3px 8px; background: #ff2bbd; color: white; border-radius: 4px; }}
    .shots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; border: 3px solid #ff2bbd; background: #2a001e; box-sizing: border-box; }}
    figcaption {{ display: inline-block; margin-top: 3px; padding: 2px 6px; font-size: 12px; color: white; background: #ff2bbd; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Show3D stress report</h1>
  <p class="status">{status}</p>
  <p>Local-only Chromium stress run for single-file Show3D HTML.</p>
  <h2>Sources</h2>
  <ul>{sources_html}</ul>
  <h2>Summary</h2>
  <table>
    <thead><tr><th>Case</th><th>Mode</th><th>Viewport</th><th>First paint ms</th><th>Final browser rAF FPS</th><th>Key settle rAF FPS</th><th>Slider visual Hz</th><th>Slider settle rAF FPS</th><th>Hist visual Hz</th><th>Hist settle rAF FPS</th><th>Hist reset panels</th><th>Zoom visual Hz</th><th>Zoom cycles</th><th>Status</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Errors</h2>
  <ul>{errors_html}</ul>
  <h2>Warnings</h2>
  <ul>{warnings_html}</ul>
  {''.join(cards)}
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(doc, encoding="utf-8")


def _build_targets(args: argparse.Namespace, artifact_dir: Path) -> list[TargetSpec]:
    targets: list[TargetSpec] = []
    for html_path in args.html:
        path = Path(html_path).expanduser().resolve()
        metadata = _show3d_metadata_from_html(path)
        targets.append(TargetSpec(name=_safe_name(path.stem), mode="single", source=str(path), metadata=metadata))
    for url in args.url:
        targets.append(TargetSpec(name=_safe_name(url.rsplit("/", 1)[-1] or "url"), mode="url", source=url, url=url))
    if not targets:
        raise SystemExit("Provide at least one --html or --url target.")
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", action="append", default=[], help="Existing standalone Show3D HTML file.")
    parser.add_argument("--url", action="append", default=[], help="Existing served Show3D HTML URL.")
    parser.add_argument("--artifact-dir", default=str(_default_artifact_dir()), help="Output directory for report and screenshots.")
    parser.add_argument("--viewports", default="desktop", help="Comma-separated viewport names: desktop,wide,narrow.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Zoom/pan stress seconds per case.")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Page load and first-paint timeout.")
    parser.add_argument("--min-fps", type=float, default=30.0, help="Minimum final requestAnimationFrame FPS.")
    parser.add_argument("--independent-contrast", action="store_true", help="Turn off linked panel contrast before screenshots/stress.")
    parser.add_argument("--headed", action="store_true", help="Open a visible browser window.")
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    targets = _build_targets(args, artifact_dir)
    viewport_names = [item.strip() for item in str(args.viewports).split(",") if item.strip()]
    unknown = [item for item in viewport_names if item not in VIEWPORTS]
    if unknown:
        raise SystemExit(f"unknown viewport(s): {', '.join(unknown)}")

    cases: list[dict[str, Any]] = []
    chrome = _chrome_executable()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.headed,
            executable_path=chrome,
            args=["--disable-web-security", "--allow-file-access-from-files"],
        )
        try:
            for target in targets:
                target_url = _target_url(target)
                for viewport_name in viewport_names:
                    page = browser.new_page()
                    try:
                        cases.append(
                            _run_case(
                                page,
                                target=target,
                                url=target_url,
                                viewport_name=viewport_name,
                                viewport=VIEWPORTS[viewport_name],
                                artifact_dir=artifact_dir,
                                seconds=args.seconds,
                                timeout_ms=args.timeout_ms,
                                min_fps=args.min_fps,
                                independent_contrast=args.independent_contrast,
                            )
                        )
                    finally:
                        page.close()
        finally:
            browser.close()

    report = {
        "created_utc": _timestamp(),
        "artifact_dir": str(artifact_dir),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "targets": [
            {
                "name": target.name,
                "mode": target.mode,
                "source": target.source,
                "metadata": target.metadata,
            }
            for target in targets
        ],
        "viewports": viewport_names,
        "cases": cases,
        "passed": all(case.get("passed") for case in cases),
    }
    _write_report(artifact_dir, report)
    print(f"Show3D stress report: {artifact_dir / 'index.html'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
