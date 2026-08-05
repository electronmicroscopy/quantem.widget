#!/usr/bin/env python3
"""Profile an already exported QuantEM widget HTML page in Chromium.

This is the local-only gate for real exported HTML pages that already exist
outside the repository, for example a Tailscale-served report from a lab
workstation. It does not generate data or exports; it opens the provided URL,
checks canvas rendering, drives common widget controls, samples
requestAnimationFrame FPS, and writes a human-readable report.
"""

from __future__ import annotations

import argparse
import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from widget_browser_smoke import (
    _chrome_executable,
    _image_nonblank,
    _measure_fps,
    _visible_canvas_boxes,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_artifact_dir() -> Path:
    return Path("/tmp") / "quantem-widget-external-html-profile" / _timestamp()


def _escape(value: object) -> str:
    return html.escape(str(value))


def _text_summary(page) -> dict[str, Any]:
    return page.evaluate(
        r"""() => {
          const text = document.body.innerText || "";
          const lines = text.split("\n").map((line) => line.trim()).filter(Boolean).slice(0, 24);
          const buttons = [...document.querySelectorAll("button")]
            .map((button) => (button.textContent || button.getAttribute("aria-label") || "").trim())
            .filter(Boolean);
          return { lines, buttons };
        }"""
    )


def _canvas_signature(page) -> list[dict[str, Any]]:
    """Return a cheap visual/layout signature for visible canvases."""

    return page.evaluate(
        r"""() => [...document.querySelectorAll("canvas")].map((canvas, index) => {
          const rect = canvas.getBoundingClientRect();
          if (rect.width < 24 || rect.height < 24 || canvas.width <= 0 || canvas.height <= 0) {
            return null;
          }
          let data = "";
          try {
            data = canvas.toDataURL("image/png").slice(0, 220);
          } catch (error) {
            data = `${canvas.width}x${canvas.height}:${rect.x},${rect.y}`;
          }
          return {
            index,
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            data,
          };
        }).filter(Boolean)"""
    )


def _click_button(page, label: str) -> bool:
    return bool(
        page.evaluate(
            r"""(label) => {
              const visible = (node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return rect.width > 0 && rect.height > 0 &&
                  style.display !== "none" && style.visibility !== "hidden";
              };
              const wanted = label.toLowerCase();
              const matches = (button) => {
                const text = (button.textContent || "").trim().toLowerCase();
                const aria = (button.getAttribute("aria-label") || "").trim().toLowerCase();
                if (text === wanted || aria === wanted) {
                  return true;
                }
                if (wanted === "panels") {
                  return text.startsWith("panels ") || aria === "choose visible panels";
                }
                return false;
              };
              const candidates = [...document.querySelectorAll("button,[role='menuitem'],li")]
                .filter((button) => matches(button) && visible(button));
              if (!candidates.length) {
                return false;
              }
              candidates[0].click();
              return true;
            }""",
            label,
        )
    )


def _button_present(summary: dict[str, Any], label: str) -> bool:
    wanted = label.lower()
    return any(str(button).strip().lower() == wanted for button in summary.get("buttons", []))


def _click_labeled_switch(page, label: str) -> bool:
    """Click the first visible switch grouped with a compact text label."""

    box = page.evaluate(
        r"""(label) => {
              const wanted = label.toLowerCase();
              const visible = (node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return rect.width > 0 && rect.height > 0 &&
                  style.display !== "none" && style.visibility !== "hidden";
              };
              const labelNodes = [...document.querySelectorAll(".show2d-root *")]
                .filter((node) => visible(node) && (node.textContent || "").trim().toLowerCase() === wanted);
              for (const node of labelNodes) {
                let scope = node.parentElement;
                for (let depth = 0; scope && depth < 5; depth += 1, scope = scope.parentElement) {
                  const switches = [...scope.querySelectorAll(".MuiSwitch-root")]
                    .filter(visible);
                  if (switches.length) {
                    const rect = switches[0].getBoundingClientRect();
                    return {
                      x: rect.x + rect.width / 2,
                      y: rect.y + rect.height / 2,
                      width: rect.width,
                      height: rect.height,
                    };
                  }
                }
              }
              return null;
            }""",
        label,
    )
    if not box:
        return False
    page.mouse.click(float(box["x"]), float(box["y"]))
    return True


def _first_hide_button(summary: dict[str, Any]) -> str | None:
    for button in summary.get("buttons", []):
        label = str(button).strip()
        if label.startswith("Hide "):
            return label
    return None


def _screenshot(page, path: Path, *, full_page: bool = False) -> dict[str, Any]:
    png = page.screenshot(path=str(path), full_page=full_page)
    nonblank, stats = _image_nonblank(png, min_unique=8, min_span=8)
    return {
        "path": str(path),
        "rel": f"screenshots/{path.name}",
        "nonblank": bool(nonblank),
        "stats": stats,
    }


def _primary_canvas_screenshot(page, path: Path) -> dict[str, Any]:
    boxes = _visible_canvas_boxes(page)
    if not boxes:
        return {"error": "no visible canvas"}
    box = sorted(boxes, key=lambda item: item["width"] * item["height"], reverse=True)[0]
    png = page.locator("canvas").nth(int(box["index"])).screenshot(path=str(path))
    nonblank, stats = _image_nonblank(png, min_unique=8, min_span=8)
    return {
        "box": box,
        "path": str(path),
        "rel": f"screenshots/{path.name}",
        "nonblank": bool(nonblank),
        "stats": stats,
    }


def _show2d_main_canvas_screenshots(page, screenshot_dir: Path, *, limit: int, prefix: str) -> list[dict[str, Any]]:
    """Capture visible Show2D image canvases and flag flat/black panels."""

    handles = page.query_selector_all("canvas[data-show2d-main-canvas]")
    viewport = page.viewport_size or {}
    viewport_w = float(viewport.get("width") or 0)
    viewport_h = float(viewport.get("height") or 0)
    reports: list[dict[str, Any]] = []
    for dom_index, handle in enumerate(handles):
        if len(reports) >= limit:
            break
        box = handle.bounding_box()
        if not box or box["width"] < 24 or box["height"] < 24:
            continue
        if viewport_w > 0 and (box["x"] >= viewport_w or box["x"] + box["width"] <= 0):
            continue
        if viewport_h > 0 and (box["y"] >= viewport_h or box["y"] + box["height"] <= 0):
            continue
        panel = handle.get_attribute("data-show2d-main-canvas") or str(dom_index)
        filename = f"{prefix}-show2d-main-panel-{panel}.png"
        path = screenshot_dir / filename
        try:
            png = handle.screenshot(path=str(path))
            nonblank, stats = _image_nonblank(png, min_unique=8, min_span=8)
        except Exception as exc:  # pragma: no cover - browser dependent
            reports.append({
                "panel": panel,
                "dom_index": dom_index,
                "box": box,
                "nonblank": False,
                "error": str(exc)[:300],
            })
            continue
        reports.append({
            "panel": panel,
            "dom_index": dom_index,
            "box": box,
            "path": str(path),
            "rel": f"screenshots/{path.name}",
            "nonblank": bool(nonblank),
            "stats": stats,
        })
    return reports


def _show2d_root_state(page) -> dict[str, Any]:
    return page.evaluate(
        r"""() => {
          const root = document.querySelector(".show2d-root");
          if (!root) return { found: false };
          const active = document.activeElement;
          return {
            found: true,
            selectedPanel: root.getAttribute("data-show2d-selected-panel"),
            selectedPanels: root.getAttribute("data-show2d-selected-panels"),
            visiblePanelCount: Number(root.getAttribute("data-show2d-visible-panel-count") || "0"),
            focused: active === root,
            activeTag: active ? active.tagName : "",
            activeClass: active ? String(active.className || "") : "",
          };
        }"""
    )


def _show2d_canvas_hashes(page, limit: int) -> list[dict[str, Any]]:
    """Return stable low-cost hashes for visible Show2D image canvases."""

    return list(
        page.evaluate(
            r"""(limit) => {
              const out = [];
              const canvases = [...document.querySelectorAll("canvas[data-show2d-main-canvas]")];
              for (let domIndex = 0; domIndex < canvases.length; domIndex += 1) {
                if (out.length >= limit) break;
                const canvas = canvases[domIndex];
                const rect = canvas.getBoundingClientRect();
                if (rect.width < 24 || rect.height < 24 || canvas.width <= 0 || canvas.height <= 0) continue;
                let hash = "";
                try {
                  const ctx = canvas.getContext("2d", { willReadFrequently: true });
                  const sampleW = Math.max(1, Math.min(64, Math.floor(canvas.width / 4)));
                  const sampleH = Math.max(1, Math.min(64, Math.floor(canvas.height / 4)));
                  const x0 = Math.max(0, Math.floor(canvas.width * 0.38 - sampleW / 2));
                  const y0 = Math.max(0, Math.floor(canvas.height * 0.38 - sampleH / 2));
                  const data = ctx.getImageData(x0, y0, sampleW, sampleH).data;
                  let h1 = 2166136261 >>> 0;
                  let h2 = 16777619 >>> 0;
                  let sum = 0;
                  for (let i = 0; i < data.length; i += 16) {
                    const v = data[i] + 3 * data[i + 1] + 5 * data[i + 2] + 7 * data[i + 3];
                    sum += v;
                    h1 ^= v & 255;
                    h1 = Math.imul(h1, 16777619) >>> 0;
                    h2 ^= (v >>> 8) & 255;
                    h2 = Math.imul(h2, 2166136261) >>> 0;
                  }
                  hash = `${h1.toString(16)}:${h2.toString(16)}:${sum}`;
                } catch (error) {
                  hash = `error:${String(error).slice(0, 80)}`;
                }
                out.push({
                  panel: canvas.getAttribute("data-show2d-main-canvas") || String(domIndex),
                  domIndex,
                  box: {
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                  },
                  hash,
                });
              }
              return out;
            }""",
            limit,
        )
    )


def _first_visible_show2d_canvas_box(page) -> dict[str, Any] | None:
    """Return the largest visible Show2D scientific canvas box in the viewport."""

    return page.evaluate(
        r"""() => {
          const viewportW = window.innerWidth || 0;
          const viewportH = window.innerHeight || 0;
          const visible = (node) => {
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return rect.width >= 32 && rect.height >= 32 &&
              rect.right > 0 && rect.bottom > 0 &&
              rect.left < viewportW && rect.top < viewportH &&
              style.display !== "none" && style.visibility !== "hidden";
          };
          const boxes = [...document.querySelectorAll("canvas[data-show2d-main-canvas]")]
            .map((canvas, domIndex) => {
              const rect = canvas.getBoundingClientRect();
              return {
                panel: canvas.getAttribute("data-show2d-main-canvas") || String(domIndex),
                domIndex,
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
                area: rect.width * rect.height,
                visible: visible(canvas),
              };
            })
            .filter((item) => item.visible)
            .sort((a, b) => b.area - a.area);
          return boxes[0] || null;
        }"""
    )


def _show2d_perf_debug(page) -> dict[str, Any]:
    """Return a JSON-safe snapshot of Show2D browser-side performance counters."""

    return dict(
        page.evaluate(
            r"""() => JSON.parse(JSON.stringify(window.__quantemShow2DPerf || {}))"""
        )
        or {}
    )


def _reset_show2d_zoom_pan_perf(page) -> None:
    """Reset Show2D zoom/pan timing counters before a measured interaction."""

    page.evaluate(
        r"""() => {
          const perf = window.__quantemShow2DPerf;
          if (!perf) return;
          perf.mainCanvasPaintCount = 0;
          perf.lastMainCanvasPaintAt = 0;
          perf.lastMainCanvasPaintPanel = null;
          perf.zoomPanEventCount = 0;
          perf.lastZoomPanEventAt = 0;
          perf.lastZoomPanEventKind = "";
          perf.lastZoomPanPaintLatencyMs = null;
          perf.zoomPanPaintLatenciesMs = [];
        }"""
    )


def _start_show2d_event_probe(page, label: str) -> dict[str, Any]:
    """Start a non-invasive event timestamp probe on the visible Show2D canvas."""

    return dict(
        page.evaluate(
            r"""(label) => {
              const viewportW = window.innerWidth || 0;
              const viewportH = window.innerHeight || 0;
              const visible = (node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return rect.width >= 32 && rect.height >= 32 &&
                  rect.right > 0 && rect.bottom > 0 &&
                  rect.left < viewportW && rect.top < viewportH &&
                  style.display !== "none" && style.visibility !== "hidden";
              };
              const item = [...document.querySelectorAll("canvas[data-show2d-main-canvas]")]
                .map((canvas, domIndex) => ({canvas, domIndex, rect: canvas.getBoundingClientRect()}))
                .filter((entry) => visible(entry.canvas))
                .sort((a, b) => (b.rect.width * b.rect.height) - (a.rect.width * a.rect.height))[0];
              if (!item) return {started: false, label, reason: "no visible Show2D canvas"};
              const start = performance.now();
              const probe = {
                active: true,
                label,
                start,
                events: [],
                target: {
                  domIndex: item.domIndex,
                  x: Math.round(item.rect.x),
                  y: Math.round(item.rect.y),
                  width: Math.round(item.rect.width),
                  height: Math.round(item.rect.height),
                },
              };
              const record = (event) => {
                if (!probe.active) return;
                probe.events.push({
                  type: event.type,
                  t: Number((performance.now() - start).toFixed(1)),
                  x: Math.round(event.clientX),
                  y: Math.round(event.clientY),
                });
              };
              const eventTypes = ["pointerdown", "pointermove", "pointerup", "mousedown", "mousemove", "mouseup", "wheel"];
              eventTypes.forEach((type) => item.canvas.addEventListener(type, record, {passive: true}));
              probe.cleanup = () => eventTypes.forEach((type) => item.canvas.removeEventListener(type, record));
              window.__quantemShow2DEventProbe = probe;
              return {started: true, label, target: probe.target};
            }""",
            label,
        )
        or {}
    )


def _stop_show2d_event_probe(page) -> dict[str, Any]:
    """Stop the non-invasive Show2D event probe."""

    return dict(
        page.evaluate(
            r"""() => {
              const probe = window.__quantemShow2DEventProbe;
              if (!probe) return {started: false, reason: "event probe was not started"};
              probe.active = false;
              if (probe.cleanup) probe.cleanup();
              const durationMs = Math.max(1, performance.now() - probe.start);
              const events = probe.events || [];
              const eventTimes = events.map((item) => Number(item.t));
              const counts = {};
              for (const item of events) counts[item.type] = (counts[item.type] || 0) + 1;
              return {
                started: true,
                label: probe.label,
                target: probe.target,
                duration_ms: Number(durationMs.toFixed(1)),
                event_count: events.length,
                event_type_counts: counts,
                first_event_ms: eventTimes.length ? eventTimes[0] : null,
                last_event_ms: eventTimes.length ? eventTimes[eventTimes.length - 1] : null,
                events: events.slice(0, 60),
                truncated_events: events.length > 60,
              };
            }"""
        )
        or {}
    )


def _scroll_show2d_histogram_slider_into_view(page) -> dict[str, Any]:
    """Scroll the first Show2D histogram drag target into view."""

    return page.evaluate(
        r"""() => {
          const visible = (node) => {
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return rect.width > 20 && rect.height > 4 &&
              style.display !== "none" && style.visibility !== "hidden";
          };
          const sliders = [...document.querySelectorAll(".show2d-root .MuiSlider-root")]
            .map((root, index) => ({ root, index, thumbs: root.querySelectorAll("[role='slider']").length }))
            .filter((item) => item.thumbs >= 2 && visible(item.root));
          const histCanvases = [...document.querySelectorAll(".show2d-root canvas:not([data-show2d-main-canvas])")]
            .map((root, index) => ({ root, index }))
            .filter((item) => {
              const rect = item.root.getBoundingClientRect();
              const ratio = rect.width / Math.max(1, rect.height);
              return visible(item.root) && rect.width >= 60 && rect.width <= 260 &&
                rect.height >= 32 && rect.height <= 140 && ratio >= 1.1 && ratio <= 8;
            });
          if (!sliders.length && !histCanvases.length) {
            return { found: false, count: 0 };
          }
          const item = sliders[0] || histCanvases[0];
          item.root.scrollIntoView({ block: "center", inline: "nearest" });
          const rect = item.root.getBoundingClientRect();
          return {
            found: true,
            kind: sliders[0] ? "mui_slider" : "histogram_canvas",
            index: item.index,
            count: sliders.length || histCanvases.length,
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
          };
        }"""
    )


def _show2d_histogram_drag_target(page) -> dict[str, Any]:
    """Find a visible Show2D histogram drag target."""

    return page.evaluate(
        r"""() => {
          const visible = (node) => {
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return rect.width > 20 && rect.height > 4 &&
              style.display !== "none" && style.visibility !== "hidden" &&
              rect.bottom > 0 && rect.right > 0 &&
              rect.left < (window.innerWidth || 0) && rect.top < (window.innerHeight || 0);
          };
          const sliders = [...document.querySelectorAll(".show2d-root .MuiSlider-root")]
            .map((root, index) => {
              const rect = root.getBoundingClientRect();
              const thumbs = [...root.querySelectorAll("[role='slider']")].map((thumb) => {
                const box = thumb.getBoundingClientRect();
                return {
                  x: box.x, y: box.y, width: box.width, height: box.height,
                  value: thumb.getAttribute("aria-valuenow"),
                  label: thumb.getAttribute("aria-label") || "",
                };
              });
              return {
                index,
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
                thumbCount: thumbs.length,
                thumbs,
                visible: visible(root),
              };
            })
            .filter((item) => item.visible && item.thumbCount >= 2 && item.width >= 40);
          sliders.sort((a, b) => {
            const aScore = (a.y > 150 ? 1 : 0) + (a.width <= 180 ? 1 : 0);
            const bScore = (b.y > 150 ? 1 : 0) + (b.width <= 180 ? 1 : 0);
            return bScore - aScore || b.y - a.y;
          });
          if (sliders[0]) {
            return { ...sliders[0], found: true, kind: "mui_slider" };
          }
          const histCanvases = [...document.querySelectorAll(".show2d-root canvas:not([data-show2d-main-canvas])")]
            .map((canvas, index) => {
              const rect = canvas.getBoundingClientRect();
              return {
                index,
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
                thumbCount: 0,
                thumbs: [],
                visible: visible(canvas),
              };
            })
            .filter((item) => {
              const ratio = item.width / Math.max(1, item.height);
              return item.visible && item.width >= 60 && item.width <= 260 &&
                item.height >= 32 && item.height <= 140 && ratio >= 1.1 && ratio <= 8;
            });
          histCanvases.sort((a, b) => b.y - a.y || b.x - a.x);
          if (histCanvases[0]) {
            return { ...histCanvases[0], found: true, kind: "histogram_canvas" };
          }
          return { found: false, candidates: sliders, histogramCanvasCandidates: histCanvases };
        }"""
    )


def _run_show2d_linked_contrast_step(page, fps_ms: int, canvas_limit: int) -> dict[str, Any]:
    """Drag the linked histogram range and verify multiple visible panels update."""

    page.keyboard.press("Escape")
    page.wait_for_timeout(120)
    try:
        first_canvas = page.locator("canvas[data-show2d-main-canvas]").first
        first_canvas.click(position={"x": 12, "y": 12}, timeout=5000)
        page.wait_for_timeout(160)
    except Exception:
        pass
    profile_clicked = _click_labeled_switch(page, "Profile")
    controls_clicked = False
    if not profile_clicked:
        controls_clicked = _click_button(page, "Controls")
        if controls_clicked:
            page.wait_for_timeout(350)
        profile_clicked = _click_labeled_switch(page, "Profile")
    if profile_clicked:
        page.wait_for_timeout(500)
    before = _show2d_canvas_hashes(page, canvas_limit)
    scroll_result = _scroll_show2d_histogram_slider_into_view(page)
    if scroll_result.get("found"):
        page.wait_for_timeout(180)
    slider = _show2d_histogram_drag_target(page)
    if not slider or slider.get("found") is False:
        return {
            "found": False,
            "clicked": controls_clicked,
            "profile_clicked": profile_clicked,
            "scroll_result": scroll_result,
            "reason": "no visible Show2D histogram drag target",
            "before_hash_count": len(before),
            "slider": slider,
        }

    if slider.get("kind") == "histogram_canvas":
        start_x = float(slider["x"]) + float(slider["width"]) * 0.95
        start_y = float(slider["y"]) + float(slider["height"]) - 10
        target_x = float(slider["x"]) + float(slider["width"]) * 0.72
        target_y = start_y
    else:
        thumbs = sorted(slider["thumbs"], key=lambda thumb: thumb["x"])
        right_thumb = thumbs[-1]
        start_x = float(right_thumb["x"]) + float(right_thumb["width"]) / 2
        start_y = float(right_thumb["y"]) + float(right_thumb["height"]) / 2
        target_x = float(slider["x"]) + float(slider["width"]) * 0.72
        target_y = start_y

    page.evaluate(
        r"""() => {
          window.__qwContrastCounting = true;
          window.__qwContrastFrames = 0;
          window.__qwContrastStart = performance.now();
          const step = () => {
            if (!window.__qwContrastCounting) return;
            window.__qwContrastFrames += 1;
            requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        }"""
    )
    started = time.perf_counter()
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(target_x, target_y, steps=16)
    page.wait_for_timeout(120)
    mid_drag = _show2d_canvas_hashes(page, canvas_limit)
    page.mouse.up()
    page.wait_for_timeout(350)
    remaining_ms = max(0, int(fps_ms - (time.perf_counter() - started) * 1000))
    if remaining_ms:
        page.wait_for_timeout(remaining_ms)
    fps = float(
        page.evaluate(
            r"""() => {
              window.__qwContrastCounting = false;
              const elapsed = performance.now() - (window.__qwContrastStart || performance.now());
              return (window.__qwContrastFrames || 0) * 1000 / Math.max(1, elapsed);
            }"""
        )
    )
    after = _show2d_canvas_hashes(page, canvas_limit)
    before_by_panel = {item["panel"]: item["hash"] for item in before}
    mid_changed = [
        item["panel"] for item in mid_drag
        if item["panel"] in before_by_panel and item["hash"] != before_by_panel[item["panel"]]
    ]
    after_changed = [
        item["panel"] for item in after
        if item["panel"] in before_by_panel and item["hash"] != before_by_panel[item["panel"]]
    ]
    return {
        "found": True,
        "clicked": True,
        "controls_clicked": controls_clicked,
        "profile_clicked": profile_clicked,
        "scroll_result": scroll_result,
        "slider": slider,
        "drag": {
            "start_x": round(start_x, 1),
            "start_y": round(start_y, 1),
            "target_x": round(target_x, 1),
            "target_y": round(target_y, 1),
        },
        "before_hash_count": len(before),
        "mid_drag_hash_count": len(mid_drag),
        "after_hash_count": len(after),
        "mid_drag_changed_panels": mid_changed,
        "after_changed_panels": after_changed,
        "changed_panel_count": len(after_changed),
        "fps": round(fps, 1),
        "duration_s": round(time.perf_counter() - started, 3),
    }


def _run_show2d_keyboard_step(page, fps_ms: int) -> dict[str, Any]:
    """Verify Show2D canvas focus plus ArrowLeft/ArrowRight/H shortcuts."""

    canvases = page.locator("canvas[data-show2d-main-canvas]")
    try:
        canvas_count = canvases.count()
    except Exception:
        canvas_count = 0
    if canvas_count <= 0:
        return {"found": False, "reason": "no Show2D main canvases"}

    before = _show2d_root_state(page)
    canvases.first.click(timeout=5000)
    page.wait_for_timeout(120)
    after_click = _show2d_root_state(page)
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(160)
    after_right = _show2d_root_state(page)
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(160)
    after_left = _show2d_root_state(page)
    before_hide_count = int(_show2d_root_state(page).get("visiblePanelCount") or 0)
    page.keyboard.press("h")
    page.wait_for_timeout(350)
    after_hide = _show2d_root_state(page)
    after_hide_count = int(after_hide.get("visiblePanelCount") or 0)
    restore_clicked = False
    if after_hide_count < before_hide_count:
        if _click_button(page, "Panels"):
            page.wait_for_timeout(250)
            restore_clicked = _click_button(page, "Show all panels")
            page.wait_for_timeout(350)
    fps = round(float(_measure_fps(page, fps_ms)), 1)
    after_restore = _show2d_root_state(page)
    return {
        "found": True,
        "canvas_count": canvas_count,
        "before": before,
        "after_click": after_click,
        "after_right": after_right,
        "after_left": after_left,
        "after_hide": after_hide,
        "after_restore": after_restore,
        "right_changed_selection": after_right.get("selectedPanel") != after_click.get("selectedPanel"),
        "left_restored_selection": after_left.get("selectedPanel") == after_click.get("selectedPanel"),
        "hide_changed_visible_count": after_hide_count < before_hide_count,
        "restore_clicked": restore_clicked,
        "fps": fps,
    }


def _run_show2d_zoom_pan_step(page, fps_ms: int, canvas_limit: int) -> dict[str, Any]:
    """Wheel-zoom and drag-pan a Show2D scientific canvas."""

    canvases = page.locator("canvas[data-show2d-main-canvas]")
    try:
        canvas_count = canvases.count()
    except Exception:
        canvas_count = 0
    if canvas_count <= 0:
        return {"found": False, "reason": "no Show2D main canvases"}

    box = _first_visible_show2d_canvas_box(page)
    if not box or box["width"] < 32 or box["height"] < 32:
        try:
            canvases.first.scroll_into_view_if_needed(timeout=5000)
            page.wait_for_timeout(120)
            box = _first_visible_show2d_canvas_box(page)
        except Exception as exc:
            return {"found": False, "reason": f"could not scroll Show2D canvas into view: {exc}"[:300]}
    if not box or box["width"] < 32 or box["height"] < 32:
        return {"found": False, "reason": f"invalid visible Show2D canvas box: {box}"}

    before = _show2d_canvas_hashes(page, canvas_limit)
    before_by_panel = {item["panel"]: item["hash"] for item in before}
    center_x = float(box["x"] + box["width"] * 0.5)
    center_y = float(box["y"] + box["height"] * 0.5)
    page.mouse.click(center_x, center_y)
    page.wait_for_timeout(120)
    page.mouse.move(center_x, center_y)
    _reset_show2d_zoom_pan_perf(page)
    wheel_probe_start = _start_show2d_event_probe(page, "show2d wheel zoom")
    for _ in range(4):
        page.mouse.wheel(0, -420)
        page.wait_for_timeout(90)
    page.wait_for_timeout(160)
    wheel_probe = _stop_show2d_event_probe(page)
    wheel_perf = _show2d_perf_debug(page)
    after_zoom = _show2d_canvas_hashes(page, canvas_limit)

    drag_to_x = center_x + min(120.0, float(box["width"]) * 0.18)
    drag_to_y = center_y + min(80.0, float(box["height"]) * 0.14)
    _reset_show2d_zoom_pan_perf(page)
    drag_probe_start = _start_show2d_event_probe(page, "show2d drag pan")
    page.mouse.move(center_x, center_y)
    page.mouse.down()
    for step_idx in range(1, 19):
        x = center_x + (drag_to_x - center_x) * step_idx / 18
        y = center_y + (drag_to_y - center_y) * step_idx / 18
        page.mouse.move(x, y)
        page.wait_for_timeout(18)
    page.mouse.up()
    page.wait_for_timeout(220)
    drag_probe = _stop_show2d_event_probe(page)
    drag_perf = _show2d_perf_debug(page)
    after_pan = _show2d_canvas_hashes(page, canvas_limit)
    fps = round(float(_measure_fps(page, fps_ms)), 1)

    def changed_panels(items: list[dict[str, Any]]) -> list[str]:
        return [
            item["panel"]
            for item in items
            if item["panel"] in before_by_panel and item["hash"] != before_by_panel[item["panel"]]
        ]

    return {
        "found": True,
        "clicked": True,
        "canvas_count": canvas_count,
        "target_panel": box.get("panel"),
        "target_dom_index": box.get("domIndex"),
        "before_hash_count": len(before),
        "after_zoom_hash_count": len(after_zoom),
        "after_pan_hash_count": len(after_pan),
        "zoom_changed_panels": changed_panels(after_zoom),
        "pan_changed_panels": changed_panels(after_pan),
        "changed_panel_count": len(set(changed_panels(after_zoom) + changed_panels(after_pan))),
        "wheel_event_probe_start": wheel_probe_start,
        "wheel_event_probe": wheel_probe,
        "wheel_perf_debug": wheel_perf,
        "drag_event_probe_start": drag_probe_start,
        "drag_event_probe": drag_probe,
        "drag_perf_debug": drag_perf,
        "wheel_events": 4,
        "drag": {
            "start_x": round(center_x, 1),
            "start_y": round(center_y, 1),
            "target_x": round(drag_to_x, 1),
            "target_y": round(drag_to_y, 1),
        },
        "browser_raf_fps_after_interaction": fps,
        "fps": fps,
    }


def _reset_show2d_zoom_pan(page) -> bool:
    """Best-effort double-click reset after capturing zoom/pan evidence."""

    canvases = page.locator("canvas[data-show2d-main-canvas]")
    try:
        if canvases.count() <= 0:
            return False
        box = _first_visible_show2d_canvas_box(page)
        if not box:
            canvases.first.scroll_into_view_if_needed(timeout=5000)
            page.wait_for_timeout(120)
            box = _first_visible_show2d_canvas_box(page)
        if not box:
            return False
        page.mouse.dblclick(float(box["x"] + box["width"] * 0.5), float(box["y"] + box["height"] * 0.5))
        page.wait_for_timeout(220)
        return True
    except Exception:
        return False


def _run_play_button_step(page, label: str, pause_labels: list[str], wait_ms: int, fps_ms: int) -> dict[str, Any]:
    before_summary = _text_summary(page)
    before_signature = _canvas_signature(page)
    clicked = _click_button(page, label)
    page.wait_for_timeout(wait_ms)
    after_summary = _text_summary(page)
    after_signature = _canvas_signature(page)
    fps = round(float(_measure_fps(page, fps_ms)), 1)
    paused_clicked = False
    for pause_label in pause_labels:
        if _click_button(page, pause_label):
            paused_clicked = True
            page.wait_for_timeout(250)
            break
    return {
        "label": label,
        "clicked": clicked,
        "paused_clicked": paused_clicked,
        "text_changed": before_summary != after_summary,
        "canvas_signature_changed": before_signature != after_signature,
        "fps": fps,
    }


def _run_hide_step(page, label: str, fps_ms: int) -> dict[str, Any]:
    before_count = len(_visible_canvas_boxes(page))
    before_signature = _canvas_signature(page)
    clicked = _click_button(page, label)
    page.wait_for_timeout(700)
    after_count = len(_visible_canvas_boxes(page))
    after_signature = _canvas_signature(page)
    fps = round(float(_measure_fps(page, fps_ms)), 1)
    restore_label = "Show " + label.removeprefix("Hide ")
    plain_label = label.removeprefix("Hide ")
    restore_clicked = _click_button(page, restore_label)
    if not restore_clicked and _click_button(page, "Panels"):
        page.wait_for_timeout(250)
        restore_clicked = _click_button(page, restore_label)
        if not restore_clicked:
            restore_clicked = _click_button(page, plain_label)
        if not restore_clicked:
            restore_clicked = _click_button(page, "Show all panels")
    page.wait_for_timeout(300)
    return {
        "label": label,
        "clicked": clicked,
        "restore_label": restore_label,
        "plain_restore_label": plain_label,
        "restore_clicked": restore_clicked,
        "canvas_count_before": before_count,
        "canvas_count_after": after_count,
        "canvas_signature_changed": before_signature != after_signature,
        "fps": fps,
    }


def _science_canvas_indices(page, limit: int) -> list[int]:
    """Return one large canvas index per visible scientific panel."""

    return list(
        page.evaluate(
            r"""(limit) => {
              const seen = new Set();
              const out = [];
              const canvases = [...document.querySelectorAll("canvas")];
              for (let index = 0; index < canvases.length; index += 1) {
                const canvas = canvases[index];
                const rect = canvas.getBoundingClientRect();
                if (canvas.width < 240 || canvas.height < 240 || rect.width < 120 || rect.height < 120) {
                  continue;
                }
                const key = [
                  Math.round((canvas.offsetLeft || rect.left) / 4),
                  Math.round((canvas.offsetTop || rect.top + window.scrollY) / 4),
                  Math.round(rect.width),
                  Math.round(rect.height),
                ].join(":");
                if (seen.has(key)) {
                  continue;
                }
                seen.add(key);
                out.push(index);
                if (out.length >= limit) {
                  break;
                }
              }
              return out;
            }""",
            limit,
        )
    )


def _run_hover_step(page, fps_ms: int, max_canvases: int) -> dict[str, Any]:
    """Sweep hover across scientific canvases while measuring rAF FPS."""

    indices = _science_canvas_indices(page, max_canvases)
    before_summary = _text_summary(page)
    page.evaluate(
        r"""() => {
          window.__qwHoverCounting = true;
          window.__qwHoverFrames = 0;
          window.__qwHoverStart = performance.now();
          const step = () => {
            if (!window.__qwHoverCounting) return;
            window.__qwHoverFrames += 1;
            requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        }"""
    )
    started = time.perf_counter()
    hovered = 0
    for index in indices:
        locator = page.locator("canvas").nth(int(index))
        try:
            locator.scroll_into_view_if_needed(timeout=5000)
            box = locator.bounding_box(timeout=5000)
        except Exception:
            continue
        if not box or box["width"] < 24 or box["height"] < 24:
            continue
        y = box["y"] + box["height"] * 0.5
        for fraction in (0.18, 0.42, 0.66, 0.84):
            x = box["x"] + box["width"] * fraction
            page.mouse.move(x, y, steps=4)
        hovered += 1
    remaining_ms = max(0, int(fps_ms - (time.perf_counter() - started) * 1000))
    if remaining_ms:
        page.wait_for_timeout(remaining_ms)
    fps = float(
        page.evaluate(
            r"""() => {
              window.__qwHoverCounting = false;
              const elapsed = performance.now() - (window.__qwHoverStart || performance.now());
              return (window.__qwHoverFrames || 0) * 1000 / Math.max(1, elapsed);
            }"""
        )
    )
    after_summary = _text_summary(page)
    return {
        "candidate_canvas_count": len(indices),
        "hovered_canvas_count": hovered,
        "max_canvases": max_canvases,
        "fps": round(fps, 1),
        "duration_s": round(time.perf_counter() - started, 3),
        "text_changed": before_summary != after_summary,
    }


def _write_report(artifact_dir: Path, metrics: dict[str, Any]) -> None:
    errors = metrics.get("errors", [])
    status_html = (
        '<p class="pass">No blocking errors found.</p>'
        if not errors
        else '<ul class="fail">' + "".join(f"<li>{_escape(error)}</li>" for error in errors) + "</ul>"
    )
    rows = [
        ("URL", metrics.get("url")),
        ("HTTP status", metrics.get("status")),
        ("Ready time", f"{metrics.get('load_to_ready_s')} s"),
        ("Initial browser rAF FPS", metrics.get("initial_fps")),
        ("Final browser rAF FPS", metrics.get("final_fps")),
        ("Canvas count", metrics.get("initial_canvas_count")),
        ("Show2D visible panels", metrics.get("initial_show2d_main_canvas_count")),
        ("Show2D blank panels", metrics.get("initial_show2d_blank_main_canvas_count")),
        ("Passed", metrics.get("passed")),
    ]
    summary_rows = "".join(f"<tr><th>{_escape(key)}</th><td>{_escape(value)}</td></tr>" for key, value in rows)
    steps_html = []
    for step in metrics.get("steps", []):
        shot = step.get("screenshot", {})
        image = f'<img src="{_escape(shot.get("rel", ""))}">' if shot.get("rel") else ""
        payload = {key: value for key, value in step.items() if key != "screenshot"}
        steps_html.append(
            f'<section class="card"><h2>{_escape(step.get("name"))}</h2>{image}'
            f"<pre>{_escape(json.dumps(payload, indent=2))}</pre></section>"
        )
    warnings = metrics.get("console_warnings_errors", [])
    warnings_html = "".join(
        f"<li>{_escape(item.get('type'))}: {_escape(item.get('text'))}</li>" for item in warnings
    ) or "<li>None captured</li>"
    show2d_main_html = ""
    if metrics.get("initial_show2d_main_canvases"):
        show2d_main_cards = []
        for item in metrics.get("initial_show2d_main_canvases", []):
            status = "PASS" if item.get("nonblank") else "FAIL"
            image = f'<img src="{_escape(item.get("rel", ""))}">' if item.get("rel") else ""
            show2d_main_cards.append(
                f'<section class="card"><h3>Panel {_escape(item.get("panel"))} · {status}</h3>'
                f"{image}<pre>{_escape(json.dumps(item.get('stats') or item.get('error'), indent=2))}</pre></section>"
            )
        show2d_main_html = (
            '<section class="card"><h2>Show2D visible main canvases</h2>'
            '<p>Every visible scientific image canvas must be nonblank.</p></section>'
            f'<div class="grid">{"".join(show2d_main_cards)}</div>'
        )
    initial = metrics.get("initial_screenshot", {})
    primary = metrics.get("initial_primary_canvas", {})
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>QuantEM external HTML profile</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172033; }}
    table {{ border-collapse: collapse; }}
    th {{ text-align: left; padding: 6px 16px 6px 0; vertical-align: top; white-space: nowrap; }}
    td {{ padding: 6px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 18px; }}
    .card {{ border: 2px solid #ff2bbd; border-radius: 8px; padding: 14px; margin: 14px 0; background: white; box-shadow: 0 0 0 3px rgba(255, 43, 189, 0.10); }}
    .card h2, .card h3 {{ display: inline-block; margin-top: 0; padding: 3px 8px; background: #ff2bbd; color: white; border-radius: 4px; }}
    img {{ max-width: 100%; border: 3px solid #ff2bbd; background: #2a001e; box-sizing: border-box; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 10px; border-radius: 6px; max-height: 360px; overflow: auto; }}
    code {{ background: #f0f2f5; padding: 1px 4px; border-radius: 4px; }}
    .pass {{ color: #087f23; font-weight: 700; }}
    .fail {{ color: #b00020; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>QuantEM external HTML profile</h1>
  {status_html}
  <section class="card"><h2>Summary</h2><table>{summary_rows}</table></section>
  <div class="grid">
    <section class="card"><h2>Initial viewport</h2><img src="{_escape(initial.get('rel', ''))}"></section>
    <section class="card"><h2>Initial primary canvas</h2><img src="{_escape(primary.get('rel', ''))}"><pre>{_escape(json.dumps(primary.get('stats'), indent=2))}</pre></section>
  </div>
  {show2d_main_html}
  <div class="grid">{''.join(steps_html)}</div>
  <section class="card"><h2>Console warnings/errors</h2><ul>{warnings_html}</ul></section>
  <section class="card"><h2>Raw data</h2><p><a href="metrics.json">metrics.json</a></p></section>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(doc, encoding="utf-8")


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - depends on local browser extras
        raise SystemExit("playwright is required for external HTML profiling") from exc

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    screenshot_dir = artifact_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    console: list[dict[str, str]] = []
    page_errors: list[str] = []
    metrics: dict[str, Any] = {
        "url": args.url,
        "artifact_dir": str(artifact_dir),
        "min_fps": args.min_fps,
        "viewport": {"width": args.viewport_width, "height": args.viewport_height},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.headed,
            executable_path=_chrome_executable(),
            args=["--disable-dev-shm-usage"],
        )
        try:
            context = browser.new_context(
                viewport={"width": args.viewport_width, "height": args.viewport_height},
                device_scale_factor=1,
            )
            page = context.new_page()
            page.on("console", lambda msg: console.append({"type": msg.type, "text": msg.text[:800]}))
            page.on("pageerror", lambda exc: page_errors.append(str(exc)[:1200]))

            started = time.perf_counter()
            response = page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)
            page.wait_for_selector("canvas", state="visible", timeout=args.timeout_ms)
            page.wait_for_timeout(args.settle_ms)
            metrics["status"] = response.status if response else None
            metrics["load_to_ready_s"] = round(time.perf_counter() - started, 3)
            metrics["initial_summary"] = _text_summary(page)
            metrics["initial_canvas_count"] = len(_visible_canvas_boxes(page))
            metrics["initial_fps"] = round(float(_measure_fps(page, args.fps_sample_ms)), 1)
            metrics["initial_screenshot"] = _screenshot(page, screenshot_dir / "00-initial.png")
            metrics["initial_primary_canvas"] = _primary_canvas_screenshot(page, screenshot_dir / "00-primary-canvas.png")
            metrics["initial_show2d_main_canvases"] = _show2d_main_canvas_screenshots(
                page,
                screenshot_dir,
                limit=args.main_canvas_check_limit,
                prefix="initial",
            )
            metrics["initial_show2d_main_canvas_count"] = len(metrics["initial_show2d_main_canvases"])
            metrics["initial_show2d_blank_main_canvas_count"] = sum(
                1 for item in metrics["initial_show2d_main_canvases"] if not item.get("nonblank")
            )

            summary = metrics["initial_summary"]
            if metrics["initial_show2d_main_canvas_count"] > 0:
                step = _run_show2d_keyboard_step(page, fps_ms=args.fps_sample_ms)
                step["name"] = "show2d_keyboard_shortcuts"
                step["screenshot"] = _screenshot(page, screenshot_dir / "01-show2d-keyboard.png")
                metrics["steps"].append(step)
                step = _run_show2d_zoom_pan_step(
                    page,
                    fps_ms=args.fps_sample_ms,
                    canvas_limit=args.contrast_canvas_check_limit,
                )
                step["name"] = "show2d_zoom_pan"
                step["screenshot"] = _screenshot(page, screenshot_dir / "02-show2d-zoom-pan.png")
                step["reset_after_screenshot"] = _reset_show2d_zoom_pan(page)
                metrics["steps"].append(step)
                step = _run_show2d_linked_contrast_step(
                    page,
                    fps_ms=args.fps_sample_ms,
                    canvas_limit=args.contrast_canvas_check_limit,
                )
                step["name"] = "show2d_linked_contrast_drag"
                step["screenshot"] = _screenshot(page, screenshot_dir / "07-show2d-linked-contrast.png")
                metrics["steps"].append(step)

            if _button_present(summary, "Play pages"):
                step = _run_play_button_step(
                    page,
                    "Play pages",
                    ["Pause pages", "Play pages"],
                    wait_ms=args.autoplay_wait_ms,
                    fps_ms=args.fps_sample_ms,
                )
                step["name"] = "page_autoplay"
                step["screenshot"] = _screenshot(page, screenshot_dir / "03-page-autoplay.png")
                metrics["steps"].append(step)

            if _button_present(summary, "Play forward"):
                step = _run_play_button_step(
                    page,
                    "Play forward",
                    ["Pause playback", "Stop and rewind to start"],
                    wait_ms=args.playback_wait_ms,
                    fps_ms=args.fps_sample_ms,
                )
                step["name"] = "frame_playback"
                step["screenshot"] = _screenshot(page, screenshot_dir / "04-frame-playback.png")
                metrics["steps"].append(step)

            hide_label = _first_hide_button(_text_summary(page))
            if hide_label:
                step = _run_hide_step(page, hide_label, fps_ms=args.fps_sample_ms)
                step["name"] = "hide_restore_panel"
                step["screenshot"] = _screenshot(page, screenshot_dir / "05-hide-panel.png")
                metrics["steps"].append(step)

            if not args.no_hover_stress:
                step = _run_hover_step(
                    page,
                    fps_ms=args.fps_sample_ms,
                    max_canvases=args.hover_canvas_limit,
                )
                step["name"] = "hover_sweep"
                step["screenshot"] = _screenshot(page, screenshot_dir / "06-hover-sweep.png")
                metrics["steps"].append(step)

            metrics["final_canvas_count"] = len(_visible_canvas_boxes(page))
            metrics["final_fps"] = round(float(_measure_fps(page, args.fps_sample_ms)), 1)
            metrics["final_summary"] = _text_summary(page)
            metrics["final_show2d_main_canvases"] = _show2d_main_canvas_screenshots(
                page,
                screenshot_dir,
                limit=args.main_canvas_check_limit,
                prefix="final",
            )
            metrics["final_show2d_main_canvas_count"] = len(metrics["final_show2d_main_canvases"])
            metrics["final_show2d_blank_main_canvas_count"] = sum(
                1 for item in metrics["final_show2d_main_canvases"] if not item.get("nonblank")
            )
            context.close()
        finally:
            browser.close()

    metrics["console_warnings_errors"] = [
        item for item in console if item["type"] in {"warning", "error"}
    ][: args.max_console_items]
    metrics["page_errors"] = page_errors

    errors: list[str] = []
    if metrics.get("status") != 200:
        errors.append(f"HTTP status was {metrics.get('status')}")
    if metrics.get("initial_canvas_count", 0) < args.min_canvases:
        errors.append(f"visible canvas count below {args.min_canvases}: {metrics.get('initial_canvas_count')}")
    if not metrics.get("initial_screenshot", {}).get("nonblank"):
        errors.append("initial viewport screenshot was blank or flat")
    if not metrics.get("initial_primary_canvas", {}).get("nonblank"):
        errors.append("initial primary canvas was blank or flat")
    blank_initial_panels = [
        item.get("panel") for item in metrics.get("initial_show2d_main_canvases", []) if not item.get("nonblank")
    ]
    if blank_initial_panels:
        errors.append(f"visible Show2D main canvases were blank/flat at first paint: {blank_initial_panels}")
    blank_final_panels = [
        item.get("panel") for item in metrics.get("final_show2d_main_canvases", []) if not item.get("nonblank")
    ]
    if blank_final_panels:
        errors.append(f"visible Show2D main canvases were blank/flat after interactions: {blank_final_panels}")
    if metrics.get("initial_fps", 0) < args.min_fps:
        errors.append(f"initial FPS {metrics.get('initial_fps')} below {args.min_fps}")
    if metrics.get("final_fps", 0) < args.min_fps:
        errors.append(f"final FPS {metrics.get('final_fps')} below {args.min_fps}")
    for step in metrics.get("steps", []):
        if step["name"] not in {"hover_sweep", "show2d_keyboard_shortcuts"} and not step.get("clicked"):
            errors.append(f"{step['name']} control was not clicked")
        if step.get("fps", args.min_fps) < args.min_fps:
            errors.append(f"{step['name']} FPS {step.get('fps')} below {args.min_fps}")
        if step["name"] == "show2d_keyboard_shortcuts":
            if not step.get("found"):
                errors.append(f"show2d_keyboard_shortcuts did not find a Show2D main canvas: {step.get('reason')}")
            if not step.get("after_click", {}).get("focused"):
                errors.append("show2d_keyboard_shortcuts did not focus the Show2D root after clicking the canvas")
            if step.get("canvas_count", 0) > 1 and not step.get("right_changed_selection"):
                errors.append("show2d_keyboard_shortcuts ArrowRight did not change the selected panel")
            if step.get("canvas_count", 0) > 1 and not step.get("left_restored_selection"):
                errors.append("show2d_keyboard_shortcuts ArrowLeft did not restore the selected panel")
            if not step.get("hide_changed_visible_count"):
                errors.append("show2d_keyboard_shortcuts H did not hide the selected panel")
            if step.get("hide_changed_visible_count") and not step.get("restore_clicked"):
                errors.append("show2d_keyboard_shortcuts did not restore the hidden panel")
        if step["name"] == "show2d_linked_contrast_drag":
            if not step.get("found"):
                errors.append(f"show2d_linked_contrast_drag did not find a histogram slider: {step.get('reason')}")
            expected = min(2, int(step.get("before_hash_count") or 0))
            if int(step.get("changed_panel_count") or 0) < expected:
                errors.append(
                    "show2d_linked_contrast_drag did not update enough visible panels: "
                    f"{step.get('changed_panel_count')} changed, expected at least {expected}"
                )
        if step["name"] == "show2d_zoom_pan":
            if not step.get("found"):
                errors.append(f"show2d_zoom_pan did not find a usable Show2D canvas: {step.get('reason')}")
            if int(step.get("changed_panel_count") or 0) <= 0:
                errors.append("show2d_zoom_pan did not change any checked scientific canvas")
            wheel_probe = step.get("wheel_event_probe") or {}
            drag_probe = step.get("drag_event_probe") or {}
            wheel_perf = step.get("wheel_perf_debug") or {}
            drag_perf = step.get("drag_perf_debug") or {}
            wheel_latencies = wheel_perf.get("zoomPanPaintLatenciesMs") or []
            drag_latencies = drag_perf.get("zoomPanPaintLatenciesMs") or []
            if int(wheel_probe.get("event_count") or 0) <= 0:
                errors.append("show2d_zoom_pan wheel zoom delivered no user input events")
            if int(drag_probe.get("event_count") or 0) <= 0:
                errors.append("show2d_zoom_pan drag pan delivered no user input events")
            if len(wheel_latencies) <= 0:
                errors.append("show2d_zoom_pan wheel zoom recorded no widget paint after input")
            if len(drag_latencies) <= 0:
                errors.append("show2d_zoom_pan drag pan recorded no widget paint after input")
        if step["name"] in {"page_autoplay", "frame_playback"} and not (
            step.get("text_changed") or step.get("canvas_signature_changed")
        ):
            errors.append(f"{step['name']} did not change visible text or canvas signature")
        if step["name"] == "hide_restore_panel" and not step.get("canvas_signature_changed"):
            errors.append("hide/restore panel did not change visible canvas signature")
        if step["name"] == "hide_restore_panel" and not step.get("restore_clicked"):
            errors.append("hide/restore panel did not restore the hidden panel")
        if step["name"] == "hover_sweep" and step.get("hovered_canvas_count", 0) <= 0:
            errors.append("hover_sweep did not hover any scientific canvases")
    if page_errors:
        errors.append("page JavaScript errors were reported")
    if args.fail_console_errors and metrics["console_warnings_errors"]:
        errors.append("console warnings/errors were reported")

    metrics["errors"] = errors
    metrics["passed"] = not errors
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_report(artifact_dir, metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Standalone exported widget HTML URL to profile.")
    parser.add_argument("--artifact-dir", default=str(_default_artifact_dir()), help="Directory for index.html and metrics.")
    parser.add_argument("--min-fps", type=float, default=30.0, help="Minimum rAF FPS for driven browser states.")
    parser.add_argument("--min-canvases", type=int, default=1, help="Minimum visible canvas count.")
    parser.add_argument("--fps-sample-ms", type=int, default=1500, help="Milliseconds for each rAF FPS sample.")
    parser.add_argument("--timeout-ms", type=int, default=180_000, help="Navigation and first-canvas timeout.")
    parser.add_argument("--settle-ms", type=int, default=2000, help="Delay after first canvas before measuring.")
    parser.add_argument("--autoplay-wait-ms", type=int, default=5000, help="Wait after clicking Play pages.")
    parser.add_argument("--playback-wait-ms", type=int, default=3500, help="Wait after clicking Play forward.")
    parser.add_argument("--viewport-width", type=int, default=1440)
    parser.add_argument("--viewport-height", type=int, default=980)
    parser.add_argument("--max-console-items", type=int, default=40)
    parser.add_argument("--fail-console-errors", action="store_true", help="Treat console warnings/errors as failures.")
    parser.add_argument("--hover-canvas-limit", type=int, default=80, help="Maximum scientific canvases to hover during the hover sweep.")
    parser.add_argument("--main-canvas-check-limit", type=int, default=80, help="Maximum visible Show2D main canvases to screenshot and verify.")
    parser.add_argument("--contrast-canvas-check-limit", type=int, default=8, help="Visible Show2D main canvases to hash before/after contrast drag.")
    parser.add_argument("--no-hover-stress", action="store_true", help="Skip the pointer hover FPS sweep.")
    parser.add_argument("--headed", action="store_true", help="Run Chromium headed for local debugging.")
    args = parser.parse_args()

    metrics = run_profile(args)
    print(f"External HTML profile: {metrics['artifact_dir']}/index.html")
    if metrics["passed"]:
        print("PASS")
        return 0
    print("FAIL")
    for error in metrics["errors"]:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
