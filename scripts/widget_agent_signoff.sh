#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/widget_agent_signoff.sh [--quick] [--full] [--artifact-dir DIR] [--docs-url URL] [--jupyter-url URL]

Creates a human-drive widget signoff packet for agent-assisted QA.

This is not a replacement for browser driving. It prepares the checklist,
artifact folders, and optional local gates. The agent then opens the listed
URLs, drives the widgets, fixes issues, rebuilds, refreshes, and redrives.

Modes:
  default           Prepare the signoff packet only.
  --quick           Prepare packet, then run frontend typecheck/build.
  --full            Prepare packet, run Python tests, frontend checks, and quick visual smoke.

Options:
  --artifact-dir    Output directory for report, screenshots, videos, and logs.
                    Default: /tmp/quantem-widget-agent-signoff/<timestamp>
  --docs-url        Docs URL to put in the checklist.
                    Default: http://127.0.0.1:8767
  --jupyter-url     Optional JupyterLab URL for notebook/live-kernel signoff.

Environment:
  QT_AGENT_SIGNOFF_DIR     Alternative default artifact directory.
  QT_AGENT_DOCS_URL        Alternative default docs URL.
  QT_AGENT_JUPYTER_URL     Alternative default JupyterLab URL.
EOF
}

mode="prepare"
artifact_dir="${QT_AGENT_SIGNOFF_DIR:-}"
docs_url="${QT_AGENT_DOCS_URL:-http://127.0.0.1:8767}"
jupyter_url="${QT_AGENT_JUPYTER_URL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --quick)
      mode="quick"
      ;;
    --full)
      mode="full"
      ;;
    --artifact-dir)
      if [[ $# -lt 2 ]]; then
        echo "--artifact-dir requires a value" >&2
        exit 2
      fi
      artifact_dir="$2"
      shift
      ;;
    --docs-url)
      if [[ $# -lt 2 ]]; then
        echo "--docs-url requires a value" >&2
        exit 2
      fi
      docs_url="$2"
      shift
      ;;
    --jupyter-url)
      if [[ $# -lt 2 ]]; then
        echo "--jupyter-url requires a value" >&2
        exit 2
      fi
      jupyter_url="$2"
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

cd "$(dirname "$0")/.."

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$artifact_dir" ]]; then
  artifact_dir="/tmp/quantem-widget-agent-signoff/$timestamp"
fi

mkdir -p "$artifact_dir"/{checklists,screenshots,videos,logs,exports}

branch="$(git branch --show-current || true)"
commit="$(git rev-parse --short HEAD)"
remote_commit="$(git rev-parse --short origin/main 2>/dev/null || true)"

git status --short > "$artifact_dir/logs/git-status.txt"
git diff --stat > "$artifact_dir/logs/git-diff-stat.txt"

cat > "$artifact_dir/README.md" <<EOF
# quantem.widget agent signoff

- Created: $timestamp
- Repo: $(pwd)
- Branch: ${branch:-unknown}
- Commit: $commit
- origin/main: ${remote_commit:-unknown}
- Docs URL: $docs_url
- JupyterLab URL: ${jupyter_url:-not provided}

## Rule

Drive the widgets in the browser. If anything is slow, stale, blank, misaligned,
or unreadable, patch it immediately, rebuild, refresh, and redrive the same
interaction. Do not mark a widget as passed until it was driven after the last
code change.

## Artifact convention

- screenshots/<widget>-after.png
- videos/<widget>-motion.mp4 or .webm
- logs/<widget>-console.txt
- logs/<widget>-notes.md

Use screenshots for final layout/theme evidence. Use short videos for
motion-sensitive interactions such as histogram center drag, ShowEDS energy-band
drag, Show4DSTEM detector drag, and Show3DSlices scrub/rotate.
EOF

cat > "$artifact_dir/report.md" <<EOF
# Widget agent signoff report

## Summary

| Widget | Surface | Result | Evidence | Notes |
|---|---|---:|---|---|
| Show2D | docs/export/Jupyter | TODO | TODO | TODO |
| Show3D | docs/export/Jupyter | TODO | TODO | TODO |
| Show3DSlices | docs/export/Jupyter | TODO | TODO | TODO |
| Show4DSTEM | docs/export/Jupyter | TODO | TODO | TODO |
| ShowEDS | docs/export/Jupyter | TODO | TODO | TODO |
| ShowDiffraction | docs/export/Jupyter | TODO | TODO | TODO |

## Fix-and-redrive log

| Time | Widget | Issue observed | Fix made | Redriven evidence |
|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO |

## Console / performance notes

- TODO

## Release decision

- [ ] PASS
- [ ] PASS WITH LIMITATION
- [ ] BLOCKED
EOF

cat > "$artifact_dir/checklists/show2d.md" <<EOF
# Show2D human-drive checklist

Open: $docs_url/tutorials/show2d.html

- [ ] Widget renders nonblank; no widget model or console errors.
- [ ] Histogram min and max handles move immediately.
- [ ] Histogram center drag moves both handles with no visible lag.
- [ ] Auto, smooth, FFT, profile/lens/ROI, colorbar, and scale controls visibly change the view where present.
- [ ] Wheel zoom and pan feel attached to the pointer.
- [ ] ROI add/drag/resize works and labels use (row, col).
- [ ] Colormap and scale menus work.
- [ ] Export button works when present and reports status/size.
- [ ] Light and dark theme text, borders, histogram, controls, and scale bar remain readable.
- [ ] Final screenshot saved to screenshots/show2d-after.png.
- [ ] Short video saved if any motion felt questionable.
EOF

cat > "$artifact_dir/checklists/show3d.md" <<EOF
# Show3D human-drive checklist

Open: $docs_url/tutorials/show3d.html

- [ ] Widget renders nonblank; no widget model or console errors.
- [ ] Frame slider scrubs immediately.
- [ ] Play/pause, reverse, loop, bounce, and FPS controls behave as labeled.
- [ ] Histogram min/max and center drag are immediate.
- [ ] Auto/smooth/color/scale/colorbar controls visibly change the view where present.
- [ ] Wheel zoom and pan work when supported.
- [ ] Export button works when present and reports status/size.
- [ ] Light and dark theme controls and labels remain readable.
- [ ] Final screenshot saved to screenshots/show3d-after.png.
- [ ] Short video saved for playback or histogram concerns.
EOF

cat > "$artifact_dir/checklists/show3dslices.md" <<EOF
# Show3DSlices human-drive checklist

Open: $docs_url/tutorials/show3dslices.html

- [ ] Widget renders all panels nonblank; no widget model or console errors.
- [ ] Three panels are top-aligned and have independent sensible sizes.
- [ ] Control groups are content-sized; labels such as "Vol Strength" do not wrap.
- [ ] Crosshair drag updates all linked panels.
- [ ] Slice, angle, and position sliders update live.
- [ ] 3D/volume view rotates, resets, and zooms where supported.
- [ ] FFT, log, smooth, colorbar, z-stretch, flip, auto, and playback controls work.
- [ ] Histogram min/max and center drag are immediate.
- [ ] Export and reset controls sit in the expected top-right action area.
- [ ] Light and dark theme controls, plots, histograms, and scale bars remain readable.
- [ ] Final screenshot saved to screenshots/show3dslices-after.png.
- [ ] Short video saved for scrub/rotate/histogram concerns.
EOF

cat > "$artifact_dir/checklists/show4dstem.md" <<EOF
# Show4DSTEM human-drive checklist

Open: $docs_url/tutorials/show4dstem.html

- [ ] Diffraction and virtual image render nonblank; no widget model or console errors.
- [ ] Detector center/radius drag updates the virtual image live.
- [ ] BF, ABF, and ADF presets update detector geometry and virtual image.
- [ ] Diffraction pan/zoom and virtual-image pan/zoom work.
- [ ] Histogram min/max and center drag are immediate.
- [ ] FFT/profile/ROI/path playback controls work where present.
- [ ] WebGPU path is active when expected; no SwiftShader fallback for GPU signoff.
- [ ] Export button works when present and reports status/size.
- [ ] Light and dark theme controls, labels, plots, detector handles, and scale bars remain readable.
- [ ] Final screenshot saved to screenshots/show4dstem-after.png.
- [ ] Short video saved for detector drag or WebGPU concerns.
EOF

cat > "$artifact_dir/checklists/showeds.md" <<EOF
# ShowEDS human-drive checklist

Open: $docs_url/tutorials/showeds.html
Optional JupyterLab real-data URL: ${jupyter_url:-provide live notebook URL}

- [ ] Widget renders map and spectrum nonblank; no widget model or console errors.
- [ ] Energy-band edge drags update the spectrum band, text, and element map.
- [ ] Energy-band center drag moves band, slider handles, labels, and element map together with no visible lag.
- [ ] ROI rectangle drag/resize updates the summed spectrum live.
- [ ] ROI ellipse/circle drag/resize works when enabled; handles match the shape.
- [ ] Log, scale, smooth, auto contrast, overlay, save ROI, and save band controls work.
- [ ] Periodic-table/element picker opens, looks polished, and peak overlays update.
- [ ] Candidate peak labels are readable and not cluttered.
- [ ] Real-data notebook path is tested when the change touches EDS loading or performance.
- [ ] No crop/bin/downsample is hidden from the user; any reduction is explicit.
- [ ] Export works for supported modes and reports status/size.
- [ ] Light and dark theme map, spectrum, histogram, controls, export UI, ROI/band handles remain readable.
- [ ] Final screenshot saved to screenshots/showeds-after.png.
- [ ] Short video saved for band drag, ROI drag, and any real-data lag concern.
EOF

cat > "$artifact_dir/checklists/showdiffraction.md" <<EOF
# ShowDiffraction human-drive checklist

Open: $docs_url/tutorials/showdiffraction.html

- [ ] Pattern renders nonblank; no widget model or console errors.
- [ ] Center auto/manual controls move the crosshair and update d-spacing readouts.
- [ ] Add/remove spot works; labels and table update.
- [ ] Add/remove ring works; ring labels update.
- [ ] Snap/refine/detect controls behave as labeled.
- [ ] Frame slider scrubs immediately for stack data.
- [ ] Contrast histogram min/max and center drag are immediate.
- [ ] Pan/zoom work and remain calibrated.
- [ ] Export button works when present and reports status/size.
- [ ] Light and dark theme labels, overlays, markers, histogram, and controls remain readable.
- [ ] Final screenshot saved to screenshots/showdiffraction-after.png.
- [ ] Short video saved for spot/ring or histogram concerns.
EOF

echo "== agent signoff packet =="
echo "$artifact_dir"
echo
echo "Wrote:"
find "$artifact_dir" -maxdepth 2 -type f | sort

if [[ "$mode" == "quick" || "$mode" == "full" ]]; then
  echo "== frontend typecheck/build =="
  npm run typecheck
  npm run build
fi

if [[ "$mode" == "full" ]]; then
  echo "== full Python test gate =="
  PYTHONPATH=src pytest -q

  echo "== quick visual smoke =="
  scripts/widget_visual_signoff.sh --quick
fi

cat <<EOF

Next:
1. Open the URLs in the in-app browser.
2. Drive each affected widget using checklists/*.md.
3. Patch anything that feels off.
4. Rebuild, refresh, and redrive after every fix.
5. Save screenshots/videos/logs into:
   $artifact_dir
6. Fill in report.md before release.
EOF

