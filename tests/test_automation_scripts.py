from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _write_notebook(path: Path, output_text: str = "") -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": output_text,
                    }
                ],
                "source": "print('ok')\n",
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


def test_notebook_size_guard_accepts_small_notebook(tmp_path: Path) -> None:
    notebook = tmp_path / "small.ipynb"
    _write_notebook(notebook, "small output")

    result = _run(sys.executable, "scripts/check_notebook_sizes.py", str(notebook), "--max-mb", "1")

    assert result.returncode == 0, result.stdout
    assert "Notebook size guard passed" in result.stdout


def test_notebook_size_guard_rejects_large_embedded_output(tmp_path: Path) -> None:
    notebook = tmp_path / "large.ipynb"
    _write_notebook(notebook, "x" * 4096)

    result = _run(
        sys.executable,
        "scripts/check_notebook_sizes.py",
        str(notebook),
        "--max-mb",
        "1",
        "--max-output-mb",
        "0.001",
    )

    assert result.returncode == 1
    assert "embeds" in result.stdout


def test_large_file_guard_accepts_small_explicit_file(tmp_path: Path) -> None:
    data = tmp_path / "small.npy"
    data.write_bytes(b"0" * 128)

    result = _run(
        sys.executable,
        "scripts/check_large_files.py",
        str(data),
        "--max-mb",
        "1",
        "--data-max-mb",
        "1",
    )

    assert result.returncode == 0, result.stdout
    assert "Tracked file size guard passed" in result.stdout


def test_large_file_guard_rejects_large_data_artifact(tmp_path: Path) -> None:
    data = tmp_path / "large.npy"
    data.write_bytes(b"0" * 4096)

    result = _run(
        sys.executable,
        "scripts/check_large_files.py",
        str(data),
        "--max-mb",
        "1",
        "--data-max-mb",
        "0.001",
    )

    assert result.returncode == 1
    assert "data/rendered artifact" in result.stdout


def test_large_file_guard_allows_show4dstem_readme_demo() -> None:
    result = _run(
        sys.executable,
        "scripts/check_large_files.py",
        "docs/_static/show4dstem-serin-gold.gif",
    )

    assert result.returncode == 0, result.stdout
    assert "Approved size exception" in result.stdout


def test_automation_documentation_names_entrypoints() -> None:
    doc = (ROOT / "docs/maintainer/automation.md").read_text(encoding="utf-8")

    for path in [
        "scripts/widget_local_signoff.sh",
        "scripts/widget_signoff_dashboard.py",
        "scripts/docs_preview.sh",
        "scripts/cleanup_browser_artifacts.py",
        "scripts/widget_html_smoke.py",
        "scripts/widget_showfolder_live_smoke.py",
        "scripts/widget_show3d_animation_smoke.py",
        "scripts/widget_browser_smoke.py",
        "scripts/widget_phone_handoff.py",
        "scripts/widget_performance_smoke.py",
        "scripts/widget_heavy_perf_signoff.py",
        "scripts/widget_show4dstem_heavy_signoff.py",
        "scripts/widget_load_bench_matrix.py",
        "scripts/widget_load_bench_sharded.py",
        "scripts/check_large_files.py",
        "scripts/check_notebook_sizes.py",
        ".github/workflows/widget-ci.yml",
        "--artifact-dir",
        "index.html",
        "browser-plan.json",
        "showfolder-live/index.html",
        "ShowFolder live-folder smoke",
        "Show3D GIF presentation smoke",
        "--dry-run",
        "--panel-gap",
        "--no-panel-labels",
        "--max-work-mb",
        "browser-smoke.html",
        "--mobile",
        "--min-fps",
        "--full --browser --performance",
        "performance/browser-smoke.html",
        "Real-data Show2D/Show3D browser smoke",
        "storyboard IDs",
        "390x844",
        "never normal CI",
        "FFT overlay cache counters",
        "FFT metric stats-toggle cache report",
        "Performance Evidence Registry",
        "fft-metric-quick-cache-guard",
        "gold-haadf-local-quick",
        "fft-metric-full-cache-guard",
        "gold-haadf-local-full",
        "stale browser artifact cleanup",
        "Which Command Should I Run?",
        "Definition Of Done",
        "Do Not Do This",
        "Report Artifacts",
        "signoff-dashboard.json",
        "everything dashboard",
        "failed gates",
        "next recommended",
        "actions/upload-artifact",
        "workflow_dispatch",
        "http://127.0.0.1:8779/index.html",
        "QUANTEM_WIDGET_REAL_DATA_ROOTS",
        "QUANTEM_WIDGET_4DSTEM_ROOTS",
        "QUANTEM_WIDGET_BENCH_MASTERS_GLOB",
        "synthetic data for real-data performance claims",
        "final answer names the exact command and report path",
        "Do not run `pkill chrome`",
    ]:
        assert path in doc


def test_ci_workflow_uploads_signoff_artifacts() -> None:
    workflow = (ROOT / ".github/workflows/widget-ci.yml").read_text(encoding="utf-8")

    assert "--artifact-dir /tmp/quantem-widget-ci-signoff" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "widget-ci-signoff" in workflow
    assert "if: always()" in workflow
    assert "signoff_mode" in workflow
    assert "inputs.browser" in workflow
    assert "inputs.mobile" in workflow
    assert "inputs.performance" in workflow


def test_github_actions_use_node24_action_generations() -> None:
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github/workflows").glob("*.yml"))
    )

    for expected in [
        "actions/checkout@v7",
        "actions/setup-python@v6",
        "actions/setup-node@v6",
        "actions/upload-artifact@v7",
        "actions/configure-pages@v6",
        "actions/upload-pages-artifact@v5",
        "actions/deploy-pages@v5",
    ]:
        assert expected in workflows
    for stale in [
        "actions/checkout@v4",
        "actions/checkout@v6",
        "actions/setup-python@v5",
        "actions/upload-artifact@v4",
        "actions/configure-pages@v5",
        "actions/upload-pages-artifact@v3",
        "actions/deploy-pages@v4",
    ]:
        assert stale not in workflows


def test_real_data_loader_benchmark_scripts_are_documented_as_local_only() -> None:
    for script in [
        ROOT / "scripts/widget_load_bench_matrix.py",
        ROOT / "scripts/widget_load_bench_sharded.py",
    ]:
        result = _run(sys.executable, str(script), "--help")
        source = script.read_text(encoding="utf-8")
        normalized = result.stdout.replace("\n", "").replace(" ", "")

        assert result.returncode == 0, result.stdout
        assert "private real" in result.stdout
        assert "masters-glob" in result.stdout
        assert "/tmp/quantem-widget-load-bench" in normalized
        assert "from quantem.gpu.io import load" in source
        assert "quantem.widget.io.hdf5" not in source


def test_sharded_loader_benchmark_exercises_public_u8_api() -> None:
    script = (ROOT / "scripts/widget_load_bench_sharded.py").read_text(encoding="utf-8")

    assert 'kwargs["dtype"] = "u8"' in script
    assert 'kwargs["output_dtype"]' not in script
    assert "_assign_indices_to_devices" in script


def test_signoff_dashboard_summarizes_available_reports(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "signoff"
    html_smoke = artifact_dir / "html-smoke"
    html_smoke.mkdir(parents=True)
    showfolder_live = artifact_dir / "showfolder-live"
    showfolder_live.mkdir()
    performance = artifact_dir / "performance"
    performance.mkdir()
    external = artifact_dir / "external-html-profile"
    external.mkdir()
    gif = artifact_dir / "show3d-gif"
    gif.mkdir()
    (artifact_dir / "signoff-manifest.json").write_text(
        json.dumps({
            "status": "pass",
            "commit": "abc123",
            "branch": "main",
            "mode": "quick",
            "browser_smoke": True,
            "duration_seconds": 12,
        }),
        encoding="utf-8",
    )
    (html_smoke / "index.html").write_text("<html>matrix</html>", encoding="utf-8")
    (html_smoke / "report.json").write_text(
        json.dumps({
            "total_size_mb": 2.0,
            "exports": [
                {"widget": "show2d", "variant": "show2d-single", "seconds": 0.2, "size_mb": 1.25},
                {"widget": "show3d", "variant": "show3d-stack", "seconds": 0.4, "size_mb": 0.75},
            ],
        }),
        encoding="utf-8",
    )
    (html_smoke / "browser-smoke.html").write_text("<html>browser</html>", encoding="utf-8")
    (html_smoke / "browser-smoke-report.json").write_text(
        json.dumps({
            "passed": 2,
            "mobile": True,
            "pages": [
                {"passed": True, "fps": 61.0, "screenshot": "screenshots/show2d.png"},
                {"passed": True, "fps": 58.4, "screenshot": "screenshots/show3d.png"},
            ],
        }),
        encoding="utf-8",
    )
    (showfolder_live / "index.html").write_text("<html>live</html>", encoding="utf-8")
    (showfolder_live / "report.json").write_text(
        json.dumps({"passed": True, "steps": [{"name": "live", "passed": True}]}),
        encoding="utf-8",
    )
    (performance / "index.html").write_text("<html>performance</html>", encoding="utf-8")
    (performance / "report.json").write_text(
        json.dumps({
            "timings": [{"case": "load_show3d_real_derived_stack", "seconds": 1.25}],
            "exports": [{"widget": "show3d", "variant": "real", "seconds": 2.5, "size_mb": 12.5}],
            "too_large": [],
        }),
        encoding="utf-8",
    )
    (performance / "browser-smoke.html").write_text("<html>performance browser</html>", encoding="utf-8")
    (performance / "browser-smoke-report.json").write_text(
        json.dumps({
            "passed": 1,
            "pages": [
                {"passed": True, "fps": 55.0, "screenshot": "screenshots/show3d-real.png"},
            ],
        }),
        encoding="utf-8",
    )
    (external / "index.html").write_text("<html>profile</html>", encoding="utf-8")
    (external / "metrics.json").write_text(
        json.dumps({
            "passed": True,
            "url": "http://127.0.0.1:8779/example.html",
            "load_to_ready_s": 1.1,
            "initial_fps": 59.5,
            "final_fps": 58.8,
            "initial_canvas_count": 3,
            "steps": [{"name": "page_autoplay", "fps": 58.2}],
        }),
        encoding="utf-8",
    )
    (gif / "index.html").write_text("<html>gif</html>", encoding="utf-8")
    (gif / "report.json").write_text(
        json.dumps({
            "source": {"kind": "synthetic CI fallback"},
            "input_shape": [5, 64, 64],
            "fps": 5,
            "panel_gap": 0,
            "planned_exports": [{"quality": "medium"}],
            "playback": "bounce",
            "total_size_mb": 0.4,
            "exports": [{"quality": "medium", "size_mb": 0.4}],
        }),
        encoding="utf-8",
    )
    (artifact_dir / "heavy-signoff-report.json").write_text(
        json.dumps({
            "passed": True,
            "browser": {"passed": 1, "pages": [{"passed": True, "fps": 55.0}]},
            "show2d_paged_scrub": {"passed": True, "page_scrub_max_ms": 120.0},
            "show3d_paged_scrub": {"passed": True, "page_scrub_max_ms": 180.0},
            "show3d_fft_idle": {"passed": True, "fps": 57.5},
        }),
        encoding="utf-8",
    )

    result = _run(sys.executable, "scripts/widget_signoff_dashboard.py", "--artifact-dir", str(artifact_dir))

    assert result.returncode == 0, result.stdout
    dashboard = (artifact_dir / "index.html").read_text(encoding="utf-8")
    dashboard_json = json.loads((artifact_dir / "signoff-dashboard.json").read_text(encoding="utf-8"))
    assert "quantem.widget everything dashboard: PASS" in dashboard
    assert "HTML export smoke" in dashboard
    assert "Browser HTML smoke" in dashboard
    assert "ShowFolder live-folder smoke" in dashboard
    assert "Real-data Show2D/Show3D performance smoke" in dashboard
    assert "Real-data Show2D/Show3D browser smoke" in dashboard
    assert "External exported HTML profile(s)" in dashboard
    assert "Show3D GIF/MP4 presentation smoke" in dashboard
    assert "Heavy Show2D/Show3D signoff" in dashboard
    assert "Measured Performance" in dashboard
    assert "Evidence Gates</div><div class='kpi-value'>" in dashboard
    assert "Min FPS: 58.4" in dashboard
    assert "Slowest backend step: load_show3d_real_derived_stack 1.25 s" in dashboard
    assert "Show2D page max" in dashboard
    assert "showfolder-live/index.html" in dashboard
    assert "html-smoke/browser-smoke.html" in dashboard
    assert "performance/index.html" in dashboard
    assert "performance/browser-smoke.html" in dashboard
    assert "external-html-profile/index.html" in dashboard
    assert "show3d-gif/index.html" in dashboard
    assert dashboard_json["manifest"]["commit"] == "abc123"
    assert dashboard_json["summary"]["status_counts"]["pass"] >= 7
    assert not dashboard_json["summary"]["failed_gates"]


def test_local_signoff_drives_performance_exports_when_browser_enabled() -> None:
    script = (ROOT / "scripts/widget_local_signoff.sh").read_text(encoding="utf-8")

    assert 'echo "== browser-drive real-data performance smoke =="' in script
    assert 'perf_browser_args=(--artifact-dir "$artifact_dir/performance")' in script
    assert 'python scripts/widget_browser_smoke.py "${perf_browser_args[@]}"' in script


def test_maintained_automation_docs_use_generic_backend_names() -> None:
    checked_paths = [
        ROOT / "docs/maintainer/automation.md",
        ROOT / "docs/maintainer/performance-ui-testing.md",
        ROOT / "docs/maintainer/storyboard.md",
        ROOT / "docs/maintainer/storyboard-show2d.md",
        ROOT / "docs/maintainer/storyboard-show4dstem.md",
        ROOT / "docs/maintainer/storyboard-showeds.md",
        ROOT / "docs/maintainer/widget-agent-signoff.md",
        ROOT / "js/show3d/index.tsx",
        ROOT / "scripts/widget_agent_signoff.sh",
        ROOT / "scripts/widget_heavy_perf_signoff.py",
        ROOT / "scripts/widget_show4dstem_heavy_signoff.py",
        ROOT / "scripts/widget_performance_smoke.py",
    ]

    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked_paths)
    for forbidden in [
        "MJ" "GOAT",
        "Mj" "goat",
        "mj" "goat",
        "Ph" "il",
        "ph" "il",
        "Rod" "man",
        "rod" "man",
    ]:
        assert forbidden not in combined


def test_browser_artifact_cleanup_removes_old_quantem_artifacts(tmp_path: Path) -> None:
    old_profile = tmp_path / "playwright-artifacts-stale"
    old_profile.mkdir()
    old_log = tmp_path / "quantem_chrome_stale.log"
    old_log.write_text("old chrome log", encoding="utf-8")
    recent_profile = tmp_path / "playwright-artifacts-recent"
    recent_profile.mkdir()

    old_mtime = 1_700_000_000
    recent_mtime = 4_000_000_000
    for path in [old_profile, old_log]:
        os.utime(path, (old_mtime, old_mtime))
    os.utime(recent_profile, (recent_mtime, recent_mtime))

    result = _run(
        sys.executable,
        "scripts/cleanup_browser_artifacts.py",
        "--tmp-root",
        str(tmp_path),
        "--older-than-hours",
        "1",
        "--json",
    )

    assert result.returncode == 0, result.stdout
    report = json.loads(result.stdout)
    assert report["removed"] == 2
    assert not old_profile.exists()
    assert not old_log.exists()
    assert recent_profile.exists()


def test_browser_artifact_cleanup_dry_run_keeps_files(tmp_path: Path) -> None:
    old_profile = tmp_path / "com.google.Chrome.stale"
    old_profile.mkdir()
    os.utime(old_profile, (1_700_000_000, 1_700_000_000))

    result = _run(
        sys.executable,
        "scripts/cleanup_browser_artifacts.py",
        "--tmp-root",
        str(tmp_path),
        "--older-than-hours",
        "1",
        "--dry-run",
        "--json",
    )

    assert result.returncode == 0, result.stdout
    assert old_profile.exists()
    report = json.loads(result.stdout)
    assert any(item["status"] == "would_remove" for item in report["records"])


def test_heavy_perf_signoff_help_documents_local_only_contract() -> None:
    result = _run(sys.executable, "scripts/widget_heavy_perf_signoff.py", "--help")

    assert result.returncode == 0, result.stdout
    assert "local-only real-data widget performance signoff" in result.stdout
    assert "--skip-browser" in result.stdout
    assert "--min-fps" in result.stdout
    assert "--idle-seconds" in result.stdout


def test_show4dstem_heavy_signoff_help_documents_local_only_contract() -> None:
    result = _run(sys.executable, "scripts/widget_show4dstem_heavy_signoff.py", "--help")

    assert result.returncode == 0, result.stdout
    assert "local-only real-data Show4DSTEM heavy performance signoff" in result.stdout
    assert "--skip-browser" in result.stdout
    assert "--min-fps" in result.stdout
    assert "--max-masters" in result.stdout
    assert "--export-det-bin" in result.stdout


def test_widget_html_smoke_writes_visual_report(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "html-smoke"

    result = _run(
        sys.executable,
        "scripts/widget_html_smoke.py",
        "--artifact-dir",
        str(artifact_dir),
        "--max-total-mb",
        "25",
    )

    assert result.returncode == 0, result.stdout
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    plan = json.loads((artifact_dir / "browser-plan.json").read_text(encoding="utf-8"))
    index = (artifact_dir / "index.html").read_text(encoding="utf-8")

    assert len(report["exports"]) == 18
    assert sum(1 for item in report["exports"] if item["widget"] == "show2d") >= 5
    assert sum(1 for item in report["exports"] if item["widget"] == "show3d") >= 6
    assert {item["widget"] for item in report["exports"]} == {
        "show2d",
        "show3d",
        "show3dslices",
        "show4dstem",
        "showptycho",
        "showeds",
        "showdiffraction",
        "showfolder",
    }
    assert "show2d-gallery-6-fft.html" in index
    assert "show3d-four-panel-downsample.html" in index
    assert "show4dstem-compare.html" in index
    assert "showptycho-webgpu-folder/index.html" in index
    assert "showfolder.html" in index
    assert "synthetic MoS2-like HAADF lattice" in index
    assert "ShowPtycho" in index
    assert {page["widget"] for page in plan["pages"]} == {
        "show2d",
        "show3d",
        "show3dslices",
        "show4dstem",
        "showptycho",
        "showeds",
        "showdiffraction",
        "showfolder",
    }
    assert any(
        page["url_path"] == "showptycho-webgpu-folder/index.html"
        for page in plan["pages"]
    )


def test_widget_showfolder_live_smoke_writes_report(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "showfolder-live"

    result = _run(
        sys.executable,
        "scripts/widget_showfolder_live_smoke.py",
        "--artifact-dir",
        str(artifact_dir),
    )

    assert result.returncode == 0, result.stdout
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    plan = json.loads(
        (artifact_dir / "browser-plan.json").read_text(encoding="utf-8")
    )
    index = (artifact_dir / "index.html").read_text(encoding="utf-8")

    assert report["passed"] is True
    assert len(report["steps"]) == 5
    image_step = report["steps"][0]
    master_step = report["steps"][1]
    assert image_step["watch_changed"] is True
    assert image_step["show2d_panels"] == 3
    assert image_step["show3d_slices"] == 3
    assert len(image_step["thumbnail_previews"]) == 3
    for preview in image_step["thumbnail_previews"]:
        assert (artifact_dir / preview["webp"]).exists()
    assert master_step["watch_changed"] is True
    assert master_step["after_frames"] == 2
    assert master_step["frame_labels"] == ["scan_000", "scan_001"]
    assert [row["status"] for row in master_step["master_qc"]] == ["ready", "ready"]
    assert master_step["uses_monkeypatch"] is True

    direct_steps = report["steps"][2:]
    assert [step["kind"] for step in direct_steps] == [
        "direct_public_from_folder",
        "direct_public_from_folder",
        "direct_public_from_folder",
    ]
    assert all(step["uses_monkeypatch"] is False for step in direct_steps)
    direct_show2d, direct_show3d, direct_show4d = direct_steps
    for step in (direct_show2d, direct_show3d):
        # C1: public image viewers mount before accepting their initial file,
        # expect probation, one stable append, later arrival, and one model ID.
        assert step["same_mounted_model"] is True
        assert step["initial_probation_added"] == [0]
        assert step["arrival_probation_added"] == []
        assert step["stable_arrival_added"] == [1]
        assert step["final_count"] == 2
        assert step["static_watch_contract"] == {
            "embedded_states": ["hidden"],
            "watching_embedded": False,
            "hidden_snapshot": True,
        }
        states = [point["state"] for point in step["timeline"]]
        for required in ("waiting", "updating", "watching", "stopped"):
            assert required in states

    # C2: direct production GPU Show4DSTEM appends after header probation,
    # expect fresh visible-page pixels before its final green state. CPU-only
    # CI skips this native-GPU integration without introducing a fallback.
    show4d_skipped = bool(direct_show4d.get("skipped", False))
    if show4d_skipped:
        assert direct_show4d["skip_reason"] == (
            "No native CUDA or MPS backend is available."
        )
    else:
        assert direct_show4d["same_mounted_model"] is True
        assert direct_show4d["arrival_probation_added"] == []
        assert direct_show4d["stable_arrival_added"] == [1]
        assert direct_show4d["authoritative_before_green"] is True
        assert direct_show4d["active_page_indices"] == [0, 1]
        if direct_show4d["backend"] == "mps":
            assert direct_show4d["active_page_loaded_count"] in {0, 1}
            assert len(direct_show4d["virtual_image_means"]) == 1
        else:
            assert direct_show4d["active_page_loaded_count"] == 2
            assert direct_show4d["virtual_image_means"][1] > direct_show4d[
                "virtual_image_means"
            ][0]
        green = [
            point
            for point in direct_show4d["timeline"]
            if point["state"] == "watching" and point["count"] == 2
        ][-1]
        assert green["compare_page_loading"] is False
        if direct_show4d["backend"] != "mps":
            assert green["compare_page_loaded_count"] == 2
        assert green["compare_panel_indices"] == [0, 1]
        assert direct_show4d["static_watch_contract"]["watching_embedded"] is False

    assert len(report["exports"]) == (6 if show4d_skipped else 7)
    assert len(plan["pages"]) == len(report["exports"])
    expected_static_variants = {
        "show2d-folder-watch-static",
        "show3d-folder-watch-static",
    }
    if not show4d_skipped:
        expected_static_variants.add("show4dstem-folder-watch-static")
    assert {
        row["variant"]
        for row in report["exports"]
        if row["variant"].endswith("folder-watch-static")
    } == expected_static_variants
    for row in report["exports"]:
        assert Path(row["path"]).exists()
    assert "ShowFolder live-folder smoke: PASS" in index
    assert "<img" in index
    assert "4D-STEM Master QC" in index
    assert "Direct Viewer Lifecycle Timeline" in index
    assert (artifact_dir / "showfolder-live-show2d.html").exists()
    assert (artifact_dir / "showfolder-live-show3d.html").exists()
    assert (artifact_dir / "showfolder-live-show4dstem.html").exists()
    assert (artifact_dir / "show2d-from-folder-stopped.html").exists()
    assert (artifact_dir / "show3d-from-folder-stopped.html").exists()
    assert (artifact_dir / "show4dstem-from-folder-stopped.html").exists() is (
        not show4d_skipped
    )


def test_widget_show3d_animation_smoke_writes_gif_report(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "show3d-gif"

    result = _run(
        sys.executable,
        "scripts/widget_show3d_animation_smoke.py",
        "--artifact-dir",
        str(artifact_dir),
        "--source",
        "synthetic",
        "--crop-size",
        "64",
        "--frames",
        "5",
        "--fps",
        "5",
        "--qualities",
        "low",
        "medium",
        "--max-total-mb",
        "5",
    )

    assert result.returncode == 0, result.stdout
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    index = (artifact_dir / "index.html").read_text(encoding="utf-8")

    assert report["source"]["kind"] == "synthetic CI fallback"
    assert report["input_shape"] == [5, 64, 64]
    assert report["panels"] == ["Raw", "Smoothed", "Change"]
    assert report["panel_gap"] == 0
    assert report["panel_labels"] is True
    assert report["scale_bar"] == {"visible": True, "sampling_nm": 0.05}
    assert report["zoom_indicator"] is True
    assert report["playback"] == "bounce"
    assert report["frame_labels"] is True
    assert [item["quality"] for item in report["exports"]] == ["low", "medium"]
    assert all(item["n_frames"] == 8 for item in report["exports"])
    assert all(item["mean_abs_delta"] > 0 for item in report["exports"])
    assert all(item["width"] > item["height"] for item in report["exports"])
    for item in report["exports"]:
        assert (artifact_dir / Path(item["path"]).name).exists()
    assert "Show3D GIF presentation smoke" in index
    assert "Raw, Smoothed, Change" in index
    assert "PowerPoint" in index
    assert "<img" in index


def test_widget_show3d_animation_smoke_supports_zero_panel_gap(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "show3d-gif-zero-gap"

    result = _run(
        sys.executable,
        "scripts/widget_show3d_animation_smoke.py",
        "--artifact-dir",
        str(artifact_dir),
        "--source",
        "synthetic",
        "--crop-size",
        "64",
        "--frames",
        "4",
        "--qualities",
        "high",
        "--panel-gap",
        "0",
        "--no-frame-labels",
        "--max-total-mb",
        "5",
    )

    assert result.returncode == 0, result.stdout
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    index = (artifact_dir / "index.html").read_text(encoding="utf-8")

    assert report["panel_gap"] == 0
    assert report["frame_labels"] is False
    assert report["planned_exports"][0]["width"] == 64 * 3
    assert report["planned_exports"][0]["height"] == 64
    assert report["exports"][0]["width"] == 64 * 3
    assert report["exports"][0]["height"] == 64
    assert "Panel gap: 0 px" in index
    assert "frame labels: off" in index


def test_widget_show3d_animation_smoke_dry_run_reports_size_plan(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "show3d-gif-dry-run"

    result = _run(
        sys.executable,
        "scripts/widget_show3d_animation_smoke.py",
        "--artifact-dir",
        str(artifact_dir),
        "--source",
        "synthetic",
        "--crop-size",
        "64",
        "--frames",
        "5",
        "--qualities",
        "low",
        "high",
        "--dry-run",
        "--max-work-mb",
        "0.001",
    )

    assert result.returncode == 0, result.stdout
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    index = (artifact_dir / "index.html").read_text(encoding="utf-8")

    assert report["dry_run"] is True
    assert report["exports"] == []
    assert [item["quality"] for item in report["planned_exports"]] == ["low", "high"]
    assert report["dry_run_decision"]["should_run"] is False
    assert report["dry_run_decision"]["total_uncompressed_rgb_mb"] > 0
    assert not list(artifact_dir.glob("*.gif"))
    assert "Show3D GIF presentation dry run" in index
    assert "Uncompressed RGB MB" in index
    assert "Dry run only" in index
    assert "<img" not in index


def test_widget_performance_smoke_writes_browser_plan(tmp_path: Path) -> None:
    data_dir = tmp_path / "real"
    artifact_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    rng = np.random.default_rng(8)
    for idx in range(2):
        image = rng.random((48, 48), dtype=np.float32)
        tifffile.imwrite(data_dir / f"real_{idx}.tif", image)

    result = _run(
        sys.executable,
        "scripts/widget_performance_smoke.py",
        "--quick",
        "--search-root",
        str(data_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--show2d-panels",
        "2",
        "--show2d-size",
        "32",
        "--show3d-panels",
        "2",
        "--show3d-frames",
        "3",
        "--show3d-size",
        "32",
        "--max-single-mb",
        "20",
    )

    assert result.returncode == 0, result.stdout
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    plan = json.loads((artifact_dir / "browser-plan.json").read_text(encoding="utf-8"))
    assert (artifact_dir / "index.html").exists()
    assert report["targets"]["show2d"] == {
        "requested_panels": 2,
        "pages": 2,
        "panels_per_page": 1,
        "total_panels": 2,
        "size": 32,
    }
    assert report["targets"]["show3d"] == {
        "requested_panels": 2,
        "pages": 2,
        "panels_per_page": 1,
        "total_panels": 2,
        "frames": 3,
        "size": 32,
    }
    assert len(report["exports"]) == 4
    assert plan["target_fps"] == 30
    assert {page["widget"] for page in plan["pages"]} == {"show2d", "show3d"}
