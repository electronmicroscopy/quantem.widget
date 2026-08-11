from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "widget_show4dstem_folder_overnight.py"
E2E_TEST = ROOT / "tests" / "show4dstem" / "test_folder_live_jupyter_e2e.py"


def _module():
    spec = importlib.util.spec_from_file_location("show4dstem_overnight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _e2e_module():
    spec = importlib.util.spec_from_file_location("show4dstem_folder_e2e", E2E_TEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _snapshot(*, utilization: int = 0, command: str = "") -> dict:
    return {
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-zero",
                "free_mib": 90000,
                "utilization_pct": utilization,
            },
            {
                "index": 1,
                "uuid": "GPU-one",
                "free_mib": 91000,
                "utilization_pct": utilization,
            },
        ],
        "apps": [
            {
                "gpu_uuid": "GPU-zero",
                "pid": 42,
                "command": command,
            }
        ],
    }


def test_gpu_idle_gate_allows_small_unmatched_background_process() -> None:
    runner = _module()

    # C1: a low-utilization persistent service does not match a blocked
    # campaign, expect enough free memory to permit consecutive idle sampling.
    idle, reasons = runner._idle_decision(
        _snapshot(command="python -m quantem.live.server.main"),
        [0, 1],
        max_utilization=15,
        min_free_mib=70000,
        block_patterns=runner.DEFAULT_BLOCK_PATTERNS,
    )

    assert idle is True
    assert reasons == []


def test_gpu_idle_gate_blocks_load_and_known_foreign_campaign() -> None:
    runner = _module()

    # C1: selected cards are busy with a named foreign campaign, expect both
    # utilization and ownership evidence in the corrective wait reason.
    idle, reasons = runner._idle_decision(
        _snapshot(
            utilization=99,
            command="python scripts/overnight_ml_calibration_campaign.py",
        ),
        [0],
        max_utilization=15,
        min_free_mib=70000,
        block_patterns=runner.DEFAULT_BLOCK_PATTERNS,
    )

    assert idle is False
    assert any("utilization" in reason for reason in reasons)
    assert any("overnight_ml_calibration_campaign.py" in reason for reason in reasons)


def test_dry_run_writes_resumable_report_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    artifacts = tmp_path / "artifacts"
    cache = tmp_path / "cache"
    source.mkdir()

    # C1: no GPU workload is requested, expect the durable report/heartbeat
    # contract to be inspectable without importing Torch or touching data.
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--artifact-dir",
            str(artifacts),
            "--cache-dir",
            str(cache),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    report = json.loads((artifacts / "report.json").read_text(encoding="utf-8"))
    status = json.loads((artifacts / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    assert report["status"] == "planned"
    assert status["current_phase"] == "dry_run"
    assert manifest["run_id"] == report["run_id"]
    assert (artifacts / "index.html").is_file()
    assert (artifacts / "gates.json").is_file()
    assert (artifacts / "environment.json").is_file()


def test_live_e2e_report_has_mobile_review_index(tmp_path: Path) -> None:
    e2e = _e2e_module()

    # C1: a completed live-browser report is written, expect one responsive
    # HTML review entry point alongside the detailed machine-readable files.
    e2e._write_reports(
        {"passed": True, "run_id": "example", "phases": {"cold": {}}},
        tmp_path,
    )

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'name="viewport"' in index
    assert "Passed" in index
    assert "show4dstem-folder-e2e-report.json" in index
    assert "show4dstem-folder-e2e-perf.json" in index
