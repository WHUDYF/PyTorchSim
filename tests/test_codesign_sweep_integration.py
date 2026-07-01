from pathlib import Path
import sys

import pytest


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "mapping_dse_codesign_sweep.py"
    spec = importlib.util.spec_from_file_location("mapping_dse_codesign_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_cells_requires_full_cross_product():
    mod = load_module()
    hw_configs = {"HW-A": Path("hw_A.yml"), "HW-B": Path("hw_B.yml")}
    mappings = [
        {"mapping_id": "000", "TILE_M": 32, "TILE_N": 64, "TILE_K": 32},
        {"mapping_id": "001", "TILE_M": 64, "TILE_N": 64, "TILE_K": 32},
    ]

    cells = mod.build_sweep_cells(hw_configs, mappings)

    assert len(cells) == 4
    assert [(cell["hw_id"], cell["mapping_id"]) for cell in cells] == [
        ("HW-A", "000"),
        ("HW-A", "001"),
        ("HW-B", "000"),
        ("HW-B", "001"),
    ]


def test_parse_args_defaults_to_v2_hw_config_set(tmp_path):
    mod = load_module()
    mappings = tmp_path / "mappings.json"

    args = mod.parse_args(["--mappings-json", str(mappings)])

    assert args.hw_config_set == "codesign_v2_2x2"


def test_parse_args_accepts_v1_hw_config_set(tmp_path):
    mod = load_module()
    mappings = tmp_path / "mappings.json"

    args = mod.parse_args(["--mappings-json", str(mappings), "--hw-config-set", "codesign_v1_2x2"])

    assert args.hw_config_set == "codesign_v1_2x2"


def test_parse_args_rejects_unknown_hw_config_set(tmp_path):
    mod = load_module()
    mappings = tmp_path / "mappings.json"

    with pytest.raises(SystemExit) as exc:
        mod.parse_args(["--mappings-json", str(mappings), "--hw-config-set", "bad"])

    assert exc.value.code == 2


def test_run_sweep_marks_unavailable_without_retry(tmp_path):
    mod = load_module()
    cells = [
        {"hw_id": "HW-A", "mapping_id": "000", "hw_config": "hw_A.yml", "tile": {"TILE_M": 32, "TILE_N": 64, "TILE_K": 32}},
    ]
    fit_rows = {
        ("HW-A", "000"): {
            "status": "unavailable",
            "budget_over_bytes": 123,
            "availability_precheck": "analytical",
        }
    }

    result = mod.run_sweep(cells, fit_rows, tmp_path, runner=lambda cell, attempt: mod.RunAttempt("measured", 0, "", ""))

    row = result["fit_availability"]["cells"][0]
    assert row["status"] == "unavailable"
    assert row["retry_count"] == 0
    assert row["budget_over_bytes"] == 123
    assert result["state_counts"]["unavailable"] == 1


def test_run_sweep_retries_runtime_failure_then_measured(tmp_path):
    mod = load_module()
    cells = [
        {"hw_id": "HW-A", "mapping_id": "000", "hw_config": "hw_A.yml", "tile": {"TILE_M": 32, "TILE_N": 64, "TILE_K": 32}},
    ]
    fit_rows = {("HW-A", "000"): {"status": "needs_togsim", "budget_over_bytes": 0}}
    calls = []

    def runner(cell, attempt):
        calls.append(attempt)
        if attempt == 0:
            return mod.RunAttempt("runtime_failed", 1, "fail", "err")
        return mod.RunAttempt("measured", 0, "ok", "")

    result = mod.run_sweep(cells, fit_rows, tmp_path, runner=runner, max_retries=2)

    row = result["fit_availability"]["cells"][0]
    assert calls == [0, 1]
    assert row["status"] == "measured"
    assert row["retry_count"] == 1
    assert result["state_counts"]["measured"] == 1
    assert result["retry_ledger"][0]["attempt_index"] == 1
    assert set(result["retry_ledger"][0]["applied_diff"]) <= {"timeout_sec", "env"}


def test_run_sweep_preserves_reproduction_fields_for_measured_cell(tmp_path):
    mod = load_module()
    cells = [
        {"hw_id": "HW-A", "mapping_id": "000", "hw_config": "hw_A.yml", "tile": {"TILE_M": 32, "TILE_N": 64, "TILE_K": 32}},
    ]
    fit_rows = {("HW-A", "000"): {"status": "needs_togsim", "budget_over_bytes": 0}}

    result = mod.run_sweep(
        cells,
        fit_rows,
        tmp_path,
        runner=lambda cell, attempt: mod.RunAttempt(
            "measured",
            0,
            "ok",
            "",
            output_dir=str(tmp_path / "run"),
            command=["python", "scripts/mapping_dse_minimal.py"],
            stdout_path=str(tmp_path / "stdout.txt"),
            stderr_path=str(tmp_path / "stderr.txt"),
            subprocess_pid=123,
            start_time="2026-07-01T00:00:00+00:00",
            end_time="2026-07-01T00:00:01+00:00",
            total_cycles=1000,
        ),
    )

    row = result["fit_availability"]["cells"][0]
    assert row["simulator_cmdline"] == ["python", "scripts/mapping_dse_minimal.py"]
    assert row["stdout_path"].endswith("stdout.txt")
    assert row["stderr_path"].endswith("stderr.txt")
    assert row["subprocess_pid"] == 123
    assert row["start_time"] == "2026-07-01T00:00:00+00:00"
    assert row["end_time"] == "2026-07-01T00:00:01+00:00"
    assert row["total_cycles"] == 1000


def test_extract_total_cycles_prefers_mapping_harness_counters_table(tmp_path):
    mod = load_module()
    output_dir = tmp_path / "mapping_run"
    output_dir.mkdir()
    (output_dir / "counters_table.json").write_text(
        '{"rows": [{"mapping_id": "000", "total_cycles": 4242}]}',
        encoding="utf-8",
    )

    assert mod.extract_total_cycles(output_dir, "000", "Total execution cycles: 1\n") == 4242


def test_default_runner_treats_successful_process_without_cycles_as_runtime_failed(tmp_path, monkeypatch):
    mod = load_module()

    class FakeProc:
        returncode = 0
        pid = 123

        def communicate(self):
            return "Wrote artifacts\n", ""

    monkeypatch.setattr(mod.subprocess, "Popen", lambda *args, **kwargs: FakeProc())
    runner = mod.default_runner(tmp_path, 1, tmp_path / "external")
    result = runner(
        {
            "hw_id": "HW-A",
            "mapping_id": "000",
            "hw_config": "hw_A.yml",
            "tile": {"TILE_M": 32, "TILE_N": 64, "TILE_K": 32},
        },
        0,
    )

    assert result.status == "runtime_failed"
    assert result.total_cycles is None


def test_run_sweep_resume_reuses_existing_measured_cell(tmp_path):
    mod = load_module()
    cells = [
        {"hw_id": "HW-A", "mapping_id": "000", "hw_config": "hw_A.yml", "tile": {"TILE_M": 32, "TILE_N": 64, "TILE_K": 32}},
    ]
    existing = {
        "fit_availability": {
            "cells": [
                {
                    "hw_id": "HW-A",
                    "mapping_id": "000",
                    "status": "measured",
                    "retry_count": 0,
                    "budget_over_bytes": 0,
                    "availability_precheck": "",
                    "total_cycles": 1234,
                    "simulator_cmdline": ["existing"],
                    "stdout_path": "stdout.txt",
                    "stderr_path": "stderr.txt",
                    "subprocess_pid": 99,
                    "start_time": "2026-07-01T00:00:00+00:00",
                    "end_time": "2026-07-01T00:00:01+00:00",
                }
            ]
        },
        "retry_ledger": [],
    }
    (tmp_path / "sweep_summary.json").write_text(__import__("json").dumps(existing), encoding="utf-8")

    def runner(cell, attempt):
        raise AssertionError("runner should not be called for resumed measured cell")

    result = mod.run_sweep(cells, {}, tmp_path, runner=runner, resume=True)

    row = result["fit_availability"]["cells"][0]
    assert row["status"] == "measured"
    assert row["total_cycles"] == 1234
    assert result["state_counts"]["measured"] == 1


def test_run_sweep_converts_repeated_runtime_failure_to_retry_exhausted(tmp_path):
    mod = load_module()
    cells = [
        {"hw_id": "HW-A", "mapping_id": "000", "hw_config": "hw_A.yml", "tile": {"TILE_M": 32, "TILE_N": 64, "TILE_K": 32}},
    ]
    fit_rows = {("HW-A", "000"): {"status": "needs_togsim", "budget_over_bytes": 0}}

    result = mod.run_sweep(
        cells,
        fit_rows,
        tmp_path,
        runner=lambda cell, attempt: mod.RunAttempt("runtime_failed", 1, "fail", "err"),
        max_retries=2,
    )

    row = result["fit_availability"]["cells"][0]
    assert row["status"] == "retry_exhausted"
    assert row["retry_count"] == 2
    assert result["state_counts"]["retry_exhausted"] == 1
    assert len(result["retry_ledger"]) == 2


def test_lint_terminal_states_rejects_runtime_failed():
    mod = load_module()

    with pytest.raises(mod.TerminalStateViolation):
        mod.lint_terminal_states([{"status": "runtime_failed"}])
