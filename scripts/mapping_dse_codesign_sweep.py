#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


TERMINAL_STATES = {"measured", "unavailable", "retry_exhausted"}
INTERMEDIATE_STATES = {"needs_togsim", "runtime_failed"}
ALLOWED_RETRY_DIFF_KEYS = {"timeout_sec", "env"}
HW_CONFIG_SETS = ("codesign_v1_2x2", "codesign_v2_2x2")


class TerminalStateViolation(ValueError):
    pass


class RetryPolicyViolation(ValueError):
    pass


@dataclass
class RunAttempt:
    status: str
    returncode: int
    stdout: str
    stderr: str
    output_dir: str = ""
    command: list[str] | None = None
    stdout_path: str = ""
    stderr_path: str = ""
    subprocess_pid: int | None = None
    start_time: str = ""
    end_time: str = ""
    total_cycles: int | None = None


def parse_total_cycles(text: str) -> int | None:
    match = re.search(r"Total execution cycles:\s*(\d+)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"total_cycles['\"]?\s*[:=]\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def extract_total_cycles(output_dir: Path, mapping_id: str, log_text: str) -> int | None:
    counters_path = output_dir / "counters_table.json"
    if counters_path.exists():
        try:
            payload = json.loads(counters_path.read_text(encoding="utf-8"))
            for row in payload.get("rows", []):
                if str(row.get("mapping_id")) == str(mapping_id) and int(row.get("total_cycles", 0) or 0) > 0:
                    return int(row["total_cycles"])
        except Exception:
            pass
    return parse_total_cycles(log_text)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_sweep_cells(hw_configs: dict[str, Path | str], mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for hw_id in sorted(hw_configs):
        for mapping in mappings:
            cells.append(
                {
                    "hw_id": hw_id,
                    "hw_config": str(hw_configs[hw_id]),
                    "mapping_id": str(mapping["mapping_id"]),
                    "tile": {
                        "TILE_M": int(mapping["TILE_M"]),
                        "TILE_N": int(mapping["TILE_N"]),
                        "TILE_K": int(mapping["TILE_K"]),
                    },
                }
            )
    expected = len(hw_configs) * len(mappings)
    if len(cells) != expected:
        raise ValueError(f"internal cross-product error: expected {expected}, got {len(cells)}")
    return cells


def lint_terminal_states(rows: list[dict[str, Any]]) -> None:
    bad = [row for row in rows if row.get("status") not in TERMINAL_STATES]
    if bad:
        states = sorted({str(row.get("status")) for row in bad})
        raise TerminalStateViolation("Non-terminal states remain: " + ", ".join(states))


def retry_diff_for_attempt(attempt_index: int) -> dict[str, Any]:
    if attempt_index == 1:
        return {"timeout_sec": "2x"}
    return {"timeout_sec": "3x"} if attempt_index >= 2 else {}


def validate_retry_diff(diff: dict[str, Any]) -> None:
    bad = sorted(key for key in diff if key not in ALLOWED_RETRY_DIFF_KEYS)
    if bad:
        raise RetryPolicyViolation("Retry attempted forbidden changes: " + ", ".join(bad))


def default_runner(output_root: Path, base_timeout_sec: int, external_mappings_dir: Path) -> Callable[[dict[str, Any], int], RunAttempt]:
    def run(cell: dict[str, Any], attempt: int) -> RunAttempt:
        mapping_json = external_mappings_dir / f"{cell['hw_id']}_{cell['mapping_id']}.json"
        mapping_json.parent.mkdir(parents=True, exist_ok=True)
        mapping_json.write_text(
            json.dumps(
                [
                    {
                        "mapping_id": cell["mapping_id"],
                        **cell["tile"],
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        timeout_sec = base_timeout_sec * (2 if attempt == 1 else 3 if attempt >= 2 else 1)
        cell_output = output_root / "runs" / cell["hw_id"] / "mappings" / cell["mapping_id"] / f"attempt_{attempt:02d}"
        command = [
            sys.executable,
            str(repo_root() / "scripts" / "mapping_dse_minimal.py"),
            "--num-mappings",
            "1",
            "--timeout-sec",
            str(timeout_sec),
            "--hw-config",
            str(cell["hw_config"]),
            "--external-mappings-json",
            str(mapping_json),
            "--output-dir",
            str(cell_output),
        ]
        cell_output.mkdir(parents=True, exist_ok=True)
        start_time = datetime.now(timezone.utc).isoformat()
        proc = subprocess.Popen(
            command,
            cwd=repo_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate()
        end_time = datetime.now(timezone.utc).isoformat()
        stdout_path = cell_output / "stdout.txt"
        stderr_path = cell_output / "stderr.txt"
        cmdline_path = cell_output / "cmdline.json"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        cmdline_path.write_text(json.dumps(command, indent=2), encoding="utf-8")
        total_cycles = extract_total_cycles(cell_output, cell["mapping_id"], stdout + "\n" + stderr)
        status = "measured" if proc.returncode == 0 and total_cycles is not None else "runtime_failed"
        return RunAttempt(
            status,
            int(proc.returncode or 0),
            stdout,
            stderr,
            str(cell_output),
            command,
            str(stdout_path),
            str(stderr_path),
            proc.pid,
            start_time,
            end_time,
            total_cycles,
        )

    return run


def run_sweep(
    cells: list[dict[str, Any]],
    fit_rows: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    *,
    runner: Callable[[dict[str, Any], int], RunAttempt],
    max_retries: int = 6,
    resume: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    retry_ledger: list[dict[str, Any]] = []
    existing_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if resume and (output_dir / "sweep_summary.json").exists():
        existing = load_json(output_dir / "sweep_summary.json")
        for row in existing.get("fit_availability", {}).get("cells", []):
            if row.get("status") in TERMINAL_STATES:
                existing_rows[(row["hw_id"], row["mapping_id"])] = row
        retry_ledger.extend(existing.get("retry_ledger", []))

    for cell in cells:
        key = (cell["hw_id"], cell["mapping_id"])
        if key in existing_rows:
            rows.append(dict(existing_rows[key]))
            continue
        fit = fit_rows.get(key, {"status": "needs_togsim", "budget_over_bytes": 0})
        row = {
            "hw_id": cell["hw_id"],
            "mapping_id": cell["mapping_id"],
            "status": "",
            "budget_over_bytes": fit.get("budget_over_bytes"),
            "retry_count": 0,
            "availability_precheck": fit.get("availability_precheck", ""),
        }
        if fit.get("status") == "unavailable":
            row["status"] = "unavailable"
            rows.append(row)
            continue

        attempt = 0
        while True:
            result = runner(cell, attempt)
            if result.status == "measured":
                row["status"] = "measured"
                row["returncode"] = result.returncode
                row["output_dir"] = result.output_dir
                row["simulator_cmdline"] = result.command or []
                row["stdout_path"] = result.stdout_path
                row["stderr_path"] = result.stderr_path
                row["subprocess_pid"] = result.subprocess_pid
                row["start_time"] = result.start_time
                row["end_time"] = result.end_time
                if result.total_cycles is not None:
                    row["total_cycles"] = result.total_cycles
                rows.append(row)
                break
            if result.status != "runtime_failed":
                row["status"] = result.status
                rows.append(row)
                break
            if attempt >= max_retries:
                row["status"] = "retry_exhausted"
                row["returncode"] = result.returncode
                row["simulator_cmdline"] = result.command or []
                row["stdout_path"] = result.stdout_path
                row["stderr_path"] = result.stderr_path
                row["subprocess_pid"] = result.subprocess_pid
                row["start_time"] = result.start_time
                row["end_time"] = result.end_time
                if result.total_cycles is not None:
                    row["total_cycles"] = result.total_cycles
                rows.append(row)
                break
            attempt += 1
            diff = retry_diff_for_attempt(attempt)
            validate_retry_diff(diff)
            row["retry_count"] = attempt
            retry_ledger.append(
                {
                    "hw_id": cell["hw_id"],
                    "mapping_id": cell["mapping_id"],
                    "attempt_index": attempt,
                    "hypothesis": "runtime failure may be timeout or environment related",
                    "applied_diff": diff,
                    "result": {
                        "status": result.status,
                        "returncode": result.returncode,
                        "stdout_tail": result.stdout[-1000:],
                        "stderr_tail": result.stderr[-1000:],
                    },
                }
            )

    lint_terminal_states(rows)
    state_counts = {state: 0 for state in sorted(TERMINAL_STATES)}
    for row in rows:
        state_counts[row["status"]] += 1
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_counts": state_counts,
        "fit_availability": {"cells": rows},
        "retry_ledger": retry_ledger,
    }
    (output_dir / "fit_availability.json").write_text(
        json.dumps(payload["fit_availability"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "retry_ledger.json").write_text(
        json.dumps(retry_ledger, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "sweep_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NPU co-design HW x mapping sweep.")
    parser.add_argument("--hw-config-summary", type=Path, default=Path("outputs/mapping_dse_codesign/hw_config_factory_summary.json"))
    parser.add_argument("--mappings-json", type=Path, required=True)
    parser.add_argument("--fit-availability-precheck", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mapping_dse_codesign/gpt2_block_prefill_s128_run1"))
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--hw-config-set", choices=HW_CONFIG_SETS, default="codesign_v2_2x2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hw_summary = load_json(args.hw_config_summary)
    hw_configs = {hw_id: Path(path) for hw_id, path in hw_summary["hw_yaml_paths"].items()}
    mappings = load_json(args.mappings_json)
    cells = build_sweep_cells(hw_configs, mappings)
    fit_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if args.fit_availability_precheck:
        for row in load_json(args.fit_availability_precheck).get("rows", []):
            fit_rows[(row["hw_id"], row["mapping_id"])] = row
    runner = default_runner(args.output_dir, args.timeout_sec, args.output_dir / "external_mappings")
    run_sweep(cells, fit_rows, args.output_dir, runner=runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
