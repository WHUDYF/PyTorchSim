#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


DEFAULT_CANARIES = [("HW-A", "006"), ("HW-C", "008")]


class CanarySelectionError(ValueError):
    pass


def median(values: list[int | float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def cycle_delta(cycles: list[int | float]) -> float:
    med = median(cycles)
    if med == 0:
        return 0.0 if max(cycles) == min(cycles) else float("inf")
    return (max(cycles) - min(cycles)) / med


def classify_delta(delta: float) -> str:
    if delta <= 0.001:
        return "deterministic"
    if delta <= 0.02:
        return "near_deterministic"
    return "non_deterministic"


def classify_cycle_runs(cycles: list[int | float]) -> dict[str, Any]:
    delta = cycle_delta(cycles)
    return {
        "cycles": [int(value) for value in cycles],
        "median_cycles": median(cycles),
        "cycle_delta": delta,
        "deterministic_class": classify_delta(delta),
    }


def summarize_canaries(canary_cycles: dict[str, list[int | float]]) -> dict[str, Any]:
    canaries = {name: classify_cycle_runs(cycles) for name, cycles in canary_cycles.items()}
    max_delta = max((row["cycle_delta"] for row in canaries.values()), default=0.0)
    return {
        "deterministic_class": classify_delta(max_delta),
        "max_cycle_delta": max_delta,
        "canaries": canaries,
    }


def summarize_canary_results(
    canary_cycles: dict[str, list[int | float]],
    canary_errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    canary_errors = canary_errors or {}
    canaries = {name: classify_cycle_runs(cycles) for name, cycles in canary_cycles.items() if cycles}
    for name, error in sorted(canary_errors.items()):
        canaries[name] = {
            "cycles": [int(value) for value in canary_cycles.get(name, [])],
            "median_cycles": None,
            "cycle_delta": None,
            "deterministic_class": "failed",
            "error": error,
        }
    if canary_errors:
        max_delta = max((row["cycle_delta"] for row in canaries.values() if row.get("cycle_delta") is not None), default=0.0)
        return {
            "deterministic_class": "blocked",
            "max_cycle_delta": max_delta,
            "canaries": canaries,
            "failed_canaries": sorted(canary_errors),
        }
    return summarize_canaries(canary_cycles)


def canary_key(hw_id: str, mapping_id: str) -> str:
    return f"{hw_id}_{mapping_id}"


def run_canary_repeats(
    cells: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    *,
    runner: Callable[[dict[str, Any], int, Path], int | float],
    repeats: int = 3,
    required_canaries: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    required = required_canaries or DEFAULT_CANARIES
    missing = [canary for canary in required if canary not in cells]
    if missing:
        raise CanarySelectionError("missing canary cells: " + ", ".join(canary_key(*item) for item in missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_cycles: dict[str, list[int | float]] = {}
    errors: dict[str, str] = {}
    for hw_id, mapping_id in required:
        cell = cells[(hw_id, mapping_id)]
        key = canary_key(hw_id, mapping_id)
        raw_cycles[key] = []
        for repeat_index in range(repeats):
            repeat_dir = output_dir / key / f"repeat_{repeat_index:02d}"
            try:
                raw_cycles[key].append(runner(cell, repeat_index, repeat_dir))
            except Exception as exc:
                errors[key] = str(exc)
                break
    summary = summarize_canary_results(raw_cycles, errors)
    payload = {"raw_cycles": raw_cycles, "summary": summary}
    (output_dir / "determinism_cycles.json").write_text(
        json.dumps(raw_cycles, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "determinism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def parse_total_cycles(stdout: str, stderr: str) -> int:
    for text in (stdout, stderr):
        for line in text.splitlines():
            if "total_cycles" not in line and "Total execution cycles" not in line:
                continue
            digits = "".join(ch if ch.isdigit() else " " for ch in line).split()
            if digits:
                return int(digits[-1])
    raise ValueError("could not parse total_cycles from subprocess output")


def extract_total_cycles(run_dir: Path, mapping_id: str, stdout: str, stderr: str) -> int:
    counters_path = run_dir / "counters_table.json"
    if counters_path.exists():
        payload = json.loads(counters_path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            if str(row.get("mapping_id")) == str(mapping_id) and int(row.get("total_cycles", 0) or 0) > 0:
                return int(row["total_cycles"])
    return parse_total_cycles(stdout, stderr)


def default_runner(repo_root: Path, timeout_sec: int) -> Callable[[dict[str, Any], int, Path], int]:
    def run(cell: dict[str, Any], repeat_index: int, output_dir: Path) -> int:
        output_dir.mkdir(parents=True, exist_ok=True)
        mapping_json = output_dir / "mapping.json"
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
        command = [
            sys.executable,
            str(repo_root / "scripts" / "mapping_dse_minimal.py"),
            "--num-mappings",
            "1",
            "--timeout-sec",
            str(timeout_sec),
            "--hw-config",
            str(cell["hw_config"]),
            "--external-mappings-json",
            str(mapping_json),
            "--output-dir",
            str(output_dir / "run"),
        ]
        proc = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (output_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (output_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        (output_dir / "cmdline.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"canary subprocess failed with returncode {proc.returncode}")
        return extract_total_cycles(output_dir / "run", cell["mapping_id"], proc.stdout, proc.stderr)

    return run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize determinism smoke-test cycle repeats.")
    parser.add_argument("--cycles-json", type=Path)
    parser.add_argument("--cells-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout-sec", type=int, default=900)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cells_json:
        cells_list = json.loads(args.cells_json.read_text(encoding="utf-8"))
        cells = {(row["hw_id"], row["mapping_id"]): row for row in cells_list}
        output_dir = args.output_dir or args.output_json.parent
        result = run_canary_repeats(
            cells,
            output_dir,
            runner=default_runner(args.repo_root, args.timeout_sec),
        )["summary"]
    elif args.cycles_json:
        data = json.loads(args.cycles_json.read_text(encoding="utf-8"))
        result = summarize_canaries(data)
    else:
        raise SystemExit("--cycles-json or --cells-json is required")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
