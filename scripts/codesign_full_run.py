#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


def _load_module(name: str):
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep = _load_module("mapping_dse_codesign_sweep")
analyzer = _load_module("codesign_analyzer")
reporter = _load_module("codesign_report")
metadata_mod = _load_module("codesign_metadata")


class FullRunContractError(ValueError):
    pass


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_analysis(output_dir: Path, analysis: dict[str, Any]) -> None:
    for key, filename in [
        ("champion_migration", "champion_migration.json"),
        ("oracle_gaps", "oracle_gaps.json"),
        ("cost_matched_pair", "cost_matched_pair.json"),
        ("interaction_analysis", "interaction_analysis.json"),
    ]:
        _write_json(output_dir / filename, analysis[key])


def _cycles_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("status") != "measured":
            continue
        if "total_cycles" not in row:
            raise FullRunContractError(
                f"measured cell {row.get('hw_id')}/{row.get('mapping_id')} missing total_cycles"
            )
        matrix.setdefault(row["hw_id"], {})[row["mapping_id"]] = int(row["total_cycles"])
    return {hw_id: dict(sorted(values.items())) for hw_id, values in sorted(matrix.items())}


def _assert_full_cross_product(hw_summary: dict[str, Any], mappings: list[dict[str, Any]], cells: list[dict[str, Any]]) -> None:
    if len(hw_summary.get("hw_yaml_paths", {})) != 4 or len(mappings) != 8 or len(cells) != 32:
        raise FullRunContractError(
            f"expected 32 cells from 4 HW configs x 8 mappings, got {len(cells)}"
        )


def synthetic_runner_factory(cycle_fn: Callable[..., int]):
    def make_runner(output_dir: Path):
        hw_order = ["HW-A", "HW-B", "HW-C", "HW-D"]
        repeat_idx = 0
        for part in output_dir.parts:
            if part.startswith("repeat_"):
                try:
                    repeat_idx = int(part.split("_", 1)[1])
                except ValueError:
                    repeat_idx = 0

        def run(cell: dict[str, Any], attempt: int):
            hw_idx = hw_order.index(cell["hw_id"])
            mapping_idx = int(cell["mapping_id"])
            try:
                cycles = cycle_fn(hw_idx, mapping_idx, repeat_idx)
            except TypeError:
                cycles = cycle_fn(hw_idx, mapping_idx)
            cell_dir = output_dir / "synthetic_runs" / cell["hw_id"] / cell["mapping_id"]
            cell_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = cell_dir / "stdout.txt"
            stderr_path = cell_dir / "stderr.txt"
            stdout = f"Total execution cycles: {cycles}\n"
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            return sweep.RunAttempt(
                "measured",
                0,
                stdout,
                "",
                output_dir=str(cell_dir),
                command=["python", "scripts/mapping_dse_minimal.py"],
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                subprocess_pid=2000 + hw_idx * 100 + mapping_idx,
                start_time="2026-07-01T00:00:00+00:00",
                end_time="2026-07-01T00:00:01+00:00",
                total_cycles=cycles,
            )

        return run

    return make_runner


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _stddev(values: list[int]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _repeat_count_for_determinism(determinism_smoke_test: dict[str, Any]) -> int:
    return 3 if determinism_smoke_test.get("deterministic_class") == "near_deterministic" else 1


def _merge_repeat_summaries(repeat_summaries: list[dict[str, Any]]) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, Any]]]:
    values: dict[tuple[str, str], list[int]] = {}
    for summary in repeat_summaries:
        for row in summary["fit_availability"]["cells"]:
            if row.get("status") != "measured" or "total_cycles" not in row:
                continue
            values.setdefault((row["hw_id"], row["mapping_id"]), []).append(int(row["total_cycles"]))
    heatmap: dict[str, dict[str, int]] = {}
    stats: dict[str, dict[str, Any]] = {}
    for (hw_id, mapping_id), cycles in sorted(values.items()):
        heatmap.setdefault(hw_id, {})[mapping_id] = int(_median(cycles))
        stats.setdefault(hw_id, {})[mapping_id] = {
            "cycles": cycles,
            "median_cycles": _median(cycles),
            "stddev": _stddev(cycles),
        }
    return heatmap, stats


def run_full_codesign(
    *,
    output_dir: Path,
    hw_summary: dict[str, Any],
    mappings: list[dict[str, Any]],
    determinism_smoke_test: dict[str, Any],
    fit_rows: dict[tuple[str, str], dict[str, Any]],
    binary_paths: dict[str, Path | str],
    repo_root: Path,
    runner_factory: Callable[[Path], Callable[[dict[str, Any], int], Any]] | None = None,
    permutation_trials: int = 1000,
    permutation_seed: int = 0,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = sweep.build_sweep_cells(hw_summary["hw_yaml_paths"], mappings)
    _assert_full_cross_product(hw_summary, mappings, cells)
    repeat_count = _repeat_count_for_determinism(determinism_smoke_test)
    repeat_summaries = []
    for repeat_idx in range(repeat_count):
        repeat_dir = output_dir if repeat_count == 1 else output_dir / f"repeat_{repeat_idx:02d}"
        runner = runner_factory(repeat_dir) if runner_factory else sweep.default_runner(
            repeat_dir,
            900,
            repeat_dir / "external_mappings",
        )
        repeat_summaries.append(sweep.run_sweep(cells, fit_rows, repeat_dir, runner=runner, resume=resume))
    sweep_summary = repeat_summaries[0]
    if repeat_count == 1:
        cycles_matrix = _cycles_from_rows(sweep_summary["fit_availability"]["cells"])
        cycle_stats = {
            hw_id: {
                mapping_id: {"cycles": [cycles], "median_cycles": cycles, "stddev": 0.0}
                for mapping_id, cycles in row.items()
            }
            for hw_id, row in cycles_matrix.items()
        }
    else:
        cycles_matrix, cycle_stats = _merge_repeat_summaries(repeat_summaries)
        measured = sum(1 for row in repeat_summaries[0]["fit_availability"]["cells"] if row.get("status") == "measured")
        unavailable = sum(1 for row in repeat_summaries[0]["fit_availability"]["cells"] if row.get("status") == "unavailable")
        retry_exhausted = sum(1 for row in repeat_summaries[0]["fit_availability"]["cells"] if row.get("status") == "retry_exhausted")
        sweep_summary = {
            "created_at": repeat_summaries[-1]["created_at"],
            "state_counts": {
                "measured": measured,
                "retry_exhausted": retry_exhausted,
                "unavailable": unavailable,
            },
            "fit_availability": {"cells": repeat_summaries[0]["fit_availability"]["cells"]},
            "retry_ledger": [item for summary in repeat_summaries for item in summary["retry_ledger"]],
            "repeat_summaries": [
                {
                    "repeat_index": idx,
                    "sweep_summary": str((output_dir / f"repeat_{idx:02d}" / "sweep_summary.json").resolve()),
                }
                for idx in range(repeat_count)
            ],
        }
        _write_json(output_dir / "fit_availability.json", sweep_summary["fit_availability"])
        _write_json(output_dir / "retry_ledger.json", sweep_summary["retry_ledger"])
        _write_json(output_dir / "sweep_summary.json", sweep_summary)
    _write_json(output_dir / "heatmap_cycles.json", cycles_matrix)
    _write_json(output_dir / "heatmap_cycle_stats.json", cycle_stats)

    analysis = analyzer.analyze_codesign(
        cycles_matrix,
        permutation_trials=permutation_trials,
        permutation_seed=permutation_seed,
    )
    analysis["state_counts"] = sweep_summary["state_counts"]
    analysis_dir = output_dir / "analysis"
    _write_analysis(analysis_dir, analysis)

    verdict = reporter.build_verdict(analysis)
    report_text = reporter.render_report(analysis, verdict, cycles_by_hw=cycles_matrix)
    report_dir = output_dir / "report"
    verdict_path = report_dir / "verdict.json"
    report_path = report_dir / "report.md"
    _write_json(verdict_path, verdict)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    manifest = metadata_mod.build_metadata_manifest(
        repo_root=repo_root,
        cli_args={"full_codesign_run": True},
        hw_summary=hw_summary,
        sweep_summary=sweep_summary,
        determinism_smoke_test=determinism_smoke_test,
        artifact_paths={
            "heatmap_cycles": str(output_dir / "heatmap_cycles.json"),
            "heatmap_cycle_stats": str(output_dir / "heatmap_cycle_stats.json"),
            "fit_availability": str(output_dir / "fit_availability.json"),
            "retry_ledger": str(output_dir / "retry_ledger.json"),
            "sweep_summary": str(output_dir / "sweep_summary.json"),
            "verdict": str(verdict_path),
            "report": str(report_path),
        },
        binary_paths=binary_paths,
    )
    metadata_path = output_dir / "00_metadata.json"
    _write_json(metadata_path, manifest)

    result = {
        "state_counts": sweep_summary["state_counts"],
        "repeat_count": repeat_count,
        "metadata_lint": metadata_mod.lint_metadata(manifest),
        "report_lint": reporter.lint_report_sections(report_text),
        "verdict": verdict,
    }
    _write_json(output_dir / "full_run_summary.json", result)
    return result


def _load_fit_rows(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(row["hw_id"], row["mapping_id"]): row for row in payload.get("rows", [])}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full 4x8 co-design measurement pipeline.")
    parser.add_argument("--hw-config-summary", type=Path, default=Path("outputs/mapping_dse_codesign/hw_config_factory_summary.json"))
    parser.add_argument("--mappings-json", type=Path, required=True)
    parser.add_argument("--fit-availability-precheck", type=Path)
    parser.add_argument("--determinism-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mapping_dse_codesign/gpt2_block_prefill_s128_run1"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ramulator2-config", type=Path, default=Path("configs/ramulator2_configs/HBM2_TPUv3.yaml"))
    parser.add_argument("--gem5-binary", type=Path, default=Path("build/RISCV/gem5.opt"))
    parser.add_argument("--togsim-binary", type=Path, default=Path("TOGSim/build/togsim"))
    parser.add_argument("--permutation-trials", type=int, default=1000)
    parser.add_argument("--permutation-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hw_summary = json.loads(args.hw_config_summary.read_text(encoding="utf-8"))
    hw_summary.setdefault("patch_matrix", {})
    mappings = json.loads(args.mappings_json.read_text(encoding="utf-8"))
    determinism = json.loads(args.determinism_json.read_text(encoding="utf-8"))
    run_full_codesign(
        output_dir=args.output_dir,
        hw_summary=hw_summary,
        mappings=mappings,
        determinism_smoke_test=determinism,
        fit_rows=_load_fit_rows(args.fit_availability_precheck),
        binary_paths={
            "ramulator2_config": args.ramulator2_config,
            "gem5_binary": args.gem5_binary,
            "togsim_binary": args.togsim_binary,
        },
        repo_root=args.repo_root,
        permutation_trials=args.permutation_trials,
        permutation_seed=args.permutation_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
