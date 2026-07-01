#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


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


def _cycles_matrix(cycles_by_cell: dict[tuple[str, str], int | float]) -> dict[str, dict[str, int | float]]:
    matrix: dict[str, dict[str, int | float]] = {}
    for (hw_id, mapping_id), cycles in cycles_by_cell.items():
        matrix.setdefault(hw_id, {})[mapping_id] = cycles
    return {hw_id: dict(sorted(values.items())) for hw_id, values in sorted(matrix.items())}


def _write_analysis(output_dir: Path, analysis: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("champion_migration", "champion_migration.json"),
        ("oracle_gaps", "oracle_gaps.json"),
        ("cost_matched_pair", "cost_matched_pair.json"),
        ("interaction_analysis", "interaction_analysis.json"),
    ]:
        (output_dir / filename).write_text(
            json.dumps(analysis[key], indent=2, sort_keys=True),
            encoding="utf-8",
        )


def run_integration_smoke(
    *,
    output_dir: Path,
    hw_summary: dict[str, Any],
    mappings: list[dict[str, Any]],
    cycles_by_cell: dict[tuple[str, str], int | float],
    determinism_smoke_test: dict[str, Any],
    binary_paths: dict[str, Path | str],
    repo_root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_hw = {hw_id: path for hw_id, path in sorted(hw_summary["hw_yaml_paths"].items())[:2]}
    selected_mappings = mappings[:2]
    cells = sweep.build_sweep_cells(selected_hw, selected_mappings)

    def runner(cell: dict[str, Any], attempt: int):
        key = (cell["hw_id"], cell["mapping_id"])
        cycles = cycles_by_cell[key]
        cell_dir = output_dir / "fake_runs" / cell["hw_id"] / cell["mapping_id"]
        cell_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = cell_dir / "stdout.txt"
        stderr_path = cell_dir / "stderr.txt"
        stdout = f"total_cycles: {cycles}\n"
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
            subprocess_pid=1000 + len(cell["mapping_id"]) + attempt,
            start_time="2026-07-01T00:00:00+00:00",
            end_time="2026-07-01T00:00:01+00:00",
        )

    fit_rows = {
        (cell["hw_id"], cell["mapping_id"]): {"status": "needs_togsim", "budget_over_bytes": 0}
        for cell in cells
    }
    sweep_summary = sweep.run_sweep(cells, fit_rows, output_dir, runner=runner)
    cycles_matrix = _cycles_matrix(cycles_by_cell)
    (output_dir / "heatmap_cycles.json").write_text(
        json.dumps(cycles_matrix, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    analysis = analyzer.analyze_codesign(cycles_matrix, permutation_trials=20, permutation_seed=0)
    analysis["state_counts"] = sweep_summary["state_counts"]
    analysis_dir = output_dir / "analysis"
    _write_analysis(analysis_dir, analysis)

    verdict = reporter.build_verdict(analysis)
    report_text = reporter.render_report(analysis, verdict, cycles_by_hw=cycles_matrix)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = report_dir / "verdict.json"
    report_path = report_dir / "report.md"
    verdict_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")

    manifest = metadata_mod.build_metadata_manifest(
        repo_root=repo_root,
        cli_args={"integration_smoke": True},
        hw_summary=hw_summary,
        sweep_summary=sweep_summary,
        determinism_smoke_test=determinism_smoke_test,
        artifact_paths={
            "heatmap_cycles": str(output_dir / "heatmap_cycles.json"),
            "fit_availability": str(output_dir / "fit_availability.json"),
            "retry_ledger": str(output_dir / "retry_ledger.json"),
            "sweep_summary": str(output_dir / "sweep_summary.json"),
            "verdict": str(verdict_path),
            "report": str(report_path),
        },
        binary_paths=binary_paths,
    )
    metadata_path = output_dir / "00_metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return {
        "state_counts": sweep_summary["state_counts"],
        "metadata_lint": metadata_mod.lint_metadata(manifest),
        "report_lint": reporter.lint_report_sections(report_text),
        "verdict": verdict,
    }


def _default_cycles() -> dict[tuple[str, str], int]:
    return {
        ("HW-A", "000"): 100,
        ("HW-A", "001"): 125,
        ("HW-B", "000"): 120,
        ("HW-B", "001"): 100,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2x2 co-design integration smoke chain.")
    parser.add_argument("--hw-config-summary", type=Path, default=Path("outputs/mapping_dse_codesign/hw_config_factory_summary.json"))
    parser.add_argument("--mappings-json", type=Path, default=Path("outputs/mapping_dse_codesign/mappings_task4_smoke.json"))
    parser.add_argument("--determinism-json", type=Path, default=Path("outputs/mapping_dse_codesign/determinism_smoke_summary_smoke.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mapping_dse_codesign/integration_task11_smoke"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ramulator2-config", type=Path, default=Path("configs/ramulator2_configs/HBM2_TPUv3.yaml"))
    parser.add_argument("--gem5-binary", type=Path, default=Path("build/RISCV/gem5.opt"))
    parser.add_argument("--togsim-binary", type=Path, default=Path("TOGSim/build/togsim"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hw_summary = json.loads(args.hw_config_summary.read_text(encoding="utf-8"))
    hw_summary.setdefault("patch_matrix", {})
    mappings = json.loads(args.mappings_json.read_text(encoding="utf-8"))
    determinism = json.loads(args.determinism_json.read_text(encoding="utf-8"))
    result = run_integration_smoke(
        output_dir=args.output_dir,
        hw_summary=hw_summary,
        mappings=mappings,
        cycles_by_cell=_default_cycles(),
        determinism_smoke_test=determinism,
        binary_paths={
            "ramulator2_config": args.ramulator2_config,
            "gem5_binary": args.gem5_binary,
            "togsim_binary": args.togsim_binary,
        },
        repo_root=args.repo_root,
    )
    (args.output_dir / "integration_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
