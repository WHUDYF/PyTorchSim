#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BASELINE_YAML = Path("configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml")
MAPPING_HARNESS = Path("scripts/mapping_dse_minimal.py")
SPEC_PATH = Path("docs/superpowers/specs/2026-07-01-npu-mapping-dse-codesign-design.md")
PLAN_PATH = Path("docs/superpowers/plans/2026-07-01-npu-mapping-dse-codesign-plan.md")
TOGSIM_CANDIDATES = [
    Path("TOGSim/build/bin/Simulator"),
    Path("TOGSim/build/togsim"),
]
REQUIRED_BASELINE_FIELDS = [
    "vpu_spad_size_kb_per_lane",
    "dram_channels",
    "icnt_injection_ports_per_core",
    "core_freq_mhz",
    "num_cores",
]


@dataclass
class PreflightError(Exception):
    message: str
    exit_code: int

    def __str__(self) -> str:
        return self.message


def _require_files(repo_root: Path, relpaths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in relpaths if not (repo_root / path).is_file()]
    if missing:
        raise PreflightError("Missing required artifacts: " + ", ".join(missing), 2)
    return {str(path): str((repo_root / path).resolve()) for path in relpaths}


def _find_togsim_binary(repo_root: Path) -> Path:
    for candidate in TOGSIM_CANDIDATES:
        path = repo_root / candidate
        if path.is_file():
            return path
    raise PreflightError(
        "Missing required artifacts: " + ", ".join(str(path) for path in TOGSIM_CANDIDATES),
        2,
    )


def _load_baseline_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PreflightError(f"Baseline YAML is not parseable: {path}: {exc}", 3) from exc
    if not isinstance(data, dict):
        raise PreflightError(f"Baseline YAML must contain a mapping: {path}", 3)
    missing_fields = [field for field in REQUIRED_BASELINE_FIELDS if field not in data]
    if missing_fields:
        raise PreflightError("Baseline YAML missing required fields: " + ", ".join(missing_fields), 3)
    return data


def verify_repo_grounding(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    artifacts = _require_files(root, [MAPPING_HARNESS, BASELINE_YAML, SPEC_PATH, PLAN_PATH])
    togsim_binary = _find_togsim_binary(root)
    baseline = _load_baseline_yaml(root / BASELINE_YAML)
    return {
        "preflight_ok": True,
        "repo_root": str(root),
        "artifacts": {
            "mapping_harness": artifacts[str(MAPPING_HARNESS)],
            "baseline_yaml": artifacts[str(BASELINE_YAML)],
            "codesign_spec": artifacts[str(SPEC_PATH)],
            "codesign_plan": artifacts[str(PLAN_PATH)],
        },
        "togsim_binary": str(togsim_binary.resolve()),
        "baseline_yaml": {field: baseline[field] for field in REQUIRED_BASELINE_FIELDS},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify repo grounding for the NPU co-design sweep.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_repo_grounding(args.repo_root)
    except PreflightError as exc:
        print(exc.message, file=sys.stderr)
        return exc.exit_code
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
