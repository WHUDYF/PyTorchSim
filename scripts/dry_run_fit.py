#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DTYPE_BYTES_DEFAULT = 4
SAFETY_FACTOR_DEFAULT = 0.9


def working_set_bytes(tile: dict[str, Any], dtype_bytes: int = DTYPE_BYTES_DEFAULT) -> int:
    tile_m = int(tile["TILE_M"])
    tile_n = int(tile["TILE_N"])
    tile_k = int(tile["TILE_K"])
    elements = tile_m * tile_k + tile_k * tile_n + tile_m * tile_n
    return elements * dtype_bytes


def spad_budget_bytes(hw_config: dict[str, Any], safety_factor: float = SAFETY_FACTOR_DEFAULT) -> int:
    spad_kb_per_lane = int(hw_config["vpu_spad_size_kb_per_lane"])
    lanes = int(hw_config["vpu_num_lanes"])
    return int(spad_kb_per_lane * 1024 * lanes * safety_factor)


def fit_classifier(
    tile: dict[str, Any],
    hw_config: dict[str, Any],
    *,
    dtype_bytes: int = DTYPE_BYTES_DEFAULT,
    safety_factor: float = SAFETY_FACTOR_DEFAULT,
) -> dict[str, Any]:
    working_set = working_set_bytes(tile, dtype_bytes)
    budget = spad_budget_bytes(hw_config, safety_factor)
    return {
        "predicted_fit": working_set <= budget,
        "working_set_bytes": working_set,
        "spad_budget_bytes": budget,
        "budget_over_bytes": max(0, working_set - budget),
        "safety_factor": safety_factor,
        "dtype_bytes": dtype_bytes,
    }


def evaluate_calibration(cases: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = []
    mismatch_count = 0
    for case in cases:
        predicted = bool(case["predicted_fit"])
        actual = bool(case["actual_fit"])
        xor_value = int(predicted ^ actual)
        mismatch_count += xor_value
        row = dict(case)
        row["predicted_xor_actual"] = xor_value
        evaluated.append(row)
    passed = mismatch_count == 0
    return {
        "calibration_passed": passed,
        "calibration_status": "calibrated" if passed else "fallback_engaged",
        "mismatch_count": mismatch_count,
        "cases": evaluated,
    }


def classify_cells(
    cells: list[dict[str, Any]],
    calibration: dict[str, Any],
    hw_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    fallback = calibration.get("calibration_status") == "fallback_engaged"
    for cell in cells:
        row = {
            "hw_id": cell.get("hw_id", ""),
            "mapping_id": cell.get("mapping_id", ""),
            "tile": cell["tile"],
        }
        if fallback:
            row.update(
                {
                    "availability_precheck": "no-precheck",
                    "status": "needs_togsim",
                    "predicted_fit": None,
                    "working_set_bytes": None,
                    "spad_budget_bytes": None,
                    "budget_over_bytes": None,
                }
            )
        else:
            fit = fit_classifier(cell["tile"], hw_config)
            row.update(fit)
            row["availability_precheck"] = "analytical"
            row["status"] = "needs_togsim" if fit["predicted_fit"] else "unavailable"
        rows.append(row)
    return rows


def load_yaml_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify mapping tiles against SPAD dry-run fit rules.")
    parser.add_argument("--cells-json", type=Path, required=True)
    parser.add_argument("--hw-config", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cells = json.loads(args.cells_json.read_text(encoding="utf-8"))
    hw_config = load_yaml_or_json(args.hw_config)
    calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    rows = classify_cells(cells, calibration, hw_config)
    payload = {"data_label": "analytical_precheck", "rows": rows, "calibration_status": calibration.get("calibration_status")}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
