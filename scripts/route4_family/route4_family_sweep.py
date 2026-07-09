#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO / "outputs/route4_fusion_pilot"
REPORT_DIR = REPO / "docs/superpowers/specs"
HARNESS = Path(__file__).resolve().with_name("route4_family_harness.py")
BASE_LD = (
    "/home/dyf/miniconda3/envs/pytorchsim-build/lib:"
    "/home/dyf/miniconda3/envs/pytorchsim-build/x86_64-conda-linux-gnu/lib64:"
    "/home/dyf/.venv/lib64/python3.11/site-packages/torch/lib:"
    "/home/dyf/opt/lib:"
    "/home/dyf/opt/protobuf-3.21.12/lib64:"
    "/home/dyf/opt/protobuf-3.21.12/lib:"
    "/usr/local/cuda-12.8/lib64"
)
BASE_PYTHONPATH = "/home/dyf/PyTorchSim:/home/dyf/PyTorchSim/PyTorchSimDevice"
HW_IDS = ["HW-A", "HW-B", "HW-C", "HW-D"]
FUSIONS = ["none", "all"]
SELECTED_WORKLOADS = ["conv3x3_large", "conv1x1"]


WORKLOAD_INFO = {
    "conv3x3_large": {
        "title": "Conv3x3-large",
        "shape": "[1,128,28,28] -> [1,128,28,28]",
        "shape_key": "conv2d_1_128_128_3_3_28_28",
        "note": "priority-1 candidate; single Conv+BN+ReLU; external mapping uses 7 Conv tile params",
        "tiles": {
            "tile_A": {"TILE_K_H": 3, "TILE_K_W": 3, "TILE_O_H": 28, "TILE_O_W": 1, "TILE_M": 1, "TILE_N": 64, "TILE_K": 16, "regime": "tiny"},
            "tile_A2": {"TILE_K_H": 3, "TILE_K_W": 3, "TILE_O_H": 28, "TILE_O_W": 2, "TILE_M": 1, "TILE_N": 64, "TILE_K": 32, "regime": "tiny_dense"},
            "tile_B": {"TILE_K_H": 3, "TILE_K_W": 3, "TILE_O_H": 28, "TILE_O_W": 4, "TILE_M": 1, "TILE_N": 128, "TILE_K": 32, "regime": "small"},
            "tile_B2": {"TILE_K_H": 3, "TILE_K_W": 3, "TILE_O_H": 28, "TILE_O_W": 7, "TILE_M": 1, "TILE_N": 128, "TILE_K": 64, "regime": "small_dense"},
            "tile_C": {"TILE_K_H": 3, "TILE_K_W": 3, "TILE_O_H": 28, "TILE_O_W": 14, "TILE_M": 1, "TILE_N": 128, "TILE_K": 64, "regime": "medium"},
            "tile_C2": {"TILE_K_H": 3, "TILE_K_W": 3, "TILE_O_H": 28, "TILE_O_W": 14, "TILE_M": 1, "TILE_N": 128, "TILE_K": 128, "regime": "large"},
            "tile_C3": {"TILE_K_H": 3, "TILE_K_W": 3, "TILE_O_H": 28, "TILE_O_W": 21, "TILE_M": 1, "TILE_N": 128, "TILE_K": 128, "regime": "large_dense"},
            "tile_D": {"TILE_K_H": 3, "TILE_K_W": 3, "TILE_O_H": 28, "TILE_O_W": 28, "TILE_M": 1, "TILE_N": 128, "TILE_K": 128, "regime": "max_valid"},
        },
    },
    "conv1x1": {
        "title": "Conv1x1",
        "shape": "[1,256,14,14] -> [1,512,14,14]",
        "shape_key": "conv2d_1_256_512_1_1_14_14",
        "note": "priority-2 candidate; ResNet bottleneck point-wise Conv+BN+ReLU",
        "tiles": {
            "tile_A": {"TILE_K_H": 1, "TILE_K_W": 1, "TILE_O_H": 14, "TILE_O_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_K": 64, "regime": "tiny"},
            "tile_A2": {"TILE_K_H": 1, "TILE_K_W": 1, "TILE_O_H": 14, "TILE_O_W": 2, "TILE_M": 1, "TILE_N": 256, "TILE_K": 64, "regime": "tiny_dense"},
            "tile_B": {"TILE_K_H": 1, "TILE_K_W": 1, "TILE_O_H": 14, "TILE_O_W": 4, "TILE_M": 1, "TILE_N": 128, "TILE_K": 128, "regime": "small"},
            "tile_B2": {"TILE_K_H": 1, "TILE_K_W": 1, "TILE_O_H": 14, "TILE_O_W": 7, "TILE_M": 1, "TILE_N": 256, "TILE_K": 128, "regime": "small_dense"},
            "tile_C": {"TILE_K_H": 1, "TILE_K_W": 1, "TILE_O_H": 14, "TILE_O_W": 7, "TILE_M": 1, "TILE_N": 256, "TILE_K": 256, "regime": "medium"},
            "tile_C2": {"TILE_K_H": 1, "TILE_K_W": 1, "TILE_O_H": 14, "TILE_O_W": 14, "TILE_M": 1, "TILE_N": 512, "TILE_K": 128, "regime": "large"},
            "tile_C3": {"TILE_K_H": 1, "TILE_K_W": 1, "TILE_O_H": 14, "TILE_O_W": 7, "TILE_M": 1, "TILE_N": 512, "TILE_K": 256, "regime": "large_dense"},
            "tile_D": {"TILE_K_H": 1, "TILE_K_W": 1, "TILE_O_H": 14, "TILE_O_W": 14, "TILE_M": 1, "TILE_N": 512, "TILE_K": 256, "regime": "max_valid"},
        },
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def shell_capture(command: str, timeout_sec: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["bash", "-lc", f"set -o pipefail; {command}"],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        return {
            "command": command,
            "returncode": None,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
        }


def tile_working_set(tile: dict[str, Any]) -> dict[str, int]:
    kh = int(tile["TILE_K_H"])
    kw = int(tile["TILE_K_W"])
    oh = int(tile["TILE_O_H"])
    ow = int(tile["TILE_O_W"])
    m = int(tile["TILE_M"])
    n = int(tile["TILE_N"])
    k = int(tile["TILE_K"])
    ih = 1 + (oh - 1) + (kh - 1)
    iw = 1 + (ow - 1) + (kw - 1)
    weight = kh * kw * k * n
    input_ = ih * iw * m * k
    output = oh * ow * m * n
    total = 4 * (weight + input_ + output)
    return {
        "tile_i_h": ih,
        "tile_i_w": iw,
        "weight_bytes": 4 * weight,
        "input_bytes": 4 * input_,
        "output_bytes": 4 * output,
        "total_bytes": total,
    }


def enrich_tiles() -> dict[str, dict[str, dict[str, Any]]]:
    smallest = 32 * 1024 * 128 // 2
    largest = 128 * 1024 * 128 // 2
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for workload, info in WORKLOAD_INFO.items():
        result[workload] = {}
        for tile_name, tile in info["tiles"].items():
            ws = tile_working_set(tile)
            payload = {key: int(tile[key]) for key in ["TILE_K_H", "TILE_K_W", "TILE_O_H", "TILE_O_W", "TILE_M", "TILE_N", "TILE_K"]}
            payload.update(
                {
                    "working_set_bytes": ws["total_bytes"],
                    "predicted_working_set_bytes": ws["total_bytes"],
                    "fit_fraction_smallest_spad": ws["total_bytes"] / smallest,
                    "fit_fraction_largest_spad": ws["total_bytes"] / largest,
                    "predicted_fit_all_hw": ws["total_bytes"] <= smallest,
                    "regime": tile["regime"],
                    "tile_i_h": ws["tile_i_h"],
                    "tile_i_w": ws["tile_i_w"],
                    "weight_bytes": ws["weight_bytes"],
                    "input_bytes": ws["input_bytes"],
                    "output_bytes": ws["output_bytes"],
                }
            )
            result[workload][tile_name] = payload
    return result


def fit_info(tile: dict[str, Any], hw_id: str) -> dict[str, Any]:
    spad_kb = 128 if hw_id in {"HW-A", "HW-B"} else 32
    usable = spad_kb * 1024 * 128 // 2
    ws = int(tile["working_set_bytes"])
    return {"usable_spad_bytes": usable, "working_set_bytes": ws, "fit_fraction": ws / usable, "fits": ws <= usable}


def read_smoke_results() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for path in sorted((OUTPUT_ROOT / "workload_family_smoke").glob("*/family_run_result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results[path.parent.name] = data
    return results


def write_discovery(tiles: dict[str, Any]) -> dict[str, Any]:
    commands = [
        'grep -rn "workload.*conv\\|conv3x3\\|conv1x1\\|depthwise\\|pointwise" scripts/ 2>/dev/null | head -30',
        'grep -rn "workload_name\\|workload_type\\|--workload" scripts/route4_fusion_pilot*.py 2>/dev/null | head',
    ]
    command_results = [shell_capture(command) for command in commands]
    smoke = read_smoke_results()
    candidates = {
        "conv3x3_large": {
            "priority": 1,
            "reachable": bool(smoke.get("conv3x3_large", {}).get("ok")),
            "selected_for_sweep": True,
            "smoke_total_cycles": smoke.get("conv3x3_large", {}).get("total_cycles"),
            "tile_schema": "conv2d_1_128_128_3_3_28_28 -> 7 Conv tile params",
        },
        "conv1x1": {
            "priority": 2,
            "reachable": bool(smoke.get("conv1x1", {}).get("ok")),
            "selected_for_sweep": True,
            "smoke_total_cycles": smoke.get("conv1x1", {}).get("total_cycles"),
            "tile_schema": "conv2d_1_256_512_1_1_14_14 -> 7 Conv tile params",
        },
        "depthwise_conv3x3": {
            "priority": 3,
            "reachable": bool(smoke.get("depthwise_conv3x3", {}).get("ok")),
            "selected_for_sweep": False,
            "smoke_total_cycles": smoke.get("depthwise_conv3x3", {}).get("total_cycles"),
            "tile_schema": "same conv2d key shape as 3x3, but grouped lowering emits multiple kernels; not selected due 2-additional-workload cap and less clean tile schema",
        },
        "resnet50_stage1_residual": {
            "priority": 4,
            "reachable": False,
            "selected_for_sweep": False,
            "smoke_total_cycles": None,
            "tile_schema": "composite multi-conv graph; Phase A allows only single-kernel smoke tests",
        },
        "mobilenetv2_inverted_residual": {
            "priority": 5,
            "reachable": False,
            "selected_for_sweep": False,
            "smoke_total_cycles": None,
            "tile_schema": "composite PW+DW+PW graph; Phase A allows only single-kernel smoke tests",
        },
    }
    lines = [
        "# Workload family discovery",
        "",
        "Phase A time-boxed discovery checked the existing `route4_fusion_pilot.py` CLI and then used a temporary no-source-modification harness for single-kernel smoke tests.",
        "",
        "## Grep evidence",
        "",
    ]
    for result in command_results:
        status = "timed out" if result["timed_out"] else f"returncode={result['returncode']}"
        lines.extend(
            [
                "```bash",
                result["command"],
                "```",
                f"Status: `{status}`",
                "",
                "```text",
                (result["stdout"].strip() or "(no stdout)")[:8000],
                "```",
                "",
            ]
        )
    lines.extend(["## Candidate table", "", "| Candidate | Reachable | Selected | Smoke cycles | Tile schema / reason |", "|---|---:|---:|---:|---|"])
    for name, row in candidates.items():
        cycles = row["smoke_total_cycles"] if row["smoke_total_cycles"] is not None else "-"
        lines.append(f"| `{name}` | `{row['reachable']}` | `{row['selected_for_sweep']}` | {cycles} | {row['tile_schema']} |")
    lines.extend(
        [
            "",
            "## Tile-span limitation",
            "",
            "The two selected workloads are smaller than the inherited Conv3x3 `[1,64,56,56]` case. Their largest valid single-batch Conv tiles do not exceed the smallest-SPAD double-buffer budget, so Phase B keeps 8 ordered tiles but records that no selected additional workload reaches the 85-110% fit-boundary target.",
            "",
        ]
    )
    (OUTPUT_ROOT / "workload_family_discovery.md").write_text("\n".join(lines), encoding="utf-8")
    return {"commands": command_results, "smoke_results": smoke, "candidates": candidates, "selected_workloads": SELECTED_WORKLOADS}


def run_one_repeat(workload: str, hw_id: str, fusion: str, tile_name: str, tile: dict[str, Any], repeat: int, timeout_sec: int) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / "workload_family_runs" / workload / hw_id / tile_name / fusion / f"run_{repeat:02d}"
    result_path = run_dir / "family_run_result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if int(existing.get("total_cycles", 0) or 0) > 0:
            return {"result": existing, "returncode": 0, "reused": True}
    run_dir.mkdir(parents=True, exist_ok=True)
    tile_path = run_dir / "tile.json"
    write_json(tile_path, {key: tile[key] for key in ["TILE_K_H", "TILE_K_W", "TILE_O_H", "TILE_O_W", "TILE_M", "TILE_N", "TILE_K"]})
    hw_config = OUTPUT_ROOT / "hw_fusion_configs_v2" / f"{hw_id}_fuse_{fusion}.yml"
    command = [
        "timeout",
        str(timeout_sec),
        sys.executable,
        str(HARNESS),
        "--workload",
        workload,
        "--hw-config",
        str(hw_config),
        "--variant",
        fusion,
        "--tile-json",
        str(tile_path),
        "--run-dir",
        str(run_dir),
        "--seed",
        "0",
    ]
    env = {
        **dict(**__import__("os").environ),
        "CUDA_VISIBLE_DEVICES": "1",
        "LD_LIBRARY_PATH": BASE_LD,
        "PYTHONPATH": BASE_PYTHONPATH,
    }
    proc = subprocess.run(command, cwd=REPO, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (run_dir / "child_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {"ok": False, "error": proc.stdout[-2000:]}
    return {"result": result, "returncode": proc.returncode, "reused": False}


def run_sweep(tiles: dict[str, dict[str, Any]], repeats: int, timeout_sec: int, max_hours: float) -> tuple[dict[str, Any], bool]:
    matrix: dict[str, Any] = {}
    start_total = time.time()
    timeboxed = False
    for workload in SELECTED_WORKLOADS:
        workload_start = time.time()
        matrix[workload] = {}
        for hw_id in HW_IDS:
            matrix[workload][hw_id] = {}
            for tile_name, tile in tiles[workload].items():
                matrix[workload][hw_id][tile_name] = {}
                info = fit_info(tile, hw_id)
                for fusion in FUSIONS:
                    if time.time() - start_total > max_hours * 3600 or time.time() - workload_start > 4 * 3600:
                        timeboxed = True
                        matrix[workload][hw_id][tile_name][fusion] = {
                            "class": "not_run_timebox",
                            "cycles_in_order": [],
                            "median": 0.0,
                            "cycle_delta": 0.0,
                            "failures": [],
                            "fit_info": info,
                        }
                        continue
                    if not info["fits"]:
                        matrix[workload][hw_id][tile_name][fusion] = {
                            "class": "unavailable",
                            "cycles_in_order": [],
                            "median": 0.0,
                            "cycle_delta": 0.0,
                            "failures": [],
                            "fit_info": info,
                        }
                        continue
                    cycles: list[int] = []
                    failures: list[dict[str, Any]] = []
                    for repeat in range(repeats):
                        outcome = run_one_repeat(workload, hw_id, fusion, tile_name, tile, repeat, timeout_sec)
                        result = outcome["result"]
                        total_cycles = int(result.get("total_cycles", 0) or 0)
                        cycles.append(total_cycles)
                        if outcome["returncode"] != 0 or total_cycles <= 0:
                            failures.append(
                                {
                                    "repeat": repeat,
                                    "returncode": outcome["returncode"],
                                    "total_cycles": total_cycles,
                                    "error": result.get("error", "") or result.get("artifact_message", ""),
                                }
                            )
                    valid = [cycle for cycle in cycles if cycle > 0]
                    median = float(statistics.median(valid)) if valid else 0.0
                    delta = (max(valid) - min(valid)) / median if valid and median else 0.0
                    matrix[workload][hw_id][tile_name][fusion] = {
                        "class": "measured" if len(valid) == repeats and not failures else "blocked",
                        "cycles_in_order": cycles,
                        "median": median,
                        "cycle_delta": delta,
                        "failures": failures,
                        "fit_info": info,
                    }
    return matrix, timeboxed


def analyze_one_workload(matrix: dict[str, Any], tiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    champion_per_hw: dict[str, Any] = {}
    measured: list[dict[str, Any]] = []
    for hw_id, tile_map in matrix.items():
        best = None
        for tile_name, fusion_map in tile_map.items():
            for fusion, cell in fusion_map.items():
                if cell.get("class") != "measured" or float(cell.get("median", 0) or 0) <= 0:
                    continue
                item = {"hw": hw_id, "tile": tile_name, "fusion": fusion, "median": float(cell["median"])}
                measured.append(item)
                if best is None or item["median"] < best["median"]:
                    best = item
        champion_per_hw[hw_id] = None if best is None else {"tile": best["tile"], "fusion": best["fusion"], "median": best["median"]}
    global_min = min(measured, key=lambda item: item["median"]) if measured else None
    fit_tiles_by_hw = {
        hw_id: [
            name
            for name, tile in sorted(tiles.items(), key=lambda pair: int(pair[1]["working_set_bytes"]))
            if fit_info(tile, hw_id)["fits"]
        ]
        for hw_id in HW_IDS
    }
    extreme = None
    if global_min:
        fit_names = fit_tiles_by_hw[global_min["hw"]]
        extreme = global_min["tile"] in {fit_names[0], fit_names[-1]} if fit_names else None
    return {
        "champion_migration": len({(item["tile"], item["fusion"]) for item in champion_per_hw.values() if item}) > 1,
        "champion_per_hw": champion_per_hw,
        "distinct_champion_tuples": len({(item["tile"], item["fusion"]) for item in champion_per_hw.values() if item}),
        "global_min_cell": global_min,
        "global_min_tile_is_extreme_for_hw": extreme,
        "tile_interior_evidence": bool(global_min and extreme is False),
        "fit_tiles_by_hw": fit_tiles_by_hw,
    }


def family_analysis(matrix_payload: dict[str, Any], tiles: dict[str, Any], timeboxed: bool) -> dict[str, Any]:
    v2_analysis = json.loads((OUTPUT_ROOT / "interior_optimum_analysis_v2.json").read_text(encoding="utf-8"))
    per_workload = {"conv3x3_probe": {**v2_analysis, "source": "inherited_from_tile_fusion_hw_matrix_v2"}}
    for workload in SELECTED_WORKLOADS:
        per_workload[workload] = analyze_one_workload(matrix_payload["matrix"][workload], tiles[workload])
    all_interior = all(bool(item.get("tile_interior_evidence")) for item in per_workload.values())
    any_new_interior = any(bool(per_workload[w].get("tile_interior_evidence")) for w in SELECTED_WORKLOADS)
    if timeboxed:
        verdict = "WORKLOAD_FAMILY_PARTIAL_SCOPE_LIMITATION"
    elif len(per_workload) >= 2 and all_interior:
        verdict = "WORKLOAD_FAMILY_INTERIOR_OPTIMUM_CONFIRMED"
    elif any_new_interior:
        verdict = "PARTIAL_WORKLOAD_FAMILY_INTERIOR"
    else:
        verdict = "WORKLOAD_FAMILY_INTERIOR_NEGATIVE"
    per_hw_stability = {}
    for hw_id in HW_IDS:
        tiles_for_hw = [data.get("champion_per_hw", {}).get(hw_id, {}).get("tile") for data in per_workload.values()]
        counts = Counter(tile for tile in tiles_for_hw if tile)
        per_hw_stability[hw_id] = (max(counts.values()) / len(tiles_for_hw)) if counts and tiles_for_hw else 0.0
    mapping = {
        f"{workload}/{hw_id}": data.get("champion_per_hw", {}).get(hw_id, {}).get("tile")
        for workload, data in per_workload.items()
        for hw_id in HW_IDS
    }
    return {
        "per_workload": per_workload,
        "family_level": {
            "workload_family_all_have_interior": all_interior,
            "champion_tile_stability_across_workloads": sum(per_hw_stability.values()) / len(per_hw_stability),
            "champion_tile_stability_per_hw": per_hw_stability,
            "family_migration_pattern": {
                "mapping": mapping,
                "distinct_champion_tiles": sorted({tile for tile in mapping.values() if tile}),
                "distinct_champion_tile_count": len({tile for tile in mapping.values() if tile}),
            },
            "tested_workloads": list(per_workload),
            "selected_additional_workloads": SELECTED_WORKLOADS,
            "timeboxed": timeboxed,
            "verdict": verdict,
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.1f}" if not value.is_integer() else f"{int(value):,}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_report(analysis: dict[str, Any], tiles: dict[str, Any], discovery: dict[str, Any], matrix_payload: dict[str, Any]) -> None:
    family = analysis["family_level"]
    lines = [
        "# NPU workload family interior-optimum verification",
        "",
        "Date: 2026-07-08",
        "",
        "## 1. Motivation",
        "",
        "Conv3x3 v2 已在 `23ea49f` 证明 single-workload `TILE_FUSION_INTERIOR_OPTIMUM_CONFIRMED_V2`。本轮把同一 tile x fusion x HW 方法扩展到 workload family，检查 interior optimum 是否只属于单个 Conv3x3 case，还是能跨 Conv-heavy family 保持。",
        "",
        "## 2. Method",
        "",
        "- Inherited workload: `conv3x3_probe`，直接引用 `tile_fusion_hw_matrix_v2.json` 与 `interior_optimum_analysis_v2.json`。",
        "- Additional workloads: `conv3x3_large` 与 `conv1x1`。",
        "- HW: `codesign_v1_2x2` 的 `HW-A/B/C/D`。",
        "- Fusion variants: `none` 与 `all`。",
        "- Repeats: 5 per measured cell，`pytorchsim_functional_mode=0`，`CUDA_VISIBLE_DEVICES=1`。",
        "- Source rule: 未修改 PyTorchSim / TOGSim / gem5 / ramulator2 / spike source；新 workload 通过临时 harness 复用 external Conv tile mapping。",
        "",
        "Discovery 文档：`outputs/route4_fusion_pilot/workload_family_discovery.md`。",
        "",
        "## 3. Per-workload results",
        "",
    ]
    for workload, result in analysis["per_workload"].items():
        lines.extend([f"### {workload}", "", "| HW | Champion tile | Fusion | Median cycles |", "|---|---|---|---:|"])
        for hw_id in HW_IDS:
            champ = result.get("champion_per_hw", {}).get(hw_id) or {}
            lines.append(f"| {hw_id} | `{champ.get('tile', '-')}` | `{champ.get('fusion', '-')}` | {fmt(champ.get('median'))} |")
        lines.extend(
            [
                "",
                f"- `tile_interior_evidence`: `{result.get('tile_interior_evidence')}`",
                f"- `global_min_cell`: `{json.dumps(result.get('global_min_cell'), sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 4. Family-level verdict",
            "",
            f"- `workload_family_all_have_interior`: `{family['workload_family_all_have_interior']}`",
            f"- `champion_tile_stability_across_workloads`: `{family['champion_tile_stability_across_workloads']:.3f}`",
            f"- `verdict`: `{family['verdict']}`",
            "",
            "## 5. Cross-workload comparison",
            "",
            "| HW | Stability | Champion tiles by workload |",
            "|---|---:|---|",
        ]
    )
    for hw_id in HW_IDS:
        entries = []
        for workload in analysis["per_workload"]:
            tile = family["family_migration_pattern"]["mapping"].get(f"{workload}/{hw_id}")
            entries.append(f"`{workload}:{tile}`")
        lines.append(f"| {hw_id} | {family['champion_tile_stability_per_hw'][hw_id]:.3f} | {'; '.join(entries)} |")
    lines.extend(
        [
            "",
            "## 6. Scope limitations",
            "",
            "- Tested family is Conv-heavy but still small: inherited Conv3x3, Conv3x3-large, Conv1x1。",
            "- Additional workloads have smaller valid tile working sets than the inherited `[1,64,56,56]` Conv3x3; neither reaches the smallest-SPAD 85-110% fit-boundary target with valid single-batch tiles。",
            "- DepthwiseConv3x3 smoke test produced cycles but has grouped/multi-kernel lowering behavior, so it was not selected under the two-additional-workload cap。",
            "- Mode=1 correctness remains deferred; this report is mode=0 timing evidence。",
            "- Existing determinism diagnosis gives a non-trivial noise floor; medians should be interpreted as timing evidence, not exact silicon prediction。",
            "",
            "## 7. Next step recommendation",
            "",
        ]
    )
    if family["verdict"] == "WORKLOAD_FAMILY_INTERIOR_OPTIMUM_CONFIRMED":
        lines.append("下一步可以转向更完整的 Conv-heavy block，例如 ResNet bottleneck stage 或 MobileNetV2 inverted residual，并加入 multi-mapping ranking，验证 family-level result 是否能支撑正式 Gate-3。")
    elif family["verdict"] == "PARTIAL_WORKLOAD_FAMILY_INTERIOR":
        lines.append("下一步应优先扩大 workload family 或增加真实 HW interior points，因为当前 family evidence 已显示 workload-dependent tile optimum。")
    elif family["verdict"] == "WORKLOAD_FAMILY_INTERIOR_NEGATIVE":
        lines.append("下一步不应声称 family-level interior optimum；应先扩大 spatial/channel 更大的 Conv-heavy workloads，或转向真实 ResNet/MobileNet block。")
    else:
        lines.append("下一步应先解除 harness/timebox 限制，再复跑完整 workload family sweep。")
    md = "\n".join(lines) + "\n"
    md_path = REPORT_DIR / "2026-07-08-workload-family-interior-optimum.md"
    md_path.write_text(md, encoding="utf-8")
    html_lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>NPU workload family interior-optimum verification</title>",
        "<style>body{font-family:Arial,sans-serif;line-height:1.55;max-width:1180px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #ccc;padding:6px 8px;text-align:left}code{background:#f3f3f3;padding:1px 4px}</style>",
        "</head><body>",
    ]
    for line in md.splitlines():
        if line.startswith("# "):
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("|"):
            continue
        elif line.startswith("- "):
            html_lines.append(f"<p>{html.escape(line)}</p>")
        elif line.strip():
            html_lines.append(f"<p>{html.escape(line)}</p>")
    html_lines.append("</body></html>")
    (REPORT_DIR / "2026-07-08-workload-family-interior-optimum.html").write_text("\n".join(html_lines) + "\n", encoding="utf-8")


def update_tracker(analysis: dict[str, Any]) -> None:
    md_path = REPORT_DIR / "2026-07-03-co-design-progress-tracker.md"
    html_path = REPORT_DIR / "2026-07-03-co-design-progress-tracker.html"
    verdict = analysis["family_level"]["verdict"]
    md = md_path.read_text(encoding="utf-8")
    status_line = (
        "当前状态：Route 4 已经完成 workload family interior-optimum generalization。"
        f"Family verdict 为 `{verdict}`；tested workloads 为 `conv3x3_probe`、`conv3x3_large`、`conv1x1`。"
    )
    md = __import__("re").sub(r"当前状态：Route 4 .*?\n", status_line + "\n", md, count=1)
    row = (
        "| **Workload family interior probe** | 在 inherited Conv3x3 v2 基础上扩展到 Conv3x3-large 与 Conv1x1，检查 tile interior optimum 是否跨 workload family 保持 | "
        "`/tmp/codesign-next-handoff.md` workload family interior-optimum generalization | 当前提交 | "
        f"**{verdict}** | tested workloads=`conv3x3_probe/conv3x3_large/conv1x1`；"
        f"all_have_interior=`{analysis['family_level']['workload_family_all_have_interior']}` |\n"
    )
    if "Workload family interior probe" not in md:
        md = md.replace(
            "| **8-tile denser interior probe** | 在 Conv 3x3 上把 tile space 从 4 加密到 8，并加入 `fusion=[\"fusion\"]` 第三 variant，检查 tile interior optimum | `/tmp/codesign-next-handoff.md` 8-tile denser sweep push interior optimum | 当前提交 | **TILE_FUSION_INTERIOR_OPTIMUM_CONFIRMED_V2** | 450 measured runs；30 unavailable slots；global min=`HW-A/tile_C/all` |\n",
            "| **8-tile denser interior probe** | 在 Conv 3x3 上把 tile space 从 4 加密到 8，并加入 `fusion=[\"fusion\"]` 第三 variant，检查 tile interior optimum | `/tmp/codesign-next-handoff.md` 8-tile denser sweep push interior optimum | `23ea49f` | **TILE_FUSION_INTERIOR_OPTIMUM_CONFIRMED_V2** | 450 measured runs；30 unavailable slots；global min=`HW-A/tile_C/all` |\n" + row,
        )
    md_path.write_text(md, encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")
    html_text = __import__("re").sub(
        r"<p><strong>当前状态：</strong>.*?</p>",
        f"<p><strong>当前状态：</strong>Route 4 已经完成 workload family interior-optimum generalization。Family verdict 为 <code>{html.escape(verdict)}</code>；tested workloads 为 <code>conv3x3_probe</code>、<code>conv3x3_large</code>、<code>conv1x1</code>。</p>",
        html_text,
        count=1,
        flags=__import__("re").S,
    )
    html_row = (
        f'<tr><td><strong>Workload family interior probe</strong></td><td>在 inherited Conv3x3 v2 基础上扩展到 Conv3x3-large 与 Conv1x1，检查 tile interior optimum 是否跨 workload family 保持</td>'
        f'<td><code>/tmp/codesign-next-handoff.md</code></td><td>当前提交</td><td class="positive">{html.escape(verdict)}</td>'
        f'<td>tested workloads=<code>conv3x3_probe/conv3x3_large/conv1x1</code>；all_have_interior=<code>{analysis["family_level"]["workload_family_all_have_interior"]}</code></td></tr>'
    )
    if "Workload family interior probe" not in html_text:
        html_text = html_text.replace("</table>", html_row + "\n</table>", 1)
    html_path.write_text(html_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--run-timeout-sec", type=int, default=900)
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args()

    tiles = enrich_tiles()
    discovery = write_discovery(tiles)
    write_json(OUTPUT_ROOT / "tile_variants_per_workload.json", tiles)
    matrix, timeboxed = run_sweep(tiles, args.repeats, args.run_timeout_sec, args.max_hours)
    matrix_payload = {
        "inherited": {
            "conv3x3_probe": {
                "matrix_path": "outputs/route4_fusion_pilot/tile_fusion_hw_matrix_v2.json",
                "analysis_path": "outputs/route4_fusion_pilot/interior_optimum_analysis_v2.json",
            }
        },
        "matrix": matrix,
        "repeats_per_cell": args.repeats,
        "fusion_variants": FUSIONS,
        "hw_config_set": "codesign_v1_2x2",
        "selected_additional_workloads": SELECTED_WORKLOADS,
        "timeboxed": timeboxed,
        "pytorchsim_functional_mode": 0,
        "cuda_visible_devices": "1",
    }
    write_json(OUTPUT_ROOT / "tile_fusion_hw_matrix_per_workload.json", matrix_payload)
    analysis = family_analysis(matrix_payload, tiles, timeboxed)
    write_json(OUTPUT_ROOT / "workload_family_interior_analysis.json", analysis)
    summary_path = OUTPUT_ROOT / "pilot_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["workload_family_interior_optimum"] = {
        "discovery": "outputs/route4_fusion_pilot/workload_family_discovery.md",
        "tile_variants_per_workload": "outputs/route4_fusion_pilot/tile_variants_per_workload.json",
        "matrix": "outputs/route4_fusion_pilot/tile_fusion_hw_matrix_per_workload.json",
        "analysis": "outputs/route4_fusion_pilot/workload_family_interior_analysis.json",
        "verdict": analysis["family_level"]["verdict"],
        "tested_workloads": analysis["family_level"]["tested_workloads"],
    }
    write_json(summary_path, summary)
    render_report(analysis, tiles, discovery, matrix_payload)
    update_tracker(analysis)
    print(json.dumps(analysis["family_level"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
