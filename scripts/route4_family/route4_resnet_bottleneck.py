#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import statistics
import subprocess
import sys
import time
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
BASE_PYTHONPATH = f"{REPO}:{REPO / 'PyTorchSimDevice'}"
HW_IDS = ["HW-A", "HW-B", "HW-C", "HW-D"]
FUSIONS = ["none", "all"]
BASE_TILE_ORDER = ["tile_A", "tile_A2", "tile_B", "tile_B2", "tile_C", "tile_C2", "tile_C3", "tile_D"]
DENSE_TILE_ORDER = ["tile_A_a", "tile_A_b", "tile_A_c"]
CORE_SHAPE_KEY = "conv2d_1_128_128_3_3_28_28"
WORKLOAD = "resnet50_bottleneck_c3"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_tiles(include_dense: bool = False) -> dict[str, dict[str, Any]]:
    path = OUTPUT_ROOT / "tile_variants_conv3x3_large_densified.json"
    all_tiles = json.loads(path.read_text(encoding="utf-8"))
    order = BASE_TILE_ORDER + (DENSE_TILE_ORDER if include_dense else [])
    return {name: all_tiles[name] for name in order if name in all_tiles}


def tile_params(tile: dict[str, Any]) -> dict[str, int]:
    keys = ["TILE_K_H", "TILE_K_W", "TILE_O_H", "TILE_O_W", "TILE_M", "TILE_N", "TILE_K"]
    return {key: int(tile[key]) for key in keys}


def fit_info(tile: dict[str, Any], hw_id: str) -> dict[str, Any]:
    spad_kb = 128 if hw_id in {"HW-A", "HW-B"} else 32
    usable = spad_kb * 1024 * 128 // 2
    ws = int(tile["working_set_bytes"])
    return {"usable_spad_bytes": usable, "working_set_bytes": ws, "fit_fraction": ws / usable, "fits": ws <= usable}


def run_one_repeat(tile_name: str, tile: dict[str, Any], hw_id: str, fusion: str, repeat: int, timeout_sec: int, *, root: Path) -> dict[str, Any]:
    run_dir = root / hw_id / tile_name / fusion / f"run_{repeat:02d}"
    result_path = run_dir / "family_run_result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if int(existing.get("total_cycles", 0) or 0) > 0:
            return {"returncode": 0, "result": existing, "reused": True, "run_dir": str(run_dir)}
    run_dir.mkdir(parents=True, exist_ok=True)
    tile_path = run_dir / "tile.json"
    write_json(tile_path, tile_params(tile))
    hw_config = OUTPUT_ROOT / "hw_fusion_configs_v2" / f"{hw_id}_fuse_{fusion}.yml"
    command = [
        "timeout",
        str(timeout_sec),
        sys.executable,
        str(HARNESS),
        "--workload",
        WORKLOAD,
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
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "1",
        "LD_LIBRARY_PATH": BASE_LD,
        "PYTHONPATH": BASE_PYTHONPATH,
    }
    proc = subprocess.run(command, cwd=REPO, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (run_dir / "child_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {"ok": False, "error": proc.stdout[-4000:]}
    return {"returncode": proc.returncode, "result": result, "reused": False, "run_dir": str(run_dir)}


def gate0_smoke(timeout_sec: int) -> dict[str, Any]:
    tiles = load_tiles(include_dense=False)
    tile_name = "tile_B2"
    tile = tiles[tile_name]
    root = OUTPUT_ROOT / "resnet_bottleneck_smoke"
    outcome = run_one_repeat(tile_name, tile, "HW-A", "all", 0, timeout_sec, root=root)
    run_dir = Path(outcome["run_dir"])
    mapping_path = run_dir / "external_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
    core_mapping = mapping.get(CORE_SHAPE_KEY)
    result = outcome["result"]
    checks = {
        "compiled_tog_togsim_without_error": outcome["returncode"] == 0 and bool(result.get("ok")),
        "total_cycles_positive": int(result.get("total_cycles", 0) or 0) > 0,
        "core_shape_key_present": CORE_SHAPE_KEY in mapping,
        "override_tile_applied_to_core": core_mapping == tile_params(tile),
    }
    payload = {
        "workload": WORKLOAD,
        "hw": "HW-A",
        "fusion": "all",
        "tile": tile_name,
        "run_dir": str(run_dir),
        "command_returncode": outcome["returncode"],
        "checks": checks,
        "passed": all(checks.values()),
        "result": result,
        "external_mapping_path": str(mapping_path),
        "core_shape_key": CORE_SHAPE_KEY,
        "core_mapping": core_mapping,
        "expected_core_tile": tile_params(tile),
    }
    write_json(OUTPUT_ROOT / "resnet_bottleneck_gate0_smoke.json", payload)
    return payload


def write_blocker(smoke: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
    result = smoke.get("result", {})
    traceback = result.get("traceback", "")
    error = result.get("error", "") or result.get("artifact_message", "")
    lines = [
        "# ResNet-50 bottleneck Gate 0 blocker",
        "",
        "Verdict: `BLOCKED`",
        "",
        f"- Workload: `{WORKLOAD}`",
        "- Command: `CUDA_VISIBLE_DEVICES=1 python scripts/route4_family/route4_resnet_bottleneck.py --smoke-only`",
        f"- Smoke run dir: `{smoke.get('run_dir')}`",
        f"- Return code: `{smoke.get('command_returncode')}`",
        f"- Checks: `{json.dumps(smoke.get('checks', {}), sort_keys=True)}`",
        f"- Error: `{error}`",
        "",
        "## Traceback",
        "",
        "```text",
        traceback or "(no traceback captured)",
        "```",
        "",
        "## Bounded troubleshooting attempts",
        "",
    ]
    if not attempts:
        lines.append("(no additional attempts were needed before blocker classification)")
    for attempt in attempts:
        lines.extend(
            [
                f"### Attempt {attempt['attempt']}",
                "",
                f"- Hypothesis: {attempt['hypothesis']}",
                f"- Command: `{attempt['command']}`",
                f"- Return code: `{attempt['returncode']}`",
                "",
                "```text",
                (attempt.get("output") or "")[-4000:],
                "```",
                "",
            ]
        )
    (OUTPUT_ROOT / "resnet_bottleneck_blocker.md").write_text("\n".join(lines), encoding="utf-8")
    analysis = {
        "workload": WORKLOAD,
        "verdict": "BLOCKED",
        "gate0_smoke": smoke,
        "bounded_troubleshooting_attempts": attempts,
    }
    write_json(OUTPUT_ROOT / "resnet_bottleneck_interior_analysis.json", analysis)


def run_sweep(tiles: dict[str, dict[str, Any]], repeats: int, timeout_sec: int, max_hours: float, *, root: Path) -> tuple[dict[str, Any], bool]:
    matrix: dict[str, Any] = {}
    start = time.time()
    timeboxed = False
    for hw_id in HW_IDS:
        matrix[hw_id] = {}
        for tile_name, tile in tiles.items():
            matrix[hw_id][tile_name] = {}
            info = fit_info(tile, hw_id)
            for fusion in FUSIONS:
                if time.time() - start > max_hours * 3600:
                    timeboxed = True
                    matrix[hw_id][tile_name][fusion] = {
                        "class": "not_run_timebox",
                        "cycles_in_order": [],
                        "median": 0.0,
                        "cycle_delta": 0.0,
                        "failures": [],
                        "fit_info": info,
                    }
                    continue
                if not info["fits"]:
                    matrix[hw_id][tile_name][fusion] = {
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
                    outcome = run_one_repeat(tile_name, tile, hw_id, fusion, repeat, timeout_sec, root=root)
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
                                "run_dir": outcome["run_dir"],
                            }
                        )
                valid = [cycle for cycle in cycles if cycle > 0]
                median = float(statistics.median(valid)) if valid else 0.0
                delta = (max(valid) - min(valid)) / median if valid and median else 0.0
                matrix[hw_id][tile_name][fusion] = {
                    "class": "measured" if len(valid) == repeats and not failures else "blocked",
                    "cycles_in_order": cycles,
                    "median": median,
                    "cycle_delta": delta,
                    "failures": failures,
                    "fit_info": info,
                }
    return matrix, timeboxed


def merge_matrix(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = {hw_id: {tile: dict(fusion_map) for tile, fusion_map in tile_map.items()} for hw_id, tile_map in base.items()}
    for hw_id, tile_map in extra.items():
        merged.setdefault(hw_id, {})
        for tile_name, fusion_map in tile_map.items():
            merged[hw_id][tile_name] = fusion_map
    return merged


def analyze_matrix(matrix: dict[str, Any], tiles: dict[str, dict[str, Any]], *, timeboxed: bool, densification_applied: bool) -> dict[str, Any]:
    measured: list[dict[str, Any]] = []
    champion_per_hw: dict[str, Any] = {}
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
    champion_tiles = {item["tile"] for item in champion_per_hw.values() if item}
    gate1 = len(champion_tiles) > 1
    gate2 = bool(global_min and extreme is False)
    if gate1 and gate2:
        verdict = "RESNET_BOTTLENECK_INTERIOR_OPTIMUM_CONFIRMED"
    elif gate1:
        verdict = "PARTIAL_RESNET_BOTTLENECK_INTERIOR"
    else:
        verdict = "NEGATIVE"
    return {
        "workload": WORKLOAD,
        "data_label": "measured",
        "timeboxed": timeboxed,
        "densification_applied": densification_applied,
        "measured_cell_count": len(measured),
        "champion_per_hw": champion_per_hw,
        "champion_migration": gate1,
        "tile_interior_evidence": gate2,
        "global_min_cell": global_min,
        "global_min_tile_is_extreme_for_hw": extreme,
        "fit_tiles_by_hw": fit_tiles_by_hw,
        "verdict": verdict,
    }


def matrix_accounting(matrix: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tile_map in matrix.values():
        for fusion_map in tile_map.values():
            for cell in fusion_map.values():
                cls = str(cell.get("class", "unknown"))
                counts[cls] = counts.get(cls, 0) + 1
    return counts


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{int(value):,}" if value.is_integer() else f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_report(analysis: dict[str, Any], matrix_payload: dict[str, Any]) -> None:
    md_path = REPORT_DIR / "2026-07-09-resnet-bottleneck-interior-optimum.md"
    html_path = REPORT_DIR / "2026-07-09-resnet-bottleneck-interior-optimum.html"
    if analysis.get("verdict") == "BLOCKED":
        md = "\n".join(
            [
                "# ResNet-50 bottleneck interior-optimum probe",
                "",
                "## 结论",
                "",
                "- `verdict`: `BLOCKED`",
                "- Gate 0 smoke 未通过，未执行 full sweep。",
                "- blocker: `outputs/route4_fusion_pilot/resnet_bottleneck_blocker.md`",
                "",
            ]
        )
    else:
        lines = [
            "# ResNet-50 bottleneck interior-optimum probe",
            "",
            "Date: 2026-07-09",
            "",
            "## 1. 目标",
            "",
            "本实验把 isolated Conv kernel 的 tile interior-optimum 结论推进到 ResNet-50 `conv3_x` bottleneck。只 sweep bottleneck 中 3x3 core 的 tile，1x1 reduce / expand 保持固定默认 tile。",
            "",
            "## 2. Workload",
            "",
            "- Input: `x[1,512,28,28]`",
            "- Block: `1x1 reduce 512->128` + `3x3 core 128->128` + `1x1 expand 128->512` + residual add + ReLU",
            f"- 3x3 core shape key: `{CORE_SHAPE_KEY}`",
            "- HW: `HW-A/B/C/D` from `codesign_v1_2x2`",
            "- Fusion: `none/all`",
            "- Repeats: 5 per measured cell, `pytorchsim_functional_mode=0`, `CUDA_VISIBLE_DEVICES=1`",
            "",
            "## 3. Gate 0",
            "",
            f"- Smoke passed: `{matrix_payload['gate0_smoke']['passed']}`",
            f"- Smoke total cycles: `{matrix_payload['gate0_smoke']['result'].get('total_cycles')}`",
            f"- Core tile override applied: `{matrix_payload['gate0_smoke']['checks'].get('override_tile_applied_to_core')}`",
            "",
            "## 4. Sweep accounting",
            "",
            f"- Base tiles: `{', '.join(BASE_TILE_ORDER)}`",
            f"- Densification applied: `{analysis['densification_applied']}`",
            f"- Matrix accounting: `{json.dumps(matrix_payload['run_accounting'], sort_keys=True)}`",
            f"- Timeboxed: `{matrix_payload['timeboxed']}`",
            "",
            "## 5. Champion per HW",
            "",
            "| HW | Champion tile | Fusion | Median cycles |",
            "|---|---|---|---:|",
        ]
        for hw_id in HW_IDS:
            champ = analysis["champion_per_hw"].get(hw_id) or {}
            lines.append(f"| {hw_id} | `{champ.get('tile', '-')}` | `{champ.get('fusion', '-')}` | {fmt(champ.get('median'))} |")
        lines.extend(
            [
                "",
                "## 6. Verdict",
                "",
                f"- `champion_migration`: `{analysis['champion_migration']}`",
                f"- `tile_interior_evidence`: `{analysis['tile_interior_evidence']}`",
                f"- `global_min_tile_is_extreme_for_hw`: `{analysis['global_min_tile_is_extreme_for_hw']}`",
                f"- `global_min_cell`: `{json.dumps(analysis['global_min_cell'], sort_keys=True)}`",
                f"- `verdict`: `{analysis['verdict']}`",
                "",
                "## 7. Scope",
                "",
                "- 该结论是 mode=0 timing evidence，不是 correctness claim。",
                "- 只 sweep 3x3 core tile；1x1 reduce / expand 固定 tile。",
                "- 不包含 MobileNetV2/depthwise/grouped-conv。",
                "- 未增加 HW interior points，也未 sweep dataflow / SPAD partition / DMA schedule。",
                "",
            ]
        )
        md = "\n".join(lines)
    md_path.write_text(md, encoding="utf-8")

    body = []
    for line in md.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<p>{html.escape(line)}</p>")
        elif line.startswith("|"):
            body.append(f"<pre>{html.escape(line)}</pre>")
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
    html_text = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ResNet bottleneck interior optimum</title>"
        "<style>body{font-family:Arial,sans-serif;line-height:1.55;max-width:1180px;margin:32px auto;padding:0 20px}"
        "code{background:#f3f3f3;padding:1px 4px}pre{background:#fafafa;border:1px solid #ddd;padding:6px}</style>"
        "</head><body>\n"
        + "\n".join(body)
        + "\n</body></html>\n"
    )
    html_path.write_text(html_text, encoding="utf-8")


def update_tracker(verdict: str, analysis: dict[str, Any]) -> None:
    md_path = REPORT_DIR / "2026-07-03-co-design-progress-tracker.md"
    html_path = REPORT_DIR / "2026-07-03-co-design-progress-tracker.html"
    md = md_path.read_text(encoding="utf-8")
    summary = f"当前状态：Route 4 已完成 ResNet-50 bottleneck interior-optimum probe。Verdict 为 `{verdict}`。"
    md = re.sub(r"当前状态：Route 4 .*?\n", summary + "\n", md, count=1)
    row = (
        "| **ResNet-50 bottleneck interior probe** | 将 isolated Conv family 的 tile interior optimum 推进到 ResNet-50 `conv3_x` composite bottleneck，只 sweep 3x3 core tile | "
        "`/tmp/codesign-next-handoff.md` Route 4 direction A ResNet-50 bottleneck interior optimum | 当前提交 | "
        f"**{verdict}** | global min=`{analysis.get('global_min_cell')}`；champion_migration=`{analysis.get('champion_migration')}`；interior=`{analysis.get('tile_interior_evidence')}` |\n"
    )
    if "ResNet-50 bottleneck interior probe" not in md:
        marker = "| **Conv3x3-large densification** |"
        idx = md.find(marker)
        if idx >= 0:
            line_end = md.find("\n", idx)
            md = md[: line_end + 1] + row + md[line_end + 1 :]
        else:
            md += "\n" + row
    md_path.write_text(md, encoding="utf-8")

    html_text = html_path.read_text(encoding="utf-8")
    html_text = re.sub(
        r"<p><strong>当前状态：</strong>.*?</p>",
        f"<p><strong>当前状态：</strong>Route 4 已完成 ResNet-50 bottleneck interior-optimum probe。Verdict 为 <code>{html.escape(verdict)}</code>。</p>",
        html_text,
        count=1,
        flags=re.S,
    )
    html_row = (
        "<tr><td><strong>ResNet-50 bottleneck interior probe</strong></td>"
        "<td>将 isolated Conv family 的 tile interior optimum 推进到 ResNet-50 <code>conv3_x</code> composite bottleneck，只 sweep 3x3 core tile</td>"
        "<td><code>/tmp/codesign-next-handoff.md</code></td>"
        "<td>当前提交</td>"
        f"<td class=\"positive\">{html.escape(verdict)}</td>"
        f"<td>global min=<code>{html.escape(str(analysis.get('global_min_cell')))}</code>；champion_migration=<code>{analysis.get('champion_migration')}</code>；interior=<code>{analysis.get('tile_interior_evidence')}</code></td></tr>"
    )
    if "ResNet-50 bottleneck interior probe" not in html_text:
        marker = "<tr><td><strong>Conv3x3-large densification</strong>"
        idx = html_text.find(marker)
        if idx >= 0:
            row_end = html_text.find("</tr>", idx)
            html_text = html_text[: row_end + 5] + html_row + html_text[row_end + 5 :]
        else:
            html_text = html_text.replace("</table>", html_row + "</table>", 1)
    html_path.write_text(html_text, encoding="utf-8")


def update_summary(analysis: dict[str, Any]) -> None:
    path = OUTPUT_ROOT / "pilot_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary["resnet50_bottleneck_interior_probe"] = {
        "matrix": "outputs/route4_fusion_pilot/resnet_bottleneck_matrix.json",
        "analysis": "outputs/route4_fusion_pilot/resnet_bottleneck_interior_analysis.json",
        "verdict": analysis.get("verdict"),
        "global_min_cell": analysis.get("global_min_cell"),
        "champion_migration": analysis.get("champion_migration"),
        "tile_interior_evidence": analysis.get("tile_interior_evidence"),
    }
    write_json(path, summary)


def run_full(args: argparse.Namespace) -> int:
    smoke = gate0_smoke(args.run_timeout_sec)
    if not smoke["passed"]:
        write_blocker(smoke, [])
        analysis = json.loads((OUTPUT_ROOT / "resnet_bottleneck_interior_analysis.json").read_text(encoding="utf-8"))
        render_report(analysis, {"gate0_smoke": smoke, "run_accounting": {}, "timeboxed": False})
        update_tracker("BLOCKED", analysis)
        update_summary(analysis)
        print(json.dumps(analysis, indent=2, sort_keys=True))
        return 2

    sweep_start = time.time()
    base_tiles = load_tiles(include_dense=False)
    base_matrix, base_timeboxed = run_sweep(
        base_tiles,
        args.repeats,
        args.run_timeout_sec,
        args.max_hours,
        root=OUTPUT_ROOT / "resnet_bottleneck_runs",
    )
    analysis = analyze_matrix(base_matrix, base_tiles, timeboxed=base_timeboxed, densification_applied=False)
    matrix = base_matrix
    tiles = base_tiles
    densification_matrix: dict[str, Any] | None = None
    dense_timeboxed = False
    if analysis["global_min_tile_is_extreme_for_hw"]:
        dense_tiles_all = load_tiles(include_dense=True)
        dense_tiles = {name: dense_tiles_all[name] for name in DENSE_TILE_ORDER if name in dense_tiles_all}
        remaining_hours = max(args.max_hours - ((time.time() - sweep_start) / 3600.0), 0.01)
        densification_matrix, dense_timeboxed = run_sweep(
            dense_tiles,
            args.repeats,
            args.run_timeout_sec,
            remaining_hours,
            root=OUTPUT_ROOT / "resnet_bottleneck_densification_runs",
        )
        matrix = merge_matrix(base_matrix, densification_matrix)
        tiles = dense_tiles_all
        analysis = analyze_matrix(matrix, tiles, timeboxed=base_timeboxed or dense_timeboxed, densification_applied=True)
    matrix_payload = {
        "workload": WORKLOAD,
        "gate0_smoke": smoke,
        "matrix": matrix,
        "base_tile_order": BASE_TILE_ORDER,
        "densification_tile_order": DENSE_TILE_ORDER if analysis["densification_applied"] else [],
        "densification_matrix": densification_matrix,
        "repeats_per_cell": args.repeats,
        "fusion_variants": FUSIONS,
        "hw_config_set": "codesign_v1_2x2",
        "pytorchsim_functional_mode": 0,
        "cuda_visible_devices": "1",
        "timeboxed": base_timeboxed or dense_timeboxed,
        "run_accounting": matrix_accounting(matrix),
    }
    write_json(OUTPUT_ROOT / "resnet_bottleneck_matrix.json", matrix_payload)
    write_json(OUTPUT_ROOT / "resnet_bottleneck_interior_analysis.json", analysis)
    render_report(analysis, matrix_payload)
    update_tracker(analysis["verdict"], analysis)
    update_summary(analysis)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--run-timeout-sec", type=int, default=900)
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only:
        smoke = gate0_smoke(args.run_timeout_sec)
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return 0 if smoke["passed"] else 2
    return run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
