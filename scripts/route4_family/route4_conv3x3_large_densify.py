#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
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
BASE_PYTHONPATH = "/home/dyf/PyTorchSim:/home/dyf/PyTorchSim/PyTorchSimDevice"
HW_IDS = ["HW-A", "HW-B", "HW-C", "HW-D"]
FUSIONS = ["none", "all"]
NEW_TILE_SPECS = {
    "tile_A_a": {"TILE_K_H": 3, "TILE_K_W": 3, "TILE_O_H": 28, "TILE_O_W": 2, "TILE_M": 1, "TILE_N": 64, "TILE_K": 16, "regime": "densified_between_A_A2"},
    "tile_A_b": {"TILE_K_H": 3, "TILE_K_W": 3, "TILE_O_H": 28, "TILE_O_W": 3, "TILE_M": 1, "TILE_N": 64, "TILE_K": 16, "regime": "densified_between_A_A2"},
    "tile_A_c": {"TILE_K_H": 3, "TILE_K_W": 3, "TILE_O_H": 28, "TILE_O_W": 4, "TILE_M": 1, "TILE_N": 64, "TILE_K": 16, "regime": "densified_between_A_A2"},
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def enrich_tile(tile: dict[str, Any]) -> dict[str, Any]:
    smallest = 32 * 1024 * 128 // 2
    largest = 128 * 1024 * 128 // 2
    ws = tile_working_set(tile)
    payload = {key: int(tile[key]) for key in ["TILE_K_H", "TILE_K_W", "TILE_O_H", "TILE_O_W", "TILE_M", "TILE_N", "TILE_K"]}
    payload.update(
        {
            "working_set_bytes": ws["total_bytes"],
            "predicted_working_set_bytes": ws["total_bytes"],
            "fit_fraction_smallest_spad": ws["total_bytes"] / smallest,
            "fit_fraction_largest_spad": ws["total_bytes"] / largest,
            "predicted_fit_all_hw": ws["total_bytes"] <= smallest,
            "regime": str(tile.get("regime", "densified")),
            "tile_i_h": ws["tile_i_h"],
            "tile_i_w": ws["tile_i_w"],
            "weight_bytes": ws["weight_bytes"],
            "input_bytes": ws["input_bytes"],
            "output_bytes": ws["output_bytes"],
        }
    )
    return payload


def fit_info(tile: dict[str, Any], hw_id: str) -> dict[str, Any]:
    spad_kb = 128 if hw_id in {"HW-A", "HW-B"} else 32
    usable = spad_kb * 1024 * 128 // 2
    ws = int(tile["working_set_bytes"])
    return {"usable_spad_bytes": usable, "working_set_bytes": ws, "fit_fraction": ws / usable, "fits": ws <= usable}


def existing_tiles() -> dict[str, Any]:
    payload = json.loads((OUTPUT_ROOT / "tile_variants_per_workload.json").read_text(encoding="utf-8"))
    return payload["conv3x3_large"]


def densified_tiles() -> dict[str, Any]:
    old = existing_tiles()
    new = {name: enrich_tile(tile) for name, tile in NEW_TILE_SPECS.items()}
    ordered: dict[str, Any] = {}
    for key in ["tile_A", "tile_A_a", "tile_A_b", "tile_A_c", "tile_A2", "tile_B", "tile_B2", "tile_C", "tile_C2", "tile_C3", "tile_D"]:
        if key in old:
            ordered[key] = old[key]
        else:
            ordered[key] = new[key]
    return ordered


def run_one_repeat(tile_name: str, tile: dict[str, Any], hw_id: str, fusion: str, repeat: int, timeout_sec: int) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / "conv3x3_large_densification_runs" / hw_id / tile_name / fusion / f"run_{repeat:02d}"
    result_path = run_dir / "family_run_result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if int(existing.get("total_cycles", 0) or 0) > 0:
            return {"returncode": 0, "result": existing, "reused": True}
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
        "conv3x3_large",
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
    return {"returncode": proc.returncode, "result": result, "reused": False}


def run_sweep(new_tiles: dict[str, Any], repeats: int, timeout_sec: int, max_minutes: float) -> tuple[dict[str, Any], bool]:
    matrix: dict[str, Any] = {}
    start = time.time()
    timeboxed = False
    for hw_id in HW_IDS:
        matrix[hw_id] = {}
        for tile_name, tile in new_tiles.items():
            matrix[hw_id][tile_name] = {}
            info = fit_info(tile, hw_id)
            for fusion in FUSIONS:
                if time.time() - start > max_minutes * 60:
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
                    outcome = run_one_repeat(tile_name, tile, hw_id, fusion, repeat, timeout_sec)
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
                matrix[hw_id][tile_name][fusion] = {
                    "class": "measured" if len(valid) == repeats and not failures else "blocked",
                    "cycles_in_order": cycles,
                    "median": median,
                    "cycle_delta": delta,
                    "failures": failures,
                    "fit_info": info,
                }
    return matrix, timeboxed


def analyze_conv3x3_large(combined_matrix: dict[str, Any], tiles: dict[str, Any]) -> dict[str, Any]:
    measured: list[dict[str, Any]] = []
    champions: dict[str, Any] = {}
    for hw_id, tile_map in combined_matrix.items():
        best = None
        for tile_name, fusion_map in tile_map.items():
            for fusion, cell in fusion_map.items():
                if cell.get("class") != "measured" or float(cell.get("median", 0) or 0) <= 0:
                    continue
                item = {"hw": hw_id, "tile": tile_name, "fusion": fusion, "median": float(cell["median"])}
                measured.append(item)
                if best is None or item["median"] < best["median"]:
                    best = item
        champions[hw_id] = None if best is None else {"tile": best["tile"], "fusion": best["fusion"], "median": best["median"]}
    global_min = min(measured, key=lambda item: item["median"]) if measured else None
    fit_tiles_by_hw = {
        hw_id: [
            name
            for name, tile in sorted(tiles.items(), key=lambda pair: int(pair[1]["working_set_bytes"]))
            if fit_info(tile, hw_id)["fits"]
        ]
        for hw_id in HW_IDS
    }
    is_extreme = None
    if global_min:
        fit_names = fit_tiles_by_hw[global_min["hw"]]
        is_extreme = global_min["tile"] in {fit_names[0], fit_names[-1]} if fit_names else None
    return {
        "champion_migration": len({(item["tile"], item["fusion"]) for item in champions.values() if item}) > 1,
        "champion_per_hw": champions,
        "distinct_champion_tuples": len({(item["tile"], item["fusion"]) for item in champions.values() if item}),
        "global_min_cell": global_min,
        "global_min_tile_is_extreme_for_hw": is_extreme,
        "tile_interior_evidence": bool(global_min and is_extreme is False),
        "fit_tiles_by_hw": fit_tiles_by_hw,
    }


def recompute_family(conv3x3_large_analysis: dict[str, Any]) -> dict[str, Any]:
    old = json.loads((OUTPUT_ROOT / "workload_family_interior_analysis.json").read_text(encoding="utf-8"))
    per = dict(old["per_workload"])
    per["conv3x3_large"] = {
        **conv3x3_large_analysis,
        "source": "combined_existing_matrix_plus_conv3x3_large_densification",
    }
    all_interior = all(bool(item.get("tile_interior_evidence")) for item in per.values())
    verdict = "WORKLOAD_FAMILY_INTERIOR_OPTIMUM_CONFIRMED" if all_interior else "PARTIAL_WORKLOAD_FAMILY_INTERIOR"
    mapping = {
        f"{workload}/{hw_id}": data.get("champion_per_hw", {}).get(hw_id, {}).get("tile")
        for workload, data in per.items()
        for hw_id in HW_IDS
    }
    per_hw_stability: dict[str, float] = {}
    for hw_id in HW_IDS:
        vals = [per[w].get("champion_per_hw", {}).get(hw_id, {}).get("tile") for w in per]
        counts: dict[str, int] = {}
        for val in vals:
            if val:
                counts[val] = counts.get(val, 0) + 1
        per_hw_stability[hw_id] = (max(counts.values()) / len(vals)) if vals and counts else 0.0
    return {
        "per_workload": per,
        "family_level": {
            **old["family_level"],
            "workload_family_all_have_interior": all_interior,
            "champion_tile_stability_across_workloads": sum(per_hw_stability.values()) / len(per_hw_stability),
            "champion_tile_stability_per_hw": per_hw_stability,
            "family_migration_pattern": {
                "mapping": mapping,
                "distinct_champion_tiles": sorted({tile for tile in mapping.values() if tile}),
                "distinct_champion_tile_count": len({tile for tile in mapping.values() if tile}),
            },
            "verdict": verdict,
            "densification_scope": "conv3x3_large_only_between_tile_A_and_tile_A2",
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{int(value):,}" if value.is_integer() else f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def update_report(new_tiles: dict[str, Any], matrix: dict[str, Any], analysis: dict[str, Any]) -> None:
    md_path = REPORT_DIR / "2026-07-08-workload-family-interior-optimum.md"
    html_path = REPORT_DIR / "2026-07-08-workload-family-interior-optimum.html"
    family = analysis["family_level"]
    conv = analysis["per_workload"]["conv3x3_large"]
    lines = [
        "## 8. Conv3x3-large densification follow-up",
        "",
        "为检查 `conv3x3_large/HW-A` 的 `tile_A` 是否只是粗粒度 tile search 造成的 boundary artifact，本轮只在 `tile_A` 与 `tile_A2` 之间加入 3 个新 tile，并只 sweep `conv3x3_large`。",
        "",
        "| New tile | TILE_O_W | TILE_K | TILE_N | Working set bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, tile in new_tiles.items():
        lines.append(f"| `{name}` | {tile['TILE_O_W']} | {tile['TILE_K']} | {tile['TILE_N']} | {tile['working_set_bytes']:,} |")
    lines.extend(["", "| HW | Champion after densification | Fusion | Median cycles |", "|---|---|---|---:|"])
    for hw_id in HW_IDS:
        champ = conv["champion_per_hw"][hw_id]
        lines.append(f"| {hw_id} | `{champ['tile']}` | `{champ['fusion']}` | {fmt(champ['median'])} |")
    lines.extend(
        [
            "",
            f"- Updated `conv3x3_large.tile_interior_evidence`: `{conv['tile_interior_evidence']}`",
            f"- Updated `conv3x3_large.global_min_cell`: `{json.dumps(conv['global_min_cell'], sort_keys=True)}`",
            f"- Updated family verdict: `{family['verdict']}`",
            "",
        ]
    )
    block = "\n".join(lines).rstrip() + "\n"
    md = md_path.read_text(encoding="utf-8")
    if "\n## 8. Conv3x3-large densification follow-up\n" in md:
        md = md.split("\n## 8. Conv3x3-large densification follow-up\n")[0].rstrip() + "\n\n" + block
    else:
        md = md.rstrip() + "\n\n" + block
    md = md.replace("`verdict`: `PARTIAL_WORKLOAD_FAMILY_INTERIOR`", f"`verdict`: `{family['verdict']}`")
    md_path.write_text(md, encoding="utf-8")

    html_text = html_path.read_text(encoding="utf-8")
    html_block = [
        "<h2>8. Conv3x3-large densification follow-up</h2>",
        "<p>为检查 <code>conv3x3_large/HW-A</code> 的 <code>tile_A</code> 是否只是粗粒度 tile search 造成的 boundary artifact，本轮只在 <code>tile_A</code> 与 <code>tile_A2</code> 之间加入 3 个新 tile，并只 sweep <code>conv3x3_large</code>。</p>",
        "<table><tr><th>New tile</th><th>TILE_O_W</th><th>TILE_K</th><th>TILE_N</th><th>Working set bytes</th></tr>",
    ]
    for name, tile in new_tiles.items():
        html_block.append(f"<tr><td><code>{html.escape(name)}</code></td><td>{tile['TILE_O_W']}</td><td>{tile['TILE_K']}</td><td>{tile['TILE_N']}</td><td>{tile['working_set_bytes']:,}</td></tr>")
    html_block.append("</table>")
    html_block.append("<table><tr><th>HW</th><th>Champion after densification</th><th>Fusion</th><th>Median cycles</th></tr>")
    for hw_id in HW_IDS:
        champ = conv["champion_per_hw"][hw_id]
        html_block.append(f"<tr><td>{hw_id}</td><td><code>{html.escape(str(champ['tile']))}</code></td><td><code>{html.escape(str(champ['fusion']))}</code></td><td>{fmt(champ['median'])}</td></tr>")
    html_block.append("</table>")
    html_block.append(f"<p>Updated <code>conv3x3_large.tile_interior_evidence</code>: <code>{conv['tile_interior_evidence']}</code></p>")
    html_block.append(f"<p>Updated family verdict: <code>{html.escape(family['verdict'])}</code></p>")
    replacement = "\n".join(html_block)
    if "<h2>8. Conv3x3-large densification follow-up</h2>" in html_text:
        html_text = html_text.split("<h2>8. Conv3x3-large densification follow-up</h2>")[0].rstrip() + "\n" + replacement + "\n</body></html>\n"
    else:
        html_text = html_text.replace("</body></html>", replacement + "\n</body></html>")
    html_text = html_text.replace("PARTIAL_WORKLOAD_FAMILY_INTERIOR", family["verdict"])
    html_path.write_text(html_text, encoding="utf-8")


def update_tracker(analysis: dict[str, Any]) -> None:
    verdict = analysis["family_level"]["verdict"]
    conv = analysis["per_workload"]["conv3x3_large"]
    md_path = REPORT_DIR / "2026-07-03-co-design-progress-tracker.md"
    html_path = REPORT_DIR / "2026-07-03-co-design-progress-tracker.html"
    md = md_path.read_text(encoding="utf-8")
    md = __import__("re").sub(
        r"当前状态：Route 4 .*?\n",
        f"当前状态：Route 4 已完成 `conv3x3_large` densification follow-up。Family verdict 为 `{verdict}`；`conv3x3_large` 的 global min 为 `{conv['global_min_cell']['hw']}/{conv['global_min_cell']['tile']}/{conv['global_min_cell']['fusion']}`。\n",
        md,
        count=1,
    )
    row = (
        "| **Conv3x3-large densification** | 只在 `conv3x3_large` 的 `tile_A` 与 `tile_A2` 之间加入 3 个 tile，检查 HW-A fit-extreme optimum 是否可被细化推翻 | "
        "`/tmp/codesign-next-handoff.md` family densification for conv3x3_large | 当前提交 | "
        f"**{verdict}** | conv3x3_large global min=`{conv['global_min_cell']['hw']}/{conv['global_min_cell']['tile']}/{conv['global_min_cell']['fusion']}`；interior=`{conv['tile_interior_evidence']}` |\n"
    )
    if "Conv3x3-large densification" not in md:
        md = md.replace(
            "| **Workload family interior probe** | 在 inherited Conv3x3 v2 基础上扩展到 Conv3x3-large 与 Conv1x1，检查 tile interior optimum 是否跨 workload family 保持 | `/tmp/codesign-next-handoff.md` workload family interior-optimum generalization | 当前提交 | **PARTIAL_WORKLOAD_FAMILY_INTERIOR** | tested workloads=`conv3x3_probe/conv3x3_large/conv1x1`；all_have_interior=`False` |\n",
            "| **Workload family interior probe** | 在 inherited Conv3x3 v2 基础上扩展到 Conv3x3-large 与 Conv1x1，检查 tile interior optimum 是否跨 workload family 保持 | `/tmp/codesign-next-handoff.md` workload family interior-optimum generalization | `51b5de7` | **PARTIAL_WORKLOAD_FAMILY_INTERIOR** | tested workloads=`conv3x3_probe/conv3x3_large/conv1x1`；all_have_interior=`False` |\n" + row,
        )
    md_path.write_text(md, encoding="utf-8")

    html_text = html_path.read_text(encoding="utf-8")
    html_text = __import__("re").sub(
        r"<p><strong>当前状态：</strong>.*?</p>",
        f"<p><strong>当前状态：</strong>Route 4 已完成 <code>conv3x3_large</code> densification follow-up。Family verdict 为 <code>{html.escape(verdict)}</code>；<code>conv3x3_large</code> 的 global min 为 <code>{html.escape(conv['global_min_cell']['hw'] + '/' + conv['global_min_cell']['tile'] + '/' + conv['global_min_cell']['fusion'])}</code>。</p>",
        html_text,
        count=1,
        flags=__import__("re").S,
    )
    html_row = (
        f'<tr><td><strong>Conv3x3-large densification</strong></td><td>只在 <code>conv3x3_large</code> 的 <code>tile_A</code> 与 <code>tile_A2</code> 之间加入 3 个 tile，检查 HW-A fit-extreme optimum 是否可被细化推翻</td>'
        f'<td><code>/tmp/codesign-next-handoff.md</code></td><td>当前提交</td><td class="positive">{html.escape(verdict)}</td>'
        f'<td>conv3x3_large global min=<code>{html.escape(conv["global_min_cell"]["hw"] + "/" + conv["global_min_cell"]["tile"] + "/" + conv["global_min_cell"]["fusion"])}</code>；interior=<code>{conv["tile_interior_evidence"]}</code></td></tr>'
    )
    if "Conv3x3-large densification" not in html_text:
        html_text = html_text.replace(
            '<td>当前提交</td><td class="positive">PARTIAL_WORKLOAD_FAMILY_INTERIOR</td><td>tested workloads=<code>conv3x3_probe/conv3x3_large/conv1x1</code>；all_have_interior=<code>False</code></td></tr>',
            '<td><code>51b5de7</code></td><td class="positive">PARTIAL_WORKLOAD_FAMILY_INTERIOR</td><td>tested workloads=<code>conv3x3_probe/conv3x3_large/conv1x1</code>；all_have_interior=<code>False</code></td></tr>' + html_row,
        )
    html_path.write_text(html_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--run-timeout-sec", type=int, default=900)
    parser.add_argument("--max-minutes", type=float, default=90.0)
    args = parser.parse_args()

    all_tiles = densified_tiles()
    new_tiles = {name: all_tiles[name] for name in NEW_TILE_SPECS}
    write_json(OUTPUT_ROOT / "tile_variants_conv3x3_large_densified.json", all_tiles)
    new_matrix, timeboxed = run_sweep(new_tiles, args.repeats, args.run_timeout_sec, args.max_minutes)
    old_matrix_all = json.loads((OUTPUT_ROOT / "tile_fusion_hw_matrix_per_workload.json").read_text(encoding="utf-8"))
    old_matrix = old_matrix_all["matrix"]["conv3x3_large"]
    combined = {hw_id: {tile: dict(fmap) for tile, fmap in old_matrix[hw_id].items()} for hw_id in HW_IDS}
    for hw_id, tile_map in new_matrix.items():
        for tile_name, fusion_map in tile_map.items():
            combined[hw_id][tile_name] = fusion_map
    matrix_payload = {
        "workload": "conv3x3_large",
        "new_tiles": new_tiles,
        "new_tile_matrix": new_matrix,
        "combined_conv3x3_large_matrix": combined,
        "repeats_per_cell": args.repeats,
        "fusion_variants": FUSIONS,
        "hw_config_set": "codesign_v1_2x2",
        "pytorchsim_functional_mode": 0,
        "cuda_visible_devices": "1",
        "timeboxed": timeboxed,
    }
    write_json(OUTPUT_ROOT / "conv3x3_large_densification_matrix.json", matrix_payload)
    conv_analysis = analyze_conv3x3_large(combined, all_tiles)
    family = recompute_family(conv_analysis)
    family["densification"] = {
        "workload": "conv3x3_large",
        "new_tiles": list(new_tiles),
        "timeboxed": timeboxed,
        "previous_global_min_cell": json.loads((OUTPUT_ROOT / "workload_family_interior_analysis.json").read_text(encoding="utf-8"))["per_workload"]["conv3x3_large"]["global_min_cell"],
        "updated_global_min_cell": conv_analysis["global_min_cell"],
        "upgraded_family_verdict": family["family_level"]["verdict"] == "WORKLOAD_FAMILY_INTERIOR_OPTIMUM_CONFIRMED",
    }
    write_json(OUTPUT_ROOT / "workload_family_interior_analysis_v2.json", family)
    summary_path = OUTPUT_ROOT / "pilot_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["conv3x3_large_densification"] = {
        "tile_variants": "outputs/route4_fusion_pilot/tile_variants_conv3x3_large_densified.json",
        "matrix": "outputs/route4_fusion_pilot/conv3x3_large_densification_matrix.json",
        "analysis": "outputs/route4_fusion_pilot/workload_family_interior_analysis_v2.json",
        "family_verdict": family["family_level"]["verdict"],
        "updated_conv3x3_large_global_min_cell": conv_analysis["global_min_cell"],
    }
    write_json(summary_path, summary)
    update_report(new_tiles, new_matrix, family)
    update_tracker(family)
    print(json.dumps(family["densification"], indent=2, sort_keys=True))
    print(json.dumps(family["family_level"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
