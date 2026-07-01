#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("outputs/mapping_dse_minimal")
DEFAULT_RUN_TAG = "gpt2_block_prefill_s128"
DEFAULT_TORCHSIM_LLVM_PATH = "/home/dyf/src/psal-llvm-project-v1.0.8/build/bin"
DEFAULT_GEM5_PATH = "/home/dyf/src/psal-gem5/build/RISCV/gem5.opt"
PYTORCHSIM_RISCV_GCC_COMPAT_DIR = Path("/home/dyf/opt/pytorchsim-riscv-gcc-compat/bin")
PYTORCHSIM_LIB_DIRS = [
    Path("/home/dyf/miniconda3/envs/pytorchsim-build/lib"),
    Path("/home/dyf/miniconda3/envs/pytorchsim-build/x86_64-conda-linux-gnu/lib64"),
    Path("/home/dyf/.venv/lib64/python3.11/site-packages/torch/lib"),
]
COUNTER_FIELDS = {
    "total_cycles": 0,
    "sa_utilization": 0.0,
    "sa_active_cycles": 0,
    "sa_idle_cycles": 0,
    "dma_active_cycles": 0,
    "dma_idle_cycles": 0,
    "vec_utilization": 0.0,
    "vec_active_cycles": 0,
    "vec_idle_cycles": 0,
    "dram_bw_gbps": 0.0,
    "dram_reads": 0,
    "dram_writes": 0,
    "inst_count_by_op": {},
}


@dataclass
class MappingResult:
    mapping_id: str
    mapping_dir: Path
    mapping_config: dict[str, Any]
    status: str
    message: str
    counters: dict[str, Any] | None = None
    features: dict[str, Any] | None = None


class MappingSchemaError(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def prepend_env_path(env: dict[str, str], key: str, paths: list[Path | str]) -> None:
    existing = env.get(key, "")
    values = []
    for path in paths:
        if isinstance(path, Path):
            if path.exists():
                values.append(str(path))
        else:
            values.append(str(path))
    if existing:
        values.append(existing)
    env[key] = os.pathsep.join(dict.fromkeys(value for value in values if value))


def pytorchsim_runtime_env(base: dict[str, str] | None = None, *, output_dir: Path | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    prepend_env_path(env, "LD_LIBRARY_PATH", PYTORCHSIM_LIB_DIRS)
    prepend_env_path(env, "PYTHONPATH", [repo_root(), repo_root() / "PyTorchSimDevice"])
    prepend_env_path(env, "PATH", [PYTORCHSIM_RISCV_GCC_COMPAT_DIR])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["TORCHSIM_DIR"] = str(repo_root())
    env.setdefault("TORCHSIM_LLVM_PATH", DEFAULT_TORCHSIM_LLVM_PATH)
    env.setdefault("GEM5_PATH", DEFAULT_GEM5_PATH)
    if output_dir is not None:
        env["TORCHSIM_DUMP_PATH"] = str((output_dir / "torchsim_outputs").resolve())
        env["TORCHSIM_LOG_PATH"] = str((output_dir / "torchsim_logs").resolve())
    return env


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _check_import(module_name: str) -> tuple[bool, str]:
    code = f"import {module_name}; print('ok')"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root(),
        env=pytorchsim_runtime_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout).strip()


def environment_preflight(require_transformers: bool = True, require_npu: bool = True) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    modules = ["torch"]
    if require_transformers:
        modules.append("transformers")
    if require_npu:
        modules.append("torch_openreg")
    for module_name in modules:
        ok, error = _check_import(module_name)
        checks[module_name] = {"ok": ok, "error": error}
        if not ok:
            failures[module_name] = error
    return {"ok": not failures, "checks": checks, "failures": failures}


def parse_togsim_log(log_text: str) -> dict[str, Any]:
    counters = dict(COUNTER_FIELDS)
    counters["inst_count_by_op"] = {}

    total = re.search(r"Total execution cycles:\s*(\d+)", log_text)
    if total:
        counters["total_cycles"] = int(total.group(1))

    sa = re.search(
        r"Systolic array \[\d+\] utilization\(%\):\s*([0-9.]+),\s*active_cycles:\s*(\d+),\s*idle_cycles:\s*(\d+)",
        log_text,
    )
    if sa:
        counters["sa_utilization"] = float(sa.group(1))
        counters["sa_active_cycles"] = int(sa.group(2))
        counters["sa_idle_cycles"] = int(sa.group(3))

    dma = re.search(
        r"DMA active_cycles:\s*(\d+),\s*DMA idle_cycles:\s*(\d+),\s*DRAM BW:\s*([0-9.]+)\s*GB/s",
        log_text,
    )
    if dma:
        counters["dma_active_cycles"] = int(dma.group(1))
        counters["dma_idle_cycles"] = int(dma.group(2))
        counters["dram_bw_gbps"] = float(dma.group(3))

    vector = re.search(
        r"Vector unit utilization\(%\):\s*([0-9.]+)(?:,\s*active cycle:\s*(\d+),\s*idle_cycle:\s*(\d+))?",
        log_text,
    )
    if vector:
        counters["vec_utilization"] = float(vector.group(1))
        if vector.group(2):
            counters["vec_active_cycles"] = int(vector.group(2))
        if vector.group(3):
            counters["vec_idle_cycles"] = int(vector.group(3))

    dram = re.search(
        r"channels\s+0\.\.\d+\s+combined\s*\|\s*([0-9.]+)\s*GB/s aggregate.*\|\s*(\d+)\s*reads,\s*(\d+)\s*writes",
        log_text,
    )
    if dram:
        counters["dram_bw_gbps"] = float(dram.group(1))
        counters["dram_reads"] = int(dram.group(2))
        counters["dram_writes"] = int(dram.group(3))

    for op, count, detail in re.findall(
        r"Core \[\d+\] :\s*([A-Z]+)\s+inst_count:\s*(\d+)(?:\s*\(([^)]*)\))?",
        log_text,
    ):
        counters["inst_count_by_op"][op] = int(count)
        if detail:
            for item in detail.split(","):
                if ":" not in item:
                    continue
                key, value = item.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value.isdigit():
                    counters["inst_count_by_op"][key] = int(value)

    return counters


def aggregate_counters(counter_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = dict(COUNTER_FIELDS)
    aggregate["inst_count_by_op"] = {}
    if not counter_rows:
        return aggregate

    sum_fields = [
        "total_cycles",
        "sa_active_cycles",
        "sa_idle_cycles",
        "dma_active_cycles",
        "dma_idle_cycles",
        "vec_active_cycles",
        "vec_idle_cycles",
        "dram_reads",
        "dram_writes",
    ]
    for field in sum_fields:
        aggregate[field] = sum(int(row.get(field, 0) or 0) for row in counter_rows)

    inst_counter: Counter[str] = Counter()
    for row in counter_rows:
        inst_counter.update({key: int(value) for key, value in row.get("inst_count_by_op", {}).items()})
    aggregate["inst_count_by_op"] = dict(sorted(inst_counter.items()))

    sa_total = aggregate["sa_active_cycles"] + aggregate["sa_idle_cycles"]
    if sa_total:
        aggregate["sa_utilization"] = aggregate["sa_active_cycles"] * 100.0 / sa_total

    vec_total = aggregate["vec_active_cycles"] + aggregate["vec_idle_cycles"]
    if vec_total:
        aggregate["vec_utilization"] = aggregate["vec_active_cycles"] * 100.0 / vec_total
    else:
        cycle_weight = sum(int(row.get("total_cycles", 0) or 0) for row in counter_rows)
        if cycle_weight:
            aggregate["vec_utilization"] = sum(
                float(row.get("vec_utilization", 0.0) or 0.0) * int(row.get("total_cycles", 0) or 0)
                for row in counter_rows
            ) / cycle_weight

    bw_weight = sum(int(row.get("total_cycles", 0) or 0) for row in counter_rows)
    if bw_weight:
        aggregate["dram_bw_gbps"] = sum(
            float(row.get("dram_bw_gbps", 0.0) or 0.0) * int(row.get("total_cycles", 0) or 0)
            for row in counter_rows
        ) / bw_weight

    aggregate["kernel_log_count"] = len(counter_rows)
    aggregate["per_kernel_total_cycles"] = [int(row.get("total_cycles", 0) or 0) for row in counter_rows]
    return aggregate


def load_raw_tog_graph(raw_tog_path: Path) -> dict[int, dict[str, Any]]:
    text = raw_tog_path.read_text(encoding="utf-8")
    match = re.search(r"graph\s*=\s*(\{.*\})", text, flags=re.S)
    if not match:
        raise ValueError(f"No graph dict found in {raw_tog_path}")
    graph = ast.literal_eval(match.group(1))
    return {int(key): value for key, value in graph.items()}


def loop_depths(graph: dict[int, dict[str, Any]]) -> dict[int, int]:
    root_ids = [node_id for node_id, node in graph.items() if not node.get("parents")]
    depths: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque((node_id, 0) for node_id in root_ids)
    while queue:
        node_id, depth = queue.popleft()
        node = graph[node_id]
        node_depth = depth + 1 if node.get("node_name") == "loopNode" else depth
        prev = depths.get(node_id)
        if prev is not None and prev >= node_depth:
            continue
        depths[node_id] = node_depth
        next_depth = node_depth
        for child in node.get("children", []):
            if int(child) in graph:
                queue.append((int(child), next_depth))
    return depths


def regular_stride_score(strides: list[list[int]]) -> float:
    if not strides:
        return 0.0
    regular = 0
    for stride in strides:
        if not stride:
            continue
        non_zero = [value for value in stride if value != 0]
        if non_zero and all(value > 0 for value in non_zero):
            regular += 1
    return regular / len(strides)


def extract_struct_features(raw_tog_path: Path, mapping_config: dict[str, Any] | None = None) -> dict[str, Any]:
    graph = load_raw_tog_graph(raw_tog_path)
    node_hist = Counter(node.get("node_name", "unknown") for node in graph.values())
    edge_count = sum(len(node.get("children", [])) for node in graph.values())
    depths = loop_depths(graph)
    loop_node_depths = [
        depths.get(node_id, 0)
        for node_id, node in graph.items()
        if node.get("node_name") == "loopNode"
    ]
    dma_nodes = [
        node
        for node in graph.values()
        if node.get("node_name") == "DMANode" or int(node.get("node_type", -1)) == 3
    ]
    compute_nodes = [
        node
        for node in graph.values()
        if node.get("node_name") == "ComputeNode" or int(node.get("node_type", -1)) == 1
    ]
    tile_sizes = [node.get("tile_size") for node in dma_nodes if isinstance(node.get("tile_size"), list)]
    tile_shape = {"M": 0, "N": 0, "K": 0}
    if mapping_config:
        tile_shape.update(
            {
                "M": int(mapping_config.get("TILE_M", mapping_config.get("tile_m", 0)) or 0),
                "N": int(mapping_config.get("TILE_N", mapping_config.get("tile_n", 0)) or 0),
                "K": int(mapping_config.get("TILE_K", mapping_config.get("tile_k", 0)) or 0),
            }
        )
    elif tile_sizes:
        first = tile_sizes[0]
        if len(first) >= 2:
            tile_shape["M"] = int(first[-2])
            tile_shape["N"] = int(first[-1])
        if len(tile_sizes) > 1 and len(tile_sizes[1]) >= 2:
            tile_shape["K"] = int(tile_sizes[1][-2])

    stride_score = regular_stride_score(
        [
            [int(value) for value in node.get("tile_stride", [])]
            for node in dma_nodes
            if isinstance(node.get("tile_stride"), list)
        ]
    )
    wait_count = node_hist.get("DMAWaitNode", 0)
    compute_count = len(compute_nodes)
    dma_count = len(dma_nodes)
    loop_count = node_hist.get("loopNode", 0)
    struct_cost_proxy = (
        compute_count * max(1, tile_shape["M"]) * max(1, tile_shape["N"])
        + dma_count * 32
        + wait_count * 16
        + loop_count * 4
    )
    cluster_key = f"m{tile_shape['M']}_n{tile_shape['N']}_k{tile_shape['K']}_d{max(loop_node_depths, default=0)}"

    return {
        "node_count": len(graph),
        "edge_count": edge_count,
        "op_type_histogram": dict(node_hist),
        "loop_nest_depth_stats": {
            "max_depth": max(loop_node_depths, default=0),
            "avg_depth": sum(loop_node_depths) / len(loop_node_depths) if loop_node_depths else 0.0,
            "loop_count": len(loop_node_depths),
        },
        "dma_node_count": dma_count,
        "dma_stride_regularity": stride_score,
        "tile_shape": tile_shape,
        "fusion_depth": max(0, compute_count - 1),
        "compute_dma_overlap_proxy": compute_count / max(1, dma_count + wait_count),
        "struct_cost_proxy": struct_cost_proxy,
        "cluster_key": cluster_key,
    }


def aggregate_struct_features(feature_rows: list[dict[str, Any]], mapping_config: dict[str, Any]) -> dict[str, Any]:
    if not feature_rows:
        return extract_struct_features_placeholder(mapping_config)

    hist: Counter[str] = Counter()
    for row in feature_rows:
        hist.update({key: int(value) for key, value in row.get("op_type_histogram", {}).items()})

    total_loops = sum(int(row.get("loop_nest_depth_stats", {}).get("loop_count", 0) or 0) for row in feature_rows)
    weighted_depth_sum = sum(
        float(row.get("loop_nest_depth_stats", {}).get("avg_depth", 0.0) or 0.0)
        * int(row.get("loop_nest_depth_stats", {}).get("loop_count", 0) or 0)
        for row in feature_rows
    )
    dma_count = sum(int(row.get("dma_node_count", 0) or 0) for row in feature_rows)
    stride_score = (
        sum(float(row.get("dma_stride_regularity", 0.0) or 0.0) * int(row.get("dma_node_count", 0) or 0) for row in feature_rows)
        / dma_count
        if dma_count
        else 0.0
    )
    overlap_den = sum(int(row.get("dma_node_count", 0) or 0) for row in feature_rows)
    compute_count = sum(int(row.get("op_type_histogram", {}).get("ComputeNode", 0) or 0) for row in feature_rows)
    tile_shape = {
        "M": int(mapping_config.get("TILE_M", 0) or 0),
        "N": int(mapping_config.get("TILE_N", 0) or 0),
        "K": int(mapping_config.get("TILE_K", 0) or 0),
    }
    max_depth = max(int(row.get("loop_nest_depth_stats", {}).get("max_depth", 0) or 0) for row in feature_rows)
    tile_hist = Counter(str(row.get("tile_shape", {})) for row in feature_rows)

    return {
        "node_count": sum(int(row.get("node_count", 0) or 0) for row in feature_rows),
        "edge_count": sum(int(row.get("edge_count", 0) or 0) for row in feature_rows),
        "op_type_histogram": dict(sorted(hist.items())),
        "loop_nest_depth_stats": {
            "max_depth": max_depth,
            "avg_depth": weighted_depth_sum / total_loops if total_loops else 0.0,
            "loop_count": total_loops,
        },
        "dma_node_count": dma_count,
        "dma_stride_regularity": stride_score,
        "tile_shape": tile_shape,
        "fusion_depth": sum(int(row.get("fusion_depth", 0) or 0) for row in feature_rows),
        "compute_dma_overlap_proxy": compute_count / max(1, overlap_den),
        "struct_cost_proxy": sum(float(row.get("struct_cost_proxy", 0.0) or 0.0) for row in feature_rows),
        "cluster_key": f"m{tile_shape['M']}_n{tile_shape['N']}_k{tile_shape['K']}_d{max_depth}_kernels{len(feature_rows)}",
        "raw_tog_count": len(feature_rows),
        "per_tog_node_count": [int(row.get("node_count", 0) or 0) for row in feature_rows],
        "tile_shape_histogram": dict(sorted(tile_hist.items())),
    }


def extract_struct_features_placeholder(mapping_config: dict[str, Any]) -> dict[str, Any]:
    tile_shape = {
        "M": int(mapping_config.get("TILE_M", 0) or 0),
        "N": int(mapping_config.get("TILE_N", 0) or 0),
        "K": int(mapping_config.get("TILE_K", 0) or 0),
    }
    return {
        "node_count": 0,
        "edge_count": 0,
        "op_type_histogram": {},
        "loop_nest_depth_stats": {"max_depth": 0, "avg_depth": 0.0, "loop_count": 0},
        "dma_node_count": 0,
        "dma_stride_regularity": 0.0,
        "tile_shape": tile_shape,
        "fusion_depth": 0,
        "compute_dma_overlap_proxy": 0.0,
        "struct_cost_proxy": 0.0,
        "cluster_key": f"m{tile_shape['M']}_n{tile_shape['N']}_k{tile_shape['K']}_d0_kernels0",
        "raw_tog_count": 0,
        "per_tog_node_count": [],
        "tile_shape_histogram": {},
    }


def rank(values: list[float]) -> list[float]:
    ordered = sorted((value, idx) for idx, value in enumerate(values))
    result = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for _, idx in ordered[i : j + 1]:
            result[idx] = avg_rank
        i = j + 1
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in rx))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ry))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def analyze_gate1(counter_rows: list[dict[str, Any]], threshold: float = 0.20) -> dict[str, Any]:
    cycles = [int(row.get("total_cycles", 0)) for row in counter_rows if int(row.get("total_cycles", 0)) > 0]
    if len(cycles) < 2:
        return {
            "cycle_spread": 0.0,
            "counter_variance": {},
            "outlier_supported": False,
            "passed": False,
            "reason": "fewer than two measured cycle rows",
        }
    min_cycle = min(cycles)
    max_cycle = max(cycles)
    spread = (max_cycle - min_cycle) / min_cycle
    sorted_cycles = sorted(cycles)
    second_max_spread = (sorted_cycles[-2] - min_cycle) / min_cycle if len(sorted_cycles) > 2 else spread
    outlier_supported = second_max_spread >= threshold * 0.5
    variance: dict[str, float] = {}
    for field in ("total_cycles", "sa_utilization", "dma_active_cycles", "vec_utilization", "dram_bw_gbps"):
        vals = [float(row.get(field, 0.0)) for row in counter_rows]
        mean = sum(vals) / len(vals) if vals else 0.0
        variance[field] = sum((val - mean) ** 2 for val in vals) / len(vals) if vals else 0.0
    return {
        "cycle_spread": spread,
        "counter_variance": variance,
        "outlier_supported": outlier_supported,
        "passed": bool(spread >= threshold and outlier_supported),
        "reason": "",
    }


def analyze_gate2(struct_rows: list[dict[str, Any]], counter_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cycle_by_id = {
        row["mapping_id"]: int(row.get("total_cycles", 0))
        for row in counter_rows
        if int(row.get("total_cycles", 0)) > 0
    }
    joined = [
        (row["mapping_id"], float(row.get("struct_cost_proxy", 0.0)), cycle_by_id[row["mapping_id"]], row)
        for row in struct_rows
        if row.get("mapping_id") in cycle_by_id
    ]
    if len(joined) < 2:
        return {
            "b_proxy_spearman": 0.0,
            "b_repr_sample_fraction": 0.0,
            "b_repr_top1_hit": False,
            "b_repr_top3_hit": False,
            "passed": False,
            "reason": "fewer than two joined rows",
        }
    proxy_costs = [item[1] for item in joined]
    cycles = [item[2] for item in joined]
    corr = spearman(proxy_costs, cycles)

    clusters: dict[str, list[tuple[str, float, int, dict[str, Any]]]] = defaultdict(list)
    for item in joined:
        clusters[str(item[3].get("cluster_key", item[0]))].append(item)
    representatives = []
    for items in clusters.values():
        representatives.append(min(items, key=lambda item: item[1]))
    sample_limit = max(1, math.floor(len(joined) * 0.30))
    representatives = sorted(representatives, key=lambda item: item[1])[:sample_limit]
    best_rep = min(representatives, key=lambda item: item[2])
    true_best = min(joined, key=lambda item: item[2])
    predicted_top3 = {item[0] for item in representatives[:3]}
    top1_hit = best_rep[0] == true_best[0]
    top3_hit = true_best[0] in predicted_top3
    passed = bool(corr >= 0.6 or top1_hit or top3_hit)
    return {
        "b_proxy_spearman": corr,
        "b_repr_sample_fraction": len(representatives) / len(joined),
        "b_repr_top1_hit": top1_hit,
        "b_repr_top3_hit": top3_hit,
        "representative_ids": [item[0] for item in representatives],
        "proxy_ranking": [item[0] for item in sorted(joined, key=lambda item: item[1])],
        "truth_ranking": [item[0] for item in sorted(joined, key=lambda item: item[2])],
        "passed": passed,
        "true_best_id": true_best[0],
        "predicted_top3_ids": [item[0] for item in representatives[:3]],
        "reason": "",
    }


def coverage_from_features(struct_rows: list[dict[str, Any]]) -> dict[str, Any]:
    op_types = sorted(
        {
            op
            for row in struct_rows
            for op, count in row.get("op_type_histogram", {}).items()
            if count
        }
    )
    tile_regimes = sorted({row.get("cluster_key", "") for row in struct_rows if row.get("cluster_key")})
    return {"op_types": op_types, "tile_regimes": tile_regimes}


def write_terminal_outputs(
    output_dir: Path,
    metadata: dict[str, Any],
    mapping_results: list[MappingResult],
    end_state: str,
    reason: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counter_rows = [
        {"mapping_id": result.mapping_id, **(result.counters or {})}
        for result in mapping_results
        if result.counters is not None
    ]
    struct_rows = [
        {"mapping_id": result.mapping_id, **(result.features or {})}
        for result in mapping_results
        if result.features is not None
    ]
    gate1 = analyze_gate1(counter_rows) if counter_rows else {
        "cycle_spread": 0.0,
        "counter_variance": {},
        "outlier_supported": False,
        "passed": False,
        "reason": reason or "no measured counters",
    }
    gate2 = analyze_gate2(struct_rows, counter_rows) if struct_rows and counter_rows else {
        "b_proxy_spearman": 0.0,
        "b_repr_sample_fraction": 0.0,
        "b_repr_top1_hit": False,
        "b_repr_top3_hit": False,
        "passed": False,
        "reason": reason or "no joined feature/counter rows",
    }
    coverage = coverage_from_features(struct_rows)
    if end_state == "AUTO":
        if not mapping_results:
            end_state = "BLOCKED"
        elif len(counter_rows) < metadata.get("args", {}).get("num_mappings", 0):
            end_state = "PARTIAL"
        elif not gate1["passed"]:
            end_state = "NEGATIVE"
        elif gate2["passed"]:
            end_state = "COMPLETE"
        else:
            end_state = "NEGATIVE"

    write_json(output_dir / "00_metadata.json", metadata)
    write_json(output_dir / "counters_table.json", {"data_label": "measured", "rows": counter_rows})
    write_json(output_dir / "struct_features.json", {"data_label": "measured", "rows": struct_rows})
    write_json(output_dir / "gate1_differentiation.json", gate1)
    write_json(output_dir / "gate2_selection.json", gate2)
    selector_artifacts = {
        "artifact_type": "npu_mapping_struct_selector_artifacts",
        "representation_mode": "tog_struct_features",
        "selector_scope": "gcl_m0_style_offline_representation_selector",
        "rows": struct_rows,
        "proxy_ranking": gate2.get("proxy_ranking", []),
        "truth_ranking": gate2.get("truth_ranking", []),
        "representative_ids": gate2.get("representative_ids", []),
        "structural_evaluation_artifacts": {
            "row_count": len(struct_rows),
            "counter_row_count": len(counter_rows),
            "coverage": coverage,
            "b_proxy_spearman": gate2["b_proxy_spearman"],
            "b_repr_sample_fraction": gate2.get("b_repr_sample_fraction", 0.0),
            "b_repr_top1_hit": gate2["b_repr_top1_hit"],
            "b_repr_top3_hit": gate2.get("b_repr_top3_hit", False),
        },
    }
    write_json(output_dir / "selector_artifacts.json", selector_artifacts)
    write_json(
        output_dir / "verdict.json",
        {
            "gate1_differentiation": {
                "cycle_spread": gate1["cycle_spread"],
                "passed": gate1["passed"],
            },
            "gate2_selection": {
                "b_proxy_spearman": gate2["b_proxy_spearman"],
                "b_repr_top1_hit": gate2["b_repr_top1_hit"],
                "passed": gate2["passed"],
            },
            "coverage": coverage,
            "end_state": end_state,
            "claim_bearing": True,
            "data_label": "measured",
            "reason": reason,
        },
    )
    report = [
        "# NPU 映射 DSE 最小闭环实验报告",
        "",
        f"- 结束状态: `{end_state}`",
        "- 数据标签: `measured`",
        f"- 实测候选数: {len(counter_rows)}",
        f"- 结构特征行数: {len(struct_rows)}",
        f"- Gate-1 映射分化: cycle_spread = {gate1['cycle_spread']:.6f}, passed = {gate1['passed']}",
        f"- Gate-2 便宜表示选择: Spearman = {gate2['b_proxy_spearman']:.6f}, top1 = {gate2['b_repr_top1_hit']}, top3 = {gate2.get('b_repr_top3_hit', False)}, passed = {gate2['passed']}",
        "",
        "## 裁决",
        "",
        "```text",
        (
            "Q1 通过: tiling 映射会造成显著性能分化。"
            if gate1["passed"]
            else "Q1 不通过: 当前 workload/tiling 维度未观察到显著分化。"
        ),
        (
            "Q2 通过: 当前便宜 TOG 结构表示能够给出有效选择信号。"
            if gate2["passed"]
            else "Q2 不通过: 当前手工 TOG 结构 proxy/代表采样不足以选出真实最优映射。"
        ),
        "```",
        "",
        "## 说明",
        "",
        "```text",
        reason or "n/a",
        "```",
        "",
        "## 产物",
        "",
        "- `00_metadata.json`",
        "- `mappings/<idx>/mapping_config.json`",
        "- `mappings/<idx>/tog.py` / `tog.onnx` / `togsim_log.txt`",
        "- `counters_table.json`",
        "- `struct_features.json`",
        "- `gate1_differentiation.json`",
        "- `gate2_selection.json`",
        "- `selector_artifacts.json`",
        "- `verdict.json`",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def default_mapping_candidates(num_mappings: int) -> list[dict[str, Any]]:
    base = [
        (32, 64, 32),
        (64, 64, 32),
        (64, 128, 32),
        (128, 64, 32),
        (128, 128, 32),
        (64, 64, 64),
        (128, 64, 64),
        (64, 128, 64),
        (16, 16, 16),
        (256, 256, 128),
        (64, 256, 64),
        (128, 256, 64),
    ]
    return [
        {"mapping_id": f"{idx:03d}", "TILE_M": m, "TILE_N": n, "TILE_K": k}
        for idx, (m, n, k) in enumerate(base[:num_mappings])
    ]


def normalize_mapping_candidate(row: dict[str, Any], idx: int) -> dict[str, Any]:
    required = ["TILE_M", "TILE_N", "TILE_K"]
    missing = [field for field in required if field not in row]
    if missing:
        raise MappingSchemaError(f"mapping[{idx}] missing required fields: {', '.join(missing)}")
    mapping_id = str(row.get("mapping_id", f"{idx:03d}"))
    return {
        "mapping_id": mapping_id,
        "TILE_M": int(row["TILE_M"]),
        "TILE_N": int(row["TILE_N"]),
        "TILE_K": int(row["TILE_K"]),
    }


def load_external_mapping_candidates(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "mappings" in data:
        data = data["mappings"]
    if not isinstance(data, list):
        raise MappingSchemaError("--external-mappings-json must contain a list or {'mappings': [...]}")
    return [normalize_mapping_candidate(row, idx) for idx, row in enumerate(data)]


def mapping_candidates_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.external_mappings_json:
        candidates = load_external_mapping_candidates(args.external_mappings_json)
        return candidates[: args.num_mappings]
    return default_mapping_candidates(args.num_mappings)


def write_external_mapping_file(path: Path, mapping_config: dict[str, Any], seq: int) -> None:
    tile = {
        "TILE_M": int(mapping_config["TILE_M"]),
        "TILE_N": int(mapping_config["TILE_N"]),
        "TILE_K": int(mapping_config["TILE_K"]),
    }
    shapes = {
        # GPT-2 block, batch=1, hidden=768, intermediate=3072.
        f"{seq}_2304_768": tile,
        f"{seq}_768_768": tile,
        f"{seq}_3072_768": tile,
        f"{seq}_768_3072": tile,
        f"{seq}_128_64": tile,
        f"{seq}_64_{seq}": tile,
    }
    write_json(path, shapes)


def write_mapping_config(base_config: Path, out_config: Path, external_mapping_path: Path) -> None:
    import yaml

    data = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    data["pytorchsim_functional_mode"] = 0
    data["pytorchsim_timing_mode"] = 1
    ramulator_config = data.get("ramulator_config_path")
    if isinstance(ramulator_config, str) and not Path(ramulator_config).is_absolute():
        normalized = (repo_root() / ramulator_config).resolve()
        if not normalized.exists():
            normalized = (repo_root() / "TOGSim" / ramulator_config).resolve()
        data["ramulator_config_path"] = str(normalized)
    data["codegen_mapping_strategy"] = "external-then-heuristic"
    data["codegen_external_mapping_file"] = str(external_mapping_path.resolve())
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def discover_files(roots: list[Path], patterns: list[str]) -> dict[str, list[str]]:
    discovered: dict[str, list[str]] = {}
    for pattern in patterns:
        matches: list[Path] = []
        for root in roots:
            if root.exists():
                matches.extend(path for path in root.rglob(pattern) if path.is_file())
        discovered[pattern] = [str(path) for path in sorted(set(matches))]
    return discovered


def pick_most_recent(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def copy_if_exists(source: Path | None, destination: Path) -> bool:
    if source is None or not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def is_togsim_summary_log(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "Total execution cycles:" in text and "DMA active_cycles:" in text


def write_concatenated_logs(log_paths: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    for idx, path in enumerate(log_paths):
        chunks.extend(
            [
                f"===== TOGSIM_LOG[{idx}] {path} =====",
                path.read_text(encoding="utf-8", errors="replace"),
                "",
            ]
        )
    destination.write_text("\n".join(chunks), encoding="utf-8")


def collect_mapping_artifacts(mapping_dir: Path, mapping_config: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    roots = [mapping_dir / "torchsim_outputs", mapping_dir / "torchsim_logs"]
    discovered = discover_files(roots, ["*_tog.py", "tile_graph.onnx", "*.log"])
    raw_tog_paths = [Path(path) for path in discovered["*_tog.py"]]
    tile_graph_paths = [Path(path) for path in discovered["tile_graph.onnx"]]
    log_paths = [Path(path) for path in discovered["*.log"] if is_togsim_summary_log(Path(path))]
    raw_tog = pick_most_recent(raw_tog_paths)
    tile_graph = pick_most_recent(tile_graph_paths)
    log_path = pick_most_recent(log_paths)
    copy_if_exists(raw_tog, mapping_dir / "tog.py")
    copy_if_exists(tile_graph, mapping_dir / "tog.onnx")
    if log_paths:
        write_concatenated_logs(log_paths, mapping_dir / "togsim_log.txt")
    else:
        copy_if_exists(log_path, mapping_dir / "togsim_log.txt")
    write_json(
        mapping_dir / "artifact_manifest.json",
        {
            "raw_tog_paths": [str(path) for path in raw_tog_paths],
            "tile_graph_paths": [str(path) for path in tile_graph_paths],
            "togsim_log_paths": [str(path) for path in log_paths],
            "representative_raw_tog": str(raw_tog) if raw_tog else "",
            "representative_tile_graph": str(tile_graph) if tile_graph else "",
            "representative_togsim_log": str(log_path) if log_path else "",
        },
    )
    if not raw_tog_paths or not log_paths:
        return None, None, "missing raw TOG or TOGSim log"
    per_kernel_counters = [
        parse_togsim_log(path.read_text(encoding="utf-8", errors="replace"))
        for path in log_paths
    ]
    per_tog_features = [extract_struct_features(path, mapping_config) for path in raw_tog_paths]
    write_json(mapping_dir / "per_kernel_counters.json", {"data_label": "measured", "rows": per_kernel_counters})
    write_json(mapping_dir / "per_tog_struct_features.json", {"data_label": "measured", "rows": per_tog_features})
    counters = aggregate_counters(per_kernel_counters)
    features = aggregate_struct_features(per_tog_features, mapping_config)
    return counters, features, ""


def run_gpt2_block_child(args: argparse.Namespace) -> int:
    result_path = Path(args.child_result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"ok": False, "error": "", "traceback": ""}
    try:
        import torch
        import torch_openreg  # noqa: F401
        from transformers.models.gpt2.configuration_gpt2 import GPT2Config
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        torch.manual_seed(args.seed)
        cfg = GPT2Config(
            n_embd=768,
            n_head=12,
            n_layer=1,
            n_positions=max(args.seq, 128),
            n_ctx=max(args.seq, 128),
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        block = GPT2Block(cfg).eval()
        x_cpu = torch.randn(1, args.seq, 768, dtype=torch.float32)
        device = torch.device("npu:0")
        block = block.to(device=device)
        x = x_cpu.to(device=device)
        compiled = torch.compile(dynamic=False)(block)
        out = compiled(x)
        result["output_summary"] = str(type(out))
        result["ok"] = True
    except Exception as exc:
        import traceback

        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    write_json(result_path, result)
    return 0 if result["ok"] else 2


def run_mapping_candidate(mapping_dir: Path, mapping_config: dict[str, Any], args: argparse.Namespace) -> MappingResult:
    mapping_id = mapping_config["mapping_id"]
    mapping_dir.mkdir(parents=True, exist_ok=True)
    write_json(mapping_dir / "mapping_config.json", mapping_config)
    external_mapping = mapping_dir / "external_mapping.json"
    config_path = mapping_dir / "togsim_config.yml"
    write_external_mapping_file(external_mapping, mapping_config, args.seq)
    write_mapping_config(
        args.hw_config,
        config_path,
        external_mapping,
    )
    child_result = mapping_dir / "child_result.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child-run-gpt2-block",
        "--child-result",
        str(child_result),
        "--seq",
        str(args.seq),
        "--seed",
        str(args.seed),
    ]
    env = pytorchsim_runtime_env(output_dir=mapping_dir)
    env["TOGSIM_CONFIG"] = str(config_path.resolve())
    proc = subprocess.run(
        command,
        cwd=repo_root(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_sec,
        check=False,
    )
    (mapping_dir / "child_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        message = f"GPT-2 block child exited {proc.returncode}"
        if child_result.exists():
            child_data = json.loads(child_result.read_text(encoding="utf-8"))
            message += f": {child_data.get('error', '')}"
        return MappingResult(mapping_id, mapping_dir, mapping_config, "failed", message)
    counters, features, message = collect_mapping_artifacts(mapping_dir, mapping_config)
    if counters is None or features is None:
        return MappingResult(mapping_id, mapping_dir, mapping_config, "partial", message)
    return MappingResult(mapping_id, mapping_dir, mapping_config, "measured", "", counters, features)


def build_metadata(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    runtime_env = pytorchsim_runtime_env()
    versions: dict[str, str] = {}
    for module_name in ("torch", "transformers"):
        versions[module_name] = module_version(module_name, runtime_env)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "args": vars(args),
        "output_dir": str(output_dir),
        "workload": {
            "model": "gpt2_single_transformer_block",
            "surrogate_model": "none",
            "phase": args.phase,
            "sequence_length": args.seq,
            "hidden_size": 768,
            "heads": 12,
            "dtype": "float32",
            "device": "npu:0",
            "tensor_shapes": {
                "input_hidden_states": [1, args.seq, 768],
                "qkv_projection": [args.seq, 2304],
                "attention_scores_per_head": [args.seq, args.seq],
                "mlp_intermediate": [1, args.seq, 3072],
            },
        },
        "environment": {
            key: runtime_env.get(key, "")
            for key in ("TORCHSIM_DIR", "TOGSIM_CONFIG", "TORCHSIM_LLVM_PATH", "GEM5_PATH", "CUDA_INSTALL_PATH")
        },
        "versions": versions,
        "stages": {},
    }


def module_version(module_name: str, env: dict[str, str] | None = None) -> str:
    code = f"import {module_name}; print(getattr({module_name}, '__version__', 'unknown'))"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root(),
            env=env or pytorchsim_runtime_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        return f"unavailable: {exc}"
    if proc.returncode == 0:
        return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "unknown"
    error = (proc.stderr or proc.stdout).strip().splitlines()
    return "unavailable: " + (error[-1] if error else f"exit {proc.returncode}")


def run_experiment(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(args, output_dir)
    preflight = environment_preflight(require_transformers=not args.allow_surrogate_block, require_npu=not args.skip_npu)
    metadata["stages"]["environment_preflight"] = preflight
    if not preflight["ok"]:
        reason = "environment preflight failed: " + "; ".join(
            f"{name}: {error}" for name, error in preflight["failures"].items()
        )
        write_terminal_outputs(output_dir, metadata, [], "BLOCKED", reason)
        return output_dir
    mapping_results: list[MappingResult] = []
    candidates = mapping_candidates_from_args(args)
    metadata["stages"]["candidate_generation"] = {"ok": True, "count": len(candidates), "source": "static_tiling_sweep"}
    for candidate in candidates:
        result = run_mapping_candidate(output_dir / "mappings" / candidate["mapping_id"], candidate, args)
        mapping_results.append(result)
        metadata.setdefault("mapping_status", {})[candidate["mapping_id"]] = {
            "status": result.status,
            "message": result.message,
        }
    measured = [result for result in mapping_results if result.status == "measured"]
    if len(measured) >= args.num_mappings:
        reason = ""
        state = "AUTO"
    elif measured:
        reason = f"only {len(measured)} of {args.num_mappings} mappings produced measured TOG/counters"
        state = "PARTIAL"
    else:
        reason = "no mapping produced measured TOG/counters"
        state = "BLOCKED"
    metadata["stages"]["harness"] = {"ok": bool(measured), "measured_count": len(measured), "reason": reason}
    write_terminal_outputs(output_dir, metadata, mapping_results, state, reason)
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal GPT-2 block NPU tiling-DSE experiment harness.")
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--num-mappings", type=int, default=10)
    parser.add_argument("--phase", choices=["prefill", "decode"], default="prefill")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT / DEFAULT_RUN_TAG),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument(
        "--hw-config",
        type=Path,
        default=repo_root() / "configs" / "systolic_ws_128x128_c1_simple_noc_tpuv3_timing_only.yml",
        help="PyTorchSim/TOGSim HW YAML config to use for generated timing config.",
    )
    parser.add_argument(
        "--external-mappings-json",
        type=Path,
        help="Optional list of mapping candidates with mapping_id, TILE_M, TILE_N, TILE_K.",
    )
    parser.add_argument(
        "--skip-npu",
        action="store_true",
        help="Run only the probe CPU/FX path; useful for diagnosing blocked NPU environments.",
    )
    parser.add_argument(
        "--allow-surrogate-block",
        action="store_true",
        help="Allow a repository-local transformer block if HuggingFace GPT-2 is unavailable. Results are not GPT-2 claim-bearing.",
    )
    parser.add_argument("--_child-run-gpt2-block", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.num_mappings <= 12:
        parser.error("--num-mappings must be in [1, 12]")
    if args.hw_config and not args.hw_config.is_file():
        parser.error(f"--hw-config does not exist: {args.hw_config}")
    if args.external_mappings_json and not args.external_mappings_json.is_file():
        parser.error(f"--external-mappings-json does not exist: {args.external_mappings_json}")
    if args.external_mappings_json:
        try:
            load_external_mapping_candidates(args.external_mappings_json)
        except (json.JSONDecodeError, MappingSchemaError, TypeError, ValueError) as exc:
            parser.error(f"--external-mappings-json invalid: {exc}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args._child_run_gpt2_block:
        return run_gpt2_block_child(args)
    output_dir = run_experiment(args)
    print(f"Wrote mapping DSE minimal-loop artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
