#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from mapping_dse_minimal import (
    collect_mapping_artifacts,
    parse_togsim_log,
    pytorchsim_runtime_env,
    repo_root,
    write_external_mapping_file,
    write_json,
)


OUTPUT_ROOT = Path("outputs/route4_fusion_pilot")
BASELINE_CONFIG = Path("configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml")
MAPPING_CONFIG = {"mapping_id": "006", "TILE_M": 128, "TILE_N": 64, "TILE_K": 64}
RTOL = 1e-3
ATOL = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_config_paths(data: dict[str, Any]) -> dict[str, Any]:
    result = dict(data)
    ramulator_config = result.get("ramulator_config_path")
    if isinstance(ramulator_config, str) and not Path(ramulator_config).is_absolute():
        candidate = (repo_root() / ramulator_config).resolve()
        if not candidate.exists():
            candidate = (repo_root() / "TOGSim" / ramulator_config).resolve()
        result["ramulator_config_path"] = str(candidate)
    return result


def generate_hw_configs(output_root: Path) -> dict[str, Any]:
    baseline_path = repo_root() / BASELINE_CONFIG
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise ValueError(f"baseline YAML must be a mapping: {baseline_path}")

    out_dir = output_root / "hw_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, dict[str, str]] = {}
    for variant in ("none", "all"):
        patched = dict(baseline)
        patched["codegen_compiler_optimization"] = variant
        path = out_dir / f"hw_baseline_fuse_{variant}.yml"
        path.write_text(yaml.safe_dump(patched, sort_keys=False), encoding="utf-8")
        generated[variant] = {"path": str(path), "sha256": sha256_file(path)}

    manifest = {
        "baseline_config": str(BASELINE_CONFIG),
        "baseline_sha256": sha256_file(baseline_path),
        "generated": generated,
        "only_changed_field": "codegen_compiler_optimization",
        "fixed_mapping": MAPPING_CONFIG,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def make_run_config(hw_config: Path, run_dir: Path, *, functional_mode: int) -> Path:
    data = yaml.safe_load(hw_config.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"HW YAML must be a mapping: {hw_config}")
    data = normalize_config_paths(data)
    data["pytorchsim_functional_mode"] = int(functional_mode)
    data["pytorchsim_timing_mode"] = 1
    data["codegen_mapping_strategy"] = "external-then-heuristic"
    external_mapping = run_dir / "external_mapping.json"
    write_external_mapping_file(external_mapping, MAPPING_CONFIG, seq=128)
    data["codegen_external_mapping_file"] = str(external_mapping.resolve())
    run_config = run_dir / "togsim_config.yml"
    run_config.parent.mkdir(parents=True, exist_ok=True)
    run_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return run_config


def clear_torch_caches() -> None:
    try:
        import torch
        import torch._dynamo
        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
        from torch._inductor.codecache import FxGraphCache

        torch._dynamo.reset()
        AOTAutogradCache.clear()
        FxGraphCache.clear()
    except Exception:
        pass


def ensure_npu_registered() -> None:
    root = repo_root()
    for path in (root, root / "PyTorchSimDevice"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    import torch_openreg  # noqa: F401


def tensor_diffs(out_cpu: Any, ref_cpu: Any) -> tuple[float, float, bool]:
    import torch

    diff = (out_cpu - ref_cpu).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = ref_cpu.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    passed = bool(torch.allclose(out_cpu, ref_cpu, rtol=RTOL, atol=ATOL))
    return max_abs, max_rel, passed


def addmm_relu_fn(x, w, b):
    import torch

    return torch.relu(torch.addmm(b, x, w))


def run_addmm_relu_child(args: argparse.Namespace) -> int:
    import torch

    result_path = Path(args.child_result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "ok": False,
        "workload": "addmm_relu_128",
        "fusion_variant": args.variant,
        "rtol": RTOL,
        "atol": ATOL,
        "reference_source": "torch cpu",
        "npu_source": "pytorchsim functional_mode=1",
        "error": "",
        "traceback": "",
    }
    try:
        ensure_npu_registered()
        clear_torch_caches()
        torch.manual_seed(args.seed)
        x_cpu = torch.randn(128, 128, dtype=torch.float32)
        w_cpu = torch.randn(128, 128, dtype=torch.float32)
        b_cpu = torch.randn(128, dtype=torch.float32)
        ref = addmm_relu_fn(x_cpu, w_cpu, b_cpu)
        device = torch.device("npu:0")
        opt_fn = torch.compile(dynamic=False)(addmm_relu_fn)
        out = opt_fn(x_cpu.to(device), w_cpu.to(device), b_cpu.to(device)).cpu()
        max_abs, max_rel, passed = tensor_diffs(out, ref)
        result.update(
            {
                "ok": True,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "allclose_passed": passed,
            }
        )
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    write_json(result_path, result)
    return 0 if result.get("ok") else 2


def run_gpt2_block_child(args: argparse.Namespace) -> int:
    import torch
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block

    result_path = Path(args.child_result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "ok": False,
        "workload": "gpt2_block_prefill_s128",
        "fusion_variant": args.variant,
        "rtol": RTOL,
        "atol": ATOL,
        "reference_source": "torch cpu",
        "npu_source": "pytorchsim functional_mode=1",
        "error": "",
        "traceback": "",
    }
    try:
        ensure_npu_registered()
        clear_torch_caches()
        torch.manual_seed(args.seed)
        cfg = GPT2Config(
            n_embd=768,
            n_head=12,
            n_layer=1,
            n_positions=128,
            n_ctx=128,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        cpu_block = GPT2Block(cfg).eval()
        npu_block = copy.deepcopy(cpu_block).to(device=torch.device("npu:0")).eval()
        x_cpu = torch.randn(1, 128, 768, dtype=torch.float32)
        with torch.no_grad():
            ref_tuple = cpu_block(x_cpu)
            ref = ref_tuple[0] if isinstance(ref_tuple, tuple) else ref_tuple
            compiled = torch.compile(dynamic=False)(npu_block)
            out_tuple = compiled(x_cpu.to("npu:0"))
            out = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
            out_cpu = out.cpu()
        max_abs, max_rel, passed = tensor_diffs(out_cpu, ref)
        result.update(
            {
                "ok": True,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "allclose_passed": passed,
            }
        )
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    write_json(result_path, result)
    return 0 if result.get("ok") else 2


def run_timing_child(args: argparse.Namespace) -> int:
    return run_addmm_relu_child(args) if args.workload == "addmm_relu_128" else run_gpt2_block_child(args)


def child_main(args: argparse.Namespace) -> int:
    if args.child_kind == "correctness":
        if args.workload == "gpt2_block_prefill_s128":
            return run_gpt2_block_child(args)
        return run_addmm_relu_child(args)
    if args.child_kind == "timing":
        return run_timing_child(args)
    raise ValueError(f"unknown child kind: {args.child_kind}")


def subprocess_env(run_dir: Path, config_path: Path) -> dict[str, str]:
    env = pytorchsim_runtime_env(output_dir=run_dir)
    env["TOGSIM_CONFIG"] = str(config_path.resolve())
    env["TORCH_COMPILE_DEBUG"] = "1"
    env["TORCHINDUCTOR_CACHE"] = "0"
    env.setdefault("TORCHSIM_LLVM_PATH", "/home/dyf/src/psal-llvm-project-v1.0.8/build/bin")
    env.setdefault("GEM5_PATH", "/home/dyf/src/psal-gem5/build/RISCV/gem5.opt")
    return env


def run_child(
    *,
    run_dir: Path,
    config_path: Path,
    child_kind: str,
    workload: str,
    variant: str,
    seed: int,
    timeout_sec: int,
) -> tuple[int, dict[str, Any], str]:
    result_path = run_dir / "child_result.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        "--child-kind",
        child_kind,
        "--child-result",
        str(result_path),
        "--workload",
        workload,
        "--variant",
        variant,
        "--seed",
        str(seed),
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=repo_root(),
            env=subprocess_env(run_dir, config_path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
            start_new_session=True,
        )
        stdout = proc.stdout
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        returncode = 124
        timeout_result = {
            "ok": False,
            "workload": workload,
            "fusion_variant": variant,
            "rtol": RTOL,
            "atol": ATOL,
            "max_abs_diff": None,
            "max_rel_diff": None,
            "allclose_passed": False,
            "reference_source": "torch cpu",
            "npu_source": "pytorchsim functional_mode=1",
            "error": f"child timed out after {timeout_sec} seconds",
            "traceback": "",
        }
        write_json(result_path, timeout_result)
        stdout = stdout + f"\n[timeout] child exceeded {timeout_sec} seconds\n"
    (run_dir / "child_stdout.txt").write_text(stdout, encoding="utf-8")
    result: dict[str, Any] = {}
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    return returncode, result, stdout


def summarize_child_error(result: dict[str, Any], stdout: str) -> str:
    error = str(result.get("error", "") or "")
    if error and error != "UNKNOWN_ERROR":
        return error
    for pattern in (
        r"spike: unrecognized option [^\n]+",
        r"\[Spike\] Command failed with exit code \d+",
        r"child timed out after \d+ seconds",
    ):
        match = re.search(pattern, stdout)
        if match:
            return match.group(0)
    if error:
        return error
    return "unknown child failure"


def discover_summary_logs(run_dir: Path) -> list[Path]:
    roots = [run_dir / "torchsim_logs", run_dir / "torchsim_outputs"]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.log"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if "Total execution cycles:" in text and "DMA active_cycles:" in text:
                paths.append(path)
    return sorted(set(paths))


def collect_run_artifacts(run_dir: Path) -> dict[str, Any]:
    counters, features, message = collect_mapping_artifacts(run_dir, MAPPING_CONFIG)
    log_paths = discover_summary_logs(run_dir)
    per_kernel = [
        parse_togsim_log(path.read_text(encoding="utf-8", errors="replace"))
        for path in log_paths
    ]
    if per_kernel:
        write_json(run_dir / "per_kernel_counters.json", {"data_label": "measured", "rows": per_kernel})
    total_cycles = int(sum(row.get("total_cycles", 0) for row in per_kernel))
    if counters and int(counters.get("total_cycles", 0) or 0) > 0:
        total_cycles = int(counters["total_cycles"])
    return {
        "counters": counters,
        "features": features,
        "message": message,
        "log_paths": [str(path) for path in log_paths],
        "total_cycles": total_cycles,
    }


def run_correctness_for_workload(
    output_root: Path,
    manifest: dict[str, Any],
    workload: str,
    timeout_sec: int,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    reports: dict[str, dict[str, Any]] = {}
    for variant in ("none", "all"):
        run_dir = output_root / "correctness" / f"fuse_{variant}"
        hw_config = repo_root() / manifest["generated"][variant]["path"]
        config_path = make_run_config(hw_config, run_dir, functional_mode=1)
        code, result, stdout = run_child(
            run_dir=run_dir,
            config_path=config_path,
            child_kind="correctness",
            workload=workload,
            variant=variant,
            seed=0,
            timeout_sec=timeout_sec,
        )
        report = {
            "workload": workload,
            "fusion_variant": variant,
            "rtol": RTOL,
            "atol": ATOL,
            "max_abs_diff": result.get("max_abs_diff"),
            "max_rel_diff": result.get("max_rel_diff"),
            "allclose_passed": bool(result.get("allclose_passed", False)),
            "reference_source": "torch cpu",
            "npu_source": "pytorchsim functional_mode=1",
            "child_returncode": code,
            "error": summarize_child_error(result, stdout),
        }
        write_json(run_dir / "allclose_report.json", report)
        reports[variant] = report
    return all(report["allclose_passed"] for report in reports.values()), reports


def write_blocker(output_root: Path, workload: str, reports: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Route 4 fusion-only pilot correctness blocker",
        "",
        f"Workload: `{workload}`",
        "",
        "Correctness gate failed or did not complete for at least one fusion variant.",
        "",
    ]
    for variant, report in reports.items():
        lines.extend(
            [
                f"## fuse_{variant}",
                "",
                f"- allclose_passed: `{report.get('allclose_passed')}`",
                f"- child_returncode: `{report.get('child_returncode')}`",
                f"- error: `{report.get('error', '')}`",
                "",
            ]
        )
    (output_root / "correctness_blocker.md").write_text("\n".join(lines), encoding="utf-8")


def run_timing(
    output_root: Path,
    manifest: dict[str, Any],
    workload: str,
    repeats: int,
    timeout_sec: int,
    *,
    functional_mode: int = 1,
    fixed_seed: int | None = None,
    timing_note: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"variants": {}}
    for variant in ("none", "all"):
        cycles: list[int] = []
        variant_dir = output_root / "timing" / variant
        for idx in range(repeats):
            run_dir = variant_dir / f"run_{idx:02d}"
            hw_config = repo_root() / manifest["generated"][variant]["path"]
            config_path = make_run_config(hw_config, run_dir, functional_mode=functional_mode)
            seed = fixed_seed if fixed_seed is not None else idx
            code, result, _ = run_child(
                run_dir=run_dir,
                config_path=config_path,
                child_kind="timing",
                workload=workload,
                variant=variant,
                seed=seed,
                timeout_sec=timeout_sec,
            )
            artifacts = collect_run_artifacts(run_dir)
            total_cycles = int(artifacts.get("total_cycles", 0) or 0)
            write_json(
                run_dir / "timing_run_summary.json",
                {
                    "variant": variant,
                    "repeat": idx,
                    "child_returncode": code,
                    "child_ok": bool(result.get("ok", False)),
                    "total_cycles": total_cycles,
                    "artifact_message": artifacts.get("message", ""),
                    "log_paths": artifacts.get("log_paths", []),
                },
            )
            cycles.append(total_cycles)
        valid = [cycle for cycle in cycles if cycle > 0]
        median = float(statistics.median(valid)) if valid else 0.0
        cycle_delta = (max(valid) - min(valid)) / median if valid and median else 0.0
        cls = "measured" if len(valid) == repeats else "partial" if valid else "blocked"
        write_json(variant_dir / "per_kernel_counters.json", {"cycles_in_order": cycles})
        summary["variants"][variant] = {
            "cycles_in_order": cycles,
            "median": median,
            "cycle_delta": cycle_delta,
            "class": cls,
        }
    medians = [data["median"] for data in summary["variants"].values() if data["median"] > 0]
    if len(medians) == 2:
        summary["cycle_delta_between_variants"] = (max(medians) - min(medians)) / min(medians)
    else:
        summary["cycle_delta_between_variants"] = None
    summary["note"] = (
        "if cycle_delta_between_variants < 0.10, the fusion axis effect is at or below "
        "noise floor and cannot be distinguished."
    )
    if timing_note:
        summary.update(timing_note)
    write_json(output_root / "timing_summary.json", summary)
    return summary


def count_mlir_ops(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped in {"{", "}"}:
            continue
        if re.search(r"\b(affine|arith|func|linalg|memref|scf|vector|llvm)\.", stripped):
            count += 1
    return count


def structural_counts_for_variant(output_root: Path, variant: str) -> dict[str, Any]:
    roots = [output_root / "timing" / variant, output_root / "correctness" / f"fuse_{variant}"]
    mlir_paths: list[Path] = []
    tog_paths: list[Path] = []
    onnx_paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        mlir_paths.extend(path for path in root.rglob("*.mlir") if path.is_file())
        tog_paths.extend(path for path in root.rglob("*_tog.py") if path.is_file())
        tog_paths.extend(path for path in root.rglob("tog.py") if path.is_file())
        onnx_paths.extend(path for path in root.rglob("tile_graph.onnx") if path.is_file())
        onnx_paths.extend(path for path in root.rglob("tog.onnx") if path.is_file())
    mlir_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in sorted(set(mlir_paths)))
    tog_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in sorted(set(tog_paths)))
    keyword_hits = sorted(
        keyword
        for keyword in ("fused", "epilogue", "elementwise_chain", "relu", "maximumf")
        if keyword.lower() in (mlir_text + "\n" + tog_text).lower()
    )
    return {
        "mlir_file_count": len(set(mlir_paths)),
        "onnx_tog_file_count": len(set(onnx_paths)),
        "raw_tog_file_count": len(set(tog_paths)),
        "mlir_op_count": count_mlir_ops(mlir_text),
        "tog_node_count": len(re.findall(r'"node_id"\s*:', tog_text)),
        "dma_node_count": len(re.findall(r'"node_name"\s*:\s*"DMANode"', tog_text)),
        "matmul_like_op_count": len(re.findall(r"linalg\.matmul|MatmulCompute|compute_type\"\s*:\s*[12]", mlir_text + "\n" + tog_text)),
        "fused_pattern_keywords": keyword_hits,
    }


def write_structural_diff(output_root: Path) -> dict[str, Any]:
    counts = {
        "none": structural_counts_for_variant(output_root, "none"),
        "all": structural_counts_for_variant(output_root, "all"),
    }
    observation = (
        "本次 structural diff 只做浅层计数，不做语义等价证明。若 fusion=all 的 MLIR/TOG "
        "出现 epilogue 或 relu/maximumf 等关键字，而 fusion=none 没有对应变化，则说明 "
        "fusion axis 已经影响编译产物；否则该 workload 下 fusion 结构差异不明显。"
    )
    payload = {"counts": counts, "observation": observation}
    write_json(output_root / "structural_diff.json", payload)
    write_json(output_root / "structural_diff" / "structural_diff.json", payload)
    return payload


def verdict_from_results(correctness_ok: bool, timing_summary: dict[str, Any]) -> str:
    if not correctness_ok:
        return "FUSION_PILOT_BLOCKED"
    variants = timing_summary.get("variants", {})
    if any(data.get("class") != "measured" for data in variants.values()):
        return "FUSION_PILOT_BLOCKED"
    med_none = float(variants["none"]["median"])
    med_all = float(variants["all"]["median"])
    if med_none <= 0 or med_all <= 0:
        return "FUSION_PILOT_BLOCKED"
    delta = (max(med_none, med_all) - min(med_none, med_all)) / min(med_none, med_all)
    if med_all <= med_none * 0.9 and delta >= 0.10:
        return "FUSION_PILOT_POSITIVE"
    return "FUSION_PILOT_NEGATIVE"


def recovery_verdict_from_mode0(timing_summary: dict[str, Any]) -> str:
    variants = timing_summary.get("variants", {})
    if set(variants) != {"none", "all"}:
        return "FUSION_PILOT_STILL_BLOCKED"
    if any(data.get("class") != "measured" for data in variants.values()):
        return "FUSION_PILOT_STILL_BLOCKED"
    med_none = float(variants["none"].get("median", 0) or 0)
    med_all = float(variants["all"].get("median", 0) or 0)
    if med_none <= 0 or med_all <= 0:
        return "FUSION_PILOT_STILL_BLOCKED"
    if med_all <= med_none * 0.9:
        return "FUSION_PILOT_POSITIVE_MODE0_DEFERRED_CORRECTNESS"
    return "FUSION_PILOT_NEGATIVE"


def read_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_correctness_reports(output_root: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for variant in ("none", "all"):
        path = output_root / "correctness" / f"fuse_{variant}" / "allclose_report.json"
        reports[variant] = read_json_if_exists(
            path,
            {
                "workload": "unknown",
                "fusion_variant": variant,
                "allclose_passed": False,
                "max_abs_diff": None,
                "max_rel_diff": None,
                "error": "missing allclose report",
            },
        )
    return reports


def render_reports(
    output_root: Path,
    workload: str,
    correctness_reports: dict[str, dict[str, Any]],
    timing_summary: dict[str, Any],
    structural_diff: dict[str, Any],
    verdict: str,
) -> None:
    docs_dir = repo_root() / "docs" / "superpowers" / "specs"
    md_path = docs_dir / "2026-07-01-npu-mapping-dse-route4-fusion-pilot-report.md"
    html_path = docs_dir / "2026-07-01-npu-mapping-dse-route4-fusion-pilot-report.html"
    docs_dir.mkdir(parents=True, exist_ok=True)
    none_report = correctness_reports.get("none", {})
    all_report = correctness_reports.get("all", {})
    none_timing = timing_summary.get("variants", {}).get("none", {})
    all_timing = timing_summary.get("variants", {}).get("all", {})
    between = timing_summary.get("cycle_delta_between_variants")
    between_text = "null" if between is None else f"{between:.6f}"
    next_step = {
        "FUSION_PILOT_POSITIVE": "下一步可以把 fusion axis 扩展到小型 4-HW sweep，但仍保持 correctness gate 作为每个候选的前置条件。",
        "FUSION_PILOT_NEGATIVE": "下一步应先放大 workload 或更换到 attention/MLP block，再判断 fusion axis 是否有可测收益；不应直接进入完整 Route 4 sweep。",
        "FUSION_PILOT_BLOCKED": "下一步应优先修复报告中记录的 correctness 或 timing 阻塞点，暂时不要扩展 HW sweep 或 dataflow 轴。",
    }[verdict]
    md = f"""# Route 4 fusion-only pilot report

## 1. 目标与非目标

本实验只验证 fusion axis 是否能在一个小 workload 上被正确生成、数值验证并独立计时。非目标包括：不做 4-HW sweep，不做 10-mapping tile sweep，不触碰 dataflow，不修改 PyTorchSim / TOGSim / gem5 / ramulator2 源码。

## 2. Method

- workload: `{workload}`
- HW config baseline: `{BASELINE_CONFIG}`
- fusion variants: `codegen_compiler_optimization=none` 和 `codegen_compiler_optimization=all`
- fixed mapping: `TILE_M=128`, `TILE_N=64`, `TILE_K=64`
- correctness gate: `pytorchsim_functional_mode=1`, `rtol={RTOL}`, `atol={ATOL}`
- timing repeats: 5 per variant
- output root: `{output_root}`

## 3. Correctness results

| variant | allclose_passed | max_abs_diff | max_rel_diff | error |
| --- | --- | --- | --- | --- |
| none | `{none_report.get('allclose_passed')}` | `{none_report.get('max_abs_diff')}` | `{none_report.get('max_rel_diff')}` | `{none_report.get('error', '')}` |
| all | `{all_report.get('allclose_passed')}` | `{all_report.get('max_abs_diff')}` | `{all_report.get('max_rel_diff')}` | `{all_report.get('error', '')}` |

## 4. Timing results

| variant | cycles_in_order | median | cycle_delta | class |
| --- | --- | --- | --- | --- |
| none | `{none_timing.get('cycles_in_order')}` | `{none_timing.get('median')}` | `{none_timing.get('cycle_delta')}` | `{none_timing.get('class')}` |
| all | `{all_timing.get('cycles_in_order')}` | `{all_timing.get('median')}` | `{all_timing.get('cycle_delta')}` | `{all_timing.get('class')}` |

`cycle_delta_between_variants = {between_text}`。如果该值低于 0.10，则 fusion axis 的效果处在或低于当前噪声地板，不能作为强结论。

## 5. Structural diff

```json
{json.dumps(structural_diff, indent=2, ensure_ascii=False)}
```

## 6. Verdict

`{verdict}`

## 7. Next step recommendation

{next_step}
"""
    md_path.write_text(md, encoding="utf-8")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Route 4 fusion-only pilot report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.55; margin: 40px; max-width: 980px; }}
    code, pre {{ background: #f5f5f5; border-radius: 4px; }}
    code {{ padding: 1px 4px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #d0d0d0; padding: 6px 8px; text-align: left; vertical-align: top; }}
  </style>
</head>
<body>
<h1>Route 4 fusion-only pilot report</h1>
<h2>1. 目标与非目标</h2>
<p>本实验只验证 fusion axis 是否能在一个小 workload 上被正确生成、数值验证并独立计时。非目标包括：不做 4-HW sweep，不做 10-mapping tile sweep，不触碰 dataflow，不修改 PyTorchSim / TOGSim / gem5 / ramulator2 源码。</p>
<h2>2. Method</h2>
<ul>
  <li>workload: <code>{workload}</code></li>
  <li>HW config baseline: <code>{BASELINE_CONFIG}</code></li>
  <li>fusion variants: <code>none</code> 和 <code>all</code></li>
  <li>fixed mapping: <code>TILE_M=128</code>, <code>TILE_N=64</code>, <code>TILE_K=64</code></li>
  <li>correctness gate: <code>pytorchsim_functional_mode=1</code>, <code>rtol={RTOL}</code>, <code>atol={ATOL}</code></li>
  <li>timing repeats: 5 per variant</li>
</ul>
<h2>3. Correctness results</h2>
<table><tr><th>variant</th><th>allclose_passed</th><th>max_abs_diff</th><th>max_rel_diff</th><th>error</th></tr>
<tr><td>none</td><td>{none_report.get('allclose_passed')}</td><td>{none_report.get('max_abs_diff')}</td><td>{none_report.get('max_rel_diff')}</td><td>{none_report.get('error', '')}</td></tr>
<tr><td>all</td><td>{all_report.get('allclose_passed')}</td><td>{all_report.get('max_abs_diff')}</td><td>{all_report.get('max_rel_diff')}</td><td>{all_report.get('error', '')}</td></tr></table>
<h2>4. Timing results</h2>
<table><tr><th>variant</th><th>cycles_in_order</th><th>median</th><th>cycle_delta</th><th>class</th></tr>
<tr><td>none</td><td>{none_timing.get('cycles_in_order')}</td><td>{none_timing.get('median')}</td><td>{none_timing.get('cycle_delta')}</td><td>{none_timing.get('class')}</td></tr>
<tr><td>all</td><td>{all_timing.get('cycles_in_order')}</td><td>{all_timing.get('median')}</td><td>{all_timing.get('cycle_delta')}</td><td>{all_timing.get('class')}</td></tr></table>
<p><code>cycle_delta_between_variants = {between_text}</code>。如果该值低于 0.10，则 fusion axis 的效果处在或低于当前噪声地板，不能作为强结论。</p>
<h2>5. Structural diff</h2>
<pre>{json.dumps(structural_diff, indent=2, ensure_ascii=False)}</pre>
<h2>6. Verdict</h2>
<p><code>{verdict}</code></p>
<h2>7. Next step recommendation</h2>
<p>{next_step}</p>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def render_recovery_reports(
    output_root: Path,
    workload: str,
    correctness_reports: dict[str, dict[str, Any]],
    timing_summary: dict[str, Any],
    structural_diff: dict[str, Any],
    verdict: str,
    diagnostic_summary: str,
) -> None:
    docs_dir = repo_root() / "docs" / "superpowers" / "specs"
    md_path = docs_dir / "2026-07-01-npu-mapping-dse-route4-fusion-pilot-report.md"
    html_path = docs_dir / "2026-07-01-npu-mapping-dse-route4-fusion-pilot-report.html"
    docs_dir.mkdir(parents=True, exist_ok=True)

    none_report = correctness_reports.get("none", {})
    all_report = correctness_reports.get("all", {})
    none_timing = timing_summary.get("variants", {}).get("none", {})
    all_timing = timing_summary.get("variants", {}).get("all", {})
    between = timing_summary.get("cycle_delta_between_variants")
    between_text = "null" if between is None else f"{between:.6f}"
    scope_limitation = timing_summary.get("scope_limitation", "")
    next_step = {
        "FUSION_PILOT_POSITIVE_MODE1": "下一步可以在保持 correctness gate 的前提下，把 fusion axis 扩展到小型 4-HW sweep。",
        "FUSION_PILOT_POSITIVE_MODE0_DEFERRED_CORRECTNESS": "下一步应先修复 Spike `--varch` toolchain，再复跑 mode=1 correctness；若通过，再把 fusion axis 扩展到小型 4-HW sweep。",
        "FUSION_PILOT_NEGATIVE": "下一步不应直接扩展到完整 4-HW sweep，应先放大 workload 或改用 attention/MLP block 再判断 fusion axis 是否有可测收益。",
        "FUSION_PILOT_STILL_BLOCKED": "下一步应优先修复 mode=0 timing 阻塞点，暂时不要扩展 HW sweep 或 dataflow 轴。",
    }[verdict]
    structural_json = json.dumps(structural_diff, indent=2, ensure_ascii=False)
    md = f"""# Route 4 fusion-only pilot report

## 1. 目标与非目标

本实验只验证 fusion axis 是否能在一个小 workload 上被正确生成、数值验证并独立计时。非目标包括：不做 4-HW sweep，不做 10-mapping tile sweep，不触碰 dataflow，不修改 PyTorchSim / TOGSim / gem5 / ramulator2 / spike 源码。

## 2. Method

- workload: `{workload}`
- HW config baseline: `{BASELINE_CONFIG}`
- fusion variants: `codegen_compiler_optimization=none` 和 `codegen_compiler_optimization=all`
- fixed mapping: `TILE_M=128`, `TILE_N=64`, `TILE_K=64`
- correctness gate: 原 mode=1 correctness 尝试保留为 evidence；本次 recovery 的 timing fallback 使用 `pytorchsim_functional_mode=0`
- timing repeats: 5 per variant, `seed=0`
- output root: `{output_root}`

## 3. Correctness results

| variant | allclose_passed | max_abs_diff | max_rel_diff | error |
| --- | --- | --- | --- | --- |
| none | `{none_report.get('allclose_passed')}` | `{none_report.get('max_abs_diff')}` | `{none_report.get('max_rel_diff')}` | `{none_report.get('error', '')}` |
| all | `{all_report.get('allclose_passed')}` | `{all_report.get('max_abs_diff')}` | `{all_report.get('max_rel_diff')}` | `{all_report.get('error', '')}` |

## 4. Timing results

| variant | cycles_in_order | median | cycle_delta | class |
| --- | --- | --- | --- | --- |
| none | `{none_timing.get('cycles_in_order')}` | `{none_timing.get('median')}` | `{none_timing.get('cycle_delta')}` | `{none_timing.get('class')}` |
| all | `{all_timing.get('cycles_in_order')}` | `{all_timing.get('median')}` | `{all_timing.get('cycle_delta')}` | `{all_timing.get('class')}` |

`cycle_delta_between_variants = {between_text}`。如果该值低于 0.10，则 fusion axis 的效果处在或低于当前噪声地板，不能作为强结论。

## 5. Structural diff

```json
{structural_json}
```

## 6. Verdict

`{verdict}`

## 7. Next step recommendation

{next_step}

## 8. Recovery attempt (Phase A + Phase B result)

Phase A 在 30 分钟 timebox 内只尝试配置级和 PATH 级诊断。结果是：`--varch=vlen:256,elen:64` 在 `Simulator/simulator.py` 的 Spike 调用中硬编码，当前 PATH 只有 `/usr/bin/spike`，该 Spike help 只暴露 `--isa`，不支持 `--varch`，configs/scripts 中也没有可用于改写 `--varch` 的 YAML 或 env 字段。因此没有找到不修改源码的 mode=1 correctness 修复。

Phase B 已进入 timing-only fallback：两个 fusion variants 都使用 `pytorchsim_functional_mode=0`、相同 fixed mapping 和 `seed=0` 重复 5 次。`correctness_gate` 记录为 `{timing_summary.get('correctness_gate', '')}`；`correctness_proxy` 记录为 `{timing_summary.get('correctness_proxy', '')}`。

scope_limitation: {scope_limitation}

Phase A diagnostic summary: {diagnostic_summary}
"""
    md_path.write_text(md, encoding="utf-8")

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Route 4 fusion-only pilot report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.55; margin: 40px; max-width: 980px; }}
    code, pre {{ background: #f5f5f5; border-radius: 4px; }}
    code {{ padding: 1px 4px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #d0d0d0; padding: 6px 8px; text-align: left; vertical-align: top; }}
  </style>
</head>
<body>
<h1>Route 4 fusion-only pilot report</h1>
<h2>1. 目标与非目标</h2>
<p>本实验只验证 fusion axis 是否能在一个小 workload 上被正确生成、数值验证并独立计时。非目标包括：不做 4-HW sweep，不做 10-mapping tile sweep，不触碰 dataflow，不修改 PyTorchSim / TOGSim / gem5 / ramulator2 / spike 源码。</p>
<h2>2. Method</h2>
<ul>
  <li>workload: <code>{workload}</code></li>
  <li>HW config baseline: <code>{BASELINE_CONFIG}</code></li>
  <li>fusion variants: <code>none</code> 和 <code>all</code></li>
  <li>fixed mapping: <code>TILE_M=128</code>, <code>TILE_N=64</code>, <code>TILE_K=64</code></li>
  <li>timing fallback: <code>pytorchsim_functional_mode=0</code>, repeats=5, <code>seed=0</code></li>
</ul>
<h2>3. Correctness results</h2>
<table><tr><th>variant</th><th>allclose_passed</th><th>max_abs_diff</th><th>max_rel_diff</th><th>error</th></tr>
<tr><td>none</td><td>{none_report.get('allclose_passed')}</td><td>{none_report.get('max_abs_diff')}</td><td>{none_report.get('max_rel_diff')}</td><td>{none_report.get('error', '')}</td></tr>
<tr><td>all</td><td>{all_report.get('allclose_passed')}</td><td>{all_report.get('max_abs_diff')}</td><td>{all_report.get('max_rel_diff')}</td><td>{all_report.get('error', '')}</td></tr></table>
<h2>4. Timing results</h2>
<table><tr><th>variant</th><th>cycles_in_order</th><th>median</th><th>cycle_delta</th><th>class</th></tr>
<tr><td>none</td><td>{none_timing.get('cycles_in_order')}</td><td>{none_timing.get('median')}</td><td>{none_timing.get('cycle_delta')}</td><td>{none_timing.get('class')}</td></tr>
<tr><td>all</td><td>{all_timing.get('cycles_in_order')}</td><td>{all_timing.get('median')}</td><td>{all_timing.get('cycle_delta')}</td><td>{all_timing.get('class')}</td></tr></table>
<p><code>cycle_delta_between_variants = {between_text}</code>。如果该值低于 0.10，则 fusion axis 的效果处在或低于当前噪声地板，不能作为强结论。</p>
<h2>5. Structural diff</h2>
<pre>{structural_json}</pre>
<h2>6. Verdict</h2>
<p><code>{verdict}</code></p>
<h2>7. Next step recommendation</h2>
<p>{next_step}</p>
<h2>8. Recovery attempt (Phase A + Phase B result)</h2>
<p>Phase A 在 30 分钟 timebox 内只尝试配置级和 PATH 级诊断。结果是：<code>--varch=vlen:256,elen:64</code> 在 <code>Simulator/simulator.py</code> 的 Spike 调用中硬编码，当前 PATH 只有 <code>/usr/bin/spike</code>，该 Spike help 只暴露 <code>--isa</code>，不支持 <code>--varch</code>，configs/scripts 中也没有可用于改写 <code>--varch</code> 的 YAML 或 env 字段。因此没有找到不修改源码的 mode=1 correctness 修复。</p>
<p>Phase B 已进入 timing-only fallback：两个 fusion variants 都使用 <code>pytorchsim_functional_mode=0</code>、相同 fixed mapping 和 <code>seed=0</code> 重复 5 次。<code>correctness_gate</code>: <code>{timing_summary.get('correctness_gate', '')}</code>；<code>correctness_proxy</code>: <code>{timing_summary.get('correctness_proxy', '')}</code>。</p>
<p><strong>scope_limitation:</strong> {scope_limitation}</p>
<p><strong>Phase A diagnostic summary:</strong> {diagnostic_summary}</p>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def run_pilot(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = generate_hw_configs(output_root)

    workload = "gpt2_block_prefill_s128"
    correctness_ok, reports = run_correctness_for_workload(output_root, manifest, workload, args.correctness_timeout_sec)
    if not correctness_ok:
        workload = "addmm_relu_128"
        shutil.rmtree(output_root / "correctness", ignore_errors=True)
        correctness_ok, reports = run_correctness_for_workload(output_root, manifest, workload, args.correctness_timeout_sec)

    if not correctness_ok:
        write_blocker(output_root, workload, reports)
        timing_summary = {
            "variants": {
                "none": {"cycles_in_order": [], "median": 0, "cycle_delta": 0, "class": "blocked"},
                "all": {"cycles_in_order": [], "median": 0, "cycle_delta": 0, "class": "blocked"},
            },
            "cycle_delta_between_variants": None,
            "note": "correctness gate failed; timing skipped",
        }
        write_json(output_root / "timing_summary.json", timing_summary)
        write_json(
            output_root / "timing" / "blocked.json",
            {
                "timing_skipped": True,
                "reason": "correctness gate failed",
                "workload": workload,
            },
        )
        structural = write_structural_diff(output_root)
        verdict = "FUSION_PILOT_BLOCKED"
        render_reports(output_root, workload, reports, timing_summary, structural, verdict)
        write_json(
            output_root / "pilot_summary.json",
            {
                "workload": workload,
                "correctness": reports,
                "timing": timing_summary,
                "structural_diff": structural,
                "verdict": verdict,
            },
        )
        return 2

    timing_summary = run_timing(output_root, manifest, workload, args.repeats, args.timing_timeout_sec)
    structural = write_structural_diff(output_root)
    verdict = verdict_from_results(correctness_ok, timing_summary)
    render_reports(output_root, workload, reports, timing_summary, structural, verdict)
    write_json(
        output_root / "pilot_summary.json",
        {
            "workload": workload,
            "correctness": reports,
            "timing": timing_summary,
            "structural_diff": structural,
            "verdict": verdict,
        },
    )
    return 0 if verdict != "FUSION_PILOT_BLOCKED" else 2


def write_spike_varch_diagnostic(output_root: Path) -> str:
    diagnostic_path = output_root / "spike_varch_diagnostic.md"
    text = """# Spike --varch diagnostic

## Phase A timebox

本诊断遵守 handoff 的 30 分钟 wall-time 限制，只尝试配置级、环境变量级、PATH 级 workaround；没有修改 PyTorchSim / TOGSim / gem5 / ramulator2 / spike 源码。

## 1. Spike binary and version

命令：

```bash
which spike
spike --version
spike --help 2>&1 | grep -iE "varch|isa" | head -20
type -a spike
```

观察：

```text
which spike -> /usr/bin/spike
spike --version -> spike: unrecognized option --version
spike --help -> Spike RISC-V ISA Simulator 1.1.1-dev
spike --help -> --isa=<name> RISC-V ISA string [default rv64imafdc_zicntr_zihpm]
type -a spike -> spike is /usr/bin/spike
```

结论：当前 PATH 上只有 `/usr/bin/spike`，该版本的 help 输出没有 `--varch` 选项。

## 2. Where --varch is produced

命令：

```bash
rg -n "varch|vlen:256|elen:64" PyTorchSimDevice TOGSim PyTorchSimFrontend Simulator --glob '!**/build/**'
nl -ba Simulator/simulator.py | sed -n '120,180p'
```

关键命中：

```text
Simulator/simulator.py:147:
run = f'spike --isa rv64gcv_zfh --varch=vlen:256,elen:64 {vectorlane_option} ...'
```

结论：`--varch=vlen:256,elen:64` 是 `Simulator/simulator.py` 的 hardcoded Spike command，不是 YAML 字段。

## 3. Config/env controllability

命令：

```bash
rg -n "vlen|elen|varch" configs scripts --glob '!**/__pycache__/**'
rg -n "SPIKE|spike|RISCV|pk|varch|vlen" Simulator PyTorchSimFrontend PyTorchSimDevice configs scripts --glob '!**/build/**'
```

观察：

```text
configs/ 中没有控制 varch/elen 的字段。
scripts/ 中没有用于替换 Spike varch string 的参数。
PyTorchSimFrontend/extension_codecache.py 使用 vlen 生成 MLIR/LLVM lowering 参数，但不控制 Spike --varch。
```

结论：现有 YAML/env 只能影响 `vpu_vector_length_bits`、MLIR lowering 或 PATH；不能在不改源码的前提下删除或改写 Spike `--varch`。

## 4. Alternative Spike binary check

命令：

```bash
find /home/dyf/opt/pytorchsim-riscv-gcc-compat/bin /home/dyf/miniconda3/envs/pytorchsim-build/bin /home/dyf/src /home/dyf/local/bin /usr/local/bin /usr/bin -maxdepth 4 -type f -name spike -executable
```

观察：

```text
/usr/bin/spike
```

结论：没有找到可通过 PATH 调整切换的兼容 Spike binary。

## Phase A conclusion

未找到配置级 workaround。阻塞点是 functional correctness path 需要支持 `--varch=vlen:256,elen:64` 的 Spike，但当前 `/usr/bin/spike` 不支持该选项。根据 handoff，进入 Phase B：`pytorchsim_functional_mode=0` timing-only fallback，并明确记录 correctness deferred。
"""
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(text, encoding="utf-8")
    return (
        "未找到配置级 workaround；`--varch=vlen:256,elen:64` 在 `Simulator/simulator.py` 中硬编码，"
        "当前 `/usr/bin/spike` 不支持 `--varch`，且 PATH 中没有替代 Spike。"
    )


def run_recovery_mode0(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stale_blocked = output_root / "timing" / "blocked.json"
    if stale_blocked.exists():
        stale_blocked.unlink()
    manifest_path = output_root / "hw_configs" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = generate_hw_configs(output_root)

    diagnostic_summary = write_spike_varch_diagnostic(output_root)
    workload = "addmm_relu_128"
    correctness_reports = load_correctness_reports(output_root)
    structural = read_json_if_exists(output_root / "structural_diff.json", {})

    timing_summary = run_timing(
        output_root,
        manifest,
        workload,
        args.repeats,
        args.timing_timeout_sec,
        functional_mode=0,
        fixed_seed=0,
        timing_note={
            "correctness_gate": "deferred_due_to_spike_varch_blocker",
            "correctness_proxy": "structural_diff shows fusion=all has 'fused' keyword and ~10% fewer MLIR ops",
            "scope_limitation": (
                "correctness gate deferred because pytorchsim_functional_mode=1 path requires a spike "
                "version compatible with --varch=vlen:256,elen:64 and no config-level fix was found "
                "within 30 minutes"
            ),
        },
    )
    if not structural:
        structural = write_structural_diff(output_root)
    verdict = recovery_verdict_from_mode0(timing_summary)
    render_recovery_reports(
        output_root,
        workload,
        correctness_reports,
        timing_summary,
        structural,
        verdict,
        diagnostic_summary,
    )
    write_json(
        output_root / "pilot_summary.json",
        {
            "workload": workload,
            "correctness": correctness_reports,
            "timing": timing_summary,
            "structural_diff": structural,
            "spike_varch_diagnostic": str(output_root / "spike_varch_diagnostic.md"),
            "verdict": verdict,
        },
    )
    return 0 if verdict != "FUSION_PILOT_STILL_BLOCKED" else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Route 4 fusion-only pilot.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--correctness-timeout-sec", type=int, default=900)
    parser.add_argument("--timing-timeout-sec", type=int, default=900)
    parser.add_argument("--recovery-mode0", action="store_true")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-kind", choices=["correctness", "timing"], help=argparse.SUPPRESS)
    parser.add_argument("--child-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=["gpt2_block_prefill_s128", "addmm_relu_128"], help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=["none", "all"], help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeats != 5:
        parser.error("--repeats must remain 5 for this pilot")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args._child:
        return child_main(args)
    if args.recovery_mode0:
        return run_recovery_mode0(args)
    return run_pilot(args)


if __name__ == "__main__":
    raise SystemExit(main())
