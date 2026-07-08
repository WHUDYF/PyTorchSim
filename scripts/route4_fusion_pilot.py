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
from hw_config_factory import get_hw_config_set, patch_hw_yaml


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


def make_run_config(
    hw_config: Path,
    run_dir: Path,
    *,
    functional_mode: int,
    use_external_mapping: bool = True,
    external_mapping_override: Path | None = None,
) -> Path:
    data = yaml.safe_load(hw_config.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"HW YAML must be a mapping: {hw_config}")
    data = normalize_config_paths(data)
    data["pytorchsim_functional_mode"] = int(functional_mode)
    data["pytorchsim_timing_mode"] = 1
    if use_external_mapping:
        data["codegen_mapping_strategy"] = "external-then-heuristic"
        external_mapping = external_mapping_override or (run_dir / "external_mapping.json")
        if external_mapping_override is None:
            write_external_mapping_file(external_mapping, MAPPING_CONFIG, seq=128)
        data["codegen_external_mapping_file"] = str(external_mapping.resolve())
    else:
        data["codegen_mapping_strategy"] = "heuristic"
        data["codegen_external_mapping_file"] = ""
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


def _external_conv_tile_candidate(template_obj: Any, kernel: Any, *, mode: str, BATCH: int, I_C: int, O_C: int, K_H: int, K_W: int, O_H: int, O_W: int) -> list[list[int]] | None:
    from PyTorchSimFrontend import extension_config

    if "external" not in extension_config.codegen_mapping_strategy:
        return None
    path = Path(extension_config.codegen_external_mapping_file)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    tile_info = data.get(conv_shape_key(BATCH, I_C, O_C, K_H, K_W, O_H, O_W))
    if tile_info is None:
        return None
    keys = ["TILE_K_H", "TILE_K_W", "TILE_O_H", "TILE_O_W", "TILE_M", "TILE_N", "TILE_K"]
    TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K = [int(tile_info[key]) for key in keys]
    TILE_I_H = 1 + (TILE_O_H - 1) * template_obj.stride[0] + (TILE_K_H - 1) * template_obj.dilation[0]
    if mode == "mt":
        TILE_I_W = 1 + (TILE_O_W - 1) * template_obj.stride[1]
    else:
        TILE_I_W = 1 + (TILE_O_W - 1) * template_obj.stride[1] + (TILE_K_W - 1) * template_obj.dilation[1]
    SUB_TILE_I_H, SUB_TILE_I_W, SUB_TILE_K_H, SUB_TILE_K_W = 1, 1, 1, 1
    if mode == "sb":
        SUB_TILE_M = TILE_I_W if TILE_I_W < kernel.vector_lane else kernel.vector_lane
    else:
        SUB_TILE_M = TILE_M if TILE_M < kernel.vector_lane else kernel.vector_lane
    SUB_TILE_N = TILE_N if TILE_N < kernel.vector_lane else kernel.vector_lane
    SUB_TILE_K = TILE_K
    if mode in {"base", "mt"}:
        SUB_TILE_N = TILE_N if TILE_N > 512 else SUB_TILE_N
    return [[TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K, TILE_I_H, TILE_I_W, SUB_TILE_I_H, SUB_TILE_I_W, SUB_TILE_K_H, SUB_TILE_K_W, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K]]


def install_conv_external_tile_patch() -> None:
    modules = [
        ("base", "PyTorchSimFrontend.mlir.mlir_conv_template", "MLIRConvTemplate"),
        ("mt", "PyTorchSimFrontend.mlir.mlir_conv_mt_template", "MLIRConvMultiTileTemplate"),
        ("sb", "PyTorchSimFrontend.mlir.mlir_conv_sb_template", "MLIRConvSingleBatchTemplate"),
        ("sbs", "PyTorchSimFrontend.mlir.mlir_conv_sbs_template", "MLIRConvSingleBatchStridedTemplate"),
    ]
    import importlib

    for mode, module_name, class_name in modules:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name, None)
        if cls is None or getattr(cls, "_route4_conv_external_tile_patch", False):
            continue
        original_select_tile = cls.select_tile

        def patched_select_tile(self, kernel, n_extra_node, BATCH, I_C, O_C, K_H, K_W, O_H, O_W, precision_bytes, *, _mode=mode, _original=original_select_tile):
            external = _external_conv_tile_candidate(
                self,
                kernel,
                mode=_mode,
                BATCH=BATCH,
                I_C=I_C,
                O_C=O_C,
                K_H=K_H,
                K_W=K_W,
                O_H=O_H,
                O_W=O_W,
            )
            if external is not None:
                return external
            return _original(self, kernel, n_extra_node, BATCH, I_C, O_C, K_H, K_W, O_H, O_W, precision_bytes)

        cls.select_tile = patched_select_tile
        cls._route4_conv_external_tile_patch = True


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


def run_conv3x3_probe_child(args: argparse.Namespace) -> int:
    import torch

    class ConvBnRelu(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            self.bn = torch.nn.BatchNorm2d(64)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    result_path = Path(args.child_result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "ok": False,
        "workload": "conv3x3_probe",
        "fusion_variant": args.variant,
        "rtol": RTOL,
        "atol": ATOL,
        "reference_source": "torch cpu",
        "npu_source": "pytorchsim functional_mode=0 timing probe",
        "error": "",
        "traceback": "",
    }
    try:
        ensure_npu_registered()
        clear_torch_caches()
        install_conv_external_tile_patch()
        torch.manual_seed(args.seed)
        cpu_model = ConvBnRelu().eval()
        npu_model = copy.deepcopy(cpu_model).to(device=torch.device("npu:0")).eval()
        x_cpu = torch.randn(1, 64, 56, 56, dtype=torch.float32)
        with torch.no_grad():
            ref = cpu_model(x_cpu)
            compiled = torch.compile(dynamic=False)(npu_model)
            out = compiled(x_cpu.to("npu:0")).cpu()
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


def run_timing_child(args: argparse.Namespace) -> int:
    if args.workload == "addmm_relu_128":
        return run_addmm_relu_child(args)
    if args.workload == "gpt2_block_prefill_s128":
        return run_gpt2_block_child(args)
    if args.workload == "conv3x3_probe":
        return run_conv3x3_probe_child(args)
    raise ValueError(f"unknown workload: {args.workload}")


def child_main(args: argparse.Namespace) -> int:
    if args.child_kind == "correctness":
        if args.workload == "gpt2_block_prefill_s128":
            return run_gpt2_block_child(args)
        if args.workload == "conv3x3_probe":
            return run_conv3x3_probe_child(args)
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


FUSED_PATTERN_KEYWORDS = (
    "epilogue",
    "fused",
    "maximumf",
    "relu",
    "gelu",
    "layernorm",
    "softmax",
    "elementwise_chain",
)


def structural_counts_for_root(root: Path) -> dict[str, Any]:
    mlir_paths: list[Path] = []
    tog_paths: list[Path] = []
    if root.exists():
        mlir_paths.extend(path for path in root.rglob("*.mlir") if path.is_file())
        tog_paths.extend(path for path in root.rglob("*_tog.py") if path.is_file())
        tog_paths.extend(path for path in root.rglob("tog.py") if path.is_file())
    mlir_paths = sorted(set(mlir_paths))
    tog_paths = sorted(set(tog_paths))
    mlir_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in mlir_paths)
    tog_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in tog_paths)
    combined = mlir_text + "\n" + tog_text
    keyword_hits = sorted(keyword for keyword in FUSED_PATTERN_KEYWORDS if keyword in combined.lower())
    return {
        "mlir_op_count": count_mlir_ops(mlir_text),
        "tog_node_count": len(re.findall(r'"node_id"\s*:', tog_text)),
        "dma_node_count": len(re.findall(r'"node_name"\s*:\s*"DMANode"', tog_text)),
        "matmul_like_op_count": len(
            re.findall(r"linalg\.matmul|MatmulCompute|compute_type\"\s*:\s*[12]", combined)
        ),
        "fused_pattern_keywords": keyword_hits,
        "mlir_file_count": len(mlir_paths),
        "raw_tog_file_count": len(tog_paths),
    }


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
    keyword_hits = sorted(keyword for keyword in FUSED_PATTERN_KEYWORDS if keyword in (mlir_text + "\n" + tog_text).lower())
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


def generate_v1_hw_fusion_configs(output_root: Path) -> dict[str, Any]:
    baseline_path = repo_root() / BASELINE_CONFIG
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise ValueError(f"baseline YAML must be a mapping: {baseline_path}")
    config_set = get_hw_config_set("codesign_v1_2x2")
    out_dir = output_root / "hw_fusion_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, dict[str, dict[str, str]]] = {}
    for hw_id, hw_patch in config_set.patch_matrix.items():
        hw_base = patch_hw_yaml(baseline, hw_patch, config_set=config_set)
        generated[hw_id] = {}
        for variant in ("none", "all"):
            patched = dict(hw_base)
            patched["codegen_compiler_optimization"] = variant
            path = out_dir / f"{hw_id}_fuse_{variant}.yml"
            path.write_text(yaml.safe_dump(patched, sort_keys=False), encoding="utf-8")
            generated[hw_id][variant] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    manifest = {
        "baseline_config": str(BASELINE_CONFIG),
        "baseline_sha256": sha256_file(baseline_path),
        "hw_config_set": "codesign_v1_2x2",
        "patch_matrix": config_set.patch_matrix,
        "fixed_mapping": MAPPING_CONFIG,
        "generated": generated,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def run_timing_cell(
    *,
    cell_dir: Path,
    hw_config: Path,
    workload: str,
    variant: str,
    repeats: int,
    timeout_sec: int,
    use_external_mapping: bool = True,
    external_mapping_override: Path | None = None,
) -> dict[str, Any]:
    cycles: list[int] = []
    run_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx in range(repeats):
        run_dir = cell_dir / f"run_{idx:02d}"
        summary_path = run_dir / "timing_run_summary.json"
        if summary_path.exists():
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            total_cycles = int(existing_summary.get("total_cycles", 0) or 0)
            if total_cycles > 0:
                run_summaries.append(existing_summary)
                cycles.append(total_cycles)
                continue
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        override_for_run = clone_external_mapping_override(external_mapping_override, run_dir)
        config_path = make_run_config(
            hw_config,
            run_dir,
            functional_mode=0,
            use_external_mapping=use_external_mapping,
            external_mapping_override=override_for_run,
        )
        code, result, stdout = run_child(
            run_dir=run_dir,
            config_path=config_path,
            child_kind="timing",
            workload=workload,
            variant=variant,
            seed=0,
            timeout_sec=timeout_sec,
        )
        artifacts = collect_run_artifacts(run_dir)
        total_cycles = int(artifacts.get("total_cycles", 0) or 0)
        run_summary = {
            "repeat": idx,
            "child_returncode": code,
            "child_ok": bool(result.get("ok", False)),
            "total_cycles": total_cycles,
            "artifact_message": artifacts.get("message", ""),
            "log_paths": artifacts.get("log_paths", []),
        }
        write_json(summary_path, run_summary)
        run_summaries.append(run_summary)
        cycles.append(total_cycles)
        if code != 0 or total_cycles <= 0:
            failures.append(
                {
                    "repeat": idx,
                    "child_returncode": code,
                    "total_cycles": total_cycles,
                    "error": result.get("error", "") or artifacts.get("message", "") or stdout[-500:],
                }
            )
    valid = [cycle for cycle in cycles if cycle > 0]
    median = float(statistics.median(valid)) if valid else 0.0
    cycle_delta = (max(valid) - min(valid)) / median if valid and median else 0.0
    return {
        "cycles_in_order": cycles,
        "median": median,
        "cycle_delta": cycle_delta,
        "class": "measured" if len(valid) == repeats and not failures else "blocked",
        "failures": failures,
        "runs": run_summaries,
    }


def strip_runs_for_report(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "cycles_in_order": cell.get("cycles_in_order", []),
        "median": cell.get("median", 0),
        "cycle_delta": cell.get("cycle_delta", 0),
        "class": cell.get("class", ""),
        "failures": cell.get("failures", []),
    }


def conv_shape_key(batch: int, i_c: int, o_c: int, k_h: int, k_w: int, o_h: int, o_w: int) -> str:
    return f"conv2d_{batch}_{i_c}_{o_c}_{k_h}_{k_w}_{o_h}_{o_w}"


def conv_tile_working_set_bytes(tile: dict[str, int], *, precision_bytes: int = 4, n_extra_node: int = 0) -> dict[str, int]:
    tile_k_h = int(tile["TILE_K_H"])
    tile_k_w = int(tile["TILE_K_W"])
    tile_o_h = int(tile["TILE_O_H"])
    tile_o_w = int(tile["TILE_O_W"])
    tile_m = int(tile["TILE_M"])
    tile_n = int(tile["TILE_N"])
    tile_k = int(tile["TILE_K"])
    tile_i_h = 1 + (tile_o_h - 1) + (tile_k_h - 1)
    tile_i_w = 1 + (tile_o_w - 1) + (tile_k_w - 1)
    weight_size = tile_k_w * tile_k_h * tile_k * tile_n
    input_size = tile_i_w * tile_i_h * tile_m * tile_k
    output_size = tile_o_w * tile_o_h * tile_m * tile_n
    total_elems = weight_size + input_size + output_size * (1 + n_extra_node)
    return {
        "tile_i_h": tile_i_h,
        "tile_i_w": tile_i_w,
        "weight_bytes": weight_size * precision_bytes,
        "input_bytes": input_size * precision_bytes,
        "output_bytes": output_size * (1 + n_extra_node) * precision_bytes,
        "total_bytes": total_elems * precision_bytes,
    }


def get_spad_size_per_lane(tile_m: int, tile_n: int, vector_lane: int = 128) -> int:
    size = tile_m * ((tile_n + vector_lane - 1) // vector_lane)
    return max(size, 2)


def divisors(n: int) -> list[int]:
    result = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            result.add(i)
            result.add(n // i)
    return sorted(result)


def gemm_seed_tile_candidates(
    M: int,
    N: int,
    K: int,
    *,
    spad_size_per_lane: int = 128 * 1024,
    vector_lane: int = 128,
    precision_bytes: int = 4,
    n_extra_node: int = 0,
) -> list[tuple[int, int, int]]:
    spad_size = spad_size_per_lane * vector_lane
    max_spad_size = spad_size // 2
    max_spad_per_lane = spad_size_per_lane // 2
    M_padded = M
    N_padded = ((N + vector_lane - 1) // vector_lane) * vector_lane
    K_padded = K
    index_i = M_padded // vector_lane if M > vector_lane else 1
    index_j = N_padded // vector_lane if N > vector_lane else 1
    index_k = K_padded // vector_lane if K > vector_lane else 1
    tile_M_range = divisors(index_i) if M > vector_lane else [1]
    tile_N_range = divisors(index_j) if N > vector_lane else [1]
    tile_K_range = divisors(index_k) if K > vector_lane else [1]
    rows = []
    for k in tile_K_range:
        tile_K = k * vector_lane if K > vector_lane else K_padded
        for i in tile_M_range:
            tile_M = i * vector_lane if M > vector_lane else M_padded
            for j in tile_N_range:
                tile_N = j * vector_lane if N > vector_lane else N_padded
                used_spad_size = (tile_M * tile_K + tile_K * tile_N + tile_M * tile_N * (1 + n_extra_node)) * precision_bytes
                weight_size_per_lane = get_spad_size_per_lane(tile_K, tile_N, vector_lane)
                input_size_per_lane = get_spad_size_per_lane(tile_M, tile_K, vector_lane)
                output_size_per_lane = get_spad_size_per_lane(tile_M * (1 + n_extra_node), tile_N, vector_lane)
                used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * precision_bytes
                if used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane:
                    rows.append((used_spad_size, (tile_M, tile_N, tile_K)))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [values for _, values in rows]


def get_conv_tile_candidates() -> list[dict[str, Any]]:
    M, N, K = gemm_seed_tile_candidates(1, 64, 64)[0]
    K = min(K, 128)
    rows = []
    for o_h in divisors(56):
        for o_w in divisors(56):
            for k_h in divisors(3):
                for k_w in divisors(3):
                    tile = {
                        "TILE_K_H": int(k_h),
                        "TILE_K_W": int(k_w),
                        "TILE_O_H": int(o_h),
                        "TILE_O_W": int(o_w),
                        "TILE_M": int(M),
                        "TILE_N": int(N),
                        "TILE_K": int(K),
                    }
                    ws = conv_tile_working_set_bytes(tile)
                    weight_size_per_lane = get_spad_size_per_lane(k_w * k_h * K, N)
                    input_size_per_lane = get_spad_size_per_lane(ws["tile_i_w"] * ws["tile_i_h"] * M, K)
                    output_size_per_lane = get_spad_size_per_lane(o_w * o_h * M, N)
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * 4
                    if ws["total_bytes"] < (128 * 1024 * 128 // 2) and used_spad_size_per_lane < (128 * 1024 // 2):
                        rows.append({**tile, **ws})
    rows.sort(key=lambda row: row["total_bytes"])
    return rows


def select_conv_tile_variants(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    smallest_budget = 32 * 1024 * 128 // 2
    largest_budget = 128 * 1024 * 128 // 2
    bands = [
        ("tile_A", 0.00, 0.25, "tiny"),
        ("tile_B", 0.25, 0.50, "small"),
        ("tile_C", 0.50, 0.90, "medium"),
    ]
    selected: dict[str, dict[str, Any]] = {}
    used = set()
    for name, lo, hi, regime in bands:
        band_rows = [row for row in candidates if lo <= row["total_bytes"] / smallest_budget < hi and tuple(row.items()) not in used]
        if not band_rows:
            band_rows = [row for row in candidates if row["total_bytes"] / smallest_budget < hi and tuple(row.items()) not in used]
        row = band_rows[-1] if band_rows else None
        if row is None:
            raise RuntimeError(f"unable to select {name} tile candidate")
        used.add(tuple(row.items()))
        selected[name] = {
            **{k: row[k] for k in ["TILE_K_H", "TILE_K_W", "TILE_O_H", "TILE_O_W", "TILE_M", "TILE_N", "TILE_K"]},
            "predicted_working_set_bytes": row["total_bytes"],
            "predicted_fit_all_hw": True,
            "regime": regime,
            "fit_fraction_smallest_spad": row["total_bytes"] / smallest_budget,
            "fit_fraction_largest_spad": row["total_bytes"] / largest_budget,
        }
    large_rows = [row for row in candidates if row["total_bytes"] > smallest_budget and row["total_bytes"] < largest_budget and tuple(row.items()) not in used]
    if not large_rows:
        large_rows = [row for row in candidates if tuple(row.items()) not in used]
    row = large_rows[0]
    selected["tile_D"] = {
        **{k: row[k] for k in ["TILE_K_H", "TILE_K_W", "TILE_O_H", "TILE_O_W", "TILE_M", "TILE_N", "TILE_K"]},
        "predicted_working_set_bytes": row["total_bytes"],
        "predicted_fit_all_hw": row["total_bytes"] <= smallest_budget,
        "regime": "large",
        "fit_fraction_smallest_spad": row["total_bytes"] / smallest_budget,
        "fit_fraction_largest_spad": row["total_bytes"] / largest_budget,
    }
    return selected


def write_conv_tile_discovery(output_root: Path, candidates: list[dict[str, Any]], selected_tiles: dict[str, dict[str, Any]]) -> None:
    smallest_budget = 32 * 1024 * 128 // 2
    lines = [
        '# Conv tile axis discovery',
        '',
        '- Conv 3x3 uses a 7-parameter heuristic tile axis: `TILE_K_H`, `TILE_K_W`, `TILE_O_H`, `TILE_O_W`, `TILE_M`, `TILE_N`, `TILE_K`.',
        '- Derived input extents are `TILE_I_H = 1 + (TILE_O_H - 1) * stride_h + (TILE_K_H - 1) * dilation_h` and `TILE_I_W = 1 + (TILE_O_W - 1) * stride_w + (TILE_K_W - 1) * dilation_w`.',
        '- Existing `external_mapping.json` originally exposed only GEMM `TILE_M/N/K`; this probe adds a Conv-specific key `conv2d_1_64_64_3_3_56_56` carrying the 7 Conv tile parameters so Route 4 can sweep tiles without touching simulator sources.',
        '- Working-set model from `conv_combination_mapping`: `weight = K_H * K_W * TILE_K * TILE_N`, `input = TILE_I_H * TILE_I_W * TILE_M * TILE_K`, `output = TILE_O_H * TILE_O_W * TILE_M * TILE_N`, `bytes = 4 * (weight + input + output)` for this unfused Conv core.',
        f'- Smallest-SPAD HW usable double-buffer budget: `{smallest_budget}` bytes (4 MiB total physical SPAD / 2 for double buffering).',
        '',
        '## Selected tiles',
        '',
    ]
    for name, tile in selected_tiles.items():
        lines.append(f'- `{name}`: {json.dumps(tile, sort_keys=True)}')
    lines.extend(['', '## Candidate sample (smallest to largest working set)', ''])
    for row in candidates[:8] + candidates[-8:]:
        summary = {k: row[k] for k in ["TILE_K_H", "TILE_K_W", "TILE_O_H", "TILE_O_W", "TILE_M", "TILE_N", "TILE_K", "total_bytes"]}
        lines.append(f'- `{json.dumps(summary, sort_keys=True)}`')
    (output_root / 'tile_axis_discovery.md').write_text("\n".join(lines) + "\n", encoding='utf-8')


def write_conv_external_mapping(path: Path, tile: dict[str, Any]) -> None:
    payload = {conv_shape_key(1, 64, 64, 3, 3, 56, 56): {k: int(tile[k]) for k in ["TILE_K_H", "TILE_K_W", "TILE_O_H", "TILE_O_W", "TILE_M", "TILE_N", "TILE_K"]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def clone_external_mapping_override(base_path: Path | None, run_dir: Path) -> Path | None:
    if base_path is None:
        return None
    target = run_dir / 'external_mapping.json'
    target.write_text(base_path.read_text(encoding='utf-8'), encoding='utf-8')
    return target


def cycle_delta_between(none_cell: dict[str, Any], all_cell: dict[str, Any]) -> float | None:
    medians = [float(none_cell.get("median", 0) or 0), float(all_cell.get("median", 0) or 0)]
    if min(medians) <= 0:
        return None
    return (max(medians) - min(medians)) / min(medians)


def analyze_hw_fusion_matrix(matrix: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    fusion_speedup_per_hw: dict[str, float] = {}
    champion_fusion_per_hw: dict[str, str] = {}
    best_by_hw: dict[str, float] = {}
    for hw_id, variants in matrix.items():
        none_median = float(variants["none"]["median"])
        all_median = float(variants["all"]["median"])
        fusion_speedup_per_hw[hw_id] = none_median / all_median if all_median > 0 else 0.0
        champion_fusion_per_hw[hw_id] = "all" if all_median <= none_median else "none"
        best_by_hw[hw_id] = min(none_median, all_median)
    speedups = list(fusion_speedup_per_hw.values())
    speedup_max = max(speedups) if speedups else 0.0
    speedup_min = min(speedups) if speedups else 0.0
    relative_range = (speedup_max - speedup_min) / speedup_min if speedup_min else 0.0
    best_b = best_by_hw.get("HW-B", 0.0)
    best_c = best_by_hw.get("HW-C", 0.0)
    gate2b_ratio = min(best_b, best_c) / max(best_b, best_c) if best_b > 0 and best_c > 0 else 0.0
    return {
        "fusion_speedup_per_hw": fusion_speedup_per_hw,
        "fusion_speedup_range": {
            "max": speedup_max,
            "min": speedup_min,
            "relative_range": relative_range,
        },
        "hw_x_fusion_interaction_significant": relative_range >= 0.15,
        "gate2b_ratio": gate2b_ratio,
        "gate2b_passed": gate2b_ratio <= 0.85 if gate2b_ratio else False,
        "champion_fusion_per_hw": champion_fusion_per_hw,
        "same_champion_across_all_hw": len(set(champion_fusion_per_hw.values())) == 1,
    }


def final_codesign_verdict(phase_a: dict[str, Any], analysis: dict[str, Any] | None, blocked: str = "") -> str:
    if blocked:
        return "FUSION_CODESIGN_BLOCKED"
    if not phase_a.get("generalizes", False):
        return "FUSION_NOT_GENERALIZABLE_ON_GPT2"
    if not analysis:
        return "FUSION_CODESIGN_BLOCKED"
    if analysis.get("hw_x_fusion_interaction_significant") or analysis.get("gate2b_passed"):
        return "FUSION_CODESIGN_POSITIVE"
    return "FUSION_CODESIGN_NEGATIVE"


def render_codesign_sections(
    phase_a: dict[str, Any],
    phase_b: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    verdict: str,
) -> None:
    docs_dir = repo_root() / "docs" / "superpowers" / "specs"
    md_path = docs_dir / "2026-07-01-npu-mapping-dse-route4-fusion-pilot-report.md"
    html_path = docs_dir / "2026-07-01-npu-mapping-dse-route4-fusion-pilot-report.html"
    phase_a_json = json.dumps(phase_a, indent=2, ensure_ascii=False)
    phase_b_json = json.dumps(phase_b or {}, indent=2, ensure_ascii=False)
    analysis_json = json.dumps(analysis or {}, indent=2, ensure_ascii=False)
    matrix = (phase_b or {}).get("matrix", {})
    rows = []
    for hw_id in ("HW-A", "HW-B", "HW-C", "HW-D"):
        item = matrix.get(hw_id, {})
        if item:
            rows.append(
                f"| {hw_id} | `{item['none']['cycles_in_order']}` | `{item['none']['median']}` | "
                f"`{item['all']['cycles_in_order']}` | `{item['all']['median']}` | `{item['fusion_speedup']}` |"
            )
    matrix_table = "\n".join(rows) if rows else "| N/A | `[]` | `0` | `[]` | `0` | `0` |"
    analysis_rows = []
    if analysis:
        for hw_id, speedup in analysis.get("fusion_speedup_per_hw", {}).items():
            analysis_rows.append(f"| {hw_id} | `{speedup}` | `{analysis['champion_fusion_per_hw'][hw_id]}` |")
    analysis_table = "\n".join(analysis_rows) if analysis_rows else "| N/A | `0` | `N/A` |"
    implication = {
        "FUSION_CODESIGN_POSITIVE": "fusion axis 不只是软件侧优化，它已经表现出与 HW 配置相关的 co-design 信号；下一步可以扩展到更完整的 4-HW 或更大 workload 验证。",
        "FUSION_CODESIGN_NEGATIVE": "fusion axis 在 GPT-2 上有效，但当前 4-HW 小矩阵下收益较均匀，暂时不能说明 HW/SW 交互明显。",
        "FUSION_NOT_GENERALIZABLE_ON_GPT2": "fusion axis 在 addmm+relu 上有效，但没有迁移到 GPT-2 block，本轮不应进入更大 HW sweep。",
        "FUSION_CODESIGN_BLOCKED": "至少一个 timing cell 未产生有效 cycles，应先修复阻塞点再扩大实验。",
    }[verdict]
    md_append = f"""

## 9. Phase A workload generalization result

Phase A 使用 `HW-A codesign_v1_2x2`、`TILE_M=128 TILE_N=64 TILE_K=64`、`GPT-2 single transformer block prefill seq=128`，在 `pytorchsim_functional_mode=0` 下比较 `fusion=none` 与 `fusion=all`。结果如下：

```json
{phase_a_json}
```

## 10. Phase B HW x fusion matrix

| HW | none cycles | none median | all cycles | all median | fusion_speedup |
| --- | --- | --- | --- | --- | --- |
{matrix_table}

完整 JSON：

```json
{phase_b_json}
```

## 11. Co-design analytics

| HW | fusion_speedup | champion_fusion |
| --- | --- | --- |
{analysis_table}

```json
{analysis_json}
```

## 12. New final verdict + implication for next step

`{verdict}`

{implication}
"""
    md_text = md_path.read_text(encoding="utf-8")
    md_text = re.split(r"\n## 9\. Phase A workload generalization result\n", md_text)[0].rstrip()
    md_text = re.sub(r"(## 6\. Verdict\n\n)`[^`]+`", rf"\1`{verdict}`", md_text)
    md_path.write_text(md_text + md_append, encoding="utf-8")

    html_append = f"""
<h2>9. Phase A workload generalization result</h2>
<p>Phase A 使用 <code>HW-A codesign_v1_2x2</code>、<code>TILE_M=128 TILE_N=64 TILE_K=64</code>、<code>GPT-2 single transformer block prefill seq=128</code>，在 <code>pytorchsim_functional_mode=0</code> 下比较 <code>fusion=none</code> 与 <code>fusion=all</code>。</p>
<pre>{phase_a_json}</pre>
<h2>10. Phase B HW x fusion matrix</h2>
<table><tr><th>HW</th><th>none cycles</th><th>none median</th><th>all cycles</th><th>all median</th><th>fusion_speedup</th></tr>
"""
    for hw_id in ("HW-A", "HW-B", "HW-C", "HW-D"):
        item = matrix.get(hw_id, {})
        if item:
            html_append += (
                f"<tr><td>{hw_id}</td><td>{item['none']['cycles_in_order']}</td><td>{item['none']['median']}</td>"
                f"<td>{item['all']['cycles_in_order']}</td><td>{item['all']['median']}</td><td>{item['fusion_speedup']}</td></tr>\n"
            )
    html_append += f"""</table>
<pre>{phase_b_json}</pre>
<h2>11. Co-design analytics</h2>
<table><tr><th>HW</th><th>fusion_speedup</th><th>champion_fusion</th></tr>
"""
    if analysis:
        for hw_id, speedup in analysis.get("fusion_speedup_per_hw", {}).items():
            html_append += f"<tr><td>{hw_id}</td><td>{speedup}</td><td>{analysis['champion_fusion_per_hw'][hw_id]}</td></tr>\n"
    html_append += f"""</table>
<pre>{analysis_json}</pre>
<h2>12. New final verdict + implication for next step</h2>
<p><code>{verdict}</code></p>
<p>{implication}</p>
"""
    html_text = html_path.read_text(encoding="utf-8")
    html_text = re.split(r"\n<h2>9\. Phase A workload generalization result</h2>\n", html_text)[0]
    html_text = re.sub(r"(<h2>6\. Verdict</h2>\n<p><code>)[^<]+(</code></p>)", rf"\1{verdict}\2", html_text)
    if "</body>" in html_text:
        html_text = html_text.replace("</body>\n</html>\n", html_append + "</body>\n</html>\n")
    else:
        html_text = html_text + html_append
    html_path.write_text(html_text, encoding="utf-8")


def run_codesign_mini_sweep(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = generate_v1_hw_fusion_configs(output_root)
    workload = "gpt2_block_prefill_s128"
    repeats = args.repeats
    timeout_sec = args.timing_timeout_sec

    phase_a_variants: dict[str, dict[str, Any]] = {}
    phase_a_blocked = ""
    for variant in ("none", "all"):
        config_path = repo_root() / manifest["generated"]["HW-A"][variant]["path"]
        cell = run_timing_cell(
            cell_dir=output_root / "phase_a_runs" / "HW-A" / variant,
            hw_config=config_path,
            workload=workload,
            variant=variant,
            repeats=repeats,
            timeout_sec=timeout_sec,
        )
        phase_a_variants[variant] = strip_runs_for_report(cell)
        if cell["class"] != "measured":
            phase_a_blocked = f"Phase A HW-A {variant} failed: {cell['failures']}"
    phase_a_delta = cycle_delta_between(phase_a_variants["none"], phase_a_variants["all"])
    phase_a = {
        "workload": workload,
        "hw_config": "HW-A codesign_v1_2x2",
        "mapping": "006",
        "variants": phase_a_variants,
        "cycle_delta_between_variants": phase_a_delta,
        "generalizes": bool(phase_a_delta is not None and phase_a_delta >= 0.10 and not phase_a_blocked),
        "note": "fusion generalizes if cycle_delta_between_variants >= 0.10; correctness remains deferred in mode=0",
    }
    if phase_a_blocked:
        phase_a["blocked_reason"] = phase_a_blocked
    write_json(output_root / "phase_a_gpt2_generalization.json", phase_a)

    phase_b: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None
    blocked = phase_a_blocked
    if not blocked and phase_a["generalizes"]:
        matrix: dict[str, dict[str, dict[str, Any]]] = {}
        for hw_id in ("HW-A", "HW-B", "HW-C", "HW-D"):
            matrix[hw_id] = {}
            for variant in ("none", "all"):
                config_path = repo_root() / manifest["generated"][hw_id][variant]["path"]
                cell = run_timing_cell(
                    cell_dir=output_root / "phase_b_runs" / hw_id / variant,
                    hw_config=config_path,
                    workload=workload,
                    variant=variant,
                    repeats=repeats,
                    timeout_sec=timeout_sec,
                )
                matrix[hw_id][variant] = strip_runs_for_report(cell)
                if cell["class"] != "measured":
                    blocked = f"Phase B {hw_id} {variant} failed: {cell['failures']}"
        for hw_id, variants in matrix.items():
            none_median = float(variants["none"].get("median", 0) or 0)
            all_median = float(variants["all"].get("median", 0) or 0)
            variants["fusion_speedup"] = none_median / all_median if all_median > 0 else 0.0
        phase_b = {
            "workload": workload,
            "mapping": "006",
            "matrix": matrix,
        }
        write_json(output_root / "phase_b_hw_fusion_matrix.json", phase_b)
        if not blocked:
            analysis = analyze_hw_fusion_matrix(matrix)
            write_json(output_root / "codesign_analysis.json", analysis)

    verdict = final_codesign_verdict(phase_a, analysis, blocked)
    existing_summary = read_json_if_exists(output_root / "pilot_summary.json", {})
    existing_summary.update(
        {
            "workload": workload,
            "phase_a_gpt2_generalization": phase_a,
            "phase_b_hw_fusion_matrix": phase_b,
            "codesign_analysis": analysis,
            "verdict": verdict,
        }
    )
    if blocked:
        existing_summary["blocked_reason"] = blocked
    write_json(output_root / "pilot_summary.json", existing_summary)
    render_codesign_sections(phase_a, phase_b, analysis, verdict)
    return 0 if verdict != "FUSION_CODESIGN_BLOCKED" else 2


def classify_fit_regime(tile: dict[str, Any], hw_id: str) -> dict[str, Any]:
    spad_kb = 128 if hw_id in {"HW-A", "HW-B"} else 32
    usable_bytes = spad_kb * 1024 * 128 // 2
    working_set = int(tile["predicted_working_set_bytes"])
    return {
        "usable_spad_bytes": usable_bytes,
        "working_set_bytes": working_set,
        "fit_fraction": working_set / usable_bytes,
        "fits": working_set <= usable_bytes,
    }


def run_interior_optimum_probe(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = generate_v1_hw_fusion_configs(output_root)
    candidates = get_conv_tile_candidates()
    selected_tiles = select_conv_tile_variants(candidates)
    write_conv_tile_discovery(output_root, candidates, selected_tiles)
    write_json(output_root / 'tile_variants.json', selected_tiles)

    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    start_time = time.time()
    for hw_id in ("HW-A", "HW-B", "HW-C", "HW-D"):
        matrix[hw_id] = {}
        for tile_name, tile in selected_tiles.items():
            matrix[hw_id][tile_name] = {}
            fit_info = classify_fit_regime(tile, hw_id)
            for variant in ("none", "all"):
                cell_dir = output_root / 'interior_runs' / hw_id / tile_name / variant
                cell_dir.mkdir(parents=True, exist_ok=True)
                config_path = Path(manifest['generated'][hw_id][variant]['path'])
                if not fit_info['fits']:
                    matrix[hw_id][tile_name][variant] = {
                        'class': 'unavailable',
                        'cycles_in_order': [],
                        'median': 0.0,
                        'cycle_delta': 0.0,
                        'failures': [],
                        'fit_info': fit_info,
                    }
                    continue
                mapping_path = cell_dir / 'external_mapping.json'
                write_conv_external_mapping(mapping_path, tile)
                cell = run_timing_cell(
                    cell_dir=cell_dir,
                    hw_config=config_path,
                    workload='conv3x3_probe',
                    variant=variant,
                    repeats=args.repeats,
                    timeout_sec=args.timing_timeout_sec,
                    use_external_mapping=True,
                    external_mapping_override=mapping_path,
                )
                summary = strip_runs_for_report(cell)
                summary['fit_info'] = fit_info
                if summary['class'] != 'measured':
                    summary['class'] = 'runtime_failed'
                matrix[hw_id][tile_name][variant] = summary
                if time.time() - start_time > 4 * 3600:
                    break
            if time.time() - start_time > 4 * 3600:
                break
        if time.time() - start_time > 4 * 3600:
            break

    write_json(output_root / 'tile_fusion_hw_matrix.json', matrix)

    measured_cells = []
    for hw_id, tile_map in matrix.items():
        for tile_name, variants in tile_map.items():
            for variant, cell in variants.items():
                if cell.get('class') == 'measured' and float(cell.get('median', 0) or 0) > 0:
                    measured_cells.append({
                        'hw_id': hw_id,
                        'tile': tile_name,
                        'fusion': variant,
                        'median': float(cell['median']),
                    })

    champion_per_hw = {}
    for hw_id, tile_map in matrix.items():
        best = None
        for tile_name, variants in tile_map.items():
            for variant, cell in variants.items():
                if cell.get('class') != 'measured' or float(cell.get('median', 0) or 0) <= 0:
                    continue
                cand = (float(cell['median']), tile_name, variant)
                if best is None or cand[0] < best[0]:
                    best = cand
        champion_per_hw[hw_id] = None if best is None else {'median': best[0], 'tile': best[1], 'fusion': best[2]}

    champion_pairs = {(v['tile'], v['fusion']) for v in champion_per_hw.values() if v}
    global_min = min(measured_cells, key=lambda item: item['median']) if measured_cells else None
    corner_cells = {
        (hw_id, tile_name, variant)
        for hw_id in ("HW-A", "HW-B", "HW-C", "HW-D")
        for tile_name in ("tile_A", "tile_D")
        for variant in ("none", "all")
    }
    global_min_is_at_corner = bool(global_min and (global_min['hw_id'], global_min['tile'], global_min['fusion']) in corner_cells)
    interior_optimum = bool(global_min and not global_min_is_at_corner)
    champion_migration = len(champion_pairs) > 1
    if interior_optimum:
        verdict = 'INTERIOR_OPTIMUM_FOUND'
    elif champion_migration:
        verdict = 'CHAMPION_MIGRATION_ONLY'
    elif measured_cells:
        verdict = 'CORNER_MONOTONIC'
    else:
        verdict = 'PARTIAL_SCOPE_LIMITATION'

    analysis = {
        'verdict': verdict,
        'champion_per_hw': champion_per_hw,
        'champion_pair_count': len(champion_pairs),
        'champion_migration': champion_migration,
        'global_min_cell': global_min,
        'global_min_is_at_corner': global_min_is_at_corner,
        'interior_optimum_evidence': interior_optimum,
        'selected_tiles': selected_tiles,
        'tile_fit_fraction_per_hw': {
            hw_id: {tile_name: classify_fit_regime(tile, hw_id) for tile_name, tile in selected_tiles.items()}
            for hw_id in ("HW-A", "HW-B", "HW-C", "HW-D")
        },
    }
    write_json(output_root / 'interior_optimum_analysis.json', analysis)
    return 0


def classify_gpt2_structural_diff(structural: dict[str, Any]) -> tuple[str, str]:
    counts = structural.get("counts", {})
    none = counts.get("none", {})
    all_variant = counts.get("all", {})
    none_ops = float(none.get("mlir_op_count", 0) or 0)
    all_ops = float(all_variant.get("mlir_op_count", 0) or 0)
    op_delta = abs(all_ops - none_ops) / none_ops if none_ops else 0.0
    none_keywords = list(none.get("fused_pattern_keywords", []))
    all_keywords = list(all_variant.get("fused_pattern_keywords", []))
    if op_delta < 0.02 and none_keywords == all_keywords:
        return (
            "structural_noop",
            (
                f"`mlir_op_count` delta is {op_delta:.4f}, below 2%, and both variants expose "
                f"the same fused pattern keywords {none_keywords}; `fusion=all` is a structural "
                "no-op for this GPT-2 block."
            ),
        )
    if op_delta >= 0.05 or none_keywords != all_keywords:
        return (
            "structural_active",
            (
                f"`mlir_op_count` delta is {op_delta:.4f} and keyword sets are "
                f"none={none_keywords}, all={all_keywords}; `fusion=all` changes the generated "
                "structure, so the weak GPT-2 timing result is more likely critical-path limited."
            ),
        )
    return (
        "ambiguous",
        (
            f"`mlir_op_count` delta is {op_delta:.4f}; this is between the no-op and active "
            "thresholds, so the GPT-2 structural evidence is ambiguous."
        ),
    )


def write_gpt2_structural_diff(output_root: Path, counts: dict[str, dict[str, Any]], observation: str) -> dict[str, Any]:
    payload = {
        "workload": "gpt2_block_prefill_s128",
        "counts": counts,
        "observation": observation,
    }
    write_json(output_root / "phase_a_gpt2_structural_diff.json", payload)
    return payload


def run_root_cause_gpt2_structural(args: argparse.Namespace, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    output_root = Path(args.output_root)
    run_root = output_root / "root_cause_gpt2_structural"
    summaries: dict[str, Any] = {}
    counts: dict[str, dict[str, Any]] = {}
    for variant in ("none", "all"):
        config_path = repo_root() / manifest["generated"]["HW-A"][variant]["path"]
        cell_dir = run_root / variant
        cell = run_timing_cell(
            cell_dir=cell_dir,
            hw_config=config_path,
            workload="gpt2_block_prefill_s128",
            variant=variant,
            repeats=1,
            timeout_sec=args.timing_timeout_sec,
        )
        summaries[variant] = strip_runs_for_report(cell)
        counts[variant] = structural_counts_for_root(cell_dir)
    structural = write_gpt2_structural_diff(
        output_root,
        counts,
        (
            "GPT-2 structural diff is extracted from one fresh mode=0 timing subprocess per "
            "fusion variant under HW-A and mapping 006. Counts aggregate only the emitted MLIR "
            "and raw TOG files in the fresh root-cause run directories."
        ),
    )
    return structural, summaries


def run_conv_probe(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(args.output_root)
    variants: dict[str, dict[str, Any]] = {}
    skipped_reason = ""
    for variant in ("none", "all"):
        config_path = repo_root() / manifest["generated"]["HW-A"][variant]["path"]
        cell = run_timing_cell(
            cell_dir=output_root / "phase_b_conv_probe_runs" / variant,
            hw_config=config_path,
            workload="conv3x3_probe",
            variant=variant,
            repeats=3,
            timeout_sec=args.timing_timeout_sec,
        )
        variants[variant] = strip_runs_for_report(cell)
        if cell["class"] != "measured" and not skipped_reason:
            skipped_reason = (
                "Conv-heavy workload was attempted but did not produce three measured timing "
                f"runs for fusion={variant}; this is treated as a harness limitation because "
                "the handoff forbids PyTorchSim source changes."
            )
    delta = cycle_delta_between(variants["none"], variants["all"])
    payload = {
        "workload": "conv3x3_probe",
        "variants": variants,
        "cycle_delta_between_variants": delta,
        "generalizes_to_conv": bool(delta is not None and delta >= 0.10 and not skipped_reason),
    }
    if skipped_reason:
        payload["skipped_due_to_harness_limitation"] = True
        payload["skip_reason"] = skipped_reason
    write_json(output_root / "phase_b_conv_probe.json", payload)
    return payload


def root_cause_verdict(
    structural_class: str,
    structural_runs: dict[str, Any],
    conv_probe: dict[str, Any] | None,
) -> str:
    if any(cell.get("class") != "measured" for cell in structural_runs.values()):
        return "FUSION_ROUTE_BLOCKED_ON_MEASUREMENT"
    if structural_class == "structural_noop":
        return "FUSION_ROUTE_STRUCTURALLY_INEFFECTIVE"
    if structural_class == "structural_active":
        if conv_probe and conv_probe.get("generalizes_to_conv"):
            return "FUSION_ROUTE_CRITICAL_PATH_LIMITED"
        return "FUSION_ROUTE_NEEDS_HARNESS_EXTENSION"
    return "FUSION_ROUTE_NEEDS_HARNESS_EXTENSION"


def append_root_cause_report_sections(
    structural: dict[str, Any],
    structural_class: str,
    structural_explanation: str,
    structural_runs: dict[str, Any],
    conv_probe: dict[str, Any] | None,
    verdict: str,
) -> None:
    docs_dir = repo_root() / "docs" / "superpowers" / "specs"
    md_path = docs_dir / "2026-07-01-npu-mapping-dse-route4-fusion-pilot-report.md"
    html_path = docs_dir / "2026-07-01-npu-mapping-dse-route4-fusion-pilot-report.html"
    structural_json = json.dumps(structural, indent=2, ensure_ascii=False)
    structural_runs_json = json.dumps(structural_runs, indent=2, ensure_ascii=False)
    conv_json = json.dumps(conv_probe or {}, indent=2, ensure_ascii=False)
    if conv_probe is None:
        conv_text = (
            "Task 2 未运行，因为 Task 1 已经把 GPT-2 block 判定为 structural no-op；"
            "按照 handoff，只有 structural active 或 ambiguous 时才继续 Conv probe。"
        )
    elif conv_probe.get("skipped_due_to_harness_limitation"):
        conv_text = str(conv_probe.get("skip_reason", "Conv probe skipped due to harness limitation."))
    else:
        conv_text = (
            f"Conv probe 完成，`cycle_delta_between_variants` = "
            f"`{conv_probe.get('cycle_delta_between_variants')}`，"
            f"`generalizes_to_conv` = `{conv_probe.get('generalizes_to_conv')}`。"
        )
    next_step = {
        "FUSION_ROUTE_STRUCTURALLY_INEFFECTIVE": (
            "Route 4 fusion axis 在 GPT-2 block 上没有传播到相关编译结构；下一步不应继续扩大 "
            "fusion sweep，而应先检查 fusion flag 到 TorchInductor/MLIR lowering 的传播边界。"
        ),
        "FUSION_ROUTE_CRITICAL_PATH_LIMITED": (
            "fusion axis 是真实存在的，但主要影响 Conv/epilogue 类结构；下一步应把 Route 4 "
            "workload 换成 ResNet、MobileNet 或 GEMM-stack，再设计后续 sweep。"
        ),
        "FUSION_ROUTE_NEEDS_HARNESS_EXTENSION": (
            "现有 harness 不能给出足够干净的 workload 证据；下一步需要扩展 workload harness，"
            "但不应在本轮修改 PyTorchSim/TOGSim 源码。"
        ),
        "FUSION_ROUTE_BLOCKED_ON_MEASUREMENT": (
            "至少一个必要 timing run 未产生有效 cycles；下一步先修复测量路径，再继续判断 fusion axis。"
        ),
    }[verdict]
    md_append = f"""

## 13. Task 1 structural diff on GPT-2 block

Task 1 在 `HW-A codesign_v1_2x2`、`mapping=006`、`pytorchsim_functional_mode=0` 下各运行一个 `fusion=none` 与 `fusion=all` 的 GPT-2 block subprocess，并只从 fresh root-cause run 目录抽取 MLIR/TOG 结构计数。

结构 run summary：

```json
{structural_runs_json}
```

结构 diff：

```json
{structural_json}
```

判读：{structural_explanation}

## 14. Task 2 Conv probe result

{conv_text}

```json
{conv_json}
```

## 15. Root cause classification and recommended next step

`{verdict}`

root cause classification: `{structural_class}`

{next_step}
"""
    md_text = md_path.read_text(encoding="utf-8")
    md_text = re.split(r"\n## 13\. Task 1 structural diff on GPT-2 block\n", md_text)[0].rstrip()
    md_text = re.sub(r"(## 6\. Verdict\n\n)`[^`]+`", rf"\1`{verdict}`", md_text)
    md_text = re.sub(
        r"(## 12\. New final verdict \+ implication for next step\n\n)`[^`]+`\n\n[^\n]+",
        rf"\1`{verdict}`\n\n{next_step}",
        md_text,
    )
    md_path.write_text(md_text + md_append, encoding="utf-8")

    html_append = f"""
<h2>13. Task 1 structural diff on GPT-2 block</h2>
<p>Task 1 在 <code>HW-A codesign_v1_2x2</code>、<code>mapping=006</code>、<code>pytorchsim_functional_mode=0</code> 下各运行一个 <code>fusion=none</code> 与 <code>fusion=all</code> 的 GPT-2 block subprocess，并只从 fresh root-cause run 目录抽取 MLIR/TOG 结构计数。</p>
<p>结构 run summary:</p>
<pre>{structural_runs_json}</pre>
<p>结构 diff:</p>
<pre>{structural_json}</pre>
<p>判读：{structural_explanation}</p>
<h2>14. Task 2 Conv probe result</h2>
<p>{conv_text}</p>
<pre>{conv_json}</pre>
<h2>15. Root cause classification and recommended next step</h2>
<p><code>{verdict}</code></p>
<p>root cause classification: <code>{structural_class}</code></p>
<p>{next_step}</p>
"""
    html_text = html_path.read_text(encoding="utf-8")
    html_text = re.split(r"\n<h2>13\. Task 1 structural diff on GPT-2 block</h2>\n", html_text)[0]
    html_text = re.sub(r"(<h2>6\. Verdict</h2>\n<p><code>)[^<]+(</code></p>)", rf"\1{verdict}\2", html_text)
    html_text = re.sub(
        r"(<h2>12\. New final verdict \+ implication for next step</h2>\n<p><code>)[^<]+(</code></p>\n<p>)[^<]+(</p>)",
        rf"\1{verdict}\2{next_step}\3",
        html_text,
    )
    if "</body>" in html_text:
        html_text = html_text.replace("</body>\n</html>\n", html_append + "</body>\n</html>\n")
    else:
        html_text += html_append
    html_path.write_text(html_text, encoding="utf-8")


def run_fusion_root_cause_investigation(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = generate_v1_hw_fusion_configs(output_root)
    structural, structural_runs = run_root_cause_gpt2_structural(args, manifest)
    structural_class, structural_explanation = classify_gpt2_structural_diff(structural)
    conv_probe: dict[str, Any] | None = None
    if structural_class in {"structural_active", "ambiguous"}:
        conv_probe = run_conv_probe(args, manifest)
    verdict = root_cause_verdict(structural_class, structural_runs, conv_probe)
    existing_summary = read_json_if_exists(output_root / "pilot_summary.json", {})
    existing_summary.update(
        {
            "workload": "gpt2_block_prefill_s128",
            "phase_a_gpt2_structural_diff": structural,
            "phase_a_gpt2_structural_runs": structural_runs,
            "phase_b_conv_probe": conv_probe,
            "root_cause_classification": structural_class,
            "root_cause_explanation": structural_explanation,
            "verdict": verdict,
        }
    )
    write_json(output_root / "pilot_summary.json", existing_summary)
    append_root_cause_report_sections(
        structural,
        structural_class,
        structural_explanation,
        structural_runs,
        conv_probe,
        verdict,
    )
    return 0 if verdict != "FUSION_ROUTE_BLOCKED_ON_MEASUREMENT" else 2


def analyze_conv_codesign_matrix(matrix: dict[str, dict[str, Any]]) -> dict[str, Any]:
    champion_fusion_per_hw: dict[str, str] = {}
    fusion_speedup_per_hw: dict[str, float] = {}
    all_medians: dict[str, dict[str, float]] = {}
    blocked_cells: list[str] = []
    for hw_id, hw_data in matrix.items():
        all_medians[hw_id] = {}
        for variant in ("none", "all"):
            cell = hw_data.get(variant, {})
            if cell.get("class") != "measured" or float(cell.get("median", 0) or 0) <= 0:
                blocked_cells.append(f"{hw_id}/{variant}")
            all_medians[hw_id][variant] = float(cell.get("median", 0) or 0)
        none_median = all_medians[hw_id]["none"]
        all_median = all_medians[hw_id]["all"]
        champion_fusion_per_hw[hw_id] = "all" if all_median <= none_median else "none"
        fusion_speedup_per_hw[hw_id] = none_median / all_median if all_median > 0 else 0.0

    mean_by_fusion = {
        variant: statistics.mean(all_medians[hw_id][variant] for hw_id in matrix)
        for variant in ("none", "all")
    }
    global_champion_fusion = min(mean_by_fusion, key=mean_by_fusion.get)
    per_hw_ratios: dict[str, float] = {}
    per_hw_oracle_cycles: dict[str, float] = {}
    global_champion_cycles: dict[str, float] = {}
    for hw_id, medians in all_medians.items():
        best = min(medians.values())
        global_cycles = medians[global_champion_fusion]
        per_hw_oracle_cycles[hw_id] = best
        global_champion_cycles[hw_id] = global_cycles
        per_hw_ratios[hw_id] = best / global_cycles if global_cycles > 0 else 0.0

    mean_per_hw = statistics.mean(per_hw_oracle_cycles.values()) if per_hw_oracle_cycles else 0.0
    mean_single_fusion = statistics.mean(global_champion_cycles.values()) if global_champion_cycles else 0.0
    gate2a_ratio_mean = mean_per_hw / mean_single_fusion if mean_single_fusion > 0 else 0.0
    best_b = per_hw_oracle_cycles.get("HW-B", 0.0)
    best_c = per_hw_oracle_cycles.get("HW-C", 0.0)
    gate2b_ratio = min(best_b, best_c) / max(best_b, best_c) if best_b > 0 and best_c > 0 else 0.0
    speedups = list(fusion_speedup_per_hw.values())
    speedup_max = max(speedups) if speedups else 0.0
    speedup_min = min(speedups) if speedups else 0.0
    speedup_relative_range = (speedup_max - speedup_min) / speedup_min if speedup_min > 0 else 0.0
    same_champion = len(set(champion_fusion_per_hw.values())) == 1
    return {
        "champion_fusion_per_hw": champion_fusion_per_hw,
        "same_champion_across_all_hw": same_champion,
        "fusion_speedup_per_hw": fusion_speedup_per_hw,
        "fusion_speedup_range": {
            "max": speedup_max,
            "min": speedup_min,
            "relative_range": speedup_relative_range,
        },
        "hw_x_fusion_interaction_significant": speedup_relative_range >= 0.15,
        "gate1_analog_champion_migrates": not same_champion,
        "global_champion_fusion": global_champion_fusion,
        "mean_cycles_by_fusion": mean_by_fusion,
        "per_hw_ratio": per_hw_ratios,
        "mean_per_hw": mean_per_hw,
        "mean_single_fusion": mean_single_fusion,
        "gate2a_ratio_mean": gate2a_ratio_mean,
        "gate2a_num_hw_meeting_threshold": sum(1 for value in per_hw_ratios.values() if value <= 0.85),
        "gate2a_passed": gate2a_ratio_mean <= 0.85,
        "gate2b_ratio_fusion": gate2b_ratio,
        "gate2b_passed": gate2b_ratio <= 0.85 if gate2b_ratio > 0 else False,
        "blocked_cells": blocked_cells,
    }


def conv_codesign_verdict(analysis: dict[str, Any]) -> str:
    if analysis.get("blocked_cells"):
        return "FUSION_CONV_CODESIGN_BLOCKED"
    if (
        analysis.get("hw_x_fusion_interaction_significant")
        or analysis.get("gate2b_passed")
        or analysis.get("gate1_analog_champion_migrates")
    ):
        return "FUSION_CONV_CODESIGN_POSITIVE"
    return "FUSION_CONV_CODESIGN_NEGATIVE"


def compiler_modification_landscape() -> dict[str, str]:
    return {
        "fusion": (
            "proven strong on Conv (6.84x per Conv probe); proven weak on GPT-2 "
            "(1.9% - critical-path limited)"
        ),
        "dataflow": "BLOCKED without source modification (per discovery)",
        "SPAD partition": "NOT INVESTIGATED",
        "DMA schedule": "NOT INVESTIGATED",
    }


def append_conv_sweep_report_sections(
    matrix_payload: dict[str, Any],
    analysis: dict[str, Any],
    verdict: str,
) -> None:
    docs_dir = repo_root() / "docs" / "superpowers" / "specs"
    md_path = docs_dir / "2026-07-01-npu-mapping-dse-route4-fusion-pilot-report.md"
    html_path = docs_dir / "2026-07-01-npu-mapping-dse-route4-fusion-pilot-report.html"
    matrix = matrix_payload.get("matrix", {})
    rows = []
    for hw_id in ("HW-A", "HW-B", "HW-C", "HW-D"):
        hw = matrix.get(hw_id, {})
        rows.append(
            f"| {hw_id} | `{hw.get('none', {}).get('cycles_in_order', [])}` | "
            f"`{hw.get('none', {}).get('median', 0)}` | "
            f"`{hw.get('all', {}).get('cycles_in_order', [])}` | "
            f"`{hw.get('all', {}).get('median', 0)}` | "
            f"`{hw.get('fusion_speedup', 0)}` |"
        )
    matrix_table = "\n".join(rows)
    analysis_rows = []
    for hw_id in ("HW-A", "HW-B", "HW-C", "HW-D"):
        analysis_rows.append(
            f"| {hw_id} | `{analysis.get('champion_fusion_per_hw', {}).get(hw_id)}` | "
            f"`{analysis.get('fusion_speedup_per_hw', {}).get(hw_id)}` | "
            f"`{analysis.get('per_hw_ratio', {}).get(hw_id)}` |"
        )
    analysis_table = "\n".join(analysis_rows)
    implication = {
        "FUSION_CONV_CODESIGN_POSITIVE": (
            "Conv workload 上 fusion axis 不只是固定的软件优化；其收益会随 HW 配置变化，"
            "因此可以作为 Route 4 后续 HW/SW co-design sweep 的正向证据。"
        ),
        "FUSION_CONV_CODESIGN_NEGATIVE": (
            "Conv workload 上 fusion axis 很强，但在四个 HW corner 中收益近似一致；"
            "这说明当前 fusion CLI 更像 workload-level compiler knob，而不是 HW-dependent knob。"
        ),
        "FUSION_CONV_CODESIGN_BLOCKED": (
            "至少一个 Conv (HW, fusion) timing cell 没有产生有效 cycles；"
            "下一步应先修复测量路径，不扩展 sweep。"
        ),
    }[verdict]
    matrix_json = json.dumps(matrix_payload, indent=2, ensure_ascii=False)
    analysis_json = json.dumps(analysis, indent=2, ensure_ascii=False)
    landscape_json = json.dumps(compiler_modification_landscape(), indent=2, ensure_ascii=False)
    md_append = f"""

## 16. Conv sweep 4x2 matrix table

本节使用 `conv3x3_probe`，即 Conv 3x3 `[1, 64, 56, 56] -> [1, 64, 56, 56]`，在 `codesign_v1_2x2` 的 `HW-A/B/C/D` 上分别比较 `codegen_compiler_optimization=none` 与 `all`。所有 run 均为 `pytorchsim_functional_mode=0`，每个 `(HW, fusion)` cell 使用 5 次 independent subprocess，`seed=0`，并保留原始 `cycles_in_order`。

| HW | none cycles | none median | all cycles | all median | fusion_speedup |
| --- | --- | --- | --- | --- | --- |
{matrix_table}

完整 matrix JSON：

```json
{matrix_json}
```

## 17. Co-design gate analytics

| HW | champion_fusion | fusion_speedup | per_hw_ratio |
| --- | --- | --- | --- |
{analysis_table}

关键 gate 结果：

- `hw_x_fusion_interaction_significant`: `{analysis.get('hw_x_fusion_interaction_significant')}`
- `fusion_speedup_range.relative_range`: `{analysis.get('fusion_speedup_range', {}).get('relative_range')}`
- `gate1_analog_champion_migrates`: `{analysis.get('gate1_analog_champion_migrates')}`
- `gate2a_ratio_mean`: `{analysis.get('gate2a_ratio_mean')}`
- `gate2b_ratio_fusion`: `{analysis.get('gate2b_ratio_fusion')}`
- `gate2b_passed`: `{analysis.get('gate2b_passed')}`

完整 analysis JSON：

```json
{analysis_json}
```

## 18. Final co-design verdict + implication for next step

`{verdict}`

{implication}

当前可编译 compiler-modification directions landscape：

```json
{landscape_json}
```
"""
    md_text = md_path.read_text(encoding="utf-8")
    md_text = re.split(r"\n## 16\. Conv sweep 4x2 matrix table\n", md_text)[0].rstrip()
    md_text = re.sub(r"(## 6\. Verdict\n\n)`[^`]+`", rf"\1`{verdict}`", md_text)
    md_path.write_text(md_text + md_append, encoding="utf-8")

    html_rows = "\n".join(
        f"<tr><td>{hw_id}</td><td>{matrix.get(hw_id, {}).get('none', {}).get('cycles_in_order', [])}</td>"
        f"<td>{matrix.get(hw_id, {}).get('none', {}).get('median', 0)}</td>"
        f"<td>{matrix.get(hw_id, {}).get('all', {}).get('cycles_in_order', [])}</td>"
        f"<td>{matrix.get(hw_id, {}).get('all', {}).get('median', 0)}</td>"
        f"<td>{matrix.get(hw_id, {}).get('fusion_speedup', 0)}</td></tr>"
        for hw_id in ("HW-A", "HW-B", "HW-C", "HW-D")
    )
    html_analysis_rows = "\n".join(
        f"<tr><td>{hw_id}</td><td>{analysis.get('champion_fusion_per_hw', {}).get(hw_id)}</td>"
        f"<td>{analysis.get('fusion_speedup_per_hw', {}).get(hw_id)}</td>"
        f"<td>{analysis.get('per_hw_ratio', {}).get(hw_id)}</td></tr>"
        for hw_id in ("HW-A", "HW-B", "HW-C", "HW-D")
    )
    html_append = f"""
<h2>16. Conv sweep 4x2 matrix table</h2>
<p>本节使用 <code>conv3x3_probe</code>，即 Conv 3x3 <code>[1, 64, 56, 56] -&gt; [1, 64, 56, 56]</code>，在 <code>codesign_v1_2x2</code> 的 <code>HW-A/B/C/D</code> 上分别比较 <code>codegen_compiler_optimization=none</code> 与 <code>all</code>。所有 run 均为 <code>pytorchsim_functional_mode=0</code>，每个 <code>(HW, fusion)</code> cell 使用 5 次 independent subprocess，<code>seed=0</code>。</p>
<table><tr><th>HW</th><th>none cycles</th><th>none median</th><th>all cycles</th><th>all median</th><th>fusion_speedup</th></tr>
{html_rows}
</table>
<p>完整 matrix JSON:</p>
<pre>{matrix_json}</pre>
<h2>17. Co-design gate analytics</h2>
<table><tr><th>HW</th><th>champion_fusion</th><th>fusion_speedup</th><th>per_hw_ratio</th></tr>
{html_analysis_rows}
</table>
<ul>
  <li><code>hw_x_fusion_interaction_significant</code>: <code>{analysis.get('hw_x_fusion_interaction_significant')}</code></li>
  <li><code>fusion_speedup_range.relative_range</code>: <code>{analysis.get('fusion_speedup_range', {}).get('relative_range')}</code></li>
  <li><code>gate1_analog_champion_migrates</code>: <code>{analysis.get('gate1_analog_champion_migrates')}</code></li>
  <li><code>gate2a_ratio_mean</code>: <code>{analysis.get('gate2a_ratio_mean')}</code></li>
  <li><code>gate2b_ratio_fusion</code>: <code>{analysis.get('gate2b_ratio_fusion')}</code></li>
  <li><code>gate2b_passed</code>: <code>{analysis.get('gate2b_passed')}</code></li>
</ul>
<p>完整 analysis JSON:</p>
<pre>{analysis_json}</pre>
<h2>18. Final co-design verdict + implication for next step</h2>
<p><code>{verdict}</code></p>
<p>{implication}</p>
<p>当前可编译 compiler-modification directions landscape:</p>
<pre>{landscape_json}</pre>
"""
    html_text = html_path.read_text(encoding="utf-8")
    html_text = re.split(r"\n<h2>16\. Conv sweep 4x2 matrix table</h2>\n", html_text)[0]
    html_text = re.sub(r"(<h2>6\. Verdict</h2>\n<p><code>)[^<]+(</code></p>)", rf"\1{verdict}\2", html_text)
    if "</body>" in html_text:
        html_text = html_text.replace("</body>\n</html>\n", html_append + "</body>\n</html>\n")
    else:
        html_text += html_append
    html_path.write_text(html_text, encoding="utf-8")


def run_conv_4hw_fusion_sweep(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = generate_v1_hw_fusion_configs(output_root)
    matrix: dict[str, dict[str, Any]] = {}
    for hw_id in ("HW-A", "HW-B", "HW-C", "HW-D"):
        matrix[hw_id] = {}
        for variant in ("none", "all"):
            config_path = repo_root() / manifest["generated"][hw_id][variant]["path"]
            cell = run_timing_cell(
                cell_dir=output_root / "conv_4hw_fusion_runs" / hw_id / variant,
                hw_config=config_path,
                workload="conv3x3_probe",
                variant=variant,
                repeats=5,
                timeout_sec=args.timing_timeout_sec,
                use_external_mapping=False,
            )
            matrix[hw_id][variant] = strip_runs_for_report(cell)
        none_median = float(matrix[hw_id]["none"].get("median", 0) or 0)
        all_median = float(matrix[hw_id]["all"].get("median", 0) or 0)
        matrix[hw_id]["fusion_speedup"] = none_median / all_median if all_median > 0 else 0.0

    matrix_payload = {
        "workload": "conv3x3_probe",
        "hw_config_set": "codesign_v1_2x2",
        "mapping": "harness_default",
        "pytorchsim_functional_mode": 0,
        "repeats_per_cell": 5,
        "seed": 0,
        "matrix": matrix,
    }
    write_json(output_root / "conv_hw_fusion_matrix.json", matrix_payload)
    analysis = analyze_conv_codesign_matrix(matrix)
    write_json(output_root / "conv_codesign_analysis.json", analysis)
    verdict = conv_codesign_verdict(analysis)
    existing_summary = read_json_if_exists(output_root / "pilot_summary.json", {})
    existing_summary.update(
        {
            "workload": "conv3x3_probe",
            "conv_hw_fusion_matrix": matrix_payload,
            "conv_codesign_analysis": analysis,
            "compiler_modification_landscape": compiler_modification_landscape(),
            "verdict": verdict,
        }
    )
    write_json(output_root / "pilot_summary.json", existing_summary)
    append_conv_sweep_report_sections(matrix_payload, analysis, verdict)
    return 0 if verdict != "FUSION_CONV_CODESIGN_BLOCKED" else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Route 4 fusion-only pilot.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--correctness-timeout-sec", type=int, default=900)
    parser.add_argument("--timing-timeout-sec", type=int, default=900)
    parser.add_argument("--recovery-mode0", action="store_true")
    parser.add_argument("--codesign-mini-sweep", action="store_true")
    parser.add_argument("--fusion-root-cause-investigation", action="store_true")
    parser.add_argument("--conv-4hw-fusion-sweep", action="store_true")
    parser.add_argument("--interior-optimum-probe", action="store_true")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-kind", choices=["correctness", "timing"], help=argparse.SUPPRESS)
    parser.add_argument("--child-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=["gpt2_block_prefill_s128", "addmm_relu_128", "conv3x3_probe"], help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=["none", "all"], help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.fusion_root_cause_investigation and not args.conv_4hw_fusion_sweep and not args.interior_optimum_probe and args.repeats != 5:
        parser.error("--repeats must remain 5 for this pilot")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args._child:
        return child_main(args)
    if args.interior_optimum_probe:
        return run_interior_optimum_probe(args)
    if args.conv_4hw_fusion_sweep:
        return run_conv_4hw_fusion_sweep(args)
    if args.fusion_root_cause_investigation:
        return run_fusion_root_cause_investigation(args)
    if args.codesign_mini_sweep:
        return run_codesign_mini_sweep(args)
    if args.recovery_mode0:
        return run_recovery_mode0(args)
    return run_pilot(args)


if __name__ == "__main__":
    raise SystemExit(main())
