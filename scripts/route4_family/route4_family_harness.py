#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import route4_fusion_pilot as r4  # noqa: E402


WORKLOADS = {
    "conv3x3_large": {
        "description": "Conv3x3-large: 3x3 conv [1,128,28,28] -> [1,128,28,28] with BN+ReLU",
        "batch": 1,
        "in_channels": 128,
        "out_channels": 128,
        "height": 28,
        "width": 28,
        "kernel_size": 3,
        "padding": 1,
        "groups": 1,
        "default_tile": {
            "TILE_K_H": 3,
            "TILE_K_W": 3,
            "TILE_O_H": 28,
            "TILE_O_W": 7,
            "TILE_M": 1,
            "TILE_N": 128,
            "TILE_K": 64,
        },
    },
    "conv1x1": {
        "description": "Conv1x1: 1x1 conv [1,256,14,14] -> [1,512,14,14] with BN+ReLU",
        "batch": 1,
        "in_channels": 256,
        "out_channels": 512,
        "height": 14,
        "width": 14,
        "kernel_size": 1,
        "padding": 0,
        "groups": 1,
        "default_tile": {
            "TILE_K_H": 1,
            "TILE_K_W": 1,
            "TILE_O_H": 14,
            "TILE_O_W": 14,
            "TILE_M": 1,
            "TILE_N": 128,
            "TILE_K": 64,
        },
    },
    "depthwise_conv3x3": {
        "description": "DepthwiseConv3x3: depthwise 3x3 conv [1,64,56,56] -> [1,64,56,56] with BN+ReLU",
        "batch": 1,
        "in_channels": 64,
        "out_channels": 64,
        "height": 56,
        "width": 56,
        "kernel_size": 3,
        "padding": 1,
        "groups": 64,
        "default_tile": {
            "TILE_K_H": 3,
            "TILE_K_W": 3,
            "TILE_O_H": 56,
            "TILE_O_W": 14,
            "TILE_M": 1,
            "TILE_N": 64,
            "TILE_K": 1,
        },
    },
    "resnet50_bottleneck_c3": {
        "description": "ResNet-50 conv3_x bottleneck: 1x1 reduce + 3x3 core + 1x1 expand + residual ReLU",
        "type": "resnet50_bottleneck_c3",
        "batch": 1,
        "in_channels": 512,
        "out_channels": 512,
        "height": 28,
        "width": 28,
        "kernel_size": 3,
        "padding": 1,
        "groups": 1,
        "core_shape_key": "conv2d_1_128_128_3_3_28_28",
        "reduce_shape_key": "conv2d_1_512_128_1_1_28_28",
        "expand_shape_key": "conv2d_1_128_512_1_1_28_28",
        "default_tile": {
            "TILE_K_H": 3,
            "TILE_K_W": 3,
            "TILE_O_H": 28,
            "TILE_O_W": 7,
            "TILE_M": 1,
            "TILE_N": 128,
            "TILE_K": 64,
        },
        "reduce_default_tile": {
            "TILE_K_H": 1,
            "TILE_K_W": 1,
            "TILE_O_H": 28,
            "TILE_O_W": 7,
            "TILE_M": 1,
            "TILE_N": 128,
            "TILE_K": 128,
        },
        "expand_default_tile": {
            "TILE_K_H": 1,
            "TILE_K_W": 1,
            "TILE_O_H": 28,
            "TILE_O_W": 7,
            "TILE_M": 1,
            "TILE_N": 128,
            "TILE_K": 128,
        },
    },
}


def shape_key(spec: dict) -> str:
    if spec.get("type") == "resnet50_bottleneck_c3":
        return str(spec["core_shape_key"])
    return r4.conv_shape_key(
        spec["batch"],
        spec["in_channels"],
        spec["out_channels"],
        spec["kernel_size"],
        spec["kernel_size"],
        spec["height"],
        spec["width"],
    )


def mapping_payload(spec: dict, tile: dict) -> dict:
    keys = ["TILE_K_H", "TILE_K_W", "TILE_O_H", "TILE_O_W", "TILE_M", "TILE_N", "TILE_K"]
    if spec.get("type") == "resnet50_bottleneck_c3":
        return {
            str(spec["reduce_shape_key"]): {key: int(spec["reduce_default_tile"][key]) for key in keys},
            str(spec["core_shape_key"]): {key: int(tile[key]) for key in keys},
            str(spec["expand_shape_key"]): {key: int(spec["expand_default_tile"][key]) for key in keys},
        }
    return {shape_key(spec): {key: int(tile[key]) for key in keys}}


def write_mapping(path: Path, spec: dict, tile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(mapping_payload(spec, tile), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_model(spec: dict):
    import torch

    if spec.get("type") == "resnet50_bottleneck_c3":
        class ResNet50BottleneckC3(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(512, 128, kernel_size=1, bias=True)
                self.bn1 = torch.nn.BatchNorm2d(128)
                self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
                self.bn2 = torch.nn.BatchNorm2d(128)
                self.conv3 = torch.nn.Conv2d(128, 512, kernel_size=1, bias=True)
                self.bn3 = torch.nn.BatchNorm2d(512)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                out = out + identity
                return self.relu(out)

        return ResNet50BottleneckC3().eval()

    kernel = int(spec["kernel_size"])
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            int(spec["in_channels"]),
            int(spec["out_channels"]),
            kernel_size=kernel,
            padding=int(spec["padding"]),
            groups=int(spec["groups"]),
            bias=True,
        ),
        torch.nn.BatchNorm2d(int(spec["out_channels"])),
        torch.nn.ReLU(),
    ).eval()


def run_workload_once(
    *,
    workload: str,
    hw_config: Path,
    variant: str,
    tile: dict,
    run_dir: Path,
    seed: int,
) -> dict:
    spec = WORKLOADS[workload]
    mapping_path = run_dir / "external_mapping.json"
    write_mapping(mapping_path, spec, tile)
    config_path = r4.make_run_config(
        hw_config,
        run_dir,
        functional_mode=0,
        use_external_mapping=True,
        external_mapping_override=mapping_path,
    )
    os.environ.update(r4.subprocess_env(run_dir, config_path))
    import torch

    r4.ensure_npu_registered()
    r4.clear_torch_caches()
    r4.install_conv_external_tile_patch()
    torch.manual_seed(seed)
    cpu_model = build_model(spec)
    npu_model = copy.deepcopy(cpu_model).to(device=torch.device("npu:0")).eval()
    x_cpu = torch.randn(
        int(spec["batch"]),
        int(spec["in_channels"]),
        int(spec["height"]),
        int(spec["width"]),
        dtype=torch.float32,
    )
    with torch.no_grad():
        ref = cpu_model(x_cpu)
        compiled = torch.compile(dynamic=False)(npu_model)
        out = compiled(x_cpu.to("npu:0")).cpu()
    max_abs, max_rel, passed = r4.tensor_diffs(out, ref)
    artifacts = r4.collect_run_artifacts(run_dir)
    total_cycles = int(artifacts.get("total_cycles", 0) or 0)
    result = {
        "ok": total_cycles > 0,
        "workload": workload,
        "description": spec["description"],
        "shape_key": shape_key(spec),
        "mapping_keys": sorted(mapping_payload(spec, tile)),
        "core_tile_applied": mapping_payload(spec, tile).get(shape_key(spec)) == {
            key: int(tile[key])
            for key in ["TILE_K_H", "TILE_K_W", "TILE_O_H", "TILE_O_W", "TILE_M", "TILE_N", "TILE_K"]
        },
        "fusion_variant": variant,
        "tile": tile,
        "total_cycles": total_cycles,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "allclose_passed": passed,
        "artifact_message": artifacts.get("message", ""),
        "log_paths": artifacts.get("log_paths", []),
    }
    r4.write_json(run_dir / "family_run_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--hw-config", type=Path, required=True)
    parser.add_argument("--variant", choices=["none", "all"], default="all")
    parser.add_argument("--tile-json", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    spec = WORKLOADS[args.workload]
    tile = json.loads(args.tile_json.read_text(encoding="utf-8")) if args.tile_json else spec["default_tile"]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run_workload_once(
            workload=args.workload,
            hw_config=args.hw_config,
            variant=args.variant,
            tile=tile,
            run_dir=args.run_dir,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 2
    except Exception as exc:
        result = {
            "ok": False,
            "workload": args.workload,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        r4.write_json(args.run_dir / "family_run_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
