#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BASELINE_YAML = Path("configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml")
PATCH_MATRIX: dict[str, dict[str, int]] = {
    "HW-A": {"vpu_spad_size_kb_per_lane": 128, "dram_channels": 32, "icnt_injection_ports_per_core": 16},
    "HW-B": {"vpu_spad_size_kb_per_lane": 128, "dram_channels": 8, "icnt_injection_ports_per_core": 4},
    "HW-C": {"vpu_spad_size_kb_per_lane": 32, "dram_channels": 32, "icnt_injection_ports_per_core": 16},
    "HW-D": {"vpu_spad_size_kb_per_lane": 32, "dram_channels": 8, "icnt_injection_ports_per_core": 4},
}
FROZEN_FIELDS = [
    "core_freq_mhz",
    "dram_freq_mhz",
    "icnt_freq_mhz",
    "num_cores",
    "num_systolic_array_per_core",
    "vpu_num_lanes",
    "vpu_vector_length_bits",
    "dram_type",
    "ramulator_config_path",
]
PATCHABLE_FIELDS = {"vpu_spad_size_kb_per_lane", "dram_channels", "icnt_injection_ports_per_core"}


class FrozenFieldViolation(ValueError):
    pass


class PlausibilityRuleViolation(ValueError):
    pass


@dataclass(frozen=True)
class GeneratedConfig:
    hw_id: str
    path: Path
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_injection_ports(dram_channels: int) -> int:
    value = max(4, min(32, dram_channels // 2))
    if value % 2:
        value += 1
    return value


def patch_hw_yaml(baseline: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    frozen_touched = sorted(field for field in patch if field in FROZEN_FIELDS)
    if frozen_touched:
        raise FrozenFieldViolation("Patch touches frozen fields: " + ", ".join(frozen_touched))
    unknown = sorted(field for field in patch if field not in PATCHABLE_FIELDS)
    if unknown:
        raise PlausibilityRuleViolation("Patch contains unsupported fields: " + ", ".join(unknown))

    result = dict(baseline)
    result.update(patch)
    dram_channels = int(result["dram_channels"])
    expected_ports = expected_injection_ports(dram_channels)
    actual_ports = int(result["icnt_injection_ports_per_core"])
    if actual_ports != expected_ports:
        raise PlausibilityRuleViolation(
            "icnt_injection_ports_per_core must equal "
            f"clip(dram_channels / 2, 4, 32) rounded to nearest even: "
            f"expected {expected_ports}, got {actual_ports}"
        )
    return result


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return data


def hw_filename(hw_id: str) -> str:
    return f"hw_{hw_id.split('-', 1)[1]}.yml"


def build_rules_payload(
    baseline_yaml: Path,
    generated: list[GeneratedConfig],
    baseline_sha256: str,
) -> dict[str, Any]:
    return {
        "baseline_yaml": str(baseline_yaml),
        "baseline_yaml_sha256": baseline_sha256,
        "patch_matrix": PATCH_MATRIX,
        "plausibility_rules": [
            "icnt_injection_ports_per_core = clip(dram_channels / 2, 4, 32) rounded to nearest even",
            "vpu_spad_size_kb_per_lane independent of fixed 128x128 systolic array size",
            "num_cores fixed at baseline value",
            "dram_type and ramulator_config_path fixed at baseline value",
        ],
        "frozen_fields": FROZEN_FIELDS,
        "frozen_fields_reason": (
            "These fields are held identical across HW-A/B/C/D so cycle differences are not "
            "explained by clock frequency or unrelated timing-model changes."
        ),
        "coupling_reason": (
            "dram_channels and icnt_injection_ports_per_core are treated as one mem_subsystem_scale "
            "factor to keep generated corners plausible."
        ),
        "hw_yaml_content_sha256": {item.hw_id: item.sha256 for item in generated},
    }


def generate_codesign_hw_configs(
    baseline_yaml: Path | str = BASELINE_YAML,
    output_dir: Path | str = Path("outputs/mapping_dse_codesign/hw_configs"),
) -> dict[str, Any]:
    baseline_path = Path(baseline_yaml)
    out_dir = Path(output_dir)
    baseline = load_yaml_mapping(baseline_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[GeneratedConfig] = []
    for hw_id, patch in PATCH_MATRIX.items():
        patched = patch_hw_yaml(baseline, patch)
        for field in FROZEN_FIELDS:
            if patched.get(field) != baseline.get(field):
                raise FrozenFieldViolation(f"Generated {hw_id} changed frozen field {field}")
        path = out_dir / hw_filename(hw_id)
        path.write_text(yaml.safe_dump(patched, sort_keys=False), encoding="utf-8")
        generated.append(GeneratedConfig(hw_id, path, sha256_file(path)))

    baseline_sha256 = sha256_file(baseline_path)
    rules = build_rules_payload(baseline_path, generated, baseline_sha256)
    rules_path = out_dir / "hw_plausibility_rules.json"
    rules_path.write_text(json.dumps(rules, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "baseline_yaml": str(baseline_path),
        "baseline_yaml_sha256": baseline_sha256,
        "hw_yaml_paths": {item.hw_id: str(item.path) for item in generated},
        "hw_yaml_content_sha256": {item.hw_id: item.sha256 for item in generated},
        "hw_plausibility_rules": str(rules_path),
        "frozen_fields_verified": True,
        "plausibility_rules_verified": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate co-design HW YAML configs.")
    parser.add_argument("--baseline-yaml", type=Path, default=BASELINE_YAML)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mapping_dse_codesign/hw_configs"))
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate_codesign_hw_configs(args.baseline_yaml, args.output_dir)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
