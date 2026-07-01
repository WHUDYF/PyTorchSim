from pathlib import Path
import sys

import pytest
import yaml


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "codesign_preflight.py"
    spec = importlib.util.spec_from_file_location("codesign_preflight", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_minimal_repo(root: Path, *, include_spad: bool = True) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "TOGSim" / "build" / "bin").mkdir(parents=True)
    (root / "docs" / "superpowers" / "specs").mkdir(parents=True)
    (root / "docs" / "superpowers" / "plans").mkdir(parents=True)
    (root / "scripts" / "mapping_dse_minimal.py").write_text("# harness\n", encoding="utf-8")
    (root / "TOGSim" / "build" / "bin" / "Simulator").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "docs" / "superpowers" / "specs" / "2026-07-01-npu-mapping-dse-codesign-design.md").write_text(
        "# spec\n",
        encoding="utf-8",
    )
    (root / "docs" / "superpowers" / "plans" / "2026-07-01-npu-mapping-dse-codesign-plan.md").write_text(
        "# plan\n",
        encoding="utf-8",
    )
    yaml_data = {
        "core_freq_mhz": 940,
        "num_cores": 1,
        "dram_channels": 16,
        "icnt_injection_ports_per_core": 16,
    }
    if include_spad:
        yaml_data["vpu_spad_size_kb_per_lane"] = 128
    (root / "configs" / "systolic_ws_128x128_c1_simple_noc_tpuv3.yml").write_text(
        yaml.safe_dump(yaml_data),
        encoding="utf-8",
    )


def test_verify_repo_grounding_accepts_minimal_valid_repo(tmp_path):
    mod = load_module()
    write_minimal_repo(tmp_path)

    result = mod.verify_repo_grounding(tmp_path)

    assert result["preflight_ok"] is True
    assert result["repo_root"] == str(tmp_path)
    assert result["artifacts"]["mapping_harness"].endswith("scripts/mapping_dse_minimal.py")
    assert result["baseline_yaml"]["vpu_spad_size_kb_per_lane"] == 128
    assert result["togsim_binary"].endswith("TOGSim/build/bin/Simulator")


def test_verify_repo_grounding_reports_missing_artifacts(tmp_path):
    mod = load_module()

    with pytest.raises(mod.PreflightError) as exc:
        mod.verify_repo_grounding(tmp_path)

    assert exc.value.exit_code == 2
    assert "scripts/mapping_dse_minimal.py" in exc.value.message
    assert "configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml" in exc.value.message


def test_verify_repo_grounding_rejects_baseline_missing_spad_field(tmp_path):
    mod = load_module()
    write_minimal_repo(tmp_path, include_spad=False)

    with pytest.raises(mod.PreflightError) as exc:
        mod.verify_repo_grounding(tmp_path)

    assert exc.value.exit_code == 3
    assert "vpu_spad_size_kb_per_lane" in exc.value.message
