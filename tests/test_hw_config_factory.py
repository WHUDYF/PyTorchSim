from pathlib import Path
import sys

import pytest
import yaml


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "hw_config_factory.py"
    spec = importlib.util.spec_from_file_location("hw_config_factory", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_baseline(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "num_cores": 1,
                "core_freq_mhz": 940,
                "num_systolic_array_per_core": 2,
                "vpu_num_lanes": 128,
                "vpu_spad_size_kb_per_lane": 128,
                "vpu_vector_length_bits": 256,
                "dram_type": "ramulator2",
                "dram_freq_mhz": 940,
                "dram_channels": 16,
                "ramulator_config_path": "../configs/ramulator2_configs/HBM2_TPUv3.yaml",
                "icnt_freq_mhz": 940,
                "icnt_injection_ports_per_core": 16,
                "pytorchsim_timing_mode": 1,
            }
        ),
        encoding="utf-8",
    )


def test_generate_codesign_hw_configs_writes_four_yaml_and_rules(tmp_path):
    mod = load_module()
    baseline = tmp_path / "baseline.yml"
    output_dir = tmp_path / "hw_configs"
    write_baseline(baseline)

    result = mod.generate_codesign_hw_configs(baseline, output_dir)

    assert sorted(result["hw_yaml_paths"]) == ["HW-A", "HW-B", "HW-C", "HW-D"]
    assert result["frozen_fields_verified"] is True
    assert result["plausibility_rules_verified"] is True
    assert set(result["hw_yaml_content_sha256"]) == {"HW-A", "HW-B", "HW-C", "HW-D"}
    rules = yaml.safe_load((output_dir / "hw_plausibility_rules.json").read_text(encoding="utf-8"))
    assert rules["patch_matrix"]["HW-A"]["dram_channels"] == 32
    assert rules["frozen_fields"] == mod.FROZEN_FIELDS

    hw_a = yaml.safe_load((output_dir / "hw_A.yml").read_text(encoding="utf-8"))
    hw_b = yaml.safe_load((output_dir / "hw_B.yml").read_text(encoding="utf-8"))
    hw_c = yaml.safe_load((output_dir / "hw_C.yml").read_text(encoding="utf-8"))
    hw_d = yaml.safe_load((output_dir / "hw_D.yml").read_text(encoding="utf-8"))

    assert hw_a["vpu_spad_size_kb_per_lane"] == 128
    assert hw_a["dram_channels"] == 32
    assert hw_a["icnt_injection_ports_per_core"] == 16
    assert hw_b["vpu_spad_size_kb_per_lane"] == 128
    assert hw_b["dram_channels"] == 8
    assert hw_b["icnt_injection_ports_per_core"] == 4
    assert hw_c["vpu_spad_size_kb_per_lane"] == 32
    assert hw_c["dram_channels"] == 32
    assert hw_c["icnt_injection_ports_per_core"] == 16
    assert hw_d["vpu_spad_size_kb_per_lane"] == 32
    assert hw_d["dram_channels"] == 8
    assert hw_d["icnt_injection_ports_per_core"] == 4

    for field in mod.FROZEN_FIELDS:
        assert hw_a[field] == hw_b[field] == hw_c[field] == hw_d[field]


def test_generate_v2_codesign_hw_configs_changes_lanes_and_freezes_spad(tmp_path):
    mod = load_module()
    baseline = tmp_path / "baseline.yml"
    output_dir = tmp_path / "hw_configs_v2"
    write_baseline(baseline)

    result = mod.generate_codesign_hw_configs(baseline, output_dir, hw_config_set="codesign_v2_2x2")

    rules = yaml.safe_load((output_dir / "hw_plausibility_rules.json").read_text(encoding="utf-8"))
    assert result["hw_config_set"] == "codesign_v2_2x2"
    assert "vpu_spad_size_kb_per_lane" in rules["frozen_fields"]
    assert "vpu_num_lanes" not in rules["frozen_fields"]

    hw_a = yaml.safe_load((output_dir / "hw_A.yml").read_text(encoding="utf-8"))
    hw_b = yaml.safe_load((output_dir / "hw_B.yml").read_text(encoding="utf-8"))
    hw_c = yaml.safe_load((output_dir / "hw_C.yml").read_text(encoding="utf-8"))
    hw_d = yaml.safe_load((output_dir / "hw_D.yml").read_text(encoding="utf-8"))

    assert hw_a["vpu_num_lanes"] == 128
    assert hw_b["vpu_num_lanes"] == 128
    assert hw_c["vpu_num_lanes"] == 8
    assert hw_d["vpu_num_lanes"] == 8
    assert hw_a["vpu_spad_size_kb_per_lane"] == hw_b["vpu_spad_size_kb_per_lane"] == 128
    assert hw_c["vpu_spad_size_kb_per_lane"] == hw_d["vpu_spad_size_kb_per_lane"] == 128


def test_v2_patch_rejects_spad_change_and_internal_lane_value(tmp_path):
    mod = load_module()
    baseline = tmp_path / "baseline.yml"
    write_baseline(baseline)
    data = yaml.safe_load(baseline.read_text(encoding="utf-8"))
    v2 = mod.get_hw_config_set("codesign_v2_2x2")

    with pytest.raises(mod.FrozenFieldViolation):
        mod.patch_hw_yaml(data, {"vpu_spad_size_kb_per_lane": 32}, config_set=v2)

    with pytest.raises(mod.PlausibilityRuleViolation) as exc:
        mod.patch_hw_yaml(
            data,
            {
                "vpu_num_lanes": 16,
                "dram_channels": 32,
                "icnt_injection_ports_per_core": 16,
            },
            config_set=v2,
        )

    assert "vpu_num_lanes" in str(exc.value)


def test_patch_hw_yaml_rejects_frozen_field_change(tmp_path):
    mod = load_module()
    baseline = tmp_path / "baseline.yml"
    write_baseline(baseline)
    data = yaml.safe_load(baseline.read_text(encoding="utf-8"))

    with pytest.raises(mod.FrozenFieldViolation) as exc:
        mod.patch_hw_yaml(data, {"core_freq_mhz": 1000})

    assert "core_freq_mhz" in str(exc.value)


def test_patch_hw_yaml_rejects_bad_mem_subsystem_coupling(tmp_path):
    mod = load_module()
    baseline = tmp_path / "baseline.yml"
    write_baseline(baseline)
    data = yaml.safe_load(baseline.read_text(encoding="utf-8"))

    with pytest.raises(mod.PlausibilityRuleViolation) as exc:
        mod.patch_hw_yaml(
            data,
            {
                "vpu_spad_size_kb_per_lane": 128,
                "dram_channels": 8,
                "icnt_injection_ports_per_core": 16,
            },
        )

    assert "icnt_injection_ports_per_core" in str(exc.value)
