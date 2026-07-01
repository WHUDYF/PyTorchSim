import json
from pathlib import Path
import sys

import pytest


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "mapping_dse_minimal.py"
    spec = importlib.util.spec_from_file_location("mapping_dse_minimal", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_togsim_log_extracts_core_counters():
    mod = load_module()
    log_text = """
[info] Core [0] : MOVIN    inst_count: 2
[info] Core [0] : MOVOUT   inst_count: 1
[info] Core [0] : COMP     inst_count: 81 (GEMM: 80, Vector: 1)
[info] Core [0] : BAR      inst_count: 80
[info] Core [0] : Systolic array [0] utilization(%): 34.37, active_cycles: 4096, idle_cycles: 7821
[info] Core [0] : DMA active_cycles: 6144, DMA idle_cycles: 5773, DRAM BW: 248.000 GB/s (98304 responses)
[info] Core [0] : Vector unit utilization(%): 2.55, active cycle: 304, idle_cycle: 0
[info] [DRAM] channels 0..15 combined | 248.13 GB/s aggregate, 51.56% of utilization (avg. per channel) | 65536 reads, 32768 writes
[info] Total execution cycles: 11917
"""

    counters = mod.parse_togsim_log(log_text)

    assert counters["total_cycles"] == 11917
    assert counters["sa_utilization"] == pytest.approx(34.37)
    assert counters["sa_active_cycles"] == 4096
    assert counters["sa_idle_cycles"] == 7821
    assert counters["dma_active_cycles"] == 6144
    assert counters["dma_idle_cycles"] == 5773
    assert counters["vec_utilization"] == pytest.approx(2.55)
    assert counters["dram_bw_gbps"] == pytest.approx(248.13)
    assert counters["dram_reads"] == 65536
    assert counters["dram_writes"] == 32768
    assert counters["inst_count_by_op"]["COMP"] == 81
    assert counters["inst_count_by_op"]["GEMM"] == 80


def test_aggregate_counters_sums_block_logs_and_weights_rates():
    mod = load_module()
    rows = [
        {
            "total_cycles": 100,
            "sa_active_cycles": 20,
            "sa_idle_cycles": 80,
            "dma_active_cycles": 30,
            "dma_idle_cycles": 70,
            "vec_active_cycles": 40,
            "vec_idle_cycles": 60,
            "dram_bw_gbps": 10.0,
            "dram_reads": 1,
            "dram_writes": 2,
            "inst_count_by_op": {"COMP": 3, "GEMM": 2},
        },
        {
            "total_cycles": 300,
            "sa_active_cycles": 180,
            "sa_idle_cycles": 120,
            "dma_active_cycles": 90,
            "dma_idle_cycles": 210,
            "vec_active_cycles": 150,
            "vec_idle_cycles": 150,
            "dram_bw_gbps": 30.0,
            "dram_reads": 4,
            "dram_writes": 8,
            "inst_count_by_op": {"COMP": 5, "Vector": 5},
        },
    ]

    result = mod.aggregate_counters(rows)

    assert result["total_cycles"] == 400
    assert result["sa_utilization"] == pytest.approx(50.0)
    assert result["vec_utilization"] == pytest.approx(47.5)
    assert result["dram_bw_gbps"] == pytest.approx(25.0)
    assert result["dram_reads"] == 5
    assert result["dram_writes"] == 10
    assert result["inst_count_by_op"]["COMP"] == 8
    assert result["inst_count_by_op"]["GEMM"] == 2
    assert result["kernel_log_count"] == 2


def test_extract_raw_tog_features_counts_loops_dma_and_edges(tmp_path):
    mod = load_module()
    raw_tog = tmp_path / "tog.py"
    raw_tog.write_text(
        """
graph = {
0: {"node_id": 0, "node_name": "root", "node_type": 0, "parents": [], "children": [1]},
1: {"node_id": 1, "node_name": "loopNode", "node_type": 2, "parents": [0], "children": [2], "loop_start": 0, "loop_end": 8, "loop_step": 8, "loop_type": "outer_loop"},
2: {"node_id": 2, "node_name": "DMANode", "node_type": 3, "parents": [1], "children": [3], "tile_size": [8, 8], "tile_stride": [8, 1]},
3: {"node_id": 3, "node_name": "ComputeNode", "node_type": 1, "parents": [2], "children": [], "compute_type": 2}
}
""",
        encoding="utf-8",
    )

    features = mod.extract_struct_features(raw_tog)

    assert features["node_count"] == 4
    assert features["edge_count"] == 3
    assert features["op_type_histogram"]["DMANode"] == 1
    assert features["dma_node_count"] == 1
    assert features["loop_nest_depth_stats"]["max_depth"] == 1
    assert features["tile_shape"]["M"] == 8
    assert features["tile_shape"]["N"] == 8
    assert features["dma_stride_regularity"] == pytest.approx(1.0)


def test_aggregate_struct_features_sums_raw_tog_features():
    mod = load_module()
    rows = [
        {
            "node_count": 4,
            "edge_count": 3,
            "op_type_histogram": {"loopNode": 1, "DMANode": 1, "ComputeNode": 1},
            "loop_nest_depth_stats": {"max_depth": 1, "avg_depth": 1.0, "loop_count": 1},
            "dma_node_count": 1,
            "dma_stride_regularity": 1.0,
            "tile_shape": {"M": 8, "N": 8, "K": 0},
            "fusion_depth": 0,
            "compute_dma_overlap_proxy": 1.0,
            "struct_cost_proxy": 100,
        },
        {
            "node_count": 6,
            "edge_count": 5,
            "op_type_histogram": {"loopNode": 2, "DMANode": 3, "ComputeNode": 1},
            "loop_nest_depth_stats": {"max_depth": 2, "avg_depth": 1.5, "loop_count": 2},
            "dma_node_count": 3,
            "dma_stride_regularity": 2 / 3,
            "tile_shape": {"M": 16, "N": 8, "K": 8},
            "fusion_depth": 0,
            "compute_dma_overlap_proxy": 0.33,
            "struct_cost_proxy": 200,
        },
    ]

    result = mod.aggregate_struct_features(rows, {"TILE_M": 32, "TILE_N": 64, "TILE_K": 32})

    assert result["node_count"] == 10
    assert result["edge_count"] == 8
    assert result["op_type_histogram"]["DMANode"] == 4
    assert result["loop_nest_depth_stats"]["max_depth"] == 2
    assert result["loop_nest_depth_stats"]["avg_depth"] == pytest.approx(4 / 3)
    assert result["dma_stride_regularity"] == pytest.approx(0.75)
    assert result["struct_cost_proxy"] == pytest.approx(300)
    assert result["tile_shape"] == {"M": 32, "N": 64, "K": 32}
    assert result["raw_tog_count"] == 2


def test_gate1_requires_spread_and_rejects_single_outlier():
    mod = load_module()
    rows = [
        {"mapping_id": "000", "total_cycles": 100},
        {"mapping_id": "001", "total_cycles": 104},
        {"mapping_id": "002", "total_cycles": 106},
        {"mapping_id": "003", "total_cycles": 180},
    ]

    result = mod.analyze_gate1(rows)

    assert result["cycle_spread"] == pytest.approx(0.8)
    assert result["outlier_supported"] is False
    assert result["passed"] is False


def test_gate2_proxy_and_representative_can_select_best():
    mod = load_module()
    counters = [
        {"mapping_id": "000", "total_cycles": 100},
        {"mapping_id": "001", "total_cycles": 130},
        {"mapping_id": "002", "total_cycles": 160},
        {"mapping_id": "003", "total_cycles": 190},
    ]
    features = [
        {"mapping_id": "000", "struct_cost_proxy": 10, "cluster_key": "a"},
        {"mapping_id": "001", "struct_cost_proxy": 13, "cluster_key": "b"},
        {"mapping_id": "002", "struct_cost_proxy": 16, "cluster_key": "c"},
        {"mapping_id": "003", "struct_cost_proxy": 19, "cluster_key": "d"},
    ]

    result = mod.analyze_gate2(features, counters)

    assert result["b_proxy_spearman"] == pytest.approx(1.0)
    assert result["b_repr_top1_hit"] is True
    assert result["passed"] is True


def test_gate2_top3_hit_means_predicted_set_contains_true_best():
    mod = load_module()
    counters = [
        {"mapping_id": "000", "total_cycles": 400},
        {"mapping_id": "001", "total_cycles": 300},
        {"mapping_id": "002", "total_cycles": 200},
        {"mapping_id": "003", "total_cycles": 100},
        {"mapping_id": "004", "total_cycles": 500},
        {"mapping_id": "005", "total_cycles": 600},
        {"mapping_id": "006", "total_cycles": 700},
        {"mapping_id": "007", "total_cycles": 800},
        {"mapping_id": "008", "total_cycles": 900},
        {"mapping_id": "009", "total_cycles": 1000},
    ]
    features = [
        {"mapping_id": "000", "struct_cost_proxy": 1, "cluster_key": "a"},
        {"mapping_id": "001", "struct_cost_proxy": 2, "cluster_key": "b"},
        {"mapping_id": "002", "struct_cost_proxy": 3, "cluster_key": "c"},
        {"mapping_id": "003", "struct_cost_proxy": 100, "cluster_key": "d"},
        {"mapping_id": "004", "struct_cost_proxy": 101, "cluster_key": "e"},
        {"mapping_id": "005", "struct_cost_proxy": 102, "cluster_key": "f"},
        {"mapping_id": "006", "struct_cost_proxy": 103, "cluster_key": "g"},
        {"mapping_id": "007", "struct_cost_proxy": 104, "cluster_key": "h"},
        {"mapping_id": "008", "struct_cost_proxy": 105, "cluster_key": "i"},
        {"mapping_id": "009", "struct_cost_proxy": 106, "cluster_key": "j"},
    ]

    result = mod.analyze_gate2(features, counters)

    assert result["true_best_id"] == "003"
    assert result["predicted_top3_ids"] == ["000", "001", "002"]
    assert result["b_repr_top3_hit"] is False


def test_gate2_representative_fraction_uses_floor_limit():
    mod = load_module()
    counters = [{"mapping_id": f"{idx:03d}", "total_cycles": idx + 1} for idx in range(8)]
    features = [
        {"mapping_id": f"{idx:03d}", "struct_cost_proxy": idx + 1, "cluster_key": f"c{idx}"}
        for idx in range(8)
    ]

    result = mod.analyze_gate2(features, counters)

    assert result["representative_ids"] == ["000", "001"]
    assert result["b_repr_sample_fraction"] == pytest.approx(0.25)


def test_default_ten_mapping_candidates_include_v2_extremes():
    mod = load_module()

    candidates = mod.default_mapping_candidates(10)

    assert candidates[8] == {"mapping_id": "008", "TILE_M": 16, "TILE_N": 16, "TILE_K": 16}
    assert candidates[9] == {"mapping_id": "009", "TILE_M": 256, "TILE_N": 256, "TILE_K": 128}


def test_write_blocked_outputs_uses_measured_label_without_fake_rows(tmp_path):
    mod = load_module()

    mod.write_terminal_outputs(
        output_dir=tmp_path,
        metadata={"stages": {"harness": {"ok": False, "message": "no npu"}}},
        mapping_results=[],
        end_state="BLOCKED",
        reason="no npu",
    )

    verdict = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
    counters = json.loads((tmp_path / "counters_table.json").read_text(encoding="utf-8"))

    assert verdict["end_state"] == "BLOCKED"
    assert verdict["data_label"] == "measured"
    assert counters["data_label"] == "measured"
    assert counters["rows"] == []


def test_write_json_serializes_path_values(tmp_path):
    mod = load_module()
    output = tmp_path / "metadata.json"

    mod.write_json(output, {"args": {"hw_config": tmp_path / "hw.yml"}})

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["args"]["hw_config"].endswith("hw.yml")


def test_environment_preflight_reports_missing_transformers_and_backend(monkeypatch):
    mod = load_module()

    def fake_import(name):
        if name == "transformers":
            return False, "missing"
        if name == "torch_openreg":
            return False, "bad glibcxx"
        return True, ""

    monkeypatch.setattr(mod, "_check_import", fake_import)

    result = mod.environment_preflight(require_transformers=True, require_npu=True)

    assert result["ok"] is False
    assert "transformers" in result["failures"]
    assert "torch_openreg" in result["failures"]


def test_runtime_env_defaults_to_known_working_toolchain(monkeypatch):
    mod = load_module()

    monkeypatch.delenv("TORCHSIM_LLVM_PATH", raising=False)
    monkeypatch.delenv("GEM5_PATH", raising=False)

    env = mod.pytorchsim_runtime_env({})

    assert env["TORCHSIM_LLVM_PATH"] == "/home/dyf/src/psal-llvm-project-v1.0.8/build/bin"
    assert env["GEM5_PATH"] == "/home/dyf/src/psal-gem5/build/RISCV/gem5.opt"
    assert env["PATH"].split(":")[0] == "/home/dyf/opt/pytorchsim-riscv-gcc-compat/bin"


def test_module_version_uses_subprocess_result(monkeypatch):
    mod = load_module()

    class Result:
        returncode = 0
        stdout = "1.2.3\n"
        stderr = ""

    def fake_run(*args, **kwargs):
        return Result()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    assert mod.module_version("torch", {}) == "1.2.3"


def test_write_mapping_config_absolutizes_ramulator_path(tmp_path):
    mod = load_module()
    base = tmp_path / "base.yml"
    out = tmp_path / "out.yml"
    external = tmp_path / "external_mapping.json"
    base.write_text(
        "\n".join(
            [
                "ramulator_config_path: ../configs/ramulator2_configs/HBM2_TPUv3.yaml",
                "pytorchsim_functional_mode: 1",
                "pytorchsim_timing_mode: 0",
            ]
        ),
        encoding="utf-8",
    )

    mod.write_mapping_config(base, out, external)

    import yaml

    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["pytorchsim_functional_mode"] == 0
    assert data["pytorchsim_timing_mode"] == 1
    assert Path(data["ramulator_config_path"]).is_absolute()
    assert data["ramulator_config_path"].endswith("configs/ramulator2_configs/HBM2_TPUv3.yaml")


def test_write_mapping_config_uses_explicit_hw_config(tmp_path):
    mod = load_module()
    hw_config = tmp_path / "hw_A.yml"
    out = tmp_path / "out.yml"
    external = tmp_path / "external_mapping.json"
    hw_config.write_text(
        "\n".join(
            [
                "vpu_spad_size_kb_per_lane: 32",
                "dram_channels: 8",
                "icnt_injection_ports_per_core: 4",
                "ramulator_config_path: ../configs/ramulator2_configs/HBM2_TPUv3.yaml",
                "pytorchsim_functional_mode: 1",
                "pytorchsim_timing_mode: 1",
            ]
        ),
        encoding="utf-8",
    )

    mod.write_mapping_config(hw_config, out, external)

    import yaml

    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["vpu_spad_size_kb_per_lane"] == 32
    assert data["dram_channels"] == 8
    assert data["codegen_external_mapping_file"] == str(external.resolve())


def test_load_external_mapping_candidates_validates_schema(tmp_path):
    mod = load_module()
    bad = tmp_path / "bad_mappings.json"
    bad.write_text(json.dumps([{"mapping_id": "000", "TILE_M": 32, "TILE_N": 64}]), encoding="utf-8")

    with pytest.raises(mod.MappingSchemaError) as exc:
        mod.load_external_mapping_candidates(bad)

    assert "TILE_K" in str(exc.value)


def test_load_external_mapping_candidates_accepts_list_schema(tmp_path):
    mod = load_module()
    path = tmp_path / "mappings.json"
    path.write_text(
        json.dumps(
            [
                {"mapping_id": "000", "TILE_M": 32, "TILE_N": 64, "TILE_K": 32},
                {"mapping_id": "001", "TILE_M": 64, "TILE_N": 64, "TILE_K": 32},
            ]
        ),
        encoding="utf-8",
    )

    rows = mod.load_external_mapping_candidates(path)

    assert rows == [
        {"mapping_id": "000", "TILE_M": 32, "TILE_N": 64, "TILE_K": 32},
        {"mapping_id": "001", "TILE_M": 64, "TILE_N": 64, "TILE_K": 32},
    ]


def test_parse_args_rejects_missing_hw_config(tmp_path):
    mod = load_module()

    with pytest.raises(SystemExit) as exc:
        mod.parse_args(["--hw-config", str(tmp_path / "missing.yml")])

    assert exc.value.code == 2


def test_togsim_log_filter_rejects_gem5_sto_log(tmp_path):
    mod = load_module()
    good = tmp_path / "togsim.log"
    bad = tmp_path / "sto.log"
    good.write_text(
        "[info] Core [0] : DMA active_cycles: 1, DMA idle_cycles: 2, DRAM BW: 3 GB/s\n"
        "[info] Total execution cycles: 4\n",
        encoding="utf-8",
    )
    bad.write_text("gem5 simulator output without TOGSim summary\n", encoding="utf-8")

    assert mod.is_togsim_summary_log(good) is True
    assert mod.is_togsim_summary_log(bad) is False
