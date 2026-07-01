from pathlib import Path
import sys


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "codesign_full_run.py"
    spec = importlib.util.spec_from_file_location("codesign_full_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def hw_summary(tmp_path):
    baseline = tmp_path / "baseline.yml"
    baseline.write_text("baseline\n", encoding="utf-8")
    paths = {}
    hashes = {}
    for hw_id in ["HW-A", "HW-B", "HW-C", "HW-D"]:
        path = tmp_path / f"{hw_id}.yml"
        path.write_text(hw_id, encoding="utf-8")
        paths[hw_id] = str(path)
        hashes[hw_id] = "sha-" + hw_id
    return {
        "baseline_yaml": str(baseline),
        "baseline_yaml_sha256": "baseline-sha",
        "hw_yaml_content_sha256": hashes,
        "hw_yaml_paths": paths,
        "patch_matrix": {hw_id: {} for hw_id in paths},
    }


def mappings():
    return [
        {"mapping_id": f"{idx:03d}", "TILE_M": 32 + idx, "TILE_N": 64, "TILE_K": 32}
        for idx in range(8)
    ]


def test_full_run_driver_emits_4x8_linted_artifacts(tmp_path):
    mod = load_module()
    output_dir = tmp_path / "full"
    ramulator = tmp_path / "ramulator.yaml"
    gem5 = tmp_path / "gem5"
    togsim = tmp_path / "togsim"
    ramulator.write_text("ramulator\n", encoding="utf-8")
    gem5.write_text("gem5\n", encoding="utf-8")
    togsim.write_text("togsim\n", encoding="utf-8")

    result = mod.run_full_codesign(
        output_dir=output_dir,
        hw_summary=hw_summary(tmp_path),
        mappings=mappings(),
        determinism_smoke_test={"deterministic_class": "deterministic"},
        fit_rows={},
        binary_paths={"ramulator2_config": ramulator, "gem5_binary": gem5, "togsim_binary": togsim},
        repo_root=tmp_path,
        runner_factory=mod.synthetic_runner_factory(
            lambda hw_idx, mapping_idx: 1000 + hw_idx * 100 + mapping_idx * 10
        ),
    )

    assert result["state_counts"] == {"measured": 32, "retry_exhausted": 0, "unavailable": 0}
    assert result["metadata_lint"]["ok"] is True
    assert result["report_lint"]["ok"] is True
    assert (output_dir / "heatmap_cycles.json").exists()
    assert (output_dir / "analysis" / "interaction_analysis.json").exists()
    assert (output_dir / "report" / "verdict.json").exists()
    assert (output_dir / "00_metadata.json").exists()


def test_full_run_driver_rejects_incomplete_cross_product(tmp_path):
    mod = load_module()

    try:
        mod.run_full_codesign(
            output_dir=tmp_path / "bad",
            hw_summary=hw_summary(tmp_path),
            mappings=mappings()[:7],
            determinism_smoke_test={"deterministic_class": "deterministic"},
            fit_rows={},
            binary_paths={},
            repo_root=tmp_path,
            runner_factory=mod.synthetic_runner_factory(lambda hw_idx, mapping_idx: 1),
        )
    except mod.FullRunContractError as exc:
        assert "8 or 10 mappings" in str(exc)
    else:
        raise AssertionError("expected FullRunContractError")


def test_full_run_near_deterministic_uses_three_repeats_and_median(tmp_path):
    mod = load_module()
    output_dir = tmp_path / "repeat"
    ramulator = tmp_path / "ramulator.yaml"
    gem5 = tmp_path / "gem5"
    togsim = tmp_path / "togsim"
    ramulator.write_text("ramulator\n", encoding="utf-8")
    gem5.write_text("gem5\n", encoding="utf-8")
    togsim.write_text("togsim\n", encoding="utf-8")

    def cycle_fn(hw_idx, mapping_idx, repeat_idx):
        return 1000 + hw_idx * 100 + mapping_idx * 10 + [0, 20, 10][repeat_idx]

    result = mod.run_full_codesign(
        output_dir=output_dir,
        hw_summary=hw_summary(tmp_path),
        mappings=mappings(),
        determinism_smoke_test={"deterministic_class": "near_deterministic"},
        fit_rows={},
        binary_paths={"ramulator2_config": ramulator, "gem5_binary": gem5, "togsim_binary": togsim},
        repo_root=tmp_path,
        runner_factory=mod.synthetic_runner_factory(cycle_fn),
    )

    heatmap = __import__("json").loads((output_dir / "heatmap_cycles.json").read_text(encoding="utf-8"))
    repeat_stats = __import__("json").loads((output_dir / "heatmap_cycle_stats.json").read_text(encoding="utf-8"))

    assert result["repeat_count"] == 3
    assert result["state_counts"] == {"measured": 32, "retry_exhausted": 0, "unavailable": 0}
    assert heatmap["HW-A"]["000"] == 1010
    assert repeat_stats["HW-A"]["000"]["cycles"] == [1000, 1020, 1010]
    assert repeat_stats["HW-A"]["000"]["stddev"] > 0


def test_compare_shared_cells_with_baseline_run_passes_within_tolerance():
    mod = load_module()
    v1_stats = {
        "HW-A": {"000": {"median_cycles": 1000}, "007": {"median_cycles": 2000}},
        "HW-B": {"000": {"median_cycles": 3000}},
    }
    v2_stats = {
        "HW-A": {"000": {"median_cycles": 1010}, "007": {"median_cycles": 1980}},
        "HW-B": {"000": {"median_cycles": 3040}},
    }

    result = mod.compare_shared_cells_with_baseline_run(v1_stats, v2_stats, max_cycle_delta=0.02)

    assert result["cross_check_status"] == "PASS"
    assert result["warning_cells"] == []
    assert result["v1_v2_cycle_delta_per_shared_cell"]["HW-A/000"]["cycle_delta"] == 0.01


def test_compare_shared_cells_with_baseline_run_warns_and_downgrades_on_large_delta():
    mod = load_module()
    v1_stats = {"HW-A": {"000": {"median_cycles": 1000}}}
    v2_stats = {"HW-A": {"000": {"median_cycles": 1300}}}

    result = mod.compare_shared_cells_with_baseline_run(v1_stats, v2_stats, max_cycle_delta=0.02)

    assert result["cross_check_status"] == "FAIL"
    assert result["force_partial"] is True
    assert result["warning_cells"] == ["HW-A/000"]


def test_full_run_emits_v2_metadata_and_cross_check_report(tmp_path):
    mod = load_module()
    output_dir = tmp_path / "full_v2"
    baseline_dir = tmp_path / "baseline_run"
    baseline_dir.mkdir()
    ramulator = tmp_path / "ramulator.yaml"
    gem5 = tmp_path / "gem5"
    togsim = tmp_path / "togsim"
    ramulator.write_text("ramulator\n", encoding="utf-8")
    gem5.write_text("gem5\n", encoding="utf-8")
    togsim.write_text("togsim\n", encoding="utf-8")
    mappings_10 = mappings() + [
        {"mapping_id": "008", "TILE_M": 16, "TILE_N": 16, "TILE_K": 16},
        {"mapping_id": "009", "TILE_M": 256, "TILE_N": 256, "TILE_K": 128},
    ]
    baseline_stats = {
        hw: {
            f"{idx:03d}": {"cycles": [1000 + hw_idx * 100 + idx * 10], "median_cycles": 1000 + hw_idx * 100 + idx * 10, "stddev": 0.0}
            for idx in range(8)
        }
        for hw_idx, hw in enumerate(["HW-A", "HW-B"])
    }
    (baseline_dir / "heatmap_cycle_stats.json").write_text(__import__("json").dumps(baseline_stats), encoding="utf-8")

    result = mod.run_full_codesign(
        output_dir=output_dir,
        hw_summary=hw_summary(tmp_path),
        mappings=mappings_10,
        determinism_smoke_test={"deterministic_class": "deterministic", "max_cycle_delta": 0.02},
        fit_rows={},
        binary_paths={"ramulator2_config": ramulator, "gem5_binary": gem5, "togsim_binary": togsim},
        repo_root=tmp_path,
        runner_factory=mod.synthetic_runner_factory(
            lambda hw_idx, mapping_idx: 1000 + hw_idx * 100 + mapping_idx * 10
        ),
        spec_version="v2",
        spec_path="docs/superpowers/specs/2026-07-01-npu-mapping-dse-codesign-v2-design.md",
        baseline_run_dir=baseline_dir,
    )

    metadata = __import__("json").loads((output_dir / "00_metadata.json").read_text(encoding="utf-8"))
    report = (output_dir / "report" / "report.md").read_text(encoding="utf-8")

    assert result["state_counts"] == {"measured": 40, "retry_exhausted": 0, "unavailable": 0}
    assert metadata["spec_version"] == "v2"
    assert metadata["hw_axis_axis1"] == "vpu_num_lanes"
    assert metadata["hw_axis_axis2"] == "mem_subsystem_scale"
    assert metadata["v1_v2_cross_check"]["cross_check_status"] == "PASS"
    assert len(metadata["v1_v2_cycle_delta_per_shared_cell"]) == 16
    assert "v1-v2 cross check" in report
