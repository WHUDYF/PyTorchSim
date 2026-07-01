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
        assert "expected 32 cells" in str(exc)
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
