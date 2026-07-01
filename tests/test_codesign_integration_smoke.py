from pathlib import Path
import sys


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "codesign_integration_smoke.py"
    spec = importlib.util.spec_from_file_location("codesign_integration_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_integration_smoke_emits_linted_artifacts(tmp_path):
    mod = load_module()
    output_dir = tmp_path / "smoke"
    hw_summary = {
        "baseline_yaml": str(tmp_path / "baseline.yml"),
        "baseline_yaml_sha256": "baseline-sha",
        "hw_yaml_content_sha256": {"HW-A": "sha-a", "HW-B": "sha-b"},
        "hw_yaml_paths": {"HW-A": str(tmp_path / "hw_A.yml"), "HW-B": str(tmp_path / "hw_B.yml")},
        "patch_matrix": {"HW-A": {}, "HW-B": {}},
    }
    (tmp_path / "baseline.yml").write_text("baseline\n", encoding="utf-8")
    (tmp_path / "hw_A.yml").write_text("hw-a\n", encoding="utf-8")
    (tmp_path / "hw_B.yml").write_text("hw-b\n", encoding="utf-8")
    ramulator = tmp_path / "ramulator.yaml"
    gem5 = tmp_path / "gem5"
    togsim = tmp_path / "togsim"
    ramulator.write_text("ramulator\n", encoding="utf-8")
    gem5.write_text("gem5\n", encoding="utf-8")
    togsim.write_text("togsim\n", encoding="utf-8")

    result = mod.run_integration_smoke(
        output_dir=output_dir,
        hw_summary=hw_summary,
        mappings=[
            {"mapping_id": "000", "TILE_M": 32, "TILE_N": 64, "TILE_K": 32},
            {"mapping_id": "001", "TILE_M": 64, "TILE_N": 64, "TILE_K": 32},
        ],
        cycles_by_cell={
            ("HW-A", "000"): 100,
            ("HW-A", "001"): 125,
            ("HW-B", "000"): 120,
            ("HW-B", "001"): 100,
        },
        determinism_smoke_test={"deterministic_class": "deterministic"},
        binary_paths={"ramulator2_config": ramulator, "gem5_binary": gem5, "togsim_binary": togsim},
        repo_root=tmp_path,
    )

    assert result["state_counts"] == {"measured": 4, "retry_exhausted": 0, "unavailable": 0}
    assert (output_dir / "sweep_summary.json").exists()
    assert (output_dir / "analysis" / "champion_migration.json").exists()
    assert (output_dir / "report" / "verdict.json").exists()
    assert (output_dir / "report" / "report.md").exists()
    assert (output_dir / "00_metadata.json").exists()
    assert result["metadata_lint"]["ok"] is True
    assert result["report_lint"]["ok"] is True
