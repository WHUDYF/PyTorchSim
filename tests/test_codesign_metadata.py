from pathlib import Path
import sys

import pytest


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "codesign_metadata.py"
    spec = importlib.util.spec_from_file_location("codesign_metadata", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_metadata():
    return {
        "spec_version": "v2",
        "spec_path": "docs/superpowers/specs/2026-07-01-npu-mapping-dse-codesign-v2-design.md",
        "timestamp": "2026-07-01T00:00:00+00:00",
        "git_commit": "abc",
        "git_status": "clean",
        "cli_args": {},
        "baseline_yaml_sha256": "sha",
        "patch_matrix": {},
        "hw_yaml_content_sha256": {},
        "hw_yaml_diff_from_baseline": {},
        "per_cell": [],
        "ramulator2_config_sha256": "sha",
        "gem5_binary_sha256": "sha",
        "togsim_binary_sha256": "sha",
        "pytorchsim_git_commit": "abc",
        "env_vars_snapshot": {},
        "versions": {"torch": "2.6.0", "transformers": "4.43.4", "python": "3.11"},
        "determinism_smoke_test": {},
        "artifact_paths": {},
    }


def test_sha256_file_hashes_content(tmp_path):
    mod = load_module()
    path = tmp_path / "x.txt"
    path.write_text("abc", encoding="utf-8")

    assert mod.sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_lint_metadata_accepts_required_schema():
    mod = load_module()

    assert mod.lint_metadata(valid_metadata())["ok"] is True


def test_lint_metadata_rejects_missing_required_field():
    mod = load_module()
    metadata = valid_metadata()
    metadata.pop("determinism_smoke_test")

    with pytest.raises(mod.MetadataSchemaError) as exc:
        mod.lint_metadata(metadata)

    assert "determinism_smoke_test" in str(exc.value)


def test_lint_claim_bearing_rejects_placeholder_values():
    mod = load_module()

    with pytest.raises(mod.ClaimBearingDataError):
        mod.lint_claim_bearing({"claim_bearing": True, "data_label": "placeholder"})

    with pytest.raises(mod.ClaimBearingDataError):
        mod.lint_claim_bearing({"claim_bearing": True, "rows": [{"status": "pending"}]})


def test_capture_versions_does_not_import_torch_module():
    mod = load_module()
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch.") or name == "numpy" or name.startswith("numpy."):
            sys.modules.pop(name, None)

    versions = mod.capture_versions()

    assert "python" in versions
    assert "torch" in versions
    assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)


def test_build_metadata_manifest_combines_run_artifacts(tmp_path):
    mod = load_module()
    baseline = tmp_path / "baseline.yml"
    baseline.write_text("core_freq_mhz: 700\n", encoding="utf-8")
    hw_a = tmp_path / "hw_A.yml"
    hw_a.write_text("core_freq_mhz: 700\nvpu_spad_size_kb_per_lane: 128\n", encoding="utf-8")
    ramulator = tmp_path / "ramulator.yaml"
    ramulator.write_text("ramulator\n", encoding="utf-8")
    gem5 = tmp_path / "gem5"
    gem5.write_text("gem5\n", encoding="utf-8")
    togsim = tmp_path / "togsim"
    togsim.write_text("togsim\n", encoding="utf-8")
    stdout = tmp_path / "stdout.txt"
    stderr = tmp_path / "stderr.txt"
    stdout.write_text("ok\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")

    hw_summary = {
        "baseline_yaml": str(baseline),
        "baseline_yaml_sha256": mod.sha256_file(baseline),
        "hw_yaml_content_sha256": {"HW-A": mod.sha256_file(hw_a)},
        "hw_yaml_paths": {"HW-A": str(hw_a)},
        "patch_matrix": {"HW-A": {"vpu_spad_size_kb_per_lane": 128}},
    }
    sweep_summary = {
        "fit_availability": {
            "cells": [
                {
                    "hw_id": "HW-A",
                    "mapping_id": "000",
                    "status": "measured",
                    "simulator_cmdline": ["python", "scripts/mapping_dse_minimal.py"],
                    "stdout_path": str(stdout),
                    "stderr_path": str(stderr),
                    "subprocess_pid": 1234,
                    "start_time": "2026-07-01T00:00:00+00:00",
                    "end_time": "2026-07-01T00:00:01+00:00",
                }
            ]
        },
        "retry_ledger": [],
    }

    metadata = mod.build_metadata_manifest(
        repo_root=tmp_path,
        cli_args={"output_dir": "out"},
        hw_summary=hw_summary,
        sweep_summary=sweep_summary,
        determinism_smoke_test={"deterministic_class": "deterministic"},
        artifact_paths={"report": "report.md"},
        binary_paths={"ramulator2_config": ramulator, "gem5_binary": gem5, "togsim_binary": togsim},
        env={"PATH": "/bin", "PYTHONPATH": "", "LD_LIBRARY_PATH": ""},
        versions={"python": "3.11", "torch": "not_captured", "transformers": "not_captured"},
    )

    assert metadata["baseline_yaml_sha256"] == mod.sha256_file(baseline)
    assert metadata["hw_yaml_content_sha256"]["HW-A"] == mod.sha256_file(hw_a)
    assert metadata["per_cell"][0]["stdout_path"] == str(stdout)
    assert metadata["per_cell"][0]["subprocess_pid"] == 1234
    assert metadata["ramulator2_config_sha256"] == mod.sha256_file(ramulator)
    assert mod.lint_metadata(metadata)["ok"] is True


def test_lint_metadata_rejects_per_cell_missing_reproduction_fields():
    mod = load_module()
    metadata = valid_metadata()
    metadata["per_cell"] = [{"hw_id": "HW-A", "mapping_id": "000", "status": "measured"}]

    with pytest.raises(mod.MetadataSchemaError) as exc:
        mod.lint_metadata(metadata)

    assert "simulator_cmdline" in str(exc.value)
