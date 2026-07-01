from pathlib import Path
import sys

import pytest


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "determinism_smoke.py"
    spec = importlib.util.spec_from_file_location("determinism_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cycle_delta_uses_relative_max_min_over_median():
    mod = load_module()

    assert mod.cycle_delta([1000, 1000, 1500]) == pytest.approx(0.5)


def test_classify_determinism_thresholds():
    mod = load_module()

    assert mod.classify_cycle_runs([1000, 1000, 1001])["deterministic_class"] == "deterministic"
    assert mod.classify_cycle_runs([1000, 1000, 1010])["deterministic_class"] == "near_deterministic"
    assert mod.classify_cycle_runs([1000, 1000, 1500])["deterministic_class"] == "non_deterministic"


def test_summarize_canaries_uses_worst_delta_for_class():
    mod = load_module()
    result = mod.summarize_canaries(
        {
            "HW-A_006": [1000, 1000, 1000],
            "HW-C_000": [1000, 1000, 1010],
        }
    )

    assert result["deterministic_class"] == "near_deterministic"
    assert result["max_cycle_delta"] == pytest.approx(0.01)
    assert result["canaries"]["HW-C_000"]["cycle_delta"] == pytest.approx(0.01)


def test_default_canaries_use_v2_boundary_cell():
    mod = load_module()

    assert mod.DEFAULT_CANARIES == [("HW-A", "006"), ("HW-C", "008")]


def test_run_canary_repeats_uses_three_fresh_output_dirs(tmp_path):
    mod = load_module()
    calls = []

    def runner(cell, repeat_index, output_dir):
        calls.append((cell["hw_id"], cell["mapping_id"], repeat_index, output_dir))
        output_dir.mkdir(parents=True, exist_ok=False)
        return 1000 + repeat_index

    cells = {
        ("HW-A", "006"): {"hw_id": "HW-A", "mapping_id": "006"},
        ("HW-C", "008"): {"hw_id": "HW-C", "mapping_id": "008"},
    }

    result = mod.run_canary_repeats(cells, tmp_path, runner=runner, repeats=3)

    assert result["raw_cycles"] == {
        "HW-A_006": [1000, 1001, 1002],
        "HW-C_008": [1000, 1001, 1002],
    }
    assert len(calls) == 6
    assert len({call[3] for call in calls}) == 6
    assert all("repeat_" in str(call[3]) for call in calls)
    assert (tmp_path / "determinism_cycles.json").exists()
    assert (tmp_path / "determinism_summary.json").exists()


def test_run_canary_repeats_requires_all_default_canaries(tmp_path):
    mod = load_module()

    with pytest.raises(mod.CanarySelectionError):
        mod.run_canary_repeats(
            {("HW-A", "006"): {"hw_id": "HW-A", "mapping_id": "006"}},
            tmp_path,
            runner=lambda cell, repeat_index, output_dir: 1,
        )


def test_extract_total_cycles_reads_mapping_harness_output(tmp_path):
    mod = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "counters_table.json").write_text(
        '{"rows": [{"mapping_id": "006", "total_cycles": 9090}]}',
        encoding="utf-8",
    )

    assert mod.extract_total_cycles(run_dir, "006", "", "") == 9090


def test_run_canary_repeats_records_failed_canary_without_traceback(tmp_path):
    mod = load_module()
    cells = {
        ("HW-A", "006"): {"hw_id": "HW-A", "mapping_id": "006"},
        ("HW-C", "008"): {"hw_id": "HW-C", "mapping_id": "008"},
    }

    def runner(cell, repeat_index, output_dir):
        if cell["mapping_id"] == "008":
            raise RuntimeError("could not parse total_cycles from subprocess output")
        return 1000 + repeat_index

    result = mod.run_canary_repeats(cells, tmp_path, runner=runner, repeats=3)

    assert result["summary"]["deterministic_class"] == "blocked"
    assert result["summary"]["canaries"]["HW-A_006"]["deterministic_class"] == "near_deterministic"
    assert result["summary"]["canaries"]["HW-C_008"]["deterministic_class"] == "failed"
    assert "could not parse total_cycles" in result["summary"]["canaries"]["HW-C_008"]["error"]
    assert (tmp_path / "determinism_summary.json").exists()
