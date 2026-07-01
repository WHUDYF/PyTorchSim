from pathlib import Path
import sys


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "dry_run_fit.py"
    spec = importlib.util.spec_from_file_location("dry_run_fit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_working_set_bytes_includes_inputs_weights_and_output():
    mod = load_module()

    assert mod.working_set_bytes({"TILE_M": 32, "TILE_N": 64, "TILE_K": 32}) == (
        32 * 32 + 32 * 64 + 32 * 64
    ) * 4


def test_fit_classifier_uses_total_spad_with_safety_factor():
    mod = load_module()
    hw = {"vpu_spad_size_kb_per_lane": 1, "vpu_num_lanes": 1}

    assert mod.fit_classifier({"TILE_M": 8, "TILE_N": 8, "TILE_K": 8}, hw)["predicted_fit"] is True
    assert mod.fit_classifier({"TILE_M": 16, "TILE_N": 16, "TILE_K": 16}, hw)["predicted_fit"] is False


def test_v2_large_tile_is_boundary_fit_on_eight_lane_hw():
    mod = load_module()
    tile = {"TILE_M": 256, "TILE_N": 256, "TILE_K": 128}
    hw = {"vpu_spad_size_kb_per_lane": 128, "vpu_num_lanes": 8}

    result = mod.fit_classifier(tile, hw)

    assert result["working_set_bytes"] == 524288
    assert result["spad_budget_bytes"] == 943718
    assert result["predicted_fit"] is True
    assert result["budget_over_bytes"] == 0


def test_default_v2_calibration_cases_use_hw_a_006_and_hw_c_009():
    mod = load_module()

    cases = mod.default_calibration_cases_v2()

    assert [case["case_id"] for case in cases] == ["HW-A/006", "HW-C/009"]
    assert [case["predicted_fit"] for case in cases] == [True, True]


def test_calibration_passes_when_predicted_xor_actual_is_zero():
    mod = load_module()
    cases = [
        {"case_id": "known_fit", "predicted_fit": True, "actual_fit": True},
        {"case_id": "known_no_fit", "predicted_fit": False, "actual_fit": False},
    ]

    result = mod.evaluate_calibration(cases)

    assert result["calibration_passed"] is True
    assert result["calibration_status"] == "calibrated"
    assert [case["predicted_xor_actual"] for case in result["cases"]] == [0, 0]


def test_calibration_falls_back_on_any_predicted_xor_actual_mismatch():
    mod = load_module()
    cases = [
        {"case_id": "known_fit", "predicted_fit": True, "actual_fit": True},
        {"case_id": "known_no_fit", "predicted_fit": True, "actual_fit": False},
    ]

    result = mod.evaluate_calibration(cases)

    assert result["calibration_passed"] is False
    assert result["calibration_status"] == "fallback_engaged"
    assert [case["predicted_xor_actual"] for case in result["cases"]] == [0, 1]


def test_classify_cells_uses_no_precheck_when_fallback_engaged():
    mod = load_module()
    cells = [
        {"hw_id": "HW-A", "mapping_id": "000", "tile": {"TILE_M": 32, "TILE_N": 64, "TILE_K": 32}},
        {"hw_id": "HW-D", "mapping_id": "004", "tile": {"TILE_M": 128, "TILE_N": 128, "TILE_K": 32}},
    ]

    rows = mod.classify_cells(cells, {"calibration_status": "fallback_engaged"}, {"vpu_spad_size_kb_per_lane": 1, "vpu_num_lanes": 1})

    assert [row["availability_precheck"] for row in rows] == ["no-precheck", "no-precheck"]
    assert all(row["status"] == "needs_togsim" for row in rows)
