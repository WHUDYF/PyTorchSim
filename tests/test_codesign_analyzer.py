from pathlib import Path
import sys

import pytest


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "codesign_analyzer.py"
    spec = importlib.util.spec_from_file_location("codesign_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_champion_migration_detects_double_swap_dominance():
    mod = load_module()
    cycles = {
        "HW-A": {"000": 100, "001": 125},
        "HW-C": {"000": 130, "001": 100},
    }

    result = mod.compute_champion_migration(cycles)

    assert result["gate1_passed"] is True
    assert result["migrating_pair_count"] == 1
    assert result["pairs"][0]["swap_dominance"] == pytest.approx(0.25)


def test_champion_migration_rejects_noise_level_swap():
    mod = load_module()
    cycles = {
        "HW-A": {"000": 1000, "001": 1005},
        "HW-C": {"000": 1005, "001": 1000},
    }

    result = mod.compute_champion_migration(cycles)

    assert result["gate1_passed"] is False
    assert result["migrating_pair_count"] == 0


def test_champion_migration_uses_custom_ten_percent_threshold():
    mod = load_module()
    five_percent = {
        "HW-A": {"000": 100, "001": 105, "002": 130},
        "HW-B": {"000": 105, "001": 100, "002": 130},
    }
    twelve_percent = {
        "HW-B": {"000": 130, "001": 100, "002": 112},
        "HW-C": {"000": 130, "001": 112, "002": 100},
    }

    below = mod.compute_champion_migration(five_percent, dominance_threshold=0.10)
    above = mod.compute_champion_migration(twelve_percent, dominance_threshold=0.10)

    assert below["dominance_threshold"] == pytest.approx(0.10)
    assert below["pairs"][0]["swap_dominance"] == pytest.approx(0.05)
    assert below["pairs"][0]["passed"] is False
    assert below["gate1_passed"] is False
    assert above["pairs"][0]["swap_dominance"] == pytest.approx(0.12)
    assert above["pairs"][0]["passed"] is True
    assert above["gate1_passed"] is True


def test_analyze_codesign_forwards_custom_gate1_threshold():
    mod = load_module()
    cycles = {
        "HW-A": {"000": 100, "001": 105},
        "HW-B": {"000": 105, "001": 100},
    }

    result = mod.analyze_codesign(cycles, gate1_dominance_threshold=0.10)

    assert result["champion_migration"]["dominance_threshold"] == pytest.approx(0.10)
    assert result["champion_migration"]["gate1_passed"] is False


def test_oracle_gap_uses_fit_all_hw_set_for_gate2a():
    mod = load_module()
    cycles = {
        "HW-A": {"000": 100, "001": 80, "002": 10},
        "HW-B": {"000": 100, "001": 80},
        "HW-C": {"000": 100, "001": 80},
        "HW-D": {"000": 100, "001": 80},
    }

    result = mod.compute_oracle_gaps(cycles)

    assert result["fit_all_hw_set"] == ["000", "001"]
    assert result["single_best_avg_mapping"] == "001"
    assert result["gate2a_ratio_mean"] == pytest.approx(1.0)
    assert result["unconstrained_per_hw_oracle"]["HW-A"]["mapping_id"] == "002"


def test_gate2b_cost_matched_pair_ratio_cases():
    mod = load_module()
    cycles = {
        "HW-B": {"000": 800, "001": 900},
        "HW-C": {"000": 1000, "001": 1100},
    }

    result = mod.compute_cost_matched_pair_upper_bound(cycles)

    assert result["cost_matched_pair"] == ["HW-B", "HW-C"]
    assert result["best_cycles"] == {"HW-B": 800, "HW-C": 1000}
    assert result["gate2b_ratio"] == pytest.approx(0.8)
    assert result["gate2b_passed"] is True


def test_gate2b_reports_insufficient_evidence_when_pair_empty():
    mod = load_module()

    result = mod.compute_cost_matched_pair_upper_bound({"HW-B": {}, "HW-C": {}})

    assert result["gate2b_status"] == "insufficient_evidence"
    assert result["gate2b_passed"] is False


def test_interaction_analysis_uses_pairwise_spearman_on_intersections():
    mod = load_module()
    cycles = {
        "HW-A": {"000": 1, "001": 2, "002": 3, "003": 4},
        "HW-B": {"000": 4, "001": 3, "002": 2, "003": 1},
        "HW-C": {"000": 1, "001": 3, "002": 2, "003": 4},
        "HW-D": {"000": 4, "001": 1, "002": 3, "003": 2},
    }

    result = mod.compute_interaction_analysis(cycles)

    assert result["numeric_pair_count"] == 6
    assert result["gate3_passed"] is True
    assert result["mean_spearman"] <= 0.7


def test_interaction_analysis_rejects_too_few_numeric_pairs():
    mod = load_module()
    cycles = {
        "HW-A": {"000": 1, "001": 2, "002": 3},
        "HW-B": {"000": 1, "001": 2, "002": 3},
        "HW-C": {"004": 1, "005": 2, "006": 3},
        "HW-D": {"007": 1, "008": 2, "009": 3},
    }

    result = mod.compute_interaction_analysis(cycles)

    assert result["numeric_pair_count"] < 5
    assert result["gate3_passed"] is False
