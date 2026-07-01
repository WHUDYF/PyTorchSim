from pathlib import Path
import sys


def load_module(name: str):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def golden_cycles():
    return {
        "HW-A": {"000": 100, "001": 100, "002": 120, "003": 130, "004": 140, "005": 150, "006": 160},
        "HW-B": {"000": 100, "001": 100, "002": 121, "003": 131, "004": 141, "005": 151, "006": 161},
        "HW-C": {"000": 100, "001": 100, "002": 122, "003": 132, "004": 142, "005": 152},
        "HW-D": {"000": 100, "001": 100, "002": 123, "003": 133, "004": 143, "005": 153},
    }


def test_golden_fixture_verdict_is_stable_negative():
    analyzer = load_module("codesign_analyzer")
    report = load_module("codesign_report")
    analysis = analyzer.analyze_codesign(golden_cycles())
    analysis["state_counts"] = {"measured": 29, "unavailable": 2, "retry_exhausted": 1}

    verdict = report.build_verdict(analysis)

    assert analysis["champion_migration"]["gate1_passed"] is False
    assert analysis["champion_migration"]["migrating_pair_count"] == 0
    assert verdict["end_state"] == "NEGATIVE"
    assert verdict["gate1"]["passed"] is False


def test_permutation_baseline_is_reference_only():
    analyzer = load_module("codesign_analyzer")
    cycles = {
        "HW-A": {"000": 1, "001": 2, "002": 3, "003": 4},
        "HW-B": {"000": 4, "001": 3, "002": 2, "003": 1},
        "HW-C": {"000": 1, "001": 3, "002": 2, "003": 4},
        "HW-D": {"000": 4, "001": 1, "002": 3, "003": 2},
    }

    without_reference = analyzer.compute_interaction_analysis(cycles)
    with_reference = analyzer.compute_interaction_analysis(cycles, permutation_trials=20, permutation_seed=7)

    assert "permutation_null_probability_reference_only" in with_reference
    assert "mean_spearman_null" in with_reference
    assert with_reference["permutation_trials"] == 20
    assert with_reference["gate3_passed"] == without_reference["gate3_passed"]
    assert 0.0 <= with_reference["permutation_null_probability_reference_only"] <= 1.0
