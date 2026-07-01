from pathlib import Path
import sys

import pytest


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "codesign_report.py"
    spec = importlib.util.spec_from_file_location("codesign_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def analysis_payload():
    return {
        "champion_migration": {
            "gate1_passed": True,
            "migrating_pair_count": 2,
            "pairs": [],
        },
        "oracle_gaps": {
            "gate2a_passed": True,
            "gate2a_ratio_mean": 0.8,
            "num_hw_meeting_threshold": 4,
            "fit_all_hw_set": ["000", "001"],
            "single_best_avg_mapping": "001",
            "per_hw_ratio": {"HW-A": 0.8, "HW-B": 0.8, "HW-C": 0.8, "HW-D": 0.8},
            "unconstrained_per_hw_oracle": {},
        },
        "cost_matched_pair": {
            "gate2b_passed": True,
            "gate2b_ratio": 0.8,
            "cost_matched_pair": ["HW-B", "HW-C"],
        },
        "interaction_analysis": {
            "gate3_passed": True,
            "mean_spearman": 0.3,
            "numeric_pair_count": 6,
        },
        "state_counts": {"measured": 32, "unavailable": 0, "retry_exhausted": 0},
    }


def test_build_verdict_positive_when_all_gates_pass():
    mod = load_module()

    verdict = mod.build_verdict(analysis_payload())

    assert verdict["end_state"] == "POSITIVE"
    assert verdict["claim_bearing"] is True
    assert verdict["data_label"] == "measured"
    assert verdict["gate1"]["passed"] is True
    assert verdict["gate2a"]["passed"] is True
    assert verdict["gate2b"]["passed"] is True
    assert verdict["gate3"]["passed"] is True
    assert len(verdict["scope_limitations"]) >= 5


def test_build_verdict_partial_when_retry_exhausted_exists():
    mod = load_module()
    payload = analysis_payload()
    payload["state_counts"]["retry_exhausted"] = 1

    verdict = mod.build_verdict(payload)

    assert verdict["end_state"] == "PARTIAL"


def test_report_contains_required_chinese_sections():
    mod = load_module()
    verdict = mod.build_verdict(analysis_payload())
    report = mod.render_report(analysis_payload(), verdict)

    for section in mod.REQUIRED_REPORT_SECTIONS:
        assert section in report
    mod.lint_report_sections(report)


def test_report_linter_rejects_missing_section():
    mod = load_module()

    with pytest.raises(mod.ReportSectionMissing):
        mod.lint_report_sections("# 缺少内容\n")
