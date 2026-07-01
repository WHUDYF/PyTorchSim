#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_REPORT_SECTIONS = [
    "cycles 表",
    "状态表",
    "migrating_pair_count/6 摘要",
    "per-HW ratio 表",
    "unconstrained vs constrained oracle 对比",
    "Gate-3 与 Gate-1 联合解读",
    "cost-matched pair 分析",
    "反 baseline (HW-D) 参考值",
    "实验边界",
]


class ReportSectionMissing(ValueError):
    pass


def _load_metadata_module():
    path = Path(__file__).resolve().with_name("codesign_metadata.py")
    spec = importlib.util.spec_from_file_location("codesign_metadata", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


metadata = _load_metadata_module()


def _gate_status(section: dict[str, Any], passed_key: str) -> dict[str, Any]:
    status = section.get(passed_key)
    raw_status = section.get(passed_key.replace("_passed", "_status"))
    return {
        "passed": bool(status),
        "status": raw_status or ("passed" if status else "failed"),
    }


def _has_insufficient_evidence(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("status") == "insufficient_evidence":
            return True
        return any(_has_insufficient_evidence(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_insufficient_evidence(item) for item in value)
    return False


def build_verdict(analysis: dict[str, Any]) -> dict[str, Any]:
    champion = analysis.get("champion_migration", {})
    oracle = analysis.get("oracle_gaps", {})
    cost_pair = analysis.get("cost_matched_pair", {})
    interaction = analysis.get("interaction_analysis", {})
    state_counts = analysis.get("state_counts", {})

    gate1 = _gate_status(champion, "gate1_passed")
    gate2a = _gate_status(oracle, "gate2a_passed")
    gate2b = _gate_status(cost_pair, "gate2b_passed")
    gate3 = _gate_status(interaction, "gate3_passed")
    gates = [gate1, gate2a, gate2b, gate3]

    has_retry_exhausted = int(state_counts.get("retry_exhausted", 0) or 0) > 0
    has_insufficient = _has_insufficient_evidence(analysis)
    if has_insufficient:
        end_state = "PARTIAL"
    elif not gate1["passed"] or (not gate2a["passed"] and not gate2b["passed"]):
        end_state = "NEGATIVE"
    elif has_retry_exhausted:
        end_state = "PARTIAL"
    elif all(gate["passed"] for gate in gates):
        end_state = "POSITIVE"
    else:
        end_state = "PARTIAL"

    verdict = {
        "end_state": end_state,
        "claim_bearing": end_state == "POSITIVE",
        "data_label": "measured",
        "gate1": gate1,
        "gate2a": gate2a,
        "gate2b": gate2b,
        "gate3": gate3,
        "state_counts": dict(state_counts),
        "scope_limitations": [
            "当前结论限定在单一 GPT-2 block prefill seq=128 类工作负载。",
            "搜索空间只覆盖 tile/mapping 选择，不覆盖 fusion、dataflow、SPAD partition 或 DMA schedule。",
            "硬件空间只覆盖 2x2 corner 组合，不能证明内部参数空间单调或最优。",
            "DRAM channel 与 ICNT port 在本实验中绑定变化，不能拆分解释二者的独立贡献。",
            "当前指标是 cycle-only，不包含面积、功耗、能耗、带宽成本或可布线性。",
        ],
    }
    cross_check = analysis.get("v1_v2_cross_check")
    if isinstance(cross_check, dict):
        verdict["v1_v2_cross_check"] = cross_check
        warning_cells = cross_check.get("warning_cells", [])
        if warning_cells:
            verdict["scope_limitations"].append(
                "v1-v2 cross check failed on cells: " + ", ".join(str(cell) for cell in warning_cells)
            )
        if cross_check.get("force_partial") and verdict["end_state"] == "POSITIVE":
            verdict["end_state"] = "PARTIAL"
            verdict["claim_bearing"] = False
    metadata.lint_claim_bearing(verdict)
    return verdict


def _fmt_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_无可展示行_"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_fmt_number(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _cycles_table(cycles_by_hw: dict[str, dict[str, Any]] | None) -> str:
    if not cycles_by_hw:
        return "_未提供 cycles-json，当前报告只展示分析摘要。_"
    mapping_ids = sorted({mapping_id for row in cycles_by_hw.values() for mapping_id in row})
    rows = []
    for hw_id in sorted(cycles_by_hw):
        rows.append([hw_id, *[cycles_by_hw[hw_id].get(mapping_id, "N/A") for mapping_id in mapping_ids]])
    return _markdown_table(["HW", *mapping_ids], rows)


def _state_table(analysis: dict[str, Any]) -> str:
    state_counts = analysis.get("state_counts", {})
    rows = [[state, count] for state, count in sorted(state_counts.items())]
    return _markdown_table(["状态", "数量"], rows)


def _migration_table(champion: dict[str, Any]) -> str:
    rows = []
    for item in champion.get("pairs", []):
        pair = item.get("hw_pair", [])
        champions = item.get("champions", {})
        champ_text = ", ".join(f"{hw}:{mapping}" for hw, mapping in sorted(champions.items()))
        rows.append(
            [
                "/".join(pair),
                item.get("status"),
                champ_text,
                item.get("swap_dominance"),
                item.get("passed", ""),
            ]
        )
    return _markdown_table(["硬件对", "状态", "champion", "swap_dominance", "通过"], rows)


def _ratio_table(oracle: dict[str, Any]) -> str:
    ratios = oracle.get("per_hw_ratio", {})
    rows = [[hw_id, ratio] for hw_id, ratio in sorted(ratios.items())]
    return _markdown_table(["HW", "per-HW ratio"], rows)


def _oracle_compare_table(oracle: dict[str, Any]) -> str:
    constrained = oracle.get("per_hw_oracle", {})
    unconstrained = oracle.get("unconstrained_per_hw_oracle", {})
    hw_ids = sorted(set(constrained) | set(unconstrained))
    rows = []
    for hw_id in hw_ids:
        con = constrained.get(hw_id, {})
        uncon = unconstrained.get(hw_id, {})
        rows.append(
            [
                hw_id,
                con.get("mapping_id"),
                con.get("cycles"),
                uncon.get("mapping_id"),
                uncon.get("cycles"),
            ]
        )
    return _markdown_table(["HW", "constrained mapping", "constrained cycles", "unconstrained mapping", "unconstrained cycles"], rows)


def render_report(
    analysis: dict[str, Any],
    verdict: dict[str, Any],
    cycles_by_hw: dict[str, dict[str, Any]] | None = None,
) -> str:
    champion = analysis.get("champion_migration", {})
    oracle = analysis.get("oracle_gaps", {})
    cost_pair = analysis.get("cost_matched_pair", {})
    interaction = analysis.get("interaction_analysis", {})
    cross_check = analysis.get("v1_v2_cross_check")
    hw_d = {}
    if cycles_by_hw:
        hw_d = cycles_by_hw.get("HW-D", {})

    lines = [
        "# NPU mapping DSE co-design 实验报告",
        "",
        f"结论状态: `{verdict['end_state']}`",
        f"数据标签: `{verdict['data_label']}`",
        "",
        "## cycles 表",
        _cycles_table(cycles_by_hw),
        "",
        "## 状态表",
        _state_table(analysis),
        "",
        "## migrating_pair_count/6 摘要",
        f"migrating_pair_count = {champion.get('migrating_pair_count', 'N/A')}/6；Gate-1 passed = {verdict['gate1']['passed']}。",
        _migration_table(champion),
        "",
        "## per-HW ratio 表",
        f"Gate-2a ratio mean = {_fmt_number(oracle.get('gate2a_ratio_mean'))}；meeting threshold = {oracle.get('num_hw_meeting_threshold', 'N/A')}。",
        _ratio_table(oracle),
        "",
        "## unconstrained vs constrained oracle 对比",
        f"fit-all-HW mapping set = {oracle.get('fit_all_hw_set', [])}；single-best-avg mapping = {oracle.get('single_best_avg_mapping', 'N/A')}。",
        _oracle_compare_table(oracle),
        "",
        "## Gate-3 与 Gate-1 联合解读",
        f"Gate-3 passed = {verdict['gate3']['passed']}；mean Spearman = {_fmt_number(interaction.get('mean_spearman'))}；numeric pair count = {interaction.get('numeric_pair_count', 'N/A')}。",
        "Gate-1 观察 champion 是否迁移，Gate-3 观察 mapping 排序在不同硬件之间是否保持一致。若二者同时成立，说明硬件参数变化不仅改变最优点，也改变了搜索空间的相对排序。",
        "",
        "## cost-matched pair 分析",
        f"pair = {cost_pair.get('cost_matched_pair', [])}；ratio = {_fmt_number(cost_pair.get('gate2b_ratio'))}；Gate-2b passed = {verdict['gate2b']['passed']}。",
        f"framing = {cost_pair.get('framing', 'N/A')}。",
        "",
        "## v1-v2 cross check",
        _cross_check_section(cross_check),
        "",
        "## 反 baseline (HW-D) 参考值",
        _markdown_table(["mapping", "cycles"], [[mapping_id, cycles] for mapping_id, cycles in sorted(hw_d.items())])
        if hw_d
        else "_未提供 HW-D cycles。_",
        "",
        "## 实验边界",
        "\n".join(f"- {item}" for item in verdict["scope_limitations"]),
    ]
    report = "\n".join(lines) + "\n"
    lint_report_sections(report)
    return report


def _cross_check_section(cross_check: dict[str, Any] | None) -> str:
    if not cross_check:
        return "_未提供 v1-v2 cross check 数据。_"
    rows = []
    for cell_id, row in sorted(cross_check.get("v1_v2_cycle_delta_per_shared_cell", {}).items()):
        rows.append([cell_id, row.get("v1_median_cycles"), row.get("v2_median_cycles"), row.get("cycle_delta")])
    prefix = (
        f"cross_check_status = {cross_check.get('cross_check_status', 'N/A')}；"
        f"max_allowed_delta = {_fmt_number(cross_check.get('max_allowed_delta'))}。"
    )
    return prefix + "\n" + _markdown_table(["cell", "v1 median", "v2 median", "cycle_delta"], rows)


def lint_report_sections(report: str) -> dict[str, Any]:
    missing = [section for section in REQUIRED_REPORT_SECTIONS if section not in report]
    if missing:
        raise ReportSectionMissing("report missing required sections: " + ", ".join(missing))
    return {"ok": True, "checked_sections": REQUIRED_REPORT_SECTIONS}


def load_analysis_dir(path: Path) -> dict[str, Any]:
    payload = {
        "champion_migration": json.loads((path / "champion_migration.json").read_text(encoding="utf-8")),
        "oracle_gaps": json.loads((path / "oracle_gaps.json").read_text(encoding="utf-8")),
        "cost_matched_pair": json.loads((path / "cost_matched_pair.json").read_text(encoding="utf-8")),
        "interaction_analysis": json.loads((path / "interaction_analysis.json").read_text(encoding="utf-8")),
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Chinese co-design DSE verdict report.")
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--sweep-summary-json", type=Path)
    parser.add_argument("--cycles-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analysis = load_analysis_dir(args.analysis_dir)
    if args.sweep_summary_json:
        sweep_summary = json.loads(args.sweep_summary_json.read_text(encoding="utf-8"))
        analysis["state_counts"] = sweep_summary.get("state_counts", {})
    cycles_by_hw = None
    if args.cycles_json:
        cycles_by_hw = json.loads(args.cycles_json.read_text(encoding="utf-8"))
    verdict = build_verdict(analysis)
    report = render_report(analysis, verdict, cycles_by_hw=cycles_by_hw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
