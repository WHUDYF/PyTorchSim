#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


def _load_stats_contract():
    path = Path(__file__).resolve().with_name("stats_contract.py")
    spec = importlib.util.spec_from_file_location("stats_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stats = _load_stats_contract()


def measured_mapping_ids(cycles_by_hw: dict[str, dict[str, int | float]]) -> list[str]:
    ids = set()
    for cycles in cycles_by_hw.values():
        ids.update(cycles)
    return sorted(ids)


def compute_champion_migration(
    cycles_by_hw: dict[str, dict[str, int | float]],
    dominance_threshold: float = 0.03,
) -> dict[str, Any]:
    pairs = []
    migrating_pair_count = 0
    for hw_i, hw_j in combinations(sorted(cycles_by_hw), 2):
        cycles_i = cycles_by_hw[hw_i]
        cycles_j = cycles_by_hw[hw_j]
        if not cycles_i or not cycles_j:
            pairs.append({"hw_pair": [hw_i, hw_j], "status": "insufficient_evidence"})
            continue
        champ_i = stats.argmin_mapping(cycles_i)
        champ_j = stats.argmin_mapping(cycles_j)
        if champ_i == champ_j:
            pairs.append(
                {
                    "hw_pair": [hw_i, hw_j],
                    "status": "same_champion",
                    "champions": {hw_i: champ_i, hw_j: champ_j},
                    "swap_dominance": 0.0,
                }
            )
            continue
        if champ_j not in cycles_i or champ_i not in cycles_j:
            pairs.append(
                {
                    "hw_pair": [hw_i, hw_j],
                    "status": "insufficient_evidence",
                    "champions": {hw_i: champ_i, hw_j: champ_j},
                }
            )
            continue
        dom_i = float(cycles_i[champ_j]) / float(cycles_i[champ_i]) - 1.0
        dom_j = float(cycles_j[champ_i]) / float(cycles_j[champ_j]) - 1.0
        swap = min(dom_i, dom_j)
        passed = swap >= dominance_threshold
        if passed:
            migrating_pair_count += 1
        pairs.append(
            {
                "hw_pair": [hw_i, hw_j],
                "status": "numeric",
                "champions": {hw_i: champ_i, hw_j: champ_j},
                "dominance": {hw_i: dom_i, hw_j: dom_j},
                "swap_dominance": swap,
                "passed": passed,
            }
        )
    return {
        "dominance_threshold": dominance_threshold,
        "migrating_pair_count": migrating_pair_count,
        "pairs": pairs,
        "gate1_passed": migrating_pair_count > 0,
    }


def fit_all_hw_set(cycles_by_hw: dict[str, dict[str, int | float]]) -> list[str]:
    hw_ids = sorted(cycles_by_hw)
    if not hw_ids:
        return []
    common = set(cycles_by_hw[hw_ids[0]])
    for hw_id in hw_ids[1:]:
        common &= set(cycles_by_hw[hw_id])
    return sorted(common)


def compute_oracle_gaps(cycles_by_hw: dict[str, dict[str, int | float]]) -> dict[str, Any]:
    common = fit_all_hw_set(cycles_by_hw)
    unconstrained = {}
    for hw_id, cycles in cycles_by_hw.items():
        if cycles:
            mapping_id = stats.argmin_mapping(cycles)
            unconstrained[hw_id] = {"mapping_id": mapping_id, "cycles": cycles[mapping_id]}
        else:
            unconstrained[hw_id] = {"mapping_id": None, "cycles": None}
    if not common:
        return {
            "fit_all_hw_set": [],
            "gate2a_status": "insufficient_evidence",
            "gate2a_passed": False,
            "unconstrained_per_hw_oracle": unconstrained,
        }

    avg_cycles = {
        mapping_id: stats.arithmetic_mean([cycles_by_hw[hw_id][mapping_id] for hw_id in sorted(cycles_by_hw)])
        for mapping_id in common
    }
    single_best = stats.argmin_mapping(avg_cycles)
    per_hw_oracle = {}
    ratios = {}
    for hw_id in sorted(cycles_by_hw):
        constrained = {mapping_id: cycles_by_hw[hw_id][mapping_id] for mapping_id in common}
        best = stats.argmin_mapping(constrained)
        best_cycles = constrained[best]
        ratio = float(best_cycles) / float(cycles_by_hw[hw_id][single_best])
        per_hw_oracle[hw_id] = {"mapping_id": best, "cycles": best_cycles}
        ratios[hw_id] = ratio
    ratio_mean = stats.arithmetic_mean(list(ratios.values()))
    meeting = sum(1 for ratio in ratios.values() if ratio <= 0.85)
    return {
        "fit_all_hw_set": common,
        "single_best_avg_mapping": single_best,
        "single_best_avg_cycles": avg_cycles[single_best],
        "per_hw_oracle": per_hw_oracle,
        "per_hw_ratio": ratios,
        "gate2a_ratio_mean": ratio_mean,
        "num_hw_meeting_threshold": meeting,
        "gate2a_passed": ratio_mean <= 0.85 and meeting >= 3,
        "gate2a_status": "numeric",
        "unconstrained_per_hw_oracle": unconstrained,
    }


def compute_cost_matched_pair_upper_bound(
    cycles_by_hw: dict[str, dict[str, int | float]],
    pair: list[str] | None = None,
) -> dict[str, Any]:
    pair = pair or ["HW-B", "HW-C"]
    best_cycles: dict[str, int | float] = {}
    best_mapping: dict[str, str] = {}
    for hw_id in pair:
        cycles = cycles_by_hw.get(hw_id, {})
        if not cycles:
            return {
                "cost_matched_pair": pair,
                "gate2b_status": "insufficient_evidence",
                "gate2b_passed": False,
            }
        mapping_id = stats.argmin_mapping(cycles)
        best_mapping[hw_id] = mapping_id
        best_cycles[hw_id] = cycles[mapping_id]
    low = min(float(value) for value in best_cycles.values())
    high = max(float(value) for value in best_cycles.values())
    ratio = low / high if high else 0.0
    return {
        "cost_matched_pair": pair,
        "best_mapping": best_mapping,
        "best_cycles": best_cycles,
        "gate2b_ratio": ratio,
        "gate2b_passed": ratio <= 0.85,
        "gate2b_status": "numeric",
        "framing": "ex-post cost-tier allocation sensitivity upper bound",
    }


def _ranking_vector(cycles: dict[str, int | float], mapping_ids: list[str]) -> list[float]:
    return stats.average_ranks([cycles[mapping_id] for mapping_id in mapping_ids])


def _mean_pairwise_spearman(ranking_by_hw: dict[str, list[float]]) -> float | None:
    values = []
    for hw_i, hw_j in combinations(sorted(ranking_by_hw), 2):
        values.append(stats.spearman(ranking_by_hw[hw_i], ranking_by_hw[hw_j]))
    return stats.arithmetic_mean(values) if values else None


def permutation_reference(
    cycles_by_hw: dict[str, dict[str, int | float]],
    trials: int,
    seed: int,
    threshold: float = 0.7,
) -> dict[str, Any]:
    common = fit_all_hw_set(cycles_by_hw)
    if trials <= 0 or len(common) < 4 or len(cycles_by_hw) < 2:
        return {
            "permutation_trials": trials,
            "mean_spearman_null": None,
            "permutation_null_probability_reference_only": None,
        }
    rng = random.Random(seed)
    base_vectors = {
        hw_id: _ranking_vector(cycles_by_hw[hw_id], common)
        for hw_id in sorted(cycles_by_hw)
    }
    null_values = []
    for _ in range(trials):
        shuffled = {}
        for hw_id, ranks in base_vectors.items():
            candidate = list(ranks)
            rng.shuffle(candidate)
            shuffled[hw_id] = candidate
        null_values.append(_mean_pairwise_spearman(shuffled))
    numeric = [value for value in null_values if value is not None]
    probability = sum(1 for value in numeric if value <= threshold) / len(numeric) if numeric else None
    return {
        "permutation_trials": trials,
        "permutation_seed": seed,
        "mean_spearman_null": stats.arithmetic_mean(numeric) if numeric else None,
        "permutation_null_probability_reference_only": probability,
    }


def compute_interaction_analysis(
    cycles_by_hw: dict[str, dict[str, int | float]],
    permutation_trials: int = 0,
    permutation_seed: int = 0,
) -> dict[str, Any]:
    pairs = []
    numeric = []
    for hw_i, hw_j in combinations(sorted(cycles_by_hw), 2):
        intersect = sorted(set(cycles_by_hw[hw_i]) & set(cycles_by_hw[hw_j]))
        if len(intersect) < 4:
            pairs.append(
                {
                    "hw_pair": [hw_i, hw_j],
                    "status": "insufficient_evidence",
                    "n_intersect": len(intersect),
                }
            )
            continue
        xs = [cycles_by_hw[hw_i][mapping_id] for mapping_id in intersect]
        ys = [cycles_by_hw[hw_j][mapping_id] for mapping_id in intersect]
        rho = stats.spearman(xs, ys)
        numeric.append(rho)
        pairs.append(
            {
                "hw_pair": [hw_i, hw_j],
                "status": "numeric",
                "n_intersect": len(intersect),
                "spearman": rho,
            }
        )
    mean_spearman = stats.arithmetic_mean(numeric) if numeric else None
    numeric_pair_count = len(numeric)
    passed = bool(mean_spearman is not None and mean_spearman <= 0.7 and numeric_pair_count >= 5)
    result = {
        "pairs": pairs,
        "numeric_pair_count": numeric_pair_count,
        "mean_spearman": mean_spearman,
        "gate3_passed": passed,
    }
    if permutation_trials:
        result.update(permutation_reference(cycles_by_hw, permutation_trials, permutation_seed))
    return result


def analyze_codesign(
    cycles_by_hw: dict[str, dict[str, int | float]],
    permutation_trials: int = 0,
    permutation_seed: int = 0,
    gate1_dominance_threshold: float = 0.03,
) -> dict[str, Any]:
    return {
        "champion_migration": compute_champion_migration(
            cycles_by_hw,
            dominance_threshold=gate1_dominance_threshold,
        ),
        "oracle_gaps": compute_oracle_gaps(cycles_by_hw),
        "cost_matched_pair": compute_cost_matched_pair_upper_bound(cycles_by_hw),
        "interaction_analysis": compute_interaction_analysis(
            cycles_by_hw,
            permutation_trials=permutation_trials,
            permutation_seed=permutation_seed,
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze measured co-design cycle matrix.")
    parser.add_argument("--cycles-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutation-trials", type=int, default=0)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--gate1-dominance-threshold", type=float, default=0.03)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cycles = json.loads(args.cycles_json.read_text(encoding="utf-8"))
    result = analyze_codesign(
        cycles,
        permutation_trials=args.permutation_trials,
        permutation_seed=args.permutation_seed,
        gate1_dominance_threshold=args.gate1_dominance_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "champion_migration.json").write_text(
        json.dumps(result["champion_migration"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "oracle_gaps.json").write_text(
        json.dumps(result["oracle_gaps"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "cost_matched_pair.json").write_text(
        json.dumps(result["cost_matched_pair"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "interaction_analysis.json").write_text(
        json.dumps(result["interaction_analysis"], indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
