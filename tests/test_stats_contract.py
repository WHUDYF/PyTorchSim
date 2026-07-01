from pathlib import Path
import sys

import pytest


def load_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "stats_contract.py"
    spec = importlib.util.spec_from_file_location("stats_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contract_constants_are_locked():
    mod = load_module()

    assert mod.ARGMIN_TIE_BREAK == "lex_by_mapping_id"
    assert mod.EQUALITY_EPSILON == 0.0
    assert mod.SPEARMAN_TIES_METHOD == "average_rank"
    assert mod.CROSS_HW_AGGREGATION == "arithmetic_mean"
    assert mod.AUX_CROSS_HW_AGGREGATION == "geometric_mean"


def test_argmin_cycles_tie_breaks_by_mapping_id():
    mod = load_module()
    cycles = {"002": 100, "001": 100, "003": 120}

    assert mod.argmin_mapping(cycles) == "001"


def test_average_rank_handles_ties():
    mod = load_module()

    assert mod.average_ranks([1, 2, 2, 3]) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_with_average_rank_ties():
    mod = load_module()

    assert mod.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)


def test_cross_hw_arithmetic_and_geometric_mean():
    mod = load_module()

    assert mod.arithmetic_mean([10, 20, 30]) == pytest.approx(20.0)
    assert mod.geometric_mean([4, 9]) == pytest.approx(6.0)
