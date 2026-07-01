#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Mapping


ARGMIN_TIE_BREAK = "lex_by_mapping_id"
EQUALITY_EPSILON = 0.0
SPEARMAN_TIES_METHOD = "average_rank"
CROSS_HW_AGGREGATION = "arithmetic_mean"
AUX_CROSS_HW_AGGREGATION = "geometric_mean"


def argmin_mapping(cycles_by_mapping: Mapping[str, int | float]) -> str:
    if not cycles_by_mapping:
        raise ValueError("argmin_mapping requires at least one mapping")
    return min(cycles_by_mapping, key=lambda mapping_id: (cycles_by_mapping[mapping_id], mapping_id))


def average_ranks(values: list[int | float]) -> list[float]:
    ordered = sorted((float(value), idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for _, idx in ordered[i : j + 1]:
            ranks[idx] = rank
        i = j + 1
    return ranks


def spearman(xs: list[int | float], ys: list[int | float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("spearman inputs must have the same length")
    if len(xs) < 2:
        raise ValueError("spearman requires at least two values")
    rx = average_ranks(xs)
    ry = average_ranks(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in rx))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ry))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def arithmetic_mean(values: list[int | float]) -> float:
    if not values:
        raise ValueError("arithmetic_mean requires at least one value")
    return sum(float(value) for value in values) / len(values)


def geometric_mean(values: list[int | float]) -> float:
    if not values:
        raise ValueError("geometric_mean requires at least one value")
    if any(float(value) <= 0 for value in values):
        raise ValueError("geometric_mean requires positive values")
    return math.exp(sum(math.log(float(value)) for value in values) / len(values))
