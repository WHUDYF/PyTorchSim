# v1 verdict re-analysis under 10% Gate-1 threshold

## 1. Motivation

commit `8c5af69` 中的 `HW-A/006` 5-repeat determinism diagnosis 将现象分类为 `case alpha`：v1 setup 和 v2 setup 在同一 measurement path 下都出现约 7% 的 `cycle_delta`。这说明原始 v1 verdict 使用的 Gate-1 dominance threshold `0.03` 低于当前 canary noise floor，不能继续作为可靠的可区分性阈值。

本次 re-analysis 只重新分析已经测得的 v1 4x8 cycle matrix，不重新运行 TOGSim sweep，不修改 `heatmap_cycles.json` 或 `heatmap_cycle_stats.json` 中的 measured cycles。

## 2. What changed

唯一变化是 Gate-1 dominance threshold：

```text
0.03 -> 0.10
```

其他 gate 保持不变：

- Gate-2a per-chip vs single mapping threshold 仍为 `0.85`。
- Gate-2a `num_hw_meeting_threshold` requirement 仍为 `3 of 4`。
- Gate-2b co-design vs cost-matched pair threshold 仍为 `0.85`。
- Gate-3 `mean_spearman` threshold 仍为 `0.70`。
- Gate-3 `numeric_pair_count` requirement 仍为 `5 of 6`。
- Gate-3 `n_intersect` per pair requirement 仍为 `>= 4`。

新的 recomputation 产物位于：

```text
outputs/mapping_dse_codesign/gpt2_block_prefill_s128_run1_reanalysis_10pct/
```

## 3. Numerical comparison

| metric | original v1 verdict (Gate-1 threshold 0.03) | re-analysis (Gate-1 threshold 0.10) |
| --- | ---: | ---: |
| Gate-1 `max_dominance_pct` | 0.000000 | 0.000000 |
| Gate-1 `migrating_pair_count` | 0 | 0 |
| Gate-1 `passed` | false | false |
| Gate-2a `ratio_mean` | 1.000000 | 1.000000 |
| Gate-2a `num_hw_meeting_threshold` | 0 | 0 |
| Gate-2a `passed` | false | false |
| Gate-2b `ratio` | 0.417737 | 0.417737 |
| Gate-2b `passed` | true | true |
| Gate-3 `mean_spearman` | 0.984127 | 0.984127 |
| Gate-3 `numeric_pair_count` | 6 | 6 |
| Gate-3 `passed` | false | false |
| `end_state` | `NEGATIVE` | `NEGATIVE` |

## 4. Interpretation

`end_state` 没有改变，仍然是 `NEGATIVE`。

原因是原始 v1 4x8 matrix 中，四个硬件配置的 champion mapping 都是 `006`。因此 Gate-1 的 `migrating_pair_count` 原本就是 `0`，`max_dominance_pct` 也是 `0`。把 threshold 从 `0.03` 提高到 `0.10` 不会改变 Gate-1 的判断，因为这里并不存在接近阈值的 champion migration。

Gate-2a 也仍然失败：`single_best_avg_mapping` 与每个 `per_hw_oracle` 都是 `006`，所以 `per_hw_ratio` 全部为 `1.0`，没有硬件从 per-HW mapping choice 中获得 `>=15%` 的收益。

Gate-3 仍然失败：`mean_spearman = 0.984127`，说明 mapping 排序在不同硬件之间高度一致，而不是出现硬件相关的排序重排。测量 noise floor 约 7% 并不能挽救 Gate-3，因为 Gate-3 失败来自整体排序过于相似，而不是单个接近阈值的 cycle 差异。

Gate-2b 仍然通过，`HW-B/HW-C` 的 cost-matched pair ratio 为 `0.417737`。但这只是 ex-post cost-tier allocation sensitivity upper bound，不能单独推翻 Gate-1、Gate-2a 和 Gate-3 给出的 NEGATIVE 判断。

## 5. Implication for the research direction

这次 re-analysis 使 v1 结论在阈值校准上更严谨，但不改变核心结论：在当前 GPT-2 single block prefill seq=128、只搜索 tiling/mapping 的空间内，没有观察到足够的 HW/SW co-design interaction。更准确的表述是：原始 3% Gate-1 threshold 不可靠，但 v1 的 NEGATIVE verdict 并不依赖这个低阈值；即使把 Gate-1 提高到高于 canary noise floor 的 10%，结论仍保持 NEGATIVE。因此，它总体上加强了“tiling-only search space 不足以支撑 co-design interaction claim”的判断。
