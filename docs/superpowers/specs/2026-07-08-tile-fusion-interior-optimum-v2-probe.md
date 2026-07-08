# tile x fusion x HW interior-optimum v2 probe

Date: 2026-07-08

## 1. Motivation

`5f16398` 的 v1 verdict 是 `CHAMPION_MIGRATION_ONLY`：best `(tile, fusion)` 会随 HW 迁移，但 global minimum 仍在 corner。v2 的目标是在不增加 HW 维度的前提下，把 software tile space 从 4 个加密到 8 个，并加入第三个 fusion variant `fusion = ["fusion"]`，观察是否能把结果推进到 tile interior optimum。

## 2. Method

- Workload: Conv 3x3 `[1, 64, 56, 56] -> [1, 64, 56, 56]`，使用既有 `conv3x3_probe`。
- HW: `codesign_v1_2x2` 的 `HW-A/B/C/D`，不增加 HW interior points。
- Tile: `tile_A/A2/B/B2/C/C2/C3/D` 共 8 个，全部保持 `TILE_O_H=56`；近边界点通过合法的 `TILE_O_W` 与 `TILE_K` 组合形成。
- Fusion: `['none', 'fusion', 'all']`，其中 `fusion` 表示 YAML list `['fusion']`。
- Repeats: `5` per cell，`pytorchsim_functional_mode=0`，seed 0，独立 subprocess。
- Fit rule: 若 `working_set > usable_spad_bytes`，该 cell 标记为 `unavailable` 并跳过 TOGSim。

Run accounting:

```json
{
  "total_cells": 96,
  "planned_run_slots": 480,
  "measured_runs": 450,
  "unavailable_run_slots": 30,
  "runtime_failed_cells": 0,
  "not_run_timebox_cells": 0
}
```

Fusion discovery 文档：`outputs/route4_fusion_pilot/fusion_variants_discovery.md`。

## 3. Full 4-HW x 8-tile x N-fusion matrix

| HW | Tile | Working set KB | Small-SPAD fit | none median/state | fusion median/state | all median/state |
|---|---|---|---|---|---|---|
| HW-A | tile_A | 487.0 | yes | 185,433 / `measured` | 181,608 / `measured` | 6,115 / `measured` |
| HW-A | tile_A2 | 657.0 | yes | 184,538 / `measured` | 180,720 / `measured` | 6,194 / `measured` |
| HW-A | tile_B | 912.0 | yes | 184,341 / `measured` | 181,506 / `measured` | 6,129 / `measured` |
| HW-A | tile_B2 | 1145.5 | yes | 290,423 / `measured` | 289,339 / `measured` | 6,140 / `measured` |
| HW-A | tile_C | 1507.0 | yes | 185,891 / `measured` | 185,336 / `measured` | 5,880 / `measured` |
| HW-A | tile_C2 | 1850.2 | yes | 536,391 / `measured` | 539,427 / `measured` | 6,174 / `measured` |
| HW-A | tile_C3 | 1976.0 | yes | 311,237 / `measured` | 314,025 / `measured` | 6,180 / `measured` |
| HW-A | tile_D | 2384.0 | yes | 162,261 / `measured` | 165,365 / `measured` | 6,089 / `measured` |
| HW-B | tile_A | 487.0 | yes | 210,648 / `measured` | 196,015 / `measured` | 17,102 / `measured` |
| HW-B | tile_A2 | 657.0 | yes | 211,200 / `measured` | 197,562 / `measured` | 17,165 / `measured` |
| HW-B | tile_B | 912.0 | yes | 211,496 / `measured` | 198,397 / `measured` | 16,827 / `measured` |
| HW-B | tile_B2 | 1145.5 | yes | 329,293 / `measured` | 318,190 / `measured` | 16,852 / `measured` |
| HW-B | tile_C | 1507.0 | yes | 216,923 / `measured` | 205,609 / `measured` | 16,716 / `measured` |
| HW-B | tile_C2 | 1850.2 | yes | 567,092 / `measured` | 559,712 / `measured` | 16,966 / `measured` |
| HW-B | tile_C3 | 1976.0 | yes | 355,138 / `measured` | 347,849 / `measured` | 16,945 / `measured` |
| HW-B | tile_D | 2384.0 | yes | 214,895 / `measured` | 207,511 / `measured` | 16,870 / `measured` |
| HW-C | tile_A | 487.0 | yes | 185,256 / `measured` | 181,391 / `measured` | 6,258 / `measured` |
| HW-C | tile_A2 | 657.0 | yes | 184,061 / `measured` | 180,853 / `measured` | 5,928 / `measured` |
| HW-C | tile_B | 912.0 | yes | 184,181 / `measured` | 181,714 / `measured` | 6,019 / `measured` |
| HW-C | tile_B2 | 1145.5 | yes | 290,188 / `measured` | 289,282 / `measured` | 6,095 / `measured` |
| HW-C | tile_C | 1507.0 | yes | 186,285 / `measured` | 185,454 / `measured` | 6,402 / `measured` |
| HW-C | tile_C2 | 1850.2 | yes | 536,287 / `measured` | 539,597 / `measured` | 5,988 / `measured` |
| HW-C | tile_C3 | 1976.0 | yes | 310,920 / `measured` | 313,994 / `measured` | 6,158 / `measured` |
| HW-C | tile_D | 2384.0 | no | unavailable / `unavailable` | unavailable / `unavailable` | unavailable / `unavailable` |
| HW-D | tile_A | 487.0 | yes | 210,632 / `measured` | 196,309 / `measured` | 17,131 / `measured` |
| HW-D | tile_A2 | 657.0 | yes | 211,474 / `measured` | 197,282 / `measured` | 17,051 / `measured` |
| HW-D | tile_B | 912.0 | yes | 211,675 / `measured` | 198,675 / `measured` | 16,916 / `measured` |
| HW-D | tile_B2 | 1145.5 | yes | 330,077 / `measured` | 318,249 / `measured` | 16,881 / `measured` |
| HW-D | tile_C | 1507.0 | yes | 217,066 / `measured` | 205,391 / `measured` | 16,836 / `measured` |
| HW-D | tile_C2 | 1850.2 | yes | 567,309 / `measured` | 559,481 / `measured` | 16,860 / `measured` |
| HW-D | tile_C3 | 1976.0 | yes | 355,101 / `measured` | 347,585 / `measured` | 16,861 / `measured` |
| HW-D | tile_D | 2384.0 | no | unavailable / `unavailable` | unavailable / `unavailable` | unavailable / `unavailable` |

完整 raw cycles 保存在 `outputs/route4_fusion_pilot/tile_fusion_hw_matrix_v2.json`。

## 4. Fit-boundary map

| Tile | Regime | Working set KB | Small-SPAD fit fraction | HW-A | HW-B | HW-C | HW-D |
|---|---|---:|---:|---|---|---|---|
| tile_A | tiny | 487.0 | 0.238 | yes | yes | yes | yes |
| tile_A2 | tiny_dense | 657.0 | 0.321 | yes | yes | yes | yes |
| tile_B | small | 912.0 | 0.445 | yes | yes | yes | yes |
| tile_B2 | small_dense | 1145.5 | 0.559 | yes | yes | yes | yes |
| tile_C | medium | 1507.0 | 0.736 | yes | yes | yes | yes |
| tile_C2 | near_boundary_fit | 1850.2 | 0.903 | yes | yes | yes | yes |
| tile_C3 | fit_boundary | 1976.0 | 0.965 | yes | yes | yes | yes |
| tile_D | large_unavailable_small_spad | 2384.0 | 1.164 | yes | yes | no | no |

`tile_C2` 和 `tile_C3` 是 v2 加密后靠近 small-SPAD fit boundary 的两个关键点；`tile_D` 仍超过 small-SPAD double-buffer budget，因此在 `HW-C/HW-D` 上不可用。

## 5. Champion analysis

| HW | Champion tile | Champion fusion | Median cycles |
|---|---|---|---:|
| HW-A | tile_C | all | 5,880 |
| HW-B | tile_C | all | 16,716 |
| HW-C | tile_A2 | all | 5,928 |
| HW-D | tile_C | all | 16,836 |

`distinct_champion_tuples = 2`，`champion_migration = True`。

Global minimum:

```json
{
  "hw": "HW-A",
  "tile": "tile_C",
  "fusion": "all",
  "median": 5880.0
}
```

## 6. Interior optimum verdict

`TILE_FUSION_INTERIOR_OPTIMUM_CONFIRMED_V2`

- `global_min_hw_is_corner_hw = True`。本 handoff 只有 2x2 HW corners，所以 HW interior 证据必然为 false。
- `global_min_tile_is_extreme_for_hw = False`。
- `tile_interior_evidence = True`。
- v1 global min median 是 `5921`；v2 对比为：

```json
{
  "v1_global_min_median": 5921.0,
  "v2_global_min_median": 5880.0,
  "absolute_cycle_delta": 41.0,
  "relative_improvement": 0.00692450599560885
}
```

## 7. Next step recommendation

本轮已经得到 tile interior optimum，下一步应换到 ResNet stage 或 GPT-2 block 做 generalization。
