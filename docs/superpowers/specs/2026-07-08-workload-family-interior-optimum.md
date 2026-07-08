# NPU workload family interior-optimum verification

Date: 2026-07-08

## 1. Motivation

Conv3x3 v2 已在 `23ea49f` 证明 single-workload `TILE_FUSION_INTERIOR_OPTIMUM_CONFIRMED_V2`。本轮把同一 tile x fusion x HW 方法扩展到 workload family，检查 interior optimum 是否只属于单个 Conv3x3 case，还是能跨 Conv-heavy family 保持。

## 2. Method

- Inherited workload: `conv3x3_probe`，直接引用 `tile_fusion_hw_matrix_v2.json` 与 `interior_optimum_analysis_v2.json`。
- Additional workloads: `conv3x3_large` 与 `conv1x1`。
- HW: `codesign_v1_2x2` 的 `HW-A/B/C/D`。
- Fusion variants: `none` 与 `all`。
- Repeats: 5 per measured cell，`pytorchsim_functional_mode=0`，`CUDA_VISIBLE_DEVICES=1`。
- Source rule: 未修改 PyTorchSim / TOGSim / gem5 / ramulator2 / spike source；新 workload 通过临时 harness 复用 external Conv tile mapping。

Discovery 文档：`outputs/route4_fusion_pilot/workload_family_discovery.md`。

## 3. Per-workload results

### conv3x3_probe

| HW | Champion tile | Fusion | Median cycles |
|---|---|---|---:|
| HW-A | `tile_C` | `all` | 5,880 |
| HW-B | `tile_C` | `all` | 16,716 |
| HW-C | `tile_A2` | `all` | 5,928 |
| HW-D | `tile_C` | `all` | 16,836 |

- `tile_interior_evidence`: `True`
- `global_min_cell`: `{"fusion": "all", "hw": "HW-A", "median": 5880.0, "tile": "tile_C"}`

### conv3x3_large

| HW | Champion tile | Fusion | Median cycles |
|---|---|---|---:|
| HW-A | `tile_A` | `all` | 6,545 |
| HW-B | `tile_B` | `all` | 19,420 |
| HW-C | `tile_C` | `all` | 6,611 |
| HW-D | `tile_C3` | `all` | 19,438 |

- `tile_interior_evidence`: `False`
- `global_min_cell`: `{"fusion": "all", "hw": "HW-A", "median": 6545.0, "tile": "tile_A"}`

### conv1x1

| HW | Champion tile | Fusion | Median cycles |
|---|---|---|---:|
| HW-A | `tile_B2` | `all` | 1,013 |
| HW-B | `tile_C` | `all` | 3,290 |
| HW-C | `tile_B2` | `all` | 1,007 |
| HW-D | `tile_B` | `all` | 3,289 |

- `tile_interior_evidence`: `True`
- `global_min_cell`: `{"fusion": "all", "hw": "HW-C", "median": 1007.0, "tile": "tile_B2"}`

## 4. Family-level verdict

- `workload_family_all_have_interior`: `False`
- `champion_tile_stability_across_workloads`: `0.417`
- `verdict`: `PARTIAL_WORKLOAD_FAMILY_INTERIOR`

## 5. Cross-workload comparison

| HW | Stability | Champion tiles by workload |
|---|---:|---|
| HW-A | 0.333 | `conv3x3_probe:tile_C`; `conv3x3_large:tile_A`; `conv1x1:tile_B2` |
| HW-B | 0.667 | `conv3x3_probe:tile_C`; `conv3x3_large:tile_B`; `conv1x1:tile_C` |
| HW-C | 0.333 | `conv3x3_probe:tile_A2`; `conv3x3_large:tile_C`; `conv1x1:tile_B2` |
| HW-D | 0.333 | `conv3x3_probe:tile_C`; `conv3x3_large:tile_C3`; `conv1x1:tile_B` |

## 6. Scope limitations

- Tested family is Conv-heavy but still small: inherited Conv3x3, Conv3x3-large, Conv1x1。
- Additional workloads have smaller valid tile working sets than the inherited `[1,64,56,56]` Conv3x3; neither reaches the smallest-SPAD 85-110% fit-boundary target with valid single-batch tiles。
- DepthwiseConv3x3 smoke test produced cycles but has grouped/multi-kernel lowering behavior, so it was not selected under the two-additional-workload cap。
- Mode=1 correctness remains deferred; this report is mode=0 timing evidence。
- Existing determinism diagnosis gives a non-trivial noise floor; medians should be interpreted as timing evidence, not exact silicon prediction。

## 7. Next step recommendation

下一步应优先扩大 workload family 或增加真实 HW interior points，因为当前 family evidence 已显示 workload-dependent tile optimum。
