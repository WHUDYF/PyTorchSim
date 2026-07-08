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
| HW-B | `tile_A_b` | `all` | 19,382 |
| HW-C | `tile_A_a` | `all` | 6,506 |
| HW-D | `tile_C3` | `all` | 19,438 |

- `tile_interior_evidence`: `True`
- `global_min_cell`: `{"fusion": "all", "hw": "HW-C", "median": 6506.0, "tile": "tile_A_a"}`

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

- `workload_family_all_have_interior`: `True`
- `champion_tile_stability_across_workloads`: `0.417`
- `verdict`: `WORKLOAD_FAMILY_INTERIOR_OPTIMUM_CONFIRMED`

## 5. Cross-workload comparison

| HW | Stability | Champion tiles by workload |
|---|---:|---|
| HW-A | 0.333 | `conv3x3_probe:tile_C`; `conv3x3_large:tile_A`; `conv1x1:tile_B2` |
| HW-B | 0.667 | `conv3x3_probe:tile_C`; `conv3x3_large:tile_A_b`; `conv1x1:tile_C` |
| HW-C | 0.333 | `conv3x3_probe:tile_A2`; `conv3x3_large:tile_A_a`; `conv1x1:tile_B2` |
| HW-D | 0.333 | `conv3x3_probe:tile_C`; `conv3x3_large:tile_C3`; `conv1x1:tile_B` |

## 6. Scope limitations

- Tested family is Conv-heavy but still small: inherited Conv3x3, Conv3x3-large, Conv1x1。
- Additional workloads have smaller valid tile working sets than the inherited `[1,64,56,56]` Conv3x3; neither reaches the smallest-SPAD 85-110% fit-boundary target with valid single-batch tiles。
- The upgraded family verdict depends on densifying only the `conv3x3_large` region between `tile_A` and `tile_A2`; it is still not a broad ResNet/MobileNet family claim。
- DepthwiseConv3x3 smoke test produced cycles but has grouped/multi-kernel lowering behavior, so it was not selected under the two-additional-workload cap。
- Mode=1 correctness remains deferred; this report is mode=0 timing evidence。
- Existing determinism diagnosis gives a non-trivial noise floor; medians should be interpreted as timing evidence, not exact silicon prediction。

## 7. Next step recommendation

下一步应优先扩大 workload family 或增加真实 HW interior points。当前 densified family evidence 已支持 Conv-heavy 小 family 的 interior optimum，但 champion tile 仍明显 workload-dependent。

## 8. Conv3x3-large densification follow-up

为检查 `conv3x3_large/HW-A` 的 `tile_A` 是否只是粗粒度 tile search 造成的 boundary artifact，本轮只在 `tile_A` 与 `tile_A2` 之间加入 3 个新 tile，并只 sweep `conv3x3_large`。

| New tile | TILE_O_W | TILE_K | TILE_N | Working set bytes |
|---|---:|---:|---:|---:|
| `tile_A_a` | 2 | 16 | 64 | 58,880 |
| `tile_A_b` | 3 | 16 | 64 | 67,968 |
| `tile_A_c` | 4 | 16 | 64 | 77,056 |

| HW | Champion after densification | Fusion | Median cycles |
|---|---|---|---:|
| HW-A | `tile_A` | `all` | 6,545 |
| HW-B | `tile_A_b` | `all` | 19,382 |
| HW-C | `tile_A_a` | `all` | 6,506 |
| HW-D | `tile_C3` | `all` | 19,438 |

- Updated `conv3x3_large.tile_interior_evidence`: `True`
- Updated `conv3x3_large.global_min_cell`: `{"fusion": "all", "hw": "HW-C", "median": 6506.0, "tile": "tile_A_a"}`
- Updated family verdict: `WORKLOAD_FAMILY_INTERIOR_OPTIMUM_CONFIRMED`
