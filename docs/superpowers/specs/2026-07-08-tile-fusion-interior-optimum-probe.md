# Route 4 Tile x Fusion x HW Interior-Optimum Probe

Date: 2026-07-08

## 1. Executive Summary

This probe extends the Route 4 Conv POSITIVE result from a fusion-only axis to a three-axis search: Conv tile size x fusion setting x `codesign_v1_2x2` HW. The verdict is `CHAMPION_MIGRATION_ONLY`.

The sweep found real champion migration: HW-A and HW-B prefer `tile_D + fusion=all`, HW-C prefers `tile_A + fusion=all`, and HW-D prefers `tile_B + fusion=all`. The global minimum is `HW-C / tile_A / fusion=all` at 5,921 cycles, but that point is still a corner of the HW/tile/fusion grid, so this is not a strict interior-global-optimum result.

## 2. Scope and Run Accounting

- Workload: Conv 3x3 `[1, 64, 56, 56] -> [1, 64, 56, 56]` with BatchNorm and ReLU in the existing `conv3x3_probe` harness.
- HW space: `codesign_v1_2x2` HW-A/B/C/D.
- Fusion variants: `codegen_compiler_optimization = none` and `all`.
- Tile variants: exactly four variants, `tile_A` through `tile_D`.
- Functional mode: `pytorchsim_functional_mode=0` timing-only.
- Repeats: 5 per measured cell, seed 0, independent subprocesses.
- Run slots: 32 cells x 5 = 160 slots. Measured: 140 runs. Dry-run unavailable: 20 slots from HW-C/HW-D `tile_D`. Runtime failed cells: 0.
- Raw `cycles_in_order` are preserved in `outputs/route4_fusion_pilot/tile_fusion_hw_matrix.json`.

## 3. Conv Tile Axis Discovery

Conv 3x3 tiling uses the 7-parameter tile tuple `TILE_K_H`, `TILE_K_W`, `TILE_O_H`, `TILE_O_W`, `TILE_M`, `TILE_N`, `TILE_K`. The derived input extents are `TILE_I_H = 1 + (TILE_O_H - 1) * stride_h + (TILE_K_H - 1) * dilation_h` and `TILE_I_W = 1 + (TILE_O_W - 1) * stride_w + (TILE_K_W - 1) * dilation_w`.

The original external mapping schema only exposed GEMM-style `TILE_M/N/K`. This probe keeps PyTorchSim source unchanged and installs a runtime patch in `scripts/route4_fusion_pilot.py` so `conv2d_1_64_64_3_3_56_56` can carry the seven Conv tile parameters during the pilot run.

Working-set model: `bytes = 4 * (K_H * K_W * TILE_K * TILE_N + TILE_I_H * TILE_I_W * TILE_M * TILE_K + TILE_O_H * TILE_O_W * TILE_M * TILE_N)`.

## 4. Tile Variants

| Tile | Regime | Parameters | Working set bytes | Small-SPAD fit fraction | Large-SPAD fit fraction | Fits all HW |
|---|---|---|---:|---:|---:|---|
| tile_A | tiny | TILE_K_H=3, TILE_K_W=3, TILE_O_H=56, TILE_O_W=4, TILE_M=1, TILE_N=128, TILE_K=64 | 498,688 | 23.8% | 5.9% | yes |
| tile_B | small | TILE_K_H=3, TILE_K_W=3, TILE_O_H=56, TILE_O_W=14, TILE_M=1, TILE_N=128, TILE_K=64 | 933,888 | 44.5% | 11.1% | yes |
| tile_C | medium | TILE_K_H=3, TILE_K_W=3, TILE_O_H=56, TILE_O_W=28, TILE_M=1, TILE_N=128, TILE_K=64 | 1,543,168 | 73.6% | 18.4% | yes |
| tile_D | large | TILE_K_H=1, TILE_K_W=1, TILE_O_H=56, TILE_O_W=56, TILE_M=1, TILE_N=128, TILE_K=64 | 2,441,216 | 116.4% | 29.1% | no |

The selected tiles span the intended fit boundary: `tile_A/B/C` fit the smallest 32KB/lane HW under the double-buffer budget, while `tile_D` exceeds it at 116.4% and remains available on 128KB/lane HW.

## 5. Cell Matrix

| HW | Tile | none state | none median | all state | all median | Fusion speedup |
|---|---|---|---:|---|---:|---:|
| HW-A | tile_A | measured | 185,248 | measured | 6,167 | 30.04x |
| HW-A | tile_B | measured | 184,460 | measured | 6,155 | 29.97x |
| HW-A | tile_C | measured | 186,042 | measured | 6,196 | 30.03x |
| HW-A | tile_D | measured | 162,723 | measured | 6,022 | 27.02x |
| HW-B | tile_A | measured | 210,531 | measured | 17,160 | 12.27x |
| HW-B | tile_B | measured | 211,673 | measured | 17,107 | 12.37x |
| HW-B | tile_C | measured | 217,085 | measured | 17,363 | 12.50x |
| HW-B | tile_D | measured | 215,429 | measured | 16,876 | 12.77x |
| HW-C | tile_A | measured | 185,126 | measured | 5,921 | 31.27x |
| HW-C | tile_B | measured | 184,115 | measured | 6,264 | 29.39x |
| HW-C | tile_C | measured | 186,014 | measured | 6,098 | 30.50x |
| HW-C | tile_D | unavailable | NA | unavailable | NA | NA |
| HW-D | tile_A | measured | 210,933 | measured | 16,858 | 12.51x |
| HW-D | tile_B | measured | 212,113 | measured | 16,815 | 12.61x |
| HW-D | tile_C | measured | 216,781 | measured | 17,198 | 12.61x |
| HW-D | tile_D | unavailable | NA | unavailable | NA | NA |

## 6. Champion Migration

| HW | Champion tile | Champion fusion | Champion median cycles |
|---|---|---|---:|
| HW-A | tile_D | all | 6,022 |
| HW-B | tile_D | all | 16,876 |
| HW-C | tile_A | all | 5,921 |
| HW-D | tile_B | all | 16,815 |

Champion pair count is 3: `(tile_D, all)`, `(tile_A, all)`, and `(tile_B, all)`. This is the key co-design signal from this probe: the best software tile choice changes with the HW point, even though the fusion choice remains `all` everywhere.

## 7. Interior-Optimum Check

Global minimum: `HW-C / tile_A / all` at `5,921` cycles.

Global minimum is at a corner: `true`. Strict interior-global-optimum evidence: `false`.

Therefore the verdict is `CHAMPION_MIGRATION_ONLY`, not `INTERIOR_OPTIMUM_FOUND`. The result is still useful: it shows tile-size fit constraints create HW-dependent software winners, but it does not yet show an interior point beating all grid corners.

## 8. Fit Boundary Effect

HW-A/HW-B have 128KB/lane SPAD, so all four tiles fit. HW-C/HW-D have 32KB/lane SPAD, so `tile_D` is unavailable before runtime because its predicted working set is larger than the double-buffered SPAD budget. This is the mechanism that forces champion migration: the large tile wins on large-SPAD HW, but small-SPAD HW must choose among `tile_A/B/C`.

## 9. Artifacts

- `outputs/route4_fusion_pilot/tile_axis_discovery.md`
- `outputs/route4_fusion_pilot/tile_variants.json`
- `outputs/route4_fusion_pilot/tile_fusion_hw_matrix.json`
- `outputs/route4_fusion_pilot/interior_optimum_analysis.json`

## 10. Next Handoff

The next handoff should investigate a route to convert champion migration into strict interior-global-optimum evidence. The most direct options are to add an actual interior HW point between the current 2x2 corners, expand the tile axis beyond four points around the small-SPAD boundary, or move to a workload where the best tile is not a boundary tile on high-SPAD HW.
