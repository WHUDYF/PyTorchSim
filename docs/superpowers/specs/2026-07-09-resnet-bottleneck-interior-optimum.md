# ResNet-50 bottleneck interior-optimum probe

Date: 2026-07-09

## 1. 目标

本实验把 isolated Conv kernel 的 tile interior-optimum 结论推进到 ResNet-50 `conv3_x` bottleneck。只 sweep bottleneck 中 3x3 core 的 tile，1x1 reduce / expand 保持固定默认 tile。

## 2. Workload

- Input: `x[1,512,28,28]`
- Block: `1x1 reduce 512->128` + `3x3 core 128->128` + `1x1 expand 128->512` + residual add + ReLU
- 3x3 core shape key: `conv2d_1_128_128_3_3_28_28`
- HW: `HW-A/B/C/D` from `codesign_v1_2x2`
- Fusion: `none/all`
- Repeats: 5 per measured cell, `pytorchsim_functional_mode=0`, `CUDA_VISIBLE_DEVICES=1`

## 3. Gate 0

- Smoke passed: `True`
- Smoke total cycles: `13897`
- Core tile override applied: `True`

## 4. Sweep accounting

- Base tiles: `tile_A, tile_A2, tile_B, tile_B2, tile_C, tile_C2, tile_C3, tile_D`
- Densification applied: `True`
- Matrix accounting: `{"blocked": 8, "measured": 80}`
- Timeboxed: `False`

## 5. Champion per HW

| HW | Champion tile | Fusion | Median cycles |
|---|---|---|---:|
| HW-A | `tile_A` | `all` | 13,035 |
| HW-B | `tile_C3` | `all` | 44,236 |
| HW-C | `tile_A_b` | `all` | 13,044 |
| HW-D | `tile_C2` | `all` | 44,233 |

## 6. Verdict

- `champion_migration`: `True`
- `tile_interior_evidence`: `False`
- `global_min_tile_is_extreme_for_hw`: `True`
- `global_min_cell`: `{"fusion": "all", "hw": "HW-A", "median": 13035.0, "tile": "tile_A"}`
- `verdict`: `PARTIAL_RESNET_BOTTLENECK_INTERIOR`

## 7. 结果解读（与孤立 kernel 对比）

本次是一个 scope-clarifying 的 PARTIAL，核心信息分两半：

**泛化成立的部分：协同交互（champion migration）**

4 个硬件各自的冠军 tile 互不相同：

```text
HW-A -> tile_A      HW-B -> tile_C3
HW-C -> tile_A_b    HW-D -> tile_C2
```

换硬件就会改变最优软件选择——这说明「最优 tile 由硬件决定、软硬件存在真实交互」这一 co-design 核心信号，从孤立 conv kernel 泛化到了真实 ResNet-50 bottleneck（前后 1x1 + 残差 + 整块 fusion 的真实上下文）。

**泛化不成立的部分：严格内部最优（interior optimum）**

全局最优落在 `HW-A / tile_A`，而 `tile_A` 是最小 tile、位于 fit 边界。即使 runner 在 `tile_A` 与 `tile_A2` 之间自动加密补扫（densification），HW-A 的冠军仍停在 `tile_A`，没有出现更优的内部点。

**关键对比**：在孤立的 `conv3x3_large` 上，同样的加密把冠军从边界 `tile_A` 推进到了内部 `tile_A_a`（内部最优成立）。但把同一个 3x3 core 嵌进完整 bottleneck 后，前后 1x1 层与残差改变了整块的 compute/memory 平衡，把最优点压到了最小 tile 边界，加密也没能找回内部甜点。

**科学含义**：co-design 的必要性（软硬件交互）能泛化到真实网络 block；但「最优点落在搜索空间内部」这个更强性质是 workload-context 依赖的——孤立 kernel 的 interior 结论不能直接外推到复合 block。这是一个诚实的边界划定，符合 evidence-first 方法论。

## 8. Scope

- 该结论是 mode=0 timing evidence，不是 correctness claim。
- 只 sweep 3x3 core tile；1x1 reduce / expand 固定 tile。
- 不包含 MobileNetV2/depthwise/grouped-conv。
- 未增加 HW interior points，也未 sweep dataflow / SPAD partition / DMA schedule。
