# NPU Mapping DSE Co-design 实测结果 Spec

## 1. 实验目标

本实验用于验证一个具体判断：在 PyTorchSim 的 GPT-2 单 transformer block、prefill、seq=128 场景下，硬件参数变化是否会显著改变 tiling mapping 的最优选择和相对排序。

实验不是为了证明某个单一硬件配置最快，而是为了判断后续是否有必要做类似 LUMINA 的 HW/SW 协同 DSE：如果不同硬件下最优 mapping 会迁移，或者 mapping 排序会明显变化，那么 GNN/LLM 需要同时理解硬件配置和编译映射；如果排序几乎不变，那么当前搜索空间还不足以支撑“强协同”的论文论断。

## 2. 实验设置

工作负载固定为 GPT-2 单 block prefill，序列长度为 128。

硬件配置为 2x2 factorial：

| HW | SPAD | DRAM/ICNT |
| --- | --- | --- |
| HW-A | 大 SPAD | 高带宽 |
| HW-B | 大 SPAD | 低带宽 |
| HW-C | 小 SPAD | 高带宽 |
| HW-D | 小 SPAD | 低带宽 |

mapping 候选为 8 组 tiling：

| mapping | TILE_M | TILE_N | TILE_K |
| --- | --- | --- | --- |
| 000 | 32 | 64 | 32 |
| 001 | 64 | 64 | 32 |
| 002 | 64 | 128 | 32 |
| 003 | 128 | 64 | 32 |
| 004 | 128 | 128 | 32 |
| 005 | 64 | 64 | 64 |
| 006 | 128 | 64 | 64 |
| 007 | 64 | 128 | 64 |

determinism canary 的最大相对波动为 0.018898，因此本次正式实验使用 3 次 repeat，并对每个 cell 取 median cycles。

## 3. 产物位置

主目录：

```text
outputs/mapping_dse_codesign/gpt2_block_prefill_s128_run1
```

核心产物：

| 文件 | 含义 |
| --- | --- |
| `heatmap_cycles.json` | 4x8 median cycles 矩阵 |
| `heatmap_cycle_stats.json` | 每个 cell 的 3 次 cycles、median、stddev |
| `fit_availability.json` | 每个 cell 的最终状态 |
| `sweep_summary.json` | 汇总后的 sweep 结果 |
| `analysis/champion_migration.json` | Gate-1 champion 是否迁移 |
| `analysis/oracle_gaps.json` | Gate-2a per-HW oracle 与单一 mapping 对比 |
| `analysis/cost_matched_pair.json` | Gate-2b cost-matched pair 对比 |
| `analysis/interaction_analysis.json` | Gate-3 ranking interaction |
| `report/verdict.json` | 最终 verdict |
| `report/report.md` | 中文报告 |
| `00_metadata.json` | 可复现实验元数据 |
| `full_run_summary.json` | lint、状态和 verdict 总结 |

## 4. 4x8 实测 cycles

| HW | 000 | 001 | 002 | 003 | 004 | 005 | 006 | 007 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HW-A | 2075876 | 1086632 | 766314 | 649730 | 542981 | 646994 | 408102 | 494088 |
| HW-B | 2940911 | 2104776 | 2037595 | 1539974 | 1402284 | 1564848 | 967906 | 1327220 |
| HW-C | 2076508 | 1089905 | 756508 | 660096 | 559079 | 652191 | 404330 | 494035 |
| HW-D | 2913219 | 2106675 | 2014717 | 1475786 | 1383767 | 1578949 | 973626 | 1320905 |

所有 32 个 cell 的最终状态都是 `measured`，没有 `unavailable`，没有 `retry_exhausted`。

## 5. 最直接的观察

四个硬件配置的最优 mapping 完全相同，都是 `006`：

| HW | best mapping | best cycles | worst mapping | worst cycles | worst/best |
| --- | --- | --- | --- | --- | --- |
| HW-A | 006 | 408102 | 000 | 2075876 | 5.087 |
| HW-B | 006 | 967906 | 000 | 2940911 | 3.038 |
| HW-C | 006 | 404330 | 000 | 2076508 | 5.136 |
| HW-D | 006 | 973626 | 000 | 2913219 | 2.992 |

这说明 mapping 选择本身很重要，因为同一硬件上最差和最好 mapping 可以相差约 3x 到 5x。但是这并不等价于 HW/SW 强协同，因为同一个 mapping `006` 在所有硬件上都是最优。

## 6. Gate 结果

最终 verdict：

```text
end_state = NEGATIVE
data_label = measured
claim_bearing = false
```

各 gate 结果：

| Gate | 结果 | 解释 |
| --- | --- | --- |
| Gate-1 champion migration | fail | 6 个硬件 pair 的 champion 都是 mapping 006，没有任何最优点迁移 |
| Gate-2a per-HW oracle gain | fail | constrained oracle 和 single-best mapping 都选 006，ratio mean = 1.0 |
| Gate-2b cost-matched pair | pass | HW-C best/HW-B best = 0.417737，说明 cost-tier 选择本身影响很大 |
| Gate-3 ranking interaction | fail | mean Spearman = 0.984127，mapping 排序在硬件之间几乎一致 |

Gate-2b 单独通过只能说明 HW-B 与 HW-C 这两个 cost-matched 配置之间的绝对性能差异很大，不能说明 mapping 和硬件之间存在强交互。总 verdict 为 `NEGATIVE` 的主要原因是 Gate-1 与 Gate-3 同时失败：最优 mapping 不迁移，整体 ranking 也不变。

## 7. 对我们 idea 的含义

这个结果对当前 idea 是一个有用的反例。

如果只把搜索空间限制在 TILE_M/TILE_N/TILE_K，并且 workload 只用 GPT-2 单 block prefill seq=128，那么当前 PyTorchSim/TOGSim 测到的现象更接近：

```text
mapping 的绝对好坏很重要，但硬件参数变化没有显著改变 mapping 排序。
```

因此，当前证据还不能支持“LLM/GNN 必须同时推理硬件参数和 mapping 才能找到不同硬件下不同最优映射”的强论断。

更准确的说法是：

```text
在当前受限搜索空间下，mapping 006 是跨硬件稳定最优点；如果要证明 HW/SW co-design 的必要性，需要扩大搜索维度，让数据流、fusion、SPAD 分区、DMA schedule 或算子组合进入搜索空间。
```

## 8. 下一步建议

下一轮实验不应该只继续增加同类 tiling 点，而应该引入会改变数据流和中间数据驻留方式的维度：

1. 增加 dataflow 维度，例如 weight-stationary、output-stationary、row-stationary 的替代实现。
2. 增加 fusion 维度，例如 matmul+bias+relu 是否融合、是否把 epilogue 留在 SPAD。
3. 增加 SPAD partition 维度，例如 X/W/Y/epilogue buffer 的容量分配。
4. 增加 DMA schedule 维度，例如 prefetch 距离、async wait 插入位置、double buffering。
5. 增加 workload 维度，例如 attention、MLP、不同 seq length、ResNet/Conv 类 workload。

这些维度比单纯 tiling 更可能改变 TOG 的结构，也更可能产生适合 GNN 压缩和 LLM 解释的图差异。

## 9. 结论

本次实验链路已经跑通：HW config 生成、mapping sweep、3-repeat median、TOGSim 测量、gate 分析、verdict/report/metadata 都生成并通过 lint。

但是实测结论是负向的：当前 4x8 tiling-only 搜索空间没有表现出足够的 HW/SW interaction。这个结论并不否定整体研究方向，而是说明下一步必须把搜索空间从“tiling 参数选择”提升到“数据流和编译策略选择”，否则 GNN/LLM 的作用会被一个跨硬件稳定最优 mapping 掩盖。

## 10. 根因分析补充：为什么本次 2x2 factorial 未暴露 HW/SW interaction

§8 的下一步建议是"扩到 fusion / dataflow / SPAD partition"。但在扩之前，需要把**本轮 NEGATIVE 的技术根因**讲清楚，否则下一版 spec 可能会重复同样的错误。

### 10.1 从行相似性看：2x2 塌成了 1x2

把 §4 的 cycles 表按 SPAD 与 DRAM BW 拆两个轴看：

| | DRAM = 32 ch（大 BW） | DRAM = 8 ch（小 BW） |
| --- | --- | --- |
| SPAD = 128 KB/lane（大） | HW-A 006 = **408 102** | HW-B 006 = **967 906** |
| SPAD = 32 KB/lane（小） | HW-C 006 = **404 330** | HW-D 006 = **973 626** |

- 沿 DRAM BW 轴：HW-A → HW-B 慢 2.37x；HW-C → HW-D 慢 2.41x。**主效应显著**。
- 沿 SPAD 轴：HW-A → HW-C 只慢 0.94%；HW-B → HW-D 只慢 0.59%。**主效应几乎为零**。

也就是说，本轮"2x2 factorial"在实测上**塌成了"1x2 factorial"**（只有 DRAM BW 一个真轴）。SPAD 从 128 降到 32 KB/lane 的**观测结果基本不变**，因此 mapping 排序不可能出现 HW 之间的分歧——因为整个"SPAD"这一轴在数据上不存在。

### 10.2 从 fit-ratio 看：SPAD 从头到尾都是空的

以本轮 8 个候选中**最大**的 tile（`TILE_M=128, TILE_N=128, TILE_K=32`，mapping 004）为例：

```text
working_set ≈ TILE_M * TILE_K + TILE_K * TILE_N + TILE_M * TILE_N
            = 128 * 32     + 32 * 128     + 128 * 128
            = 4096 + 4096 + 16384
            = 24 576 elements
            = 24 576 * 4 bytes (fp32)
            ≈ 96 KB
```

即使在**最小 SPAD 的 HW**（HW-C、HW-D，`vpu_spad_size_kb_per_lane = 32`），总容量也是 `32 KB/lane * 128 lanes = 4096 KB`。fit-ratio ≈ `96 / 4096 = 2.3%`。

也就是说，**我们从来没有把 SPAD 装满过**。工作集只占 SPAD 的 2%~5%。SPAD 尺寸变化根本不 binding。这一点在 `oracle_gaps.json` 中也间接反映：`fit_all_hw_set` 是全部 8 个 mapping——没有任何一个 tile 因为 SPAD 太小而被 `dry_run_fit.py` 判为 `unavailable`。

### 10.3 结论：本轮的两个设计缺陷

1. **HW 两个轴都在 memory 子系统**（SPAD size 和 DRAM channels+NoC ports）。缺一个 **compute 侧的轴**（例如 `num_systolic_array_per_core`、systolic 阵列大小 8x8 vs 128x128）。没有 compute vs memory 的实质张力，mapping 排序自然不会随 HW 迁移。
2. **SPAD 取值范围（32~128 KB/lane）远超工作集需求**。要让 SPAD 成为真正的 constraint，应该压到 4~8 KB/lane 让最大 tile 塞不下、最小 tile 才能活下来。

只解决其中任一条，都可以让"champion migration"至少在原理上有出现的可能。

### 10.4 修正后的下一版 spec 建议（优先级顺序）

在采纳 §8 的 dataflow/fusion/SPAD-partition 扩展之前，**先做 HW 侧的两项修正**，代价小、直接命中根因：

1. **换掉 SPAD 轴，改用 systolic 阵列大小**：`8x8` vs `128x128` 作为 2x2 的另一轴，与 DRAM BW 组成新的 2x2 factorial。仓库已有 `systolic_ws_8x8_c1_simple_noc.yml`，直接可用。
2. **保留 SPAD 轴但压到真正 binding 范围**：`vpu_spad_size_kb_per_lane` 从 `128` / `32` 改成 `8` / `2`，或者把 `vpu_num_lanes` 减半。让最大 tile 塞不下，classifier 会把它们判 `unavailable`，champion 就必须在 HW 间迁移。
3. **在 mapping 池里加两端极值**：现有 8 个候选都在 `M/N/K ∈ [32, 128]` 里，太集中。加 `(16, 16, 16)` 和 `(256, 256, 128)`——极小的能在极端 SPAD 上活、极大的只能在大 SPAD 上活。

**只有** §10.3 的两个根因得到解决之后，再叠加 §8 的 dataflow/fusion/SPAD-partition 扩展才有意义；否则新增的 SW 维度依然会被"跨硬件稳定最优 mapping"掩盖。

### 10.5 Gate-2b 通过的意外收获（附加读者提示）

Gate-2b 用 cost-matched pair `{HW-B, HW-C}` 判定，实测 `gate2b_ratio = 0.417737`。虽然这不是 HW×SW interaction 的证据，但它**独立成立**：

```text
在同一 cost 预算下, DRAM BW (HW-C: 32 ch) 相比 SPAD 尺寸 (HW-B: 128 KB/lane) 快 2.4x
```

这是一条纯 HW 侧的 allocation 建议：**对 GPT-2 单 block prefill 类工作负载，把 chip 预算堆到 DRAM 带宽比堆到片上 SPAD 更划算**。这条结论不依赖 HW/SW interaction 成立，可以独立引用。
