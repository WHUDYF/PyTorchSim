# NPU 软硬件协同价值验证设计 v2：GPT-2 单 block × systolic 阵列 × mem_subsystem_scale 2×2 factorial

## 1. 目标与研究问题

本 spec 是 `2026-07-01-npu-mapping-dse-codesign-design.md`（下称 v1 spec）的**根因修正版**。v1 spec 在 GPT-2 单 block prefill seq=128 上的实测结论是 `NEGATIVE`（见 `2026-07-01-npu-mapping-dse-codesign-result.md` §10）：

```text
v1 的 2x2 factorial (SPAD size x DRAM channels) 实测塌成了 1x2 (只 DRAM BW 一轴).
根因: SPAD 32~128 KB/lane 范围远超工作集 (最大 tile working set ~96 KB, 最小 SPAD 总容量 4 MB),
从头到尾 fit ratio 只 2.3%, SPAD 尺寸根本不 binding.
+
HW 两个轴都在 memory 子系统, 缺 compute 侧张力, mapping 排序不可能随 HW 迁移.
```

v2 spec **不改变 workload、不改变 gate 判据、不改变统计契约、不改变 retry 白名单**。**只在两处做硬修正**：

1. **HW 第 1 轴从 SPAD size 换成 systolic 阵列大小**（`vpu_num_lanes`：128 vs 8），引入 compute 侧张力。第 2 轴（mem_subsystem_scale = dram_channels + icnt_ports 联动）保持不变。
2. **Mapping 池从 8 个 tiling 扩到 10 个**，新增两端极值（`(16, 16, 16)` 与 `(256, 256, 128)`），让不同 systolic 大小对应不同 MAC 利用率区间，champion migration 才有出现的可能。

研究问题（与 v1 相同，只重述以自包含）：

> 在 PyTorchSim / TOGSim 上，对 GPT-2 单 transformer block、prefill、seq=128 workload：
>
> - **Q1（冠军迁移）**：改变 systolic 阵列大小（128 lanes vs 8 lanes）与 mem_subsystem_scale（32ch+16ports vs 8ch+4ports），tiling mapping 的冠军是否会**换人**（double-swap dominance ≥ 3%）？
> - **Q2a（per-chip SW autotune 收益）**：`fit_all_hw_set` 池内，per-chip 单独选最优是否比"统一 mapping 卖给所有芯片"节省 ≥ 15%，且 ≥ 3 个 HW 上单独 ratio ≤ 0.85？
> - **Q2b（cost-matched pair 上界）**：cost-matched pair `{HW-B, HW-C}`（`compute-heavy + 低 BW` 与 `compute-light + 高 BW`）之间，`min/max ≤ 0.85`？
> - **Q3（interaction）**：跨 HW 的 mapping ranking 相似度 `mean_spearman ≤ 0.7`，且 numeric pair ≥ 5/6，per-pair `n_intersect ≥ 4`？

判定逻辑（与 v1 相同）：

```text
Q1 & Q2a & Q2b & Q3 全通过 -> co-design 得到 measured 支撑, POSITIVE
Q1 不通过                   -> 换 compute 轴之后冠军仍不迁移, 更强 NEGATIVE
Q2a & Q2b 都不通过           -> 增量收益不够, NEGATIVE
其它组合                    -> PARTIAL
```

## 2. 非目标

明确**不做**（与 v1 一致）：

- fusion / dataflow / SPAD partition / DMA schedule 等 mapping 维度扫描（仍只扫 tiling）；
- 跨 workload 泛化（GPT-2 单 block prefill seq=128 之外的留独立 spec）；
- 引入 GNN / embedding / LLM；
- 修改 PyTorchSim / TOGSim / gem5 / ramulator2 源码（HW 变化全部通过替换 YAML config 实现）；
- 探索**内部** systolic 大小或 DRAM channel 值（仅 4 个 corner）；
- 引入 area / power / cost 归一化（verdict 仍只看 cycle）。

## 3. Workload 输入

**与 v1 完全一致**，不做任何变化：

```text
model: GPT-2 single transformer block
phase: prefill (forward only)
sequence_length: 128
hidden_size: 768
heads: 12
dtype: float32
device: npu:0
surrogate_model: none
```

CLI flags（默认覆盖）：

```text
--seq            (default 128)
--num-mappings K (default 10, 与 v2 mapping 池一致)
--hw-config-set  (default codesign_v2_2x2)
--output-dir     (default outputs/mapping_dse_codesign/gpt2_block_prefill_s128_v2_run1/)
--seed           (default 0)
```

## 4. HW config 扫描设计（v2 的核心修改）

### 4.1 新 factorial 结构

沿两个**真正**能引出 compute-vs-memory 张力的轴：

| 轴 | 参数 | 大档 | 小档 |
| --- | --- | ---: | ---: |
| compute 侧 | `vpu_num_lanes`（systolic 阵列大小） | 128 | 8 |
| memory 侧 | `mem_subsystem_scale` = `dram_channels` + `icnt_injection_ports_per_core` 联动 | 32 ch / 16 ports | 8 ch / 4 ports |

4 组 config：

| id | 语义 | vpu_num_lanes | dram_channels | icnt_ports |
| --- | --- | ---: | ---: | ---: |
| `HW-A` | compute-heavy + 高 BW（与 v1 HW-A 相同 config） | 128 | 32 | 16 |
| `HW-B` | compute-heavy + 低 BW（与 v1 HW-B 相同 config） | 128 | 8  | 4  |
| `HW-C` | compute-light + 高 BW（v1 HW-C 语义变了） | 8   | 32 | 16 |
| `HW-D` | compute-light + 低 BW（v1 HW-D 语义变了） | 8   | 8  | 4  |

`HW-A` 与 `HW-B` 与 v1 完全对应（同 `vpu_num_lanes=128`、同 SPAD、同 DRAM 组合）。**这是刻意保留的交叉验证机会**：v2 跑完后可以对比 v1 和 v2 在 `HW-A` / `HW-B` 上的 cycles，验证 harness 行为一致（差异应该 ≤ determinism smoke test 的 `cycle_delta` 上限）。

`HW-C` 与 `HW-D` 的语义在 v2 里换了：现在是"lanes 少"而非"SPAD 小"。命名保持是为了 run_tag 目录结构对齐，读者只需要通过 `patch_matrix` 记录辨认实际内容。

### 4.2 HW config 生成规则

所有 4 份 YAML 都从 baseline `configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml` **patch** 而来：

```json
{
  "baseline_yaml": "configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml",
  "patch_matrix": {
    "HW-A": {"vpu_num_lanes": 128, "dram_channels": 32, "icnt_injection_ports_per_core": 16},
    "HW-B": {"vpu_num_lanes": 128, "dram_channels": 8,  "icnt_injection_ports_per_core": 4},
    "HW-C": {"vpu_num_lanes": 8,   "dram_channels": 32, "icnt_injection_ports_per_core": 16},
    "HW-D": {"vpu_num_lanes": 8,   "dram_channels": 8,  "icnt_injection_ports_per_core": 4}
  },
  "plausibility_rules": [
    "icnt_injection_ports_per_core = clip(dram_channels / 2, 4, 32) rounded to nearest even",
    "vpu_spad_size_kb_per_lane 固定 128 (frozen), 总 SPAD 容量 = 128 * vpu_num_lanes KB, 随 lanes 缩放",
    "num_cores 固定为 1",
    "dram_type / ramulator_config_path 固定为 HBM2_TPUv3",
    "vpu_num_lanes 取值仅限 {8, 128} (本 spec 不扫内部值)"
  ],
  "frozen_fields": [
    "core_freq_mhz",
    "dram_freq_mhz",
    "icnt_freq_mhz",
    "num_cores",
    "num_systolic_array_per_core",
    "vpu_spad_size_kb_per_lane",
    "vpu_vector_length_bits",
    "dram_type",
    "ramulator_config_path"
  ],
  "frozen_fields_reason": "v2 中 vpu_num_lanes 从 frozen 移出成为 HW 第 1 轴; vpu_spad_size_kb_per_lane 从 v1 的轴变量转为 frozen (v1 实测证明它在 32~128 KB/lane 范围内不 binding). 其余 frozen 字段与 v1 一致, 确保 HW 之间的 cycle 差异不能被时钟 / DRAM 类型 / core 数量等分外因素解释.",
  "coupling_reason": "Reducing vpu_num_lanes to 8 naturally reduces total SPAD (from 16 MB to 1 MB, 128 KB/lane * 8 lanes) and total systolic array throughput. This is a physically realistic co-scaling (real chips downsize on-chip memory with compute). Not treating it as a separate factor."
}
```

### 4.3 factorial 局限的诚实声明

本 spec 只采样 4 个 corner，不扫 `vpu_num_lanes` 的内部值（16 / 32 / 64）、不扫 `dram_channels` 的内部值（12 / 16 / 24）。因此**不能识别内部非单调 interaction**。若 4×8 上得到 POSITIVE，只能陈述"在两个 corner 极值上 co-design 有 measured 支撑"，不能外推到内部空间。

### 4.4 vpu_num_lanes 与 SPAD 总容量的耦合声明

减少 `vpu_num_lanes` 从 128 到 8，总 SPAD 从 `128 * 128 = 16 384 KB (16 MB)` 缩到 `128 * 8 = 1 024 KB (1 MB)`。这是**特意允许**的物理合理耦合（实际 chip 也是把 compute 和 on-chip memory 一起缩放）。verdict 与 report 引用本 spec 结论时，措辞必须包含这一耦合，不能声称"compute 独立主效应被识别"。

## 5. Mapping 候选（v2 的第二处修改）

v1 的 8 个 tiling 全部保留，新增 2 个两端极值，共 **10 个候选**：

```text
000  32   64   32     (v1 保留)
001  64   64   32     (v1 保留)
002  64  128   32     (v1 保留)
003 128   64   32     (v1 保留)
004 128  128   32     (v1 保留)
005  64   64   64     (v1 保留)
006 128   64   64     (v1 保留, v1 冠军)
007  64  128   64     (v1 保留)
008  16   16   16     (v2 新增, 极小 tile, 用于 8-lane HW 上验证小 tile 会不会成为冠军)
009 256  256  128     (v2 新增, 极大 tile, 用于 128-lane HW 上验证大 tile 能否吃满 systolic)
```

**为什么这两个极值有可能触发冠军迁移**：

- `008 (16, 16, 16)`：working set ≈ 3 KB，在 8-lane HW（1 MB SPAD）上占比 <0.5%，能吃进 SPAD。但 in 128-lane systolic 里，`TILE_M=16` 只能占用 16/128 = 12.5% 的 lane，MAC 利用率极低。8-lane 反而饱和。
- `009 (256, 256, 128)`：working set ≈ 512 KB，在 128-lane HW（16 MB SPAD）上占比 3%，能塞下；但在 8-lane HW（1 MB SPAD）上占 **50%**，会占用相当一部分 SPAD 甚至导致 fit classifier 判 unavailable。128-lane systolic 能吃满这个大 tile；8-lane 只能大量拆迭代。

预期行为：`HW-A` / `HW-B`（128 lanes）冠军应偏向大 tile（`006` 或 `009`）；`HW-C` / `HW-D`（8 lanes）冠军应偏向中/小 tile。若真实测出这种迁移 → Gate-1 通过。

### 5.1 fit 约束（沿用 v1 §5.1 的四态状态机）

四态状态机 `{measured, unavailable, runtime_failed, retry_exhausted}` 与 v1 完全一致，不重述。dry-run 预检的解析公式不变：

```text
working_set_bytes = (TILE_M * TILE_K + TILE_K * TILE_N + TILE_M * TILE_N) * dtype_bytes
fit iff working_set_bytes <= vpu_spad_size_kb_per_lane * 1024 * vpu_num_lanes * safety_factor (=0.9)
```

**calibration 需要重跑**（因为 SPAD 总容量范围从 v1 的 4~16 MB 变到 v2 的 1~16 MB）：

- known-fit 校准组合：`(HW-A, 006)` — 128 lanes / 16 MB SPAD / working set 64 KB → 预期 fit ✓
- known-no-fit 校准组合：`(HW-C, 009)` — 8 lanes / 1 MB SPAD / working set 512 KB → 预期 **勉强 fit**（ratio 50%）。若 safety_factor=0.9 判 fit 而 TOGSim 崩，说明公式偏乐观；若 safety_factor=0.9 判 unavailable 而 TOGSim 能跑，说明公式偏保守。
- 若 `predicted_fit XOR actual_fit == 1` on 任一组 → 自动 fallback 到 no-precheck。

## 6. 总体数据流

与 v1 完全一致，不重述（HWConfigMaker → SweepHarness → Extractor → CoDesignAnalyzer → Report+Verdict）。

## 7. 组件设计

### 7.1 复用 v1 的组件（不改代码或极小改动）

- `scripts/codesign_preflight.py`（AC-1）：**不改**。
- `scripts/mapping_dse_codesign_sweep.py`（AC-6）：**不改**（HW config path 通过 CLI 传入）。
- `scripts/codesign_analyzer.py`（AC-8）：**不改**。纯函数，只吃 4×N cycles 矩阵，不关心 HW 语义。
- `scripts/codesign_report.py`（AC-10）：**不改**。表格式和强制段与 HW 语义解耦。
- `scripts/codesign_metadata.py`（AC-12）：**不改**。
- `scripts/stats_contract.py`（AC-7）：**不改**（tie / epsilon / spearman / aggregation 全部沿用）。
- `scripts/codesign_full_run.py`（AC-11、AC-12 集成）：**不改**。

### 7.2 需要小改动的组件

**`scripts/hw_config_factory.py`（AC-2）**：
- 更新默认 `patch_matrix` 与 `frozen_fields`（见 §4.2 JSON）。
- 保留旧 v1 patch_matrix 的支持，通过 CLI flag `--hw-config-set {codesign_v1_2x2, codesign_v2_2x2}` 或直接传新 patch_matrix JSON 切换。
- 变更：**`vpu_num_lanes` 从 frozen 移出、`vpu_spad_size_kb_per_lane` 加入 frozen**。工厂 assert 逻辑要更新。

**`scripts/dry_run_fit.py`（AC-3）**：
- 公式与 safety_factor **完全不变**。
- Calibration 需要用新的 known-fit / known-no-fit 组合（见 §5.1）重跑；calibration fixture 新增 (HW-C, 009) 组合的 TOGSim 真跑校准结果。
- Fallback 逻辑不变。

### 7.3 需要小改动的 tests

- `tests/test_hw_config_factory.py`：新增 fixture 验证 `vpu_num_lanes` 变化被接受、`vpu_spad_size_kb_per_lane` 变化被拒绝为 frozen 违反。
- `tests/test_dry_run_fit.py`：新增 fixture 覆盖 (256, 256, 128) 大 tile 在 8-lane HW 上的边界判断。
- `tests/test_codesign_analyzer_golden.py`：golden fixture 无需变（analyzer 纯函数不 care 语义）。若要显式测 v2 场景，可加一个 fixture 让 (HW-A/B, 009) 冠军、(HW-C/D, 008) 冠军 → 断言 champion migration 通过。
- 其余测试全部沿用 v1，无需改动。

### 7.4 harness CLI 拓展

`scripts/mapping_dse_minimal.py` 的 `default_mapping_candidates` 列表当前是 12 个候选的 superset，包含本 spec 的 10 个候选（v1 8 个 + v2 2 个新增）。若不包含，`--num-mappings 10` + `--external-mappings-json` 的 CLI 路径要能覆盖，**这条路径 v1 已经实现**（AC-4），不需改。

## 8. go/no-go 判据

阈值与结构**完全与 v1 一致**，不重述。判据表：

| Gate | 判据 | 阈值 |
| --- | --- | --- |
| Gate-1 champion migration | 至少一对 (i,j) `swap_dominance ≥ 3%` | 3% |
| Gate-2a per-chip SW vs single mapping | `gate2a_ratio_mean ≤ 0.85` **且** `num_hw_meeting_threshold ≥ 3`（fit_all_hw_set 内） | 15% + 3/4 HW |
| Gate-2b co-design vs cost-matched pair | `min(best_B, best_C) / max(best_B, best_C) ≤ 0.85` | 15% |
| Gate-3 interaction | `mean_spearman ≤ 0.7` 且 `numeric_pair_count ≥ 5/6` 且每对 `n_intersect ≥ 4` | 0.7 / 5 / 4 |

**cost-matched pair 语义在 v2 里变了**：v1 是 `{SPAD大 + BW低, SPAD小 + BW高}`；v2 是 `{compute-heavy + BW低, compute-light + BW高}`。物理上更能真的"compute-vs-memory 决胜"。framing 仍然写"ex-post allocation sensitivity upper bound"。

## 9. 产物与 schema

运行目录：`outputs/mapping_dse_codesign/gpt2_block_prefill_s128_v2_run1/`（与 v1 严格分开）。

产物结构与 v1 完全一致（`00_metadata.json`、`hw_configs/`、`runs/`、`heatmap_cycles.json`、`fit_availability.json`、`analysis/*.json`、`verdict.json`、`report.md` 等）。

`heatmap_cycles.json` 变化：`mapping_ids` 从 8 项变 10 项（多 `008`、`009`）；`cycles` 从 4×8 矩阵变 4×10 矩阵。

`00_metadata.json` 新增字段：

```json
{
  "spec_version": "v2",
  "spec_path": "docs/superpowers/specs/2026-07-01-npu-mapping-dse-codesign-v2-design.md",
  "hw_axis_axis1": "vpu_num_lanes",
  "hw_axis_axis2": "mem_subsystem_scale",
  "v1_run_tag_for_cross_check": "gpt2_block_prefill_s128_run1",
  "v1_v2_shared_cells": ["HW-A/000", "HW-A/001", "...", "HW-B/007"],
  "v1_v2_cycle_delta_per_shared_cell": {}
}
```

`v1_v2_cycle_delta_per_shared_cell` 由 v2 run 结束时自动计算：对每个 HW-A/B × mapping 000~007 (共 16 cell)，计算 `abs(v2_cycles - v1_cycles) / min(v1, v2)`，写入 metadata。断言：**所有 16 cell 的 delta 应 ≤ determinism_smoke_test 里的 `max_cycle_delta`**（v1 是 0.019）。任一 cell 超出 → verdict 加 `cross_check_warning`，不阻塞 verdict 但入 `scope_limitations`。

## 10. Milestone gates

与 v1 完全一致（Gate α HW 配置就绪、Gate β 采集就绪、Gate γ 分析已算、Gate δ 裁决完成）。**唯一新增**：Gate γ 之前必须先跑 v1-v2 cross check（对比 16 个 shared cell 的 cycle_delta），验证 harness 无 regression。

## 11. 风险与缓解

### R1 SPAD 小的 HW 上 tile 塞不下（v1 R1 变严重）

在 v1 里，最小 SPAD HW 也有 4 MB，任何 tile 都能塞下。v2 里最小 SPAD HW 只有 1 MB，最大 tile (256, 256, 128) 就有 512 KB working set，占 50%。加上 double buffering + activation + intermediate + bias 的实际开销，很可能被 fit classifier 判 unavailable 或 TOGSim 报错。缓解：
- 保留 v1 §5.1 的四态状态机。
- Calibration 用 `(HW-C, 009)` 组合真跑一次 TOGSim，明确边界。
- 若最大 tile 在 HW-C / HW-D 上都不 fit，Gate-2a 的 `fit_all_hw_set` 会缩到只包含较小 tile → 可能进 `insufficient_evidence`，end_state 只能 `PARTIAL`，不能 `POSITIVE`。这是诚实结果不是失败。

### R2 vpu_num_lanes 减到 8 后可能触发 TOGSim / gem5 / ramulator2 里未测试的路径

仓库虽有 `systolic_ws_8x8_c1_simple_noc.yml` 参考，但那份 config 的 `dram_type=ramulator2 DDR4` 和 `core_freq_mhz=800`，与本 spec 的 HBM2_TPUv3 + 940 MHz + `dram_channels=32` 组合从未跑过。缓解：
- Gate α 内做 smoke run（每份新 YAML + 单 addmm workload 单 kernel）确认能出 timing 结果，任何 config 通不过 smoke → 进 R2 6-attempt 排障（bounded）。
- 若 8-lane + 32-channel 组合在 TOGSim 里根本无法加载 → end_state `BLOCKED`。

### R3 v1-v2 cross check 失败

v2 中 HW-A / HW-B 与 v1 完全对应，000~007 mapping 也一致。任何 HW-A/B × 000~007 的 cycle_delta 显著超过 determinism smoke `max_cycle_delta`（v1 是 0.019）→ 说明 harness 有 regression。缓解：
- Cross check 结果写 `cross_check_warning` 字段进 verdict。
- 超出 20% 的 cell → 直接把 v2 end_state 强制降级 `PARTIAL`，report 里说明"v2 harness 与 v1 存在系统性差异，需先定位"。

### R4 与 v1 一致的其它风险（不重述）

R2 patch loading、R3 单次超时、R4 HW-A vendor baseline 循环质疑（v2 已用 cost-matched 消除）、R5 permutation 显著性、R6 corner-only 局限、R7 determinism、R8 rank-only 软 caveat——**全部沿用 v1 §11**，不重复陈述。determinism smoke 在 v2 中要**重新做一遍**（canary cell 用 `(HW-A, 006)` 和 `(HW-C, 008)` 两个新组合）。

## 12. 测试

大部分沿用 v1 §12。新增：

- HWConfigMaker：新 fixture 验证 `vpu_num_lanes` patch 被接受、`vpu_spad_size_kb_per_lane` patch 被拒绝。
- dry_run_fit：新 fixture 覆盖 (256, 256, 128) 大 tile 在 8-lane HW 上的边界（predicted fit 边缘、ground truth 校准结果）。
- CoDesignAnalyzer：**可选**新增 fixture — 4 HW × 10 mapping 合成矩阵，让 (HW-A/B, 009) 冠军、(HW-C/D, 008) 冠军 → 断言 `champion migration` 通过、Gate-3 mean_spearman 落到 0.5 附近。此 fixture 仅证明 analyzer 能处理 4×10 输入，不参与 verdict。
- v1-v2 cross check：新增单测覆盖 metadata 里 `v1_v2_cycle_delta_per_shared_cell` 的计算与断言逻辑。

## 13. 诚实结束状态

沿用 v1 §13。**新增说明**：

```text
POSITIVE  仍然只表示 cycle-sense co-design 有 measured 支撑, 且在两个 corner 极值上;
          不代表 vpu_num_lanes 或 mem_subsystem_scale 的独立主效应被识别 (联动缩放);
          不代表内部空间 co-design 均有价值;
          不代表结论可推广到 GPT-2 block 以外的 workload;
          不代表结论可推广到 tiling 以外的 mapping 维度.
```

若 v2 仍然 NEGATIVE：意味着**换 compute 轴、加两端极值 tile 都不足以暴露 HW/SW interaction**，此时应严肃考虑放弃 tiling-only DSE 路线，改走 codex 结果 spec §8 建议的 fusion / dataflow / SPAD partition 扩展，或者干脆放弃 co-design 主张、转做 SW-only mapping DSE 加强 proxy 表征。

## 14. 后续 spec 展望

v2 若 POSITIVE：
- 3x3 或 4x4 factorial 加 `vpu_num_lanes ∈ {8, 32, 128}` 的内部值，识别非单调 interaction。
- 跨 workload：ResNet-50 一层 + Llama 一个 block，复用同一 4×10 结构。
- 加 area / power / cost，从 cycle-only 升级到联合 verdict。
- 引入 fusion 维度作为 mapping 的第 2 轴，观察 `(HW × tiling × fusion)` 三向 interaction。
- co-design 自动搜索：GP / BO 替代 dense factorial。

v2 若 NEGATIVE：
- 独立 spec：转做 SW-only tiling DSE 加强 proxy（GCL / GNN 表征替代手工 struct_feat）。
- 独立 spec：如果一定要保留 co-design 叙事，扩到 fusion / dataflow / SPAD partition（工程代价高，可能要动 TOGSim）。
- 结果 spec 里明确记录"tiling-only 空间在 systolic × BW 2x2 corner 上仍无法暴露强 interaction"作为负面证据。
