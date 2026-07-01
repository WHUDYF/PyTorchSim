# NPU 软硬件协同价值验证设计：GPT-2 单 block × HW 2×2 factorial

## 1. 目标与研究问题

本实验的作用是在**已经证实"SW mapping 在固定 HW 下会造成显著性能差异"**（详见
`docs/superpowers/specs/2026-07-01-npu-mapping-dse-minimal-loop-result.md`，实测
`cycle_spread ≈ 4.03`，最优冠军 = `TILE_M=128, TILE_N=64, TILE_K=64`）
之后，回答**更进一步**、**更能立"软硬件协同"故事**的一件事：

研究问题（自包含）：

> 在 NPU（以 PyTorchSim / TOGSim 作为硬件确认）上，对同一个 workload
> （GPT-2 单个 transformer block，prefill，sequence length = 128）：
>
> - **Q1（冠军迁移是否存在）**：随 HW 配置变化，最优 tiling mapping 是否会**换人**？
> - **Q2a（per-chip SW autotune 有增量收益吗）**：与"一份统一 mapping 卖给所有芯片"相比，
>   为每个 HW 单独选最优 mapping，节省是否 ≥ 15%？
> - **Q2b（真正的 co-design 有增量收益吗）**：与"vendor baseline HW + 该 HW 的最优
>   mapping"相比，联合选择 `(HW, mapping)` 是否节省 ≥ 15%？
> - **Q3（HW×mapping 交互是否显著）**：跨 HW 的 mapping 排名相似度是否足够低？

判定逻辑：

```text
Q1 & Q2a & Q2b & Q3 全通过 -> co-design 在本 workload 上得到 measured 支撑, POSITIVE
Q1 不通过                   -> 冠军永远是同一个 mapping, co-design 无独立价值, NEGATIVE
Q2a & Q2b 都不通过           -> 就算 argmin 会换, 增量收益也不够立故事, NEGATIVE
其它组合                    -> PARTIAL, 记录哪些 gate 通过哪些不通过, 不写 POSITIVE
```

外部概念解释（避免调用方假设已知）：

- **HW/SW co-design**：不再"先冻结硬件、只调软件"，而是**同时在 HW 参数空间和 SW mapping 空间搜索**，
  用两者的联合最优点回答"如果重新设计 chip、并允许 SW 一起调，会不会更好"。co-design 的**独立价值**
  取决于"最优 mapping 会不会随 HW 变化"——如果不会，co-design 就退化成"先定 HW 再单独 autotune"。
- **factorial design**：从统计学 DOE 借来的概念，扫描多个 HW 因子的**所有组合**（这里是 2×2 = 4 个
  corner），比"一次只变一个因子"（OFAT）更能识别**交互效应**（interaction effect）。
  参考：Montgomery, "Design and Analysis of Experiments"。
- **champion migration**：术语来自 auto-tuning 文献，指"最优候选随配置漂移"的现象，是
  co-design 有价值的**必要条件**（不是充分条件）。
- **double-swap dominance**：本 spec 引入的严格化——为了排除"冠军换人只是浮点噪声"，要求两个冠军
  在各自主场上都以 ≥ 3% 优势胜出。

## 2. 非目标

本 spec 明确**不做**：

- fusion / dataflow 变体 / SPAD 内部分区 / DMA 调度扫描（仍只扫 tiling，与
  `2026-06-30-npu-mapping-dse-minimal-loop-design.md` 严格对齐，为的是能干净地隔离
  HW×tiling 的交互项）；
- 跨 workload 泛化（仅 GPT-2 单 block prefill seq=128；ResNet-50 / Llama block 留独立 spec）；
- 引入 GNN / embedding / LLM；
- 修改 PyTorchSim / TOGSim / gem5 / ramulator2 源码（HW 变化全部通过替换 YAML config 实现）；
- 探索**内部** HW 点（本 spec 仅采样 2×2 的 4 个 corner，非单调 interaction 留独立 spec）；
- 引入 area / power / cost 维度（当前 verdict 只看 cycle 收益，co-design 的成本侧留独立 spec）。

## 3. Workload 输入

沿用上一次实验的确切配置（保证与其结果**直接可比**）：

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

CLI flags（默认值可覆盖）：

```text
--seq            (default 128)
--num-mappings K (default 8, 与上一次严格一致)
--hw-config-set  (default codesign_2x2, 见 §4)
--output-dir     (default outputs/mapping_dse_codesign/gpt2_block_prefill_s128_run1/)
--seed           (default 0)
```

## 4. HW config 扫描设计

### 4.1 factorial 结构

沿两个正交轴扫，每轴取"贵/廉"两档，共 4 个 corner：

| 轴 | 参数 | 大档 | 小档 |
| --- | --- | ---: | ---: |
| 算-存平衡：片上 | `vpu_spad_size_kb_per_lane` | `128` | `32` |
| 算-存平衡：片外 | `dram_channels`             | `32` | `8`  |

4 组 config：

| id | 语义 | SPAD (KB/lane) | DRAM ch |
| --- | --- | ---: | ---: |
| `HW-A` | vendor baseline (旗舰) | 128 | 32 |
| `HW-B` | 算宽 / 存瓶颈           | 128 | 8  |
| `HW-C` | 存宽 / 算瓶颈           | 32  | 32 |
| `HW-D` | 双紧                    | 32  | 8  |

**HW-A 被指定为 vendor baseline**，因为它在 4 个 corner 里对应"用最贵芯片配最优 SW"的场景，是
Gate-2b 的自然对手。选它不是循环论证，因为 Gate-2b 判据是"co-design 的联合最优是否显著优于
HW-A 的最优 SW"，而 HW-A 本身也在 co-design 搜索空间里——如果最终冠军 (HW, mapping) 恰好是
HW-A 上的某个 tiling，`gain=0` 会诚实地反映"co-design 无收益"，不会被有利于 HW-A 的选择偏袒。

### 4.2 HW config 生成规则

所有 4 个 YAML 均从 baseline `configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml` **patch** 而来，
不允许手工新造 YAML。patch 规则用 JSON 显式记录（见 `hw_plausibility_rules.json`）：

```json
{
  "baseline_yaml": "configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml",
  "patch_matrix": {
    "HW-A": {"vpu_spad_size_kb_per_lane": 128, "dram_channels": 32, "icnt_injection_ports_per_core": 16},
    "HW-B": {"vpu_spad_size_kb_per_lane": 128, "dram_channels": 8,  "icnt_injection_ports_per_core": 4},
    "HW-C": {"vpu_spad_size_kb_per_lane": 32,  "dram_channels": 32, "icnt_injection_ports_per_core": 16},
    "HW-D": {"vpu_spad_size_kb_per_lane": 32,  "dram_channels": 8,  "icnt_injection_ports_per_core": 4}
  },
  "plausibility_rules": [
    "icnt_injection_ports_per_core = clip(dram_channels / 2, 4, 32) rounded to nearest even",
    "vpu_spad_size_kb_per_lane 与 systolic array size 独立 (systolic 固定 128x128)",
    "num_cores 固定为 1",
    "dram_type / ramulator_config_path 固定为 HBM2_TPUv3"
  ],
  "frozen_fields": [
    "core_freq_mhz",
    "dram_freq_mhz",
    "icnt_freq_mhz",
    "num_cores",
    "num_systolic_array_per_core",
    "vpu_num_lanes",
    "vpu_vector_length_bits",
    "dram_type",
    "ramulator_config_path"
  ],
  "frozen_fields_reason": "These fields are held identical across HW-A/B/C/D. Any HW-to-HW cycle difference otherwise could be trivially explained by clock frequency or timing-model divergence rather than the SPAD/DRAM factors under study. HWConfigMaker MUST error if any patch touches a frozen field.",
  "coupling_reason": "Reducing DRAM channels without shrinking NoC injection ports produces an unrealistic corner (NoC starved chip with excess ports). Scaling ports with channels keeps the config on the plausibility manifold of chips someone would actually tape out."
}
```

### 4.3 factorial 局限的诚实声明

2×2 只采样 4 个 corner，能识别 **main effect** + **corner-level interaction**，**不能**识别
**内部非单调** interaction（例如 SPAD=64 时冠军换人、SPAD=32 和 SPAD=128 时不换）。这属于
本 spec 的**已知 scope limitation**，处理方式：

- 在 §12 显式记录；
- 后续独立 spec 若要覆盖，做 3×3 或 4×4 factorial，或 Latin square。
- 本 spec 的 POSITIVE 结论**仅陈述** "在 2×2 corner 上 co-design 有 measured 支撑"，
  不外推到"整个 HW 内部空间 co-design 均有价值"。

### 4.4 DRAM 与 NoC 联动 -> 复合因子的诚实声明

`icnt_injection_ports_per_core` 按 §4.2 的 plausibility rule 与 `dram_channels` 联动，
这是**特意**的（避免"8 channels + 16 ports"这种不真实 corner），代价是：**本 spec 无法
分离 DRAM channels 主效应与 NoC injection ports 主效应**。

处理方式：

- 本 spec 把 `{dram_channels, icnt_injection_ports_per_core}` 视为**单一复合因子**（记作
  `mem_subsystem_scale`），axis 2 的语义应当读作"片外内存子系统整体带宽"，而不是"DRAM
  channels 单变量"；
- verdict、report、后续 spec 引用本实验结论时，措辞不得暗示"DRAM channels 的独立主效应
  被识别"；
- 若后续 spec 需要分离两者，必须做 3×3 或 4×4 factorial 且允许"不联动"的 corner，同时
  显式接受 corner 的物理不合理代价。

## 5. Mapping 候选

**与上一次严格一致**的 8 个 tiling 候选（`TILE_M`, `TILE_N`, `TILE_K`）：

```text
000  32   64   32
001  64   64   32
002  64  128   32
003 128   64   32
004 128  128   32
005  64   64   64
006 128   64   64
007  64  128   64
```

`006` 是上次的冠军（HW = TPUv3 baseline）。**保持候选池不变**是本 spec 能干净比较的前提：
如果候选池变了，任何冠军迁移都可能是候选池差异造成的，与 HW 无关。

### 5.1 fit 约束与 (HW, mapping) 状态机（Risk R1）

每个 `(HW, mapping)` 组合有以下四种终态之一，**语义必须清晰分开**，混淆会污染 gate 判据：

| 状态 | 含义 | 何时触发 | 是否入 ranking |
| --- | --- | --- | --- |
| `measured` | TOGSim 成功跑完，产出 `total_cycles` | dry-run 预检通过 且 TOGSim 正常返回 | 是 |
| `unavailable` | 物理约束下 tile 塞不下 SPAD | dry-run 预检不 fit | **否**（合法排除） |
| `runtime_failed` | tile fit 但 TOGSim 崩溃 / 超时 | dry-run 通过 但 TOGSim 报错 / hang | **否**（视为待办） |
| `retry_exhausted` | `runtime_failed` 经 6-attempt 排障仍无法转 measured | R2 排障 loop 走满 | **否** |

关键规则：

- 每个 `(HW, mapping)` 组合先跑一次 **dry-run 预检**（不做 timing 模拟，仅计算 tile 静态
  占用是否超 SPAD 预算），预检失败的组合标为 `unavailable`；
- `unavailable` 是**物理事实**，不触发 bounded troubleshooting，从该 HW 的 ranking 与
  `per_hw_oracle` 池里干净排除；
- `runtime_failed` 是**工具链故障**，**必须**触发 R2 的 6-attempt 排障 loop；只要还有
  attempt 余额，该组合**不允许**留在 `runtime_failed` 状态就下 verdict；
- 若最终存在 `retry_exhausted` 组合，`end_state` **只能** 是 `PARTIAL` 或 `BLOCKED`，
  不允许 `POSITIVE`；
- fit / 状态全部记入 `fit_availability.json`（4×8 状态矩阵 + 预算超出量 + retry 次数）；
- Gate-3 的 Spearman 计算**只在每对 HW 的 `measured` 交集**上做（详见 §7.4）。

## 6. 总体数据流

```text
 baseline YAML  +  patch_matrix
        |
        v
 [1] HWConfigMaker: 生成 4 份 HW YAML + plausibility check
        | 4 * hw_<id>.yml
        v
 [2] SweepHarness: 对每份 HW YAML, 跑 mapping_dse_minimal 8 candidates
        |          (复用现有 harness, 不重写)
        | 32 份 (HW, mapping) 完整 measured 输出
        v
 [3] Extractor: 从 32 份输出解出 4 x 8 total_cycles 矩阵
        |          + fit_availability + per-mapping counter_vec
        v
 [4] CoDesignAnalyzer:
        Gate-1 champion migration (double-swap >= 3%)
        Gate-2a per-HW SW vs single mapping oracle (>= 15%)
        Gate-2b co-design vs HW-A best (>= 15%)
        Gate-3 mean pairwise Spearman on fit intersection (<= 0.7)
        |
        v
 [5] Report + Verdict: markdown 表 + heatmap + verdict.json
```

## 7. 组件设计

沿用上一次 5 组件结构。**只有** HWConfigMaker 与 CoDesignAnalyzer 是新增；SweepHarness 是对
现有 `mapping_dse_minimal` 的循环调用；Extractor / Report 大部分逻辑复用。

### 7.1 HWConfigMaker（HW 配置生成器）

- **做什么**：读 baseline YAML + `patch_matrix`，产出 4 份 HW YAML，运行 plausibility rules
  校验并生成 `hw_plausibility_rules.json`。
- **输入**：baseline YAML 路径 + `patch_matrix` JSON。
- **输出**：`hw_configs/hw_<A|B|C|D>_<label>.yml` + `hw_plausibility_rules.json`。
- **失败模式**：patch 后的 YAML 加载失败、或 plausibility rule 违反 → 状态 `BLOCKED`。
- **可单元测试**：给定 fixture baseline + patch，断言输出字段正确、rules 校验通过。

### 7.2 SweepHarness（扫描驱动器）

- **做什么**：对每个 HW YAML，调用现有 `scripts/mapping_dse_minimal.py` 以相同 8 tilings 跑一遍；
  按 §5.1 的四态状态机记录每个 (HW, mapping) 结果。
- **输入**：`hw_configs/*.yml` + 8 tilings 的 `external_mapping.json`。
- **输出**：`runs/hw_<id>/mappings/<idx>/...` 结构（与现有 harness 输出对齐，便于复用）。
- **依赖**：PyTorchSim / TOGSim / ramulator2 / gem5 环境；每个 HW YAML 需能被 TOGSim 加载。
- **状态处置**：
  - `measured`：直接归档；
  - `unavailable`：dry-run 预检失败，记录预算超出量，**不** retry；
  - `runtime_failed`：进入 R2 的 6-attempt 排障，每次 retry 记录假设/改动/输出；转 measured 后即
    退出 loop，否则 6 次后转 `retry_exhausted`；
  - `retry_exhausted`：视为该 HW 上该 mapping 无 measured，写 verdict 时 `end_state` 强制降级。
- **cold-start policy**：每个 (HW, mapping) 都在新进程里跑；gem5 / ramulator2 / TOGSim 从 clean
  state 起，不复用上一次运行的 DRAM row-buffer / cache / prefetch 内部状态。`00_metadata.json`
  记录每次 `subprocess.Popen` 的 pid 与启动时间，可事后 audit。

### 7.3 Extractor（矩阵提取器）

- **做什么**：从 32 份运行输出解析出 4×8 `total_cycles` 矩阵、`per_kernel_counters`（附录用）、
  `fit_availability` 布尔矩阵。
- **输入**：`runs/hw_<id>/mappings/<idx>/togsim_log.txt` 与 `per_kernel_counters.json`。
- **输出**：`heatmap_cycles.json`、`fit_availability.json`、`counter_vec_by_hw.json`。
- **单元测试**：给定 4×8 fixture 输出（含 `unavailable`）断言矩阵与 fit 表正确。

### 7.4 CoDesignAnalyzer（协同分析器）

- **做什么**：从 4×8 cycles 矩阵计算 4 个 gate 的所有数值。禁止依赖任何 modeled/proxy 数据。
- **输入**：`heatmap_cycles.json` + `fit_availability.json`。
- **输出**：`champion_migration.json`、`oracle_gaps.json`、`interaction_analysis.json`。

**Gate-1 champion migration（double-swap dominance ≥ 3%）**：

```text
for each unordered pair (HW_i, HW_j):
  M_i = argmin_m cycles(m, HW_i)  # in HW_i's fit set
  M_j = argmin_m cycles(m, HW_j)  # in HW_j's fit set
  if M_i == M_j:
    swap_dominance_ij = 0
    continue
  if M_j not fit on HW_i OR M_i not fit on HW_j:
    swap_dominance_ij = insufficient_evidence
    continue
  dom_i = cycles(M_j, HW_i) / cycles(M_i, HW_i) - 1  # % worse M_j is on HW_i vs local champion
  dom_j = cycles(M_i, HW_j) / cycles(M_j, HW_j) - 1
  swap_dominance_ij = min(dom_i, dom_j)

Gate-1_passed = exists (i,j) with swap_dominance_ij >= 0.03 (numeric, not insufficient_evidence)
```

**Gate-2a per-chip SW vs single mapping**：

对手是"一份统一 mapping 卖给所有 HW"；本 gate 判"per-chip 单独选 mapping 相对于统一 mapping
的收益"。为了让分子分母**用同一个候选池**（否则 fit 不对称会让分子系统性偏低，把 feasibility
asymmetry 误读为 SW autotune 收益），分子分母都被约束到 `fit_all_hw_set`。

```text
# fit_all_hw_set = 在 4 个 HW 上都 measured (不含 unavailable / runtime_failed / retry_exhausted) 的 mapping
fit_all_hw_set = {m : status(m, HW) == "measured" for all HW in [A,B,C,D]}

if fit_all_hw_set is empty:
  Gate-2a = insufficient_evidence   # 走 R2 / R1 排障, 不允许 POSITIVE

# 分母: 从 fit_all_hw_set 里挑跨 HW 平均最快的一个 mapping
single_best_avg_mapping = argmin_{m in fit_all_hw_set} mean_HW cycles(m, HW)

# 分子: 每 HW 单独选最优, 但仍限制在 fit_all_hw_set 内 (与分母同池)
per_hw_oracle_cycles(HW_i) = min_{m in fit_all_hw_set} cycles(m, HW_i)
per_hw_ratio(HW_i)         = per_hw_oracle_cycles(HW_i) / cycles(single_best_avg_mapping, HW_i)

mean_per_hw          = mean_HW of per_hw_oracle_cycles(HW)
mean_single_mapping  = mean_HW of cycles(single_best_avg_mapping, HW)
gate2a_ratio_mean    = mean_per_hw / mean_single_mapping

# 附加约束: 避免单一慢 corner 主导 mean, 要求至少 3 个 HW 上单独也满足阈值
num_hw_meeting_threshold = |{HW_i : per_hw_ratio(HW_i) <= 0.85}|

Gate-2a_passed = (gate2a_ratio_mean <= 0.85) AND (num_hw_meeting_threshold >= 3)
```

**附加信息（不入 gate 判据）**：另存一份 `per_hw_unconstrained_oracle_cycles`——每个 HW 用
它自己**完整** fit set（不限 `fit_all_hw_set`）选最优 mapping。此值代表"per-chip SKU-tuning
在完整 fit set 下的能力"，供 report.md 参考，但因为分母池已固定，它不能与 `single_best_avg_mapping`
直接相除去算 gate。

**Gate-2b co-design vs vendor baseline HW-A**：

```text
codesign_optimum = min over (m, HW) of cycles(m, HW)  # over all fit combos
vendor_baseline_optimum = min_m cycles(m, HW-A)       # over HW-A's fit set

gate2b_ratio = codesign_optimum / vendor_baseline_optimum
Gate-2b_passed = (gate2b_ratio <= 0.85)
```

**Gate-3 mean pairwise Spearman on fit intersection**：

4 个 HW 共 `C(4,2) = 6` 对 pair。为防止 adversarial fit 模式把不利 pair 剔掉后拿"3 pair 均值"
勉强通过阈值，本 gate 要求 numeric pair **数量下限**较严 且 **每对 pair** 的交集都足够大。

```text
for each unordered pair (HW_i, HW_j) in 6 pairs:
  I_ij = { m : status(m, HW_i) == "measured" AND status(m, HW_j) == "measured" }
  if |I_ij| < 4:
    spearman_ij = insufficient_evidence
    continue
  rank_i = rank of cycles restricted to I_ij on HW_i
  rank_j = rank of cycles restricted to I_ij on HW_j
  spearman_ij = spearman_rho(rank_i, rank_j)

numeric_pair_count = |{ pair : spearman_ij is numeric }|
mean_spearman      = mean over numeric spearman_ij
all_numeric_pairs_have_min_intersect_4 = all numeric pairs have |I_ij| >= 4

Gate-3_passed = (mean_spearman <= 0.7)
              AND (numeric_pair_count >= 5)          # 6 pair 中最多允许 1 pair insufficient
              AND all_numeric_pairs_have_min_intersect_4
```

如果 `numeric_pair_count < 5` → `Gate-3 = insufficient_evidence`，end_state 只能是
`PARTIAL / BLOCKED`。

### 7.5 Report + Verdict

- **做什么**：产出 `verdict.json` + `report.md`（中文），含 4×8 cycle 表、4×8 状态表、
  各 gate 数值与通过状态、诚实 end_state。
- 所有 claim-bearing 行标 `data_label = measured`；禁止 `placeholder / modeled / pending_measurement`。
- **report.md 强制字段（读者防误读用）**：
  - `migrating_pair_count / 6`：Gate-1 里 6 对 pair 中真正满足 double-swap ≥3% 的对数。1/6 与
    5/6 都可以过 Gate-1，但读者需要看到分布；
  - `per_hw_ratio 表`：Gate-2a 里 4 个 HW 各自的 ratio，让读者判断"15% 均值收益"是均匀分布
    还是被某个 corner 一票通过；
  - `unconstrained per-HW oracle vs fit_all_hw_set oracle` 对比：让读者看到 fit 约束的实际
    代价；
  - **Gate-3 与 Gate-1 联合解读 caveat**：Gate-3 只看 rank，不看绝对 cycle 差；如果 Gate-1
    的 max dominance 只有 3~5%（勉强通过），report.md 必须显式提醒 "ranking 差异对应的
    绝对性能差异不大"；
  - **反 baseline 参照数值**（R4）：以 HW-D 为 baseline 时的 `codesign_optimum /
    min_m cycles(m, HW-D)`，仅供参考，不入 verdict。

## 8. go/no-go 判据（初值，可调）

| Gate | 判据 | 阈值 |
| --- | --- | --- |
| Gate-1 champion migration | 至少一对 (i,j) `swap_dominance ≥ 3%` | 3% |
| Gate-2a per-chip SW vs single mapping | `gate2a_ratio_mean ≤ 0.85` **且** `num_hw_meeting_threshold ≥ 3`（fit-all-HW 池内） | 15% 节省 + 3/4 HW |
| Gate-2b co-design vs vendor baseline | `codesign_optimum / hw_A_best ≤ 0.85` | 15% 节省 |
| Gate-3 interaction | `mean_spearman ≤ 0.7` **且** `numeric_pair_count ≥ 5`（6 中至少 5）**且** 每对 `n_intersect ≥ 4` | 0.7 / 5 / 4 |

**POSITIVE** 需要 4 个 gate 全部 `passed = true`，且不存在任何 `insufficient_evidence` 或
`retry_exhausted` 状态。任一硬性子条件不满足 → 只能 `PARTIAL / NEGATIVE / BLOCKED`。

## 9. 产物与 schema

运行目录：`outputs/mapping_dse_codesign/<run_tag>/`

```text
00_metadata.json
hw_configs/
  hw_A_bigmem_bigbw.yml
  hw_B_bigmem_lowbw.yml
  hw_C_smallmem_bigbw.yml
  hw_D_smallmem_lowbw.yml
hw_plausibility_rules.json
runs/
  hw_A/mappings/{000..007}/mapping_config.json, tog.onnx, togsim_log.txt, per_kernel_counters.json, ...
  hw_B/...
  hw_C/...
  hw_D/...
heatmap_cycles.json
fit_availability.json
counter_vec_by_hw.json
champion_migration.json
oracle_gaps.json
interaction_analysis.json
verdict.json
report.md
```

`heatmap_cycles.json`：

```json
{
  "data_label": "measured",
  "hw_ids": ["HW-A", "HW-B", "HW-C", "HW-D"],
  "mapping_ids": ["000","001","002","003","004","005","006","007"],
  "cycles": [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]
  ],
  "unavailable_marker": -1
}
```

`champion_migration.json`：

```json
{
  "data_label": "measured",
  "champions_per_hw": {"HW-A": "006", "HW-B": null, "HW-C": null, "HW-D": null},
  "pair_dominance": [
    {"pair": ["HW-A","HW-B"], "M_i": "006", "M_j": null, "dom_i_pct": 0, "dom_j_pct": 0, "swap_dominance_pct": 0, "status": "pending"}
  ],
  "gate1_max_dominance_pct": 0,
  "gate1_passed": false
}
```

`oracle_gaps.json`：

```json
{
  "data_label": "measured",
  "fit_all_hw_set": [],
  "single_best_avg_mapping_id": null,
  "single_best_avg_mapping_cycles_per_hw": {"HW-A": 0, "HW-B": 0, "HW-C": 0, "HW-D": 0},
  "per_hw_oracle_cycles_constrained": {"HW-A": 0, "HW-B": 0, "HW-C": 0, "HW-D": 0},
  "per_hw_ratio_constrained":         {"HW-A": 0.0, "HW-B": 0.0, "HW-C": 0.0, "HW-D": 0.0},
  "gate2a_ratio_mean": 0.0,
  "gate2a_num_hw_meeting_threshold": 0,
  "gate2a_passed": false,
  "per_hw_unconstrained_oracle_cycles": {"HW-A": 0, "HW-B": 0, "HW-C": 0, "HW-D": 0},
  "per_hw_unconstrained_oracle_mapping": {"HW-A": null, "HW-B": null, "HW-C": null, "HW-D": null},
  "codesign_optimum_cycles": 0,
  "codesign_optimum_pair": {"hw": null, "mapping": null},
  "vendor_baseline_hw": "HW-A",
  "vendor_baseline_optimum_cycles": 0,
  "vendor_baseline_optimum_mapping": null,
  "gate2b_ratio": 0.0,
  "gate2b_passed": false,
  "reverse_baseline_hw": "HW-D",
  "reverse_baseline_ratio_reference_only": 0.0
}
```

`interaction_analysis.json`：

```json
{
  "data_label": "measured",
  "pair_spearman": [
    {"pair": ["HW-A","HW-B"], "n_intersect": 0, "spearman": null, "status": "pending"}
  ],
  "mean_spearman_numeric_only": 0.0,
  "numeric_pair_count": 0,
  "all_numeric_pairs_have_min_intersect_4": false,
  "permutation_null_probability_reference_only": 0.0,
  "gate3_passed": false
}
```

`permutation_null_probability_reference_only` 由 R5 描述的 1000 次 permutation 得出，仅供
report 解读，**不入** `gate3_passed` 判据。

`verdict.json`：

```json
{
  "data_label": "measured",
  "claim_bearing": true,
  "gate1_champion_migration": {
    "max_dominance_pct": 0,
    "migrating_pair_count": 0,
    "total_pair_count": 6,
    "passed": false
  },
  "gate2a_per_chip_vs_single": {
    "ratio_mean": 0.0,
    "num_hw_meeting_threshold": 0,
    "fit_all_hw_set_size": 0,
    "passed": false
  },
  "gate2b_codesign_vs_vendor": {
    "ratio": 0.0,
    "vendor_baseline_hw": "HW-A",
    "reverse_baseline_ratio_reference_only": 0.0,
    "passed": false
  },
  "gate3_interaction": {
    "mean_spearman": 0.0,
    "numeric_pair_count": 0,
    "all_numeric_pairs_have_min_intersect_4": false,
    "passed": false
  },
  "state_counts": {
    "measured": 0,
    "unavailable": 0,
    "runtime_failed": 0,
    "retry_exhausted": 0
  },
  "scope_limitations": [
    "single workload: GPT-2 single block prefill seq=128",
    "corner-only factorial (2x2): non-monotonic interior interactions not covered",
    "cycle-only verdict: area / power / cost not considered",
    "axis-2 = mem_subsystem_scale (dram_channels bundled with icnt_injection_ports); main effects not separable",
    "POSITIVE means cycle-sense co-design value only; area / power / cost-normalized verdict is a separate spec"
  ],
  "end_state": "POSITIVE | NEGATIVE | PARTIAL | BLOCKED"
}
```

`00_metadata.json` 字段（为独立复现服务，必须齐全）：

- `timestamp` / `run_tag` / `git_commit` / `git_status` (dirty 与否);
- `cli_args`（完整 argv）;
- `baseline_yaml_path` + `baseline_yaml_sha256`;
- `patch_matrix`（原样引用 §4.2）+ `frozen_fields_verified: true` + `plausibility_rules_verified: true`;
- 4 份 `hw_yaml_content_sha256`（patch 后的完整 YAML 内容 hash）+ `hw_yaml_diff_from_baseline`（unified diff）;
- 每个 (HW, mapping) 的 `simulator_cmdline`（完整 argv 字符串）、`stderr_path`、`stdout_path`、
  `subprocess_pid`、`start_time`、`end_time`、状态 (`measured / unavailable / runtime_failed /
  retry_exhausted`)、retry 次数、retry 每轮的假设与结果;
- `ramulator2_config_path` + `ramulator2_config_sha256`;
- `gem5_binary_path` + `gem5_binary_sha256`;
- `togsim_binary_sha256` + `pytorchsim_git_commit`;
- `env_vars_snapshot`: `{CUDA_INSTALL_PATH, TORCHSIM_LLVM_PATH, GEM5_PATH, PYTHONPATH,
  LD_LIBRARY_PATH, PATH}` 的实际值；
- `torch_version` / `transformers_version` / `python_version`;
- `determinism_smoke_test`: `{repeats: 3, cycle_deltas: [...], deterministic: bool}` （见 R7）;
- `artifact_paths`（本次 run 所有 artifact 的绝对路径列表）。

## 10. Milestone gates

```text
Gate α HW 配置就绪:   4 份 HW YAML + plausibility rules 校验通过, 无 frozen_field 被动过
                      [BLOCKED 若 patch 加载失败或触碰 frozen field]

Gate β 采集就绪:      1) R7 determinism smoke test 完成, 且不为 non-deterministic;
                      2) 4 x 8 = 32 组 (HW, mapping) 每组终态 ∈ {measured, unavailable};
                      3) 无组合停留在 runtime_failed 或 retry_exhausted
                      [data_label=measured, unavailable 允许]

Gate γ 分析已算:      champion migration / oracle gaps / interaction / 反 baseline 参考值
                      全部算出                                                          [data_label=measured]

Gate δ 裁决完成:      4-gate 判决 -> end_state, scope_limitations 齐全
```

各 gate 允许的数据标签：claim-bearing 行仅 `measured`；不允许 `placeholder / modeled /
pending_measurement`。`unavailable` 是**合法**状态（表示物理约束下 tile 塞不下 SPAD），不视为
placeholder，但必须在 `fit_availability.json` 里显式记录。`runtime_failed` 与 `retry_exhausted`
**不是**合法终态——存在这两种状态 → Gate β 未过 → end_state 只能 `PARTIAL / BLOCKED`。

## 11. 风险与缓解

### R1 SPAD 小的 HW 上部分 tile 塞不下

- 现象：预测 HW-C / HW-D 上 `TILE_M=128, TILE_K=64` 之类的组合可能超 SPAD 预算。
- 缓解：§5.1 的 dry-run 预检 + `unavailable` 标记；Gate-3 的 fit-intersection 处理；
  Gate-2a 需要 `fit-all-HW` mapping 集合非空——若全 8 个都在某个 HW 上不 fit，
  Gate-2a `insufficient_evidence`，最终 end_state 只能 `PARTIAL`，绝不写 `POSITIVE`。

### R2 patch 后的 HW config 不能被 TOGSim 加载

- 现象：DRAM channels 从 16 改到 8 或 32 可能触发 ramulator2 / booksim2 config 联通性问题。
- 缓解：先跑一次 `smoke run`——每个 HW YAML 用一个最简 addmm workload 单独 sanity check，
  确认能出 timing 结果。任何 HW YAML 通不过 smoke → 进 bounded troubleshooting：
  ```text
  max_attempts_per_plan_gap_issue = 6
  每次记录: 阻塞问题 / 假设 / patch 改动 / TOGSim 报错 / 下一步
  6 次未解 -> BLOCKED (不写 POSITIVE / PARTIAL)
  ```

### R3 单次 TOGSim 运行时间过长导致总时长失控

- 上限估计：4 HW × 8 mapping = 32 次，若单次 15 min，总时长 ≈ 8 h。
- 缓解：先跑 HW-A 全部 8 个 mapping 拿到"单次时间预算 P95"；若 P95 > 20 min 则收缩：
  - 优先 3 个 HW（HW-A / HW-B / HW-C），跳 HW-D，能保 factorial 主效应但丢一个 corner；
  - 或候选减到 K=6（去掉离冠军最远的 3 个），保 4 HW 但减小 mapping 池。
- 收缩后必须在 `verdict.json` 的 `scope_limitations` 里显式写清。

### R4 HW-A 作为 vendor baseline 的选择可能被质疑循环论证

- 潜在质疑：Gate-2b 以 HW-A 为 baseline，如果 co-design 冠军就是 HW-A + 某 mapping，
  `gate2b_ratio = 1.0`，Gate-2b fail。这不是漏洞，是**诚实结果**——说明"再怎么 co-design
  也无法从最贵芯片再提升"，co-design 就没独立价值。
- 附加防线：在 `report.md` 里同时报告 **反 baseline** 结果：以 HW-D 为 baseline 的
  `codesign_optimum / min_m cycles(m, HW-D)`。这个数值**不计入** verdict，仅供读者参考，
  避免只看 HW-A baseline 得出误导结论。

### R5 Gate-3 阈值 0.7 在 K=8 下的显著性

- 隐忧：在 8 个点上，Spearman 有非平凡的抽样分布，`0.7` 是否显著需要小心。
- 缓解：在 `interaction_analysis.json` 里同时输出每对 pair 的 `spearman` 与 pair 数量；
  在 `report.md` 附录里做 permutation baseline（把 4 个 HW 的 ranking 全部打散重排 1000 次，
  报告在 null 下 `mean_spearman ≤ 0.7` 的概率）。这个 permutation 结果**不用于**判 Gate-3，
  但用于**读者解读**。permutation 逻辑纯数学，用合成数据可单测。

### R6 factorial 只有 4 个 corner，非单调 interaction 无法识别

- 已在 §4.3 显式声明；本 spec 不试图解决，交后续独立 spec。

### R7 Simulator 非确定性可能翻转冠军

- 隐忧：PyTorchSim / TOGSim / gem5 / ramulator2 若在 timing 上非确定（受 OS 调度、内存
  分配顺序、autotune 内部随机化影响），同一 (HW, mapping) 两次结果可能有差异；Gate-1 的
  3% 阈值附近的冠军判定会被翻转。
- 缓解流程（**Gate β 采集就绪前**强制执行的 smoke test）：
  ```text
  1. 选 (HW-A, mapping 006) 作为 canary 组合；
  2. 顺序跑 3 次, 完全 cold-start (新进程 / 新 tmp 目录);
  3. 记录 3 次 total_cycles, 计算 cycle_delta = (max - min) / min;
  4. 若 cycle_delta <= 0.001 (0.1%): 判定 deterministic, 32 组每组跑 1 次即可;
  5. 若 0.001 < cycle_delta <= 0.02 (2%): 判定 near-deterministic, 32 组每组跑 3 次取中位数,
     heatmap_cycles.json 加 stddev 字段;
  6. 若 cycle_delta > 0.02: 判定 non-deterministic, end_state 强制降级为 PARTIAL,
     verdict.json 附上 canary 数据, 由后续 spec 决定是否引入更强重复方案;
  ```
- `determinism_smoke_test` 结果必须存 `00_metadata.json`；缺少即 Gate β 不允许通过。

### R8 Gate 判据可能被"看似高交互但实际噪声级"的排名颠倒骗过（软 caveat）

- 隐忧：Gate-3 只看 rank，不看绝对 cycle 差。极端情况下 4 个 HW 的 ranking 完全不同，但
  cycles 在每个 HW 上都只差 ~1%——Gate-3 会 pass，但实际 co-design 收益极小。
- 缓解：Gate-1 的 3% dominance 已经从 cycles 侧提供了下界保护（要求至少一对 pair 有实质
  cycle 差）。此外 report.md 强制"Gate-3 与 Gate-1 联合解读"（详见 §7.5）。本 spec 不追加
  新的硬 gate。

## 12. 测试

- HWConfigMaker：
  - 给定 baseline YAML + patch_matrix fixture，断言输出 YAML 字段正确、plausibility rules
    违反时抛错；
  - **frozen_fields 验证**：构造 patch 触碰 `core_freq_mhz` / `dram_freq_mhz` 之一 → 断言抛错；
  - 验证 `hw_yaml_content_sha256` 与 `hw_yaml_diff_from_baseline` 正确生成。
- Extractor：4×8 fixture（含 `unavailable` / `runtime_failed`）→ 断言矩阵、状态表正确；
  `runtime_failed` 不进入 ranking，`unavailable` 从池里排除。
- CoDesignAnalyzer 单元测试要覆盖的合成矩阵：
  - all-champion-same：Gate-1 fail，Gate-2a fail，Gate-2b fail。
  - obvious-migration：Gate-1 pass with dominance = 20%，migrating_pair_count = 6。
  - noise-only migration（0.5% dominance）：Gate-1 fail（噪声不算迁移）。
  - **Gate-2a 分子分母池不对称**：per-HW oracle 在完整 fit set 下会更低，但被 fit_all_hw_set
    约束后与分母同池 → 验证不会误判。
  - **Gate-2a 单点主导**：3 个 HW ratio = 1.0，1 个 HW ratio = 0.5，mean_ratio = 0.875 ≤ 0.85
    但 `num_hw_meeting_threshold` = 1 → Gate-2a fail。
  - **fit_all_hw_set 为空**：Gate-2a 判 `insufficient_evidence`，end_state 强制 `PARTIAL`。
  - **Gate-3 numeric_pair_count = 4**：即使 mean_spearman = 0.3，Gate-3 也 fail（<5）。
  - **Gate-3 每对 n_intersect 边界**：某对 n_intersect = 3 → 该对 `insufficient_evidence`，
    检查 `all_numeric_pairs_have_min_intersect_4` 逻辑。
  - Spearman = 1（所有 HW ranking 完全一致）：Gate-3 fail。
  - Spearman = -1（6 对均 numeric）：Gate-3 pass。
- Determinism smoke test：合成 3 组重复数据，断言 `cycle_delta` 分类正确（deterministic /
  near-deterministic / non-deterministic）。
- Permutation baseline（R5）：合成已知 ranking 集合，断言 permutation 概率与解析近似值吻合。
- 真正的 measured 运行**是证据，不是单元测试**，单独留痕于运行目录。

## 13. 诚实结束状态

```text
POSITIVE: 4 个 gate 全过, 无 insufficient_evidence 与 retry_exhausted,
          co-design 在本 workload 上有 measured 支撑
NEGATIVE: Gate-1 fail (冠军不迁移)  或  Gate-2a & Gate-2b 都 fail (增量收益不够)
PARTIAL : 部分 gate 过部分不过, 或 fit_all_hw_set 为空导致 Gate-2a insufficient_evidence,
          或 Gate-3 numeric_pair_count < 5, 或存在 retry_exhausted 组合,
          或 R7 determinism smoke test 判定 non-deterministic
BLOCKED : HW YAML 无法加载 / TOGSim 环境不可用 / 6-attempt 排障未解
```

任何 gate 出现 `insufficient_evidence` 且不能通过更多 measured 数据消除 → 最终不允许写
`POSITIVE`；最好写 `PARTIAL` 并列出哪些 gate 缺证据。

**POSITIVE 的语义边界（防误读硬 caveat）**：

- POSITIVE **仅**表示"cycle 意义上，在 2×2 corner 的 mem_subsystem_scale × SPAD_size 空间
  内，co-design 相对于 vendor baseline (HW-A) 与统一 mapping 都有 ≥ 15% 收益、且 mapping
  ranking 跨 corner 相关性弱"。
- POSITIVE **不表示**：
  - co-design 在 area / power / cost 归一化后仍有价值；
  - HW 内部空间（非 corner）co-design 一样有价值；
  - DRAM channels 或 NoC ports 各自独立主效应被识别；
  - 结论可推广到 GPT-2 block 以外的 workload；
  - 结论可推广到 tiling 以外的 mapping 维度。
- 引用本 spec 结论时，措辞必须包含上述 scope 限定；`scope_limitations` 字段是引用本
  verdict 的**强制上下文**。

## 14. 后续 spec 展望

本 spec 若得到 POSITIVE，按独立 spec 逐项扩展：

```text
- 3x3 或 4x4 factorial: 覆盖内部非单调 interaction
- 加 systolic array 大小轴 (128x128 / 64x64 / 32x32) 变第 3 因子
- 跨 workload 泛化: ResNet-50 一个 stage + Llama block 复用同一 4x8 结构
- 引入 area / power / cost, 从 verdict 单纯看 cycle 升级到看性能/面积/功耗联合
- 引入 fusion 维度作为 mapping 的第 2 轴, 观察 (HW × tiling × fusion) 三向 interaction
- co-design 空间的自动搜索: 引入 GP / BO 之类替代 dense factorial
```

若本 spec 得到 NEGATIVE，同样是有效科学结论：说明在本 workload 上 co-design 无独立价值，
主线转回 SW-only mapping DSE，后续投入优先做**更强 proxy 表征**（GCL / GNN），而非扩展 HW 空间。
