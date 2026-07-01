# NPU 映射 DSE 协同价值验证 v2 实现计划：GPT-2 单 block × systolic × mem_subsystem 2×2 factorial

## Goal Description

落实 v2 设计 spec `docs/superpowers/specs/2026-07-01-npu-mapping-dse-codesign-v2-design.md`。**本 plan 是 v1 plan `docs/superpowers/plans/2026-07-01-npu-mapping-dse-codesign-plan.md` 的增量修订版**，v1 的 10 scripts + 11 tests 大部分**直接复用**，只在**两处**做代码修改，**新增一处**跨版本验证逻辑。

**必须继承 v1 的所有已锁决策**（DEC-1 ~ DEC-5）：

- Gate-2b 用 cost-matched pair `{HW-B, HW-C}`，仍是 ex-post allocation sensitivity upper bound（v2 里语义变成"compute-heavy+低 BW vs compute-light+高 BW"，更合理）
- Gate-2a 分子分母硬约束 `fit_all_hw_set`，unconstrained per_hw_oracle 附加参考
- retry 白名单严格（timeout + env 变化，禁止 HW YAML / mapping_config / seed / mapping_id 变动）
- SPAD fit classifier 解析公式 + `predicted_fit XOR actual_fit` 校准 + 自动 fallback
- 统计契约冻结（lex tie-break / epsilon=0 / average rank / arith mean + aux geomean）

**v2 相对 v1 的三处硬修正**：

1. **HW 第 1 轴换成 `vpu_num_lanes`**（128 vs 8），`vpu_spad_size_kb_per_lane` 加入 frozen_fields（v1 实测证明它在 32~128 KB/lane 范围内不 binding）。
2. **Mapping 池从 8 扩到 10**：`008 = (16, 16, 16)`、`009 = (256, 256, 128)`。
3. **新增 v1-v2 cross check**：对 v1 v2 都有的 16 个 shared cell（HW-A/B × mapping 000~007）在 v2 跑完后计算 cycle_delta，验证 harness 无 regression。

## Acceptance Criteria

TDD 原则：每条 AC 都有 positive + negative 测试。**大部分 AC 直接沿用 v1**，本 plan 只列 v2 需要新增或修改的 AC。

### 直接沿用 v1 的 AC（不列细节，见 v1 plan）

`AC-1`（preflight）、`AC-4`（CLI），`AC-7`（statistical contract），`AC-11`（terminal state completeness），`AC-14`（golden matrix regression fixture）。

### v2 需要修改的 AC

- **AC-2-v2**：HW config factory 更新 patch_matrix + frozen_fields
  - Positive: 传入 v2 patch_matrix（HW-A/B/C/D，`vpu_num_lanes ∈ {128, 8}`）→ 生成 4 份可 PyYAML 加载的 YAML。
  - Positive: patch 只改 `vpu_num_lanes` 和 `dram_channels` + `icnt_injection_ports_per_core` → 工厂接受。
  - Negative: patch 试图改 `vpu_spad_size_kb_per_lane`（v2 新 frozen）→ raise `FrozenFieldViolation`。
  - Negative: patch 试图设 `vpu_num_lanes = 16`（v2 只允许 {8, 128}）→ raise `PlausibilityRuleViolation`（因为本 spec 明确不扫内部值）。

- **AC-3-v2**：SPAD dry-run fit classifier 重新校准
  - Positive: 用新校准组合 `(HW-A, 006)` 和 `(HW-C, 009)` 真跑 TOGSim，比对 `predicted_fit XOR actual_fit`，两组都 `== 0` → `calibration_passed = true`。
  - Positive: `(HW-C, 009)` 的 fit ratio ≈ 50%（512 KB / 1 MB），classifier 预测 `fit`（safety_factor=0.9 之下）——若 TOGSim 也能跑 → 一致。
  - Negative: 若 TOGSim 崩（e.g., 实际占用超过 SPAD） → 至少一组 XOR = 1 → 自动 fallback 到 no-precheck。

- **AC-5-v2**：Determinism smoke test 用新 canary 组合
  - Canary cells 换成 `(HW-A, 006)`（v2 里预期 fit-all，与 v1 相同）和 `(HW-C, 008)`（v2 新，`8-lane HW + 极小 tile`，接近 tile 边界）。
  - `cycle_delta = (max - min) / median` 分类表沿用 v1（deterministic / near_deterministic / non_deterministic）。

- **AC-6-v2**：SweepHarness driver 处理 10 mapping × 4 HW = 40 cell
  - Positive: `heatmap_cycles.json` 4×10 矩阵，每 cell 终态 ∈ `{measured, unavailable, retry_exhausted}`。
  - Negative: 若 `(HW-C, 009)` 或 `(HW-D, 009)` 被判 unavailable，`fit_availability.json` 记录预算超出量、状态矩阵中标注、不阻塞 verdict emit。

- **AC-8-v2**：CoDesignAnalyzer 处理 4×10 矩阵
  - Positive: 4 gate 计算逻辑**代码不改**，纯函数对新形状矩阵一样能跑。
  - Positive（可选新增 fixture）: 合成 4×10 矩阵，让 `(HW-A/B, 009)` 冠军、`(HW-C/D, 008)` 冠军 → 断言 `champion_migration.gate1_passed=true`、`migrating_pair_count ≥ 4`。

### v2 新增的 AC

- **AC-15（新增）**：v1-v2 cross check on 16 shared cells
  - Positive: v2 跑完后自动读取 v1 run（`outputs/mapping_dse_codesign/gpt2_block_prefill_s128_run1/heatmap_cycle_stats.json`），对每个 shared cell（HW-A/B × mapping 000~007，共 16 个）计算 `cycle_delta = abs(v2_median - v1_median) / min(v1_median, v2_median)`，写入 `00_metadata.json.v1_v2_cycle_delta_per_shared_cell`。
  - Positive: 若 16 个 delta 全部 ≤ v2 determinism smoke `max_cycle_delta`（约 0.019）→ cross check `PASS`。
  - Negative: 若任一 cell delta > 0.20（20%）→ verdict 强制降级为 `PARTIAL` + `scope_limitations` 添加 `"v1-v2 cross check failed on cells: [...]"` 字符串。
  - Negative: 若 v1 run 目录缺失或已删 → `cross_check_status = "v1_baseline_missing"`，仍允许 verdict 走 `POSITIVE / NEGATIVE / PARTIAL`，但 `scope_limitations` 追加相应说明。

- **AC-16（新增）**：`--hw-config-set` CLI 切换 v1/v2 patch_matrix
  - Positive: `python scripts/mapping_dse_codesign_sweep.py --hw-config-set codesign_v2_2x2` → 使用 v2 patch_matrix；`--hw-config-set codesign_v1_2x2` → 使用 v1（可复现 v1 结果）。
  - Positive: 不传 flag → 默认使用 `codesign_v2_2x2`（v2 是当前主线）。
  - Negative: 传未知 set 名 → argparse 报错 + exit 2。

### 直接沿用 v1 的 AC（不修改）

`AC-9`（retry 白名单）、`AC-10`（verdict + report schema）、`AC-12`（extended metadata schema）、`AC-13`（permutation baseline，upper bound only）。

## Path Boundaries

### Upper Bound

- 全部沿用 + 修改的 AC 落地，含 AC-13 permutation baseline 与 AC-14 golden fixture 保留 + AC-15 cross check 全字段 + AC-16 CLI 切换。
- `report.md` 增加"v1-v2 cross check 章节"（列出 16 shared cell 的 cycle_delta 表）。
- 完整跑完 4×10×3 repeat = 120 次 measured TOGSim runs。
- Codex 独立复核 verdict interpretation（analyze task）通过。

### Lower Bound

- AC-1 沿用 + AC-2-v2/3-v2/5-v2/6-v2/8-v2 + AC-15/16 全部落地。
- AC-13 permutation baseline 可跳过（`scope_limitations` 声明）。
- 若 `(HW-C/D, 009)` 全部 unavailable，`fit_all_hw_set` 池缩到 ≤ 8 → Gate-2a 仍能算，但结果可能是 `insufficient_evidence` → end_state `PARTIAL` 是合法交付。

### Allowed Choices

- **可用**：v1 已 commit 的 10 scripts + 11 tests（`scripts/codesign_*.py`、`scripts/dry_run_fit.py`、`scripts/hw_config_factory.py`、`scripts/mapping_dse_codesign_sweep.py`、`scripts/stats_contract.py`；同名 tests），继承 argparse / PyYAML / dataclasses / pytest 现有约定。
- **不允许**：与 v1 相同（不改 PyTorchSim/TOGSim/gem5/ramulator2 源码、不引入 numpy/scipy/GNN/LLM、不扩 workload、不扩 mapping 维度到 tiling 以外、不在 claim-bearing 用 placeholder/modeled、不在 retry 内改 HW YAML / mapping_config / seed / mapping_id）。

## Feasibility Hints and Suggestions

### Conceptual Approach

**代码改动量估计**（对 v1 已 commit 版本）：

```
scripts/hw_config_factory.py     ~30 行改动 (新 patch_matrix + frozen_fields 集合)
scripts/dry_run_fit.py           ~10 行改动 (calibration 组合更新, 逻辑不变)
scripts/mapping_dse_codesign_sweep.py  ~15 行改动 (加 --hw-config-set flag)
scripts/codesign_full_run.py     ~50 行新增 (cross check 逻辑)
scripts/codesign_metadata.py     ~20 行新增 (v1_v2_cycle_delta_per_shared_cell 字段)
scripts/codesign_report.py       ~30 行新增 (v1-v2 cross check 章节)

tests/test_hw_config_factory.py         ~40 行新增 (v2 fixture)
tests/test_dry_run_fit.py               ~30 行新增 (v2 calibration)
tests/test_codesign_analyzer_golden.py  ~20 行新增 (可选 v2 fixture)
tests/test_codesign_full_run.py         ~40 行新增 (cross check 测试)
tests/test_codesign_metadata.py         ~20 行新增 (新字段)

其它 scripts / tests: 0 改动
```

总量 ~305 行改动，远小于 v1 的 9010 行。

### Relevant References

- v1 已 commit 代码：`92623de` — 全部脚本 + 测试 + 完整 4×8 measured run，可以直接读来复用
- v1 result spec `docs/superpowers/specs/2026-07-01-npu-mapping-dse-codesign-result.md` §10 根因分析（**必读**——说明为什么 v2 要做这三处修改）
- v2 design spec `docs/superpowers/specs/2026-07-01-npu-mapping-dse-codesign-v2-design.md`（本 plan 的 authoritative reference）
- baseline YAML `configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml`
- v1 run 产物：`outputs/mapping_dse_codesign/gpt2_block_prefill_s128_run1/`（AC-15 cross check 要读它的 `heatmap_cycle_stats.json`）

## Dependencies and Sequence

### Milestones

1. **Infrastructure Delta Ready (Gate α)**
   - Phase A: task-v2-a（hw_config_factory 更新）+ task-v2-b（dry_run_fit 重校准）
   - Phase B: task-v2-c（CLI flag `--hw-config-set`）+ task-v2-d（metadata 加 cross check 字段）

2. **v2 Acquisition Ready (Gate β)**
   - Phase A: task-v2-e（v2 canary determinism smoke）
   - Phase B: task-v2-f（4×10 sweep 跑 3 repeat, 状态矩阵齐）

3. **Claim-bearing Measurement Complete (Gate γ)**
   - Phase A: task-v2-g（4 gate 计算）+ task-v2-h（v1-v2 cross check 计算）
   - Phase B: task-v2-i（verdict + report 加 v1-v2 章节 emit）

4. **Go/No-Go Decision Complete (Gate δ)**
   - Phase A: task-v2-j（2 HW × 2 mapping smoke integration）+ task-v2-k（4×10×3 完整 measured run）
   - Phase B: task-v2-l（codex verdict interpretation review）

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---|---|---|---|---|
| task-v2-a | Update `hw_config_factory.py` v2 patch_matrix + frozen_fields (vpu_num_lanes 移出, vpu_spad 移入) | AC-2-v2 | coding | v1 committed baseline |
| task-v2-b | Update `dry_run_fit.py` calibration to use `(HW-A, 006)` and `(HW-C, 009)`, verify no logic change | AC-3-v2 | coding | task-v2-a |
| task-v2-c | Add `--hw-config-set {codesign_v1_2x2, codesign_v2_2x2}` CLI to `mapping_dse_codesign_sweep.py` | AC-16 | coding | task-v2-a |
| task-v2-d | Update `codesign_metadata.py` add `v1_v2_cycle_delta_per_shared_cell` field + `spec_version=v2` | AC-15 | coding | task-v2-a |
| task-v2-e | Determinism smoke test with new v2 canaries; sanity check `(HW-C, 008)` 边界 | AC-5-v2 | coding | task-v2-c |
| task-v2-f | Sweep 4 HW × 10 mapping × 3 repeat via existing driver; verify 40 cell 状态齐 | AC-6-v2 | coding | task-v2-b, task-v2-c |
| task-v2-g | Verify `codesign_analyzer.py` handles 4×10 matrix (no code change, add optional v2 fixture test) | AC-8-v2 | coding | task-v2-f |
| task-v2-h | Implement v1-v2 cross check logic in `codesign_full_run.py` + report 章节 emit in `codesign_report.py` | AC-15 | coding | task-v2-d, task-v2-g |
| task-v2-i | Verdict + report emit including v1-v2 cross check section | AC-10 (v1 reuse), AC-15 | coding | task-v2-h |
| task-v2-j | Integration smoke test 2 HW × 2 mapping mini-run with v2 configs | AC-6-v2, AC-15 | coding | task-v2-f, task-v2-i |
| task-v2-k | Full 4×10×3 measured run → claim-bearing verdict.json + report.md | all v2 AC | coding | task-v2-e, task-v2-j |
| task-v2-l | Codex independent verdict interpretation review; output appended to report.md | — | analyze | task-v2-k |

## Claude-Codex Deliberation

### Agreements（继承自 v1 plan）

- 四态状态机、statistical contract 冻结、fit classifier 双阶段（解析公式 + calibration + fallback）、retry 白名单严格、task deps 反映 metadata 耦合，全部沿用。

### Resolved Disagreements（v1 已解决，v2 无新分歧）

- v1 的 DEC-1 ~ DEC-5 全部继承，不重新讨论。
- v2 新增的 3 处修正（换 systolic 轴 / 加两端 tile / 加 cross check）都是**用户 + Claude 已合议**的结果（见 v1 result spec §10 根因分析），不再送 codex convergence 检查。

### Convergence Status

- Final Status: `converged`（继承自 v1，v2 delta 无新歧义）
- Convergence rounds: 0（本 plan 是 v1 增量修订，不再从零走 gen-plan 全流程）

## Pending User Decisions

无。所有决策 v1 已定，v2 delta 已合议。

## Implementation Notes

### Code Style Requirements

- **不允许**在代码 / 注释 / commit message / PR body 出现 plan-progress terminology（`AC-*`、`Milestone`、`Step`、`Phase`、`Gate α/β/γ/δ` 等），也不允许 AI 工具名（`Codex` / `Claude` 等）。
- 用 domain-appropriate 命名：例如 v1-v2 cross check 对应的函数应叫 `compare_shared_cells_with_baseline_run()` 而非 `check_ac15()`。
- Report.md 中文正文，JSON 字段 / CLI flag / 文件路径 / 模块名英文。

### Cold-Start Policy

- 与 v1 一致：每个 (HW, mapping) 都新 subprocess，gem5 / ramulator2 / TOGSim clean state 起。
- `00_metadata.json` 记录每次 `subprocess.Popen` 的 pid + start/end time。

### Reproducibility Contract

- 4 份 v2 HW YAML 生成后立即写 SHA-256 到 `hw_yaml_content_sha256`；每次 retry 前 assert hash 未变。
- v1-v2 cross check 依赖 v1 run 产物存在于 `outputs/mapping_dse_codesign/gpt2_block_prefill_s128_run1/`。若已删或不存在 → cross check status = `v1_baseline_missing`，不阻塞 verdict emit。
- 通过 v2 `spec_version` 字段确保跨版本 artifact 可辨识。

### 特别提醒：不要重复造轮子

v1 已 commit 的 10 scripts + 11 tests **是 v2 的 baseline**。任何"重写一遍"、"从头做"的冲动都是错的。v2 的实现应该是：

1. `git log --oneline` 找到 v1 commit `92623de`
2. `git show 92623de --stat` 看 v1 落地的完整清单
3. 只改本 plan 列出的 5 个 scripts + 5 个 tests，其它一个字都不动
4. 如果某个 v1 的 script 已经能满足 v2 需求（例如 `codesign_analyzer.py` 对 4×10 矩阵原生支持），**不要"为了看起来完整"再改一遍**
