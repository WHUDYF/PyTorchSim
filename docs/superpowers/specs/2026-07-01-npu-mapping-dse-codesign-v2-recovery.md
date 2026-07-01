# v2 阻塞恢复行动手册

## 0. 使用说明

本文档不是 spec，也不是 plan。它是一份**面向操作者的行动手册**，说明在 v2 撞到 8-lane BLOCKER 之后，该做什么、按什么顺序、每一步的判据是什么。文档产出的最终物是：

- 一次 debug（回答"HW-A determinism 塌陷是不是真事"）
- 一份 v2.1 delta spec（如果 debug 通过）
- 一份 v2.1 handoff（给 codex）

## 1. 现状盘点

### 1.1 Codex 已完成的实现层改动（85 tests 全绿）

- `hw_config_factory.py`：支持 `codesign_v1_2x2` 与 `codesign_v2_2x2` 两套 patch_matrix，通过 `--hw-config-set` 切换
- `mapping_dse_codesign_sweep.py`：
  - 加了 `--hw-config-set` CLI flag（默认 `codesign_v2_2x2`）
  - **修了一个 v1 时期就存在的 bug**：`returncode=0` 但 `total_cycles` 缺失时，从"错标 measured"改成"正确标 runtime_failed"
- `codesign_full_run.py` / `codesign_metadata.py` / `codesign_report.py`：支持 4×N 输入（原来硬编码 4×8）、v2 metadata 字段、v1-v2 cross check
- 新增 `scripts/determinism_smoke.py`
- v2 config 生成、10 mapping 池、synthetic 4×10 smoke 全过

### 1.2 Codex 已经证实的 BLOCKER

```text
HW-C/000  failed  (无 counters)
HW-C/006  failed  (无 counters)
HW-C/008  failed  (无 counters)
HW-D/000  failed  (无 counters)
retry timeout 900 -> 1800 仍失败
mlir-opt 报错: "Mismatched subtile K between A and B: A(128) != B(8)"
```

**结论**：`vpu_num_lanes = 8` 在当前 PyTorchSim mlir lowering 下整体不可测；v2 plan 禁改源码，`BLOCKED` 是诚实终态。

### 1.3 v2 意外暴露的次要问题

**HW-A/006 canary 的 determinism 显著变差**：

| 版本 | 3 次 cycles | cycle_delta | class |
|---|---|---:|---|
| v1 | 410615 / 409399 / 402878 | 0.019 | near_deterministic |
| v2 | 396118 / 430617 / 431845 | **0.083** | **non_deterministic** |

v1 3 次都在 2% 带内；v2 有**一次跑得快 8%**（396K），另两次接近 431K。这**不是随机噪声**，看起来像"第一次冷启动 vs 后续暖启动"的系统性模式。

**为什么重要**：即使我们 fix 了 8-lane 问题，如果 HW-A（128-lane）都是 non_deterministic，Gate-1 double-swap dominance ≥ 3% 阈值形同虚设——测量本身的噪声就已经 > 3%。

### 1.4 未 commit 的 codex 修改

```text
M scripts/mapping_dse_codesign_sweep.py    (sweep bug fix + --hw-config-set)
M scripts/codesign_full_run.py             (4xN 支持 + cross check)
M scripts/codesign_metadata.py             (v2 字段)
M scripts/codesign_report.py               (v2 章节)
?? scripts/determinism_smoke.py            (新)
??（可能还有 tests/、v2 outputs/）
```

session 一挂就丢。**优先级最高**。

## 2. 分成 5 步的恢复流水线

### 步骤 1（最高优先级）：让 codex 提交当前所有 v2 工作

**为什么先做**：avoid session-death 数据丢失。

**给 codex 的 prompt**（复制粘贴到 codex）：

```text
在 /home/dyf/PyTorchSim 里，把当前所有 uncommitted 的 v2 相关改动 commit 到 master:
- scripts/mapping_dse_codesign_sweep.py (bug fix + --hw-config-set)
- scripts/codesign_full_run.py
- scripts/codesign_metadata.py
- scripts/codesign_report.py
- scripts/determinism_smoke.py (new)
- 所有新增的 tests
- outputs/mapping_dse_codesign_v2/ 下的小 JSON summary 与 report.md
  (raw TOGSim 日志被 .gitignore 排除, 用 git add -f 加显式小文件)
- docs/superpowers/specs/2026-07-01-npu-mapping-dse-codesign-v2-progress.md (codex 写的)
- docs/superpowers/specs/2026-07-01-npu-mapping-dse-codesign-v2-recovery.md (本文档)

不要动 mtime 是 2026-06-24 的 M-status 文件 (mlir_*.py, OpenReg*.cpp 等), 那些不是 v2 的工作.

commit message 遵守项目规则:
- 不出现 Codex / Claude / AI 工具名
- 不用 Phase / Step / AC-x 等 progress 术语
- 主题行 <70 字符, 正文说 "why" 而不是 "what"

建议 commit message 主题:
  Add v2 harness delta and 8-lane BLOCKED evidence

正文可以写 v2 delta 实现完成 + 8-lane 阻塞诊断 + 未修 lowering 的原因.
```

**验证**：`git log --oneline -3` 里应出现新 commit，`git status` 里 v2 相关脚本状态变干净。

**判据**：commit 成功 → 进步骤 2。commit 失败（比如 pre-commit hook 拦） → 让 codex 修完再来。

---

### 步骤 2：Debug HW-A determinism 塌陷（1~2 小时）

**假设 1**：v2 canary 是**首次运行**在 fresh subprocess，而 v1 canary 可能被"上一次 sweep 遗留的 warmup 状态"救了。若假设成立，v2 的 396K 是真实的冷启动数字，v1 的 402~410K 都是"暖机后的"数字。这不是 bug，是**v1 一直有的度量偏差**，只不过 v2 才暴露出来。

**假设 2**：codex 的 sweep bug fix 引入的副作用——`extract_total_cycles` 现在被调用两次（一次在 status 判定、一次在 return），如果这个函数有副作用（比如读日志 seek），可能改变了下游行为。看代码：

```python
total_cycles = extract_total_cycles(cell_output, cell["mapping_id"], stdout + "\n" + stderr)
status = "measured" if proc.returncode == 0 and total_cycles is not None else "runtime_failed"
```

只是变量绑定，没有副作用。**假设 2 排除**。

**假设 3**：v1 与 v2 的 canary 跑法不一样——v1 直接用 `mapping_dse_minimal.py`，v2 用 codex 新加的 `determinism_smoke.py`。两条路径的 subprocess 环境差异可能引入了额外的冷启动开销（例如 v2 canary 是 clean tmp dir + fresh PYTHONPATH，v1 canary 是 sweep 的 second-order 上下文）。

#### 步骤 2 具体动作（自己做，不交给 codex）

**动作 A**：读 codex 新写的 `scripts/determinism_smoke.py`，找**它跟 v1 canary 逻辑的差异**：

```bash
# 找 v1 时代 determinism smoke 的原始实现（在 codesign_full_run.py 里）
git show 92623de:scripts/codesign_full_run.py | grep -n -A 30 'determinism_smoke\|canary' | head -80

# 对比 codex 现在的 determinism_smoke.py
cat /home/dyf/PyTorchSim/scripts/determinism_smoke.py | head -100
```

**关键看点**：
- 3 次 canary run 是否共享 tmp dir？还是每次新 dir？
- 是否有 warmup run（不计数的）？
- subprocess 的 env 是否每次都 reset？

**动作 B**：在 v2 setup 下 **手工重跑 HW-A/006 5 次**，看是否第一次总是最快：

```bash
cd /home/dyf/PyTorchSim
source /home/dyf/.venv/bin/activate
for i in 1 2 3 4 5; do
  echo "=== run $i ==="
  time python scripts/mapping_dse_minimal.py \
    --hw-config outputs/mapping_dse_codesign_v2/hw_configs/hw_A.yml \
    --num-mappings 1 \
    --external-mappings-json outputs/mapping_dse_codesign_v2/mappings_10.json \
    --output-dir /tmp/canary_run_$i
  grep total_cycles /tmp/canary_run_$i/counters_table.json | head
done
```

**判据**：
- 若 5 次 cycles 全部 ≤ 2% 波动（跟 v1 一致） → v2 canary 逻辑本身有问题，需要 codex 修 `determinism_smoke.py`
- 若 5 次 cycles 都 8% 左右波动，第一次总是快 → **确认冷启动系统性偏差**，需要在 canary 里加 warmup run
- 若模式随机 → 真非确定性，可能要考虑 SEED 设置或 sub-process 隔离

**动作 C**：跟 v1 的同一 cell 5 次跑对比：

```bash
for i in 1 2 3 4 5; do
  # v1 config: HW-A 用 vpu_num_lanes=128, spad=128 (v1 patch)
  time python scripts/mapping_dse_minimal.py \
    --hw-config <v1 时期 HW-A yml> \
    --num-mappings 1 \
    --external-mappings-json <v1 mappings_8.json> \
    --output-dir /tmp/v1_canary_run_$i
  grep total_cycles /tmp/v1_canary_run_$i/counters_table.json | head
done
```

**判据**：若 v1 config 下 5 次跑也是 8% 波动 → 是 canary 独立跑本身有问题（跟 v1/v2 无关）；若 v1 config 下稳定 2% → v2 的 8-lane 相关改动（可能是 fit classifier calibration）引入了系统影响。

---

### 步骤 3：根据 debug 结果决定 v2.1 是走 Route A' 还是 Route B

Debug 结果**必然**属于以下一种：

#### Case α：HW-A determinism 8% 是 canary 独立跑固有问题（不是 v2 独有）

- 结论：v1 结果里 HW-A 的 near_deterministic 是**巧合**，真实测量噪声可能一直更大
- 影响：Gate-1 double-swap dominance ≥ 3% 阈值需要抬到 ≥ 10%（覆盖 8% 噪声 + 边际）
- 行动：先把 canary 加 1 次 warmup run（不入 median）+ 抬 Gate-1 阈值到 10%，再 update v2.1 spec

#### Case β：v1 setup 稳定、v2 setup 抖，是 codex 加的什么东西引入了系统偏差

- 结论：codex 的 determinism_smoke.py 或者 sweep bug fix 有隐性副作用
- 行动：让 codex 定位并回滚，恢复到 v1 的 near_deterministic 水平后再讨论 v2.1

#### Case γ：v1 和 v2 都抖，说明 TOGSim 本身对某种 config 敏感

- 结论：需要在 sweep 层引入 warmup run + 4 repeats（取 median of 3 discarding first）
- 行动：升级 spec §7 Determinism smoke test 的 tolerance table，用 median of 3-discarding-first

在 debug 完成之前**不要**丢新 handoff 给 codex——不然又是重跑同样的 8-lane BLOCKED。

---

### 步骤 4：写 v2.1 delta spec + plan

假设步骤 3 完成，我们知道 determinism 问题的性质。下一步是写 v2.1 delta（相对 v2）：

**必改**：
1. `vpu_num_lanes` 小档从 8 改为 **32**（避开 mlir lowering bug）。参数选择理由：
   - 128 / 32 = 4× 差距，仍有 compute 侧张力
   - 32 lanes 大概率已在 v1 时代被 exercise 过（`mapping_dse_minimal` 默认路径），不会触发 A(128) != B(8) 那种边界
   - 若 32 仍不 work，再改 64（1/2 差距）

2. 更新 §5.1 fit classifier calibration：
   - v2 用 `(HW-A, 006)` + `(HW-C, 009)`，但 HW-C 现在是 32 lanes，SPAD 总量 = 32 × 128 = 4 MB，(256,256,128) 的 512 KB working set fit ratio = 12.5%，会**轻松 fit**——`(HW-C, 009)` 失去 known-no-fit 的角色
   - 换成 `(HW-A, 006)`（fit）+ `(HW-C, some_impossible_tile)`——例如构造一个 known-no-fit tile 或者去 unavailable 的 known-fit 分类

3. Update `frozen_fields` 和 patch_matrix 数值

**可能改**（取决于步骤 2 结果）：
4. Gate-1 dominance 阈值：3% → 10%（如果确认 canary 噪声本来就 ~8%）
5. Determinism smoke 加 warmup run，只算 3 个"暖机后"cycles
6. mapping 池要不要保留 008 = (16, 16, 16)：在 32-lane HW 上小 tile 的 MAC 利用率还是很低，但不像 8-lane 那样触发 mlir bug；保留是安全的

**建议路径**：不重写 v2 spec/plan，而是**追加一份 v2.1 delta doc**——写清"v2 → v2.1 的最小改动清单"，代码那边只 patch 相应字段。

---

### 步骤 5：Fresh handoff 给 codex

**关键点**：不能再用 `/tmp/codesign-v2-goal-handoff.md`。它引用的 v2 plan 已过时（lanes=8）。

需要一份 `/tmp/codesign-v2.1-goal-handoff.md`，包含：

- 明确说 "**v2 的 8-lane 路径已确认 BLOCKED，本 handoff 是 v2.1 恢复轮次**"
- 指向 v2 recovery brief（本文档）
- 指向 v2.1 delta spec（步骤 4 产出）
- 强调："**先读 v2 progress spec 已完成的实现，只 patch delta，不要重跑 sweep 直到 canary determinism 通过**"
- 目标：先跑一次 4 HW × 3 mapping（`006`、`007`、`008`）的 mini-sweep 验证 32-lane 通道能出 counters + determinism 达标；通过再展开完整 4×10

---

## 3. 决策点摘要（供用户 checklist）

| # | 决策 | 你需要做的 |
|---|---|---|
| 1 | 步骤 1 那段 prompt 是否可以直接丢给 codex？ | 复制粘贴到 codex |
| 2 | 步骤 2 的 debug 是我（Claude）做，还是让 codex 做？ | 我建议我做，codex 是"实现者"，debug 需要跨 v1/v2 上下文；等我做完给你 case α/β/γ 结果 |
| 3 | 步骤 3 里 v2.1 用 32 lanes 是否 OK？ | 先按 32 走；32 不 work 再降到 64 |
| 4 | 步骤 4 的 v2.1 是"追加 delta doc" 还是"重写 v2 spec/plan"？ | 追加 delta doc（v1→v2→v2.1 保留完整决策链） |
| 5 | Debug 期间要不要停止 codex？ | 停。codex 现在在原地打转，让它先做步骤 1 commit，然后 idle 等 v2.1 handoff |

## 4. 时间估计

| 阶段 | 估计耗时 |
|---|---:|
| 步骤 1 codex commit | 5~15 min |
| 步骤 2 debug（5 次手工重跑 + 分析） | 45~90 min |
| 步骤 3 case 判定 | 5 min |
| 步骤 4 v2.1 delta spec + plan | 30 min |
| 步骤 5 handoff 写 + 给 codex | 15 min |
| **codex 跑 v2.1 4×3 mini-sweep** | 30~60 min |
| **codex 跑完整 4×10 measured** | 4~8 小时 |

## 5. 风险与回退

- **风险 R1**：32 lanes 也触发未知 lowering bug → 降到 64 lanes 再试。
- **风险 R2**：determinism 无论怎么改都下不来 → 承认 v2.1 也 BLOCKED，放弃 Route A，转 Route B（v1 的 SPAD 压到 4 KB 让 SPAD 真的 binding）。
- **风险 R3**：跑 mini-sweep 时 32 lanes + 高 BW 组合出 crash（HW-C mid tier） → 逐步降级到 32 lanes + baseline BW，观察是否只是 BW 极值组合的问题。

若三条都撞墙：说明"想在 tiling-only 空间里立 co-design 故事"这个方向本身在 PyTorchSim 上不可行（不是 spec 的问题，是 simulator 的成熟度问题）。此时应当停止 co-design 主线，把工作转成"SW-only tiling DSE + 强 proxy 表征"论文角度。

## 6. 结束状态判据

**v2.1 交付成功** = 满足以下**所有**：
1. 4 HW × 10 mapping = 40 cell 全 measured / unavailable / retry_exhausted（无 runtime_failed 残留）
2. determinism smoke 分类 ∈ {deterministic, near_deterministic}（若 Case α 走了新阈值，`cycle_delta ≤ 10%` 视为通过）
3. verdict.json 有 end_state（POSITIVE / NEGATIVE / PARTIAL / BLOCKED 之一，不是 pending）
4. v1-v2.1 cross check 完成，`v1_v2_cycle_delta_per_shared_cell` 全部数字化（不含 null）
5. codex 完成 verdict interpretation analyze task

**中间状态**（可接受但需明说）：
- 若 32 lanes 也 fail → v2.1 = BLOCKED，切换 Route B
- 若 Case α 抬阈值后 Gate-3 通不过（noise > interaction） → NEGATIVE，co-design 主线正式终结

## 7. 一句话总结

> Codex 交付了 v2 delta 实现（可用）+ 8-lane BLOCKED 证据（诚实）+ 一个 HW-A determinism 塌陷线索（需 debug）。
> 下一步不是继续给 codex 同样的 handoff，而是先 commit + debug + 决定 v2.1（lanes 换 32），再给 fresh handoff。
