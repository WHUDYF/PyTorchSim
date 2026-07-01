# NPU Mapping DSE Co-design v2 执行进度与阻塞诊断

## 1. 当前完成项

本轮已经按 v2 plan 完成实现层面的增量改造：

1. `hw_config_factory.py` 支持 `codesign_v1_2x2` 与 `codesign_v2_2x2` 两套 HW config set。
2. v2 HW 轴从 `vpu_spad_size_kb_per_lane` 切换为 `vpu_num_lanes`：
   - HW-A/HW-B: `vpu_num_lanes = 128`
   - HW-C/HW-D: `vpu_num_lanes = 8`
   - 四个 HW 都保持 `vpu_spad_size_kb_per_lane = 128`
3. `mapping_dse_minimal.py` 的默认 10 个 mapping 已包含 v2 极值：
   - `008 = (16, 16, 16)`
   - `009 = (256, 256, 128)`
4. `determinism_smoke.py` 的默认 canary 已切到 v2 组合：
   - `HW-A/006`
   - `HW-C/008`
5. `codesign_full_run.py` 支持 4x10 输入，并支持 v2 metadata：
   - `spec_version = v2`
   - `hw_axis_axis1 = vpu_num_lanes`
   - `hw_axis_axis2 = mem_subsystem_scale`
   - `v1_v2_cycle_delta_per_shared_cell`
6. `codesign_report.py` 已加入 `v1-v2 cross check` 章节，并把 cycles/status 表从固定 `4x8` 泛化为 4xN。
7. `mapping_dse_codesign_sweep.py` 修正了一个关键状态传播问题：子进程 returncode 为 0 但没有 `total_cycles` 时，不再误标为 `measured`，而是标为 `runtime_failed` 进入 retry/终态逻辑。

## 2. 已生成产物

v2 准备产物：

```text
outputs/mapping_dse_codesign_v2/hw_config_factory_summary.json
outputs/mapping_dse_codesign_v2/hw_configs/hw_A.yml
outputs/mapping_dse_codesign_v2/hw_configs/hw_B.yml
outputs/mapping_dse_codesign_v2/hw_configs/hw_C.yml
outputs/mapping_dse_codesign_v2/hw_configs/hw_D.yml
outputs/mapping_dse_codesign_v2/mappings_10.json
outputs/mapping_dse_codesign_v2/dry_run_fit_v2.json
outputs/mapping_dse_codesign_v2/determinism_cells_v2.json
```

synthetic 4x10 smoke 产物：

```text
outputs/mapping_dse_codesign_v2/synthetic_smoke/run/full_run_summary.json
outputs/mapping_dse_codesign_v2/synthetic_smoke/run/00_metadata.json
outputs/mapping_dse_codesign_v2/synthetic_smoke/run/report/report.md
```

真实 v2 canary 产物：

```text
outputs/mapping_dse_codesign_v2/determinism_real_canary_retry/summary.json
```

## 3. 单测与 smoke 结果

相关回归测试通过：

```text
85 passed in 0.48s
```

synthetic 4x10 full-run smoke 通过：

```text
state_counts = {measured: 40, retry_exhausted: 0, unavailable: 0}
metadata_lint.ok = true
report_lint.ok = true
v1-v2 cross_check_status = PASS
```

这说明 v2 的 Python harness、metadata、report、analysis 链路已经能处理 4x10 输入。

## 4. 真实 canary 阻塞结果

真实 v2 determinism canary 结果：

```json
{
  "deterministic_class": "blocked",
  "max_cycle_delta": 0.08296699851608273,
  "failed_canaries": ["HW-C_008"]
}
```

其中：

| canary | 结果 |
| --- | --- |
| HW-A/006 | cycles = [396118, 430617, 431845]，cycle_delta = 0.08297，non_deterministic |
| HW-C/008 | failed，无法解析 total_cycles，因为底层 mapping 没有产生 counters |

这意味着当前不能进入 claim-bearing 的 4x10x3 measured run。原因有两个：

1. `HW-C/008` 在 8-lane 配置下真实编译/采样失败。
2. 即使只看 `HW-A/006`，cycle 波动也超过 v1/v2 设定的 non-deterministic 阈值 0.02。

## 5. HW-C/008 的根因

`HW-C/008` 的 `mapping_dse_minimal.py` 主进程 returncode 为 0，但内部 `mapping_status` 是 failed，`counters_table.json` 为空：

```json
{
  "data_label": "measured",
  "rows": []
}
```

子进程错误来自 MLIR lowering 的 gem5 sample 阶段。手动复现 `mlir-opt` 命令得到的关键错误是：

```text
error: Mismatched subtile K between A and B: A(128) != B(8)
error: failed to legalize operation 'linalg.matmul' that was explicitly marked illegal
```

对应命令使用了：

```text
-dma-fine-grained=systolic-array-size=8
-test-pytorchsim-to-vcix=systolic-array-size=8 vlen=256
-test-tile-operation-graph=vectorlane=8 sample-mode=1
-test-memref-to-gemmini=vectorlane=8 timing=1
```

这说明 v2 引入的 8-lane HW 触发了当前 PyTorchSim MLIR lowering 未覆盖的路径：某个 matmul 的 A/B subtile K 被 lowering 成了不一致的 128 vs 8。

## 6. 当前判断

这不是 v2 Python harness 的问题，也不是 TOGSim 结果解析问题。当前状态更接近 v2 spec 中的 R2 风险：

```text
vpu_num_lanes 减到 8 后可能触发 TOGSim / gem5 / lowering 中未测试的路径。
```

由于 plan 明确禁止修改 PyTorchSim / TOGSim / gem5 / ramulator2 源码，本轮不能通过改 lowering pass 来绕过该问题。

### 6.1 额外探针：8-lane 不是单个 mapping 的个例

为了确认 `HW-C/008` 是否只是极小 tile 的个例，又额外跑了三个 8-lane 探针：

```text
HW-C/000
HW-C/006
HW-D/000
```

产物位置：

```text
outputs/mapping_dse_codesign_v2/eight_lane_probe/summary.json
```

三个探针都出现相同终态：

```json
{
  "mapping_status": {"status": "failed"},
  "harness": {
    "ok": false,
    "measured_count": 0,
    "reason": "no mapping produced measured TOG/counters"
  },
  "counters_rows": []
}
```

因此，当前阻塞不是 `008=(16,16,16)` 特有，也不是 `HW-C` 的高带宽组合特有；`HW-C` 与 `HW-D` 这两个 `vpu_num_lanes=8` corner 都无法稳定通过 GPT-2 block 的 PyTorchSim lowering / gem5 sample 路径。v2 的核心 HW 轴正是 `128 lanes vs 8 lanes`，所以在不修改 PyTorchSim lowering 的约束下，完整 4x10 measured run 无法成立。

### 6.2 Retry 白名单验证：增加 timeout 不能解决

按 DEC-3，retry 只能改变 timeout 或 env，不能改 HW YAML、mapping、seed 或 tile。为了确认该失败不是 timeout 偶发，又对 `HW-C/000` 做了一次 timeout=1800 秒的白名单内重试：

```text
outputs/mapping_dse_codesign_v2/eight_lane_retry_probe/summary.json
```

结果仍然是：

```json
{
  "mapping_status": {"status": "failed"},
  "harness": {
    "ok": false,
    "measured_count": 0,
    "reason": "no mapping produced measured TOG/counters"
  },
  "counters_rows": []
}
```

因此，在当前允许的 retry 空间内没有继续推进到 measured run 的路径。要继续完成 v2 full run，必须改变 plan 约束：要么修改 8-lane lowering，要么修改 v2 的小档硬件取值。

## 7. 下一步选择

有三条可选路线：

1. **严格按 plan**：将 v2 当前真实执行状态判为 `BLOCKED`，因为 8-lane HW canary 不能产生 measured counters，且 determinism 超阈值。
2. **调整 v2 HW 小档**：把 `vpu_num_lanes = 8` 改为当前 lowering 支持的更温和值，例如 16/32/64。但这会违反当前 v2 plan 的 locked axis，需要更新 spec/plan。
3. **允许修改 PyTorchSim lowering**：定位并修复 `Mismatched subtile K between A and B` 的 lowering 逻辑。但这违反当前 plan 的“不修改 PyTorchSim 源码”规则，需要用户明确改变约束。

在当前约束下，最保守且可复现的结论是：v2 harness 已实现，但真实 measured run 被 8-lane lowering 路径阻塞。
