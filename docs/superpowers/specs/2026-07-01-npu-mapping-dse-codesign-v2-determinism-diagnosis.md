# v2 HW-A/006 determinism regression diagnosis

## 1. Method

本诊断只检查 `HW-A/006` 这个 canary cell，不重新运行完整 4x10 sweep，也不修改 PyTorchSim / TOGSim / gem5 / ramulator2 源码。实验使用同一条 `scripts/mapping_dse_minimal.py` measurement path，对 v2 setup 和 v1 setup 分别执行 5 次独立子进程运行。

v2 setup 使用：

```text
--hw-config outputs/mapping_dse_codesign_v2/hw_configs/hw_A.yml
--num-mappings 1
--external-mappings-json /tmp/determinism_diag_v2/mapping_006.json
--seed 0
--timeout-sec 900
```

v1 setup 使用由 `scripts/hw_config_factory.py --hw-config-set codesign_v1_2x2` 重新生成的 `/tmp/v1_hw_configs/hw_A.yml`，并使用同一个 `006=(128,64,64)` mapping：

```text
--hw-config /tmp/v1_hw_configs/hw_A.yml
--num-mappings 1
--external-mappings-json /tmp/determinism_diag_v1/mapping_006.json
--seed 0
--timeout-sec 900
```

每次运行都写入独立目录 `/tmp/determinism_diag_v*/run_i`。统计口径为读取对应 `counters_table.json` 中 `mapping_id = "006"` 的 `total_cycles`，然后计算：

```text
cycle_delta = (max(cycles) - min(cycles)) / median(cycles)
```

## 2. Results

v1 summary 文件：`/tmp/determinism_diag_v1/summary.json`。

| setup | run_1 | run_2 | run_3 | run_4 | run_5 | median | cycle_delta | class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v1 | 398589 | 429014 | 409713 | 424222 | 411006 | 411006 | 0.074026 | `non_deterministic` |

v2 summary 文件：`/tmp/determinism_diag_v2/summary.json`。

| setup | run_1 | run_2 | run_3 | run_4 | run_5 | median | cycle_delta | class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v2 | 417396 | 425598 | 397695 | 403582 | 402268 | 403582 | 0.069138 | `non_deterministic` |

两组都产生了 measured counters，没有出现 `runtime_failed` 或缺失 `total_cycles`。

## 3. Classification

分类为 **case alpha**。

原因是 v1 和 v2 在 5-repeat 受控实验中都显示 `cycle_delta > 0.05`：v1 为 `0.074026`，v2 为 `0.069138`。这说明此前 v1 `cycle_delta = 0.019` 更可能是小样本偶然结果，而不是 v2 setup 特有的测量噪声回归。

它不符合 case beta，因为 v1 没有稳定停留在约 `0.02`；它也不符合 case gamma，因为 v2 并没有表现出“run_1 明显更快、run_2..run_5 相互接近”的 cold/warm 模式。v2 的最小值出现在 run_3，v1 的最小值出现在 run_1，但两者后续运行并没有形成稳定的 warm cluster。

## 4. Implication for v2.1 gate calibration

当前 `HW-A/006` canary 的测量噪声底线约为 7%。因此，如果 v2.1 继续使用同一 measurement path 和同类 workload，Gate-1 dominance threshold 不能继续设为 3%。3% 低于本次观察到的测量噪声，会把随机波动误判为 mapping 或硬件配置差异。

更合理的起点是把 Gate-1 dominance threshold 提高到约 10%，并在报告中把它解释为“高于当前 canary noise floor 的保守阈值”。这不直接推翻 v1 已记录的 NEGATIVE verdict，但会削弱 v1 gate calibration 的可信度：v1 的阈值校准需要按新的噪声观测重新解释。

## 5. Recommendation

v2.1 spec 应采用 case alpha 的结论：`HW-A/006` 的非确定性不是 v2 patch matrix 独有问题，而是当前真实 measurement path 的噪声底线问题。后续设计不应优先 bisect v2 HW-A YAML，也不应只加入 warmup run 后沿用 3% threshold；更稳妥的做法是把 canary 5-repeat 作为 gate calibration 的前置步骤，将 dominance threshold 提高到约 10%，并要求 claim-bearing sweep 在报告中同时披露 canary noise floor。
