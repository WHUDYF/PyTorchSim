# NPU mapping DSE co-design v2 阻塞报告

结论状态: `BLOCKED`

## 已完成

- v2 HW config set、10 mapping pool、4x10 full-run harness、metadata/report/cross-check 链路已实现。
- synthetic 4x10 smoke 通过。
- 相关回归测试通过。

## 阻塞证据

- `HW-C/008` 真实 canary failed，没有产生 `total_cycles`。
- `HW-C/000` 与 `HW-C/006` 额外探针同样 failed，说明不是单个 tile 的问题。
- 手动复现 `mlir-opt` 得到 `Mismatched subtile K between A and B: A(128) != B(8)`。

## 结论

在当前 plan 禁止修改 PyTorchSim / TOGSim / gem5 / ramulator2 源码的约束下，v2 的核心 8-lane HW 轴无法完成 claim-bearing measured run。下一步需要更新约束：要么允许修 lowering，要么修改 v2 小档 lanes。

## Retry 白名单验证

对 `HW-C/000` 使用 timeout=1800 秒重试，仍然没有产生 counters，说明当前白名单内 retry 不能绕过 8-lane lowering 失败。

## 低带宽 8-lane corner 验证

补跑 `HW-D/000` 后同样 failed，没有 counters。这说明 8-lane 阻塞不是 HW-C 高带宽组合特有，而是 v2 小档 HW 轴整体不可测。
