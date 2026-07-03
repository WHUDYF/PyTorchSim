# PyTorchSim 编译器修改维度全景

## 1. 背景与范围

Route 4 的目标是从 tiling-only search space 扩展到更接近真实编译器控制面的修改维度，判断 PyTorchSim 上是否存在 measured HW/SW co-design POSITIVE 证据。

证据链从 v1 NEGATIVE 开始：`92623de` 在 GPT-2 single block prefill seq=128 的 tiling-only 2x2 SPAD/BW 空间上得到 NEGATIVE，`09b9ac3` 在 10% Gate-1 threshold 下重算后仍保持 NEGATIVE。随后 `42e57b6` 做 Route 4 discovery，确认 fusion 是无需源码修改即可切换的编译器轴，而 dense systolic dataflow 不是 YAML 暴露轴。最终 `79e6259` 在 Conv 3x3 workload 上得到 `FUSION_CONV_CODESIGN_POSITIVE`：fusion speedup across HW 的 relative range 为 `42.7%`，`gate2b_ratio_fusion=0.523`。

本文只总结当前可实现的编译器修改维度，不新增 TOGSim sweep，不修改 PyTorchSim / TOGSim / gem5 / ramulator2 / spike 源码。

## 2. 四个编译器修改维度的当前状态

| 维度 | 当前 status | 修改成本 | 是否已实测 | POSITIVE 证据 |
| --- | --- | --- | --- | --- |
| fusion | `codegen_compiler_optimization` YAML field，可设 `none` / `all` / subset；已证明会改变 MLIR/TOG 结构 | 0，无源码修改 | YES | Conv 4-HW sweep：`fusion_speedup_range.relative_range=0.427`，`gate2b_ratio_fusion=0.523`，commit `79e6259` |
| dataflow | Dense systolic 主路径是 weight-stationary / `WS_MESH`，未暴露 OS/IS YAML 字段 | HIGH，需要 codegen template、TOG/cycle assumptions 和 config plumbing overhaul | NO | none |
| SPAD partition | `SPAD_PARTITION_HARDCODED`：只发现容量字段 `vpu_spad_size_kb_per_lane`，没有 partition strategy 字段；模板内固定 `input_buffer` / `weight_buffer` / `output_buffer` | HIGH，需要改 template tile descriptors / scratchpad allocation policy | NO | none |
| DMA schedule | `DMA_SCHEDULE_HARDCODED`：未发现 `prefetch` / `double_buffer` / `dma_schedule` / `dma_pipeline` 配置字段；DMA 插入由 codegen buffer 顺序和模板逻辑决定 | MEDIUM-HIGH，需要改 MLIR DMA emission、wait/dependency insertion 和 possibly TOG generation | NO | none |

## 3. Fusion 维度深入

Fusion 是当前唯一已经跑通并产生 POSITIVE 的 compiler axis。

`42e57b6` 的 discovery 确认 fusion 控制面来自 `codegen_compiler_optimization`，配置可设为 `all`、`none` 或部分 fusion family。相关能力在 `PyTorchSimFrontend/mlir/mlir_scheduling.py` 和 GEMM/BMM/Conv templates 中实现。限制是：它不是任意 pattern-level 开关，template-template fusion 被禁止，Conv 主要支持 epilogue fusion。

后续实验链如下：

- `59f44ba` recovery 后，addmm+relu 128 在 mode=0 timing 下出现强信号：`fusion=none` median `2183` cycles，`fusion=all` median `1118` cycles，约 `2x`。
- `3489f94` 在 GPT-2 block seq=128 上做 fusion x HW mini check：`fusion=none` median `430717`，`fusion=all` median `422711`，差异只有 `1.9%`，低于 10% gate，也接近 7% noise floor。
- `ce5f0e3` 做 root-cause investigation：GPT-2 structural diff 显示 `fusion=all` 的 MLIR op count 比 `fusion=none` 少约 `7.8%`，所以 fusion 不是 structural no-op；但该变化不在 GPT-2 主导 critical path 上。Conv 3x3 single-cell probe 则显示 `fusion=none` median `189987`，`fusion=all` median `27769`，约 `6.84x`。
- `79e6259` 做 Conv 4-HW x 2-fusion sweep，得到 `FUSION_CONV_CODESIGN_POSITIVE`。四个 HW 的 fusion speedup 为：`HW-A=6.895x`，`HW-B=4.830x`，`HW-C=6.262x`，`HW-D=5.253x`。`fusion_speedup_range.relative_range=0.427`，超过 15% threshold。

结论：fusion 轴是实测有效的，但其价值高度 workload-dependent。GPT-2 block 上弱，Conv-heavy workload 上强，并且在 Conv 上与 HW 配置存在 measured interaction。

## 4. Dataflow 维度死锁

`42e57b6` 的 discovery 已确认 dense systolic main path 实际上是 weight-stationary。

关键证据包括：

- config family 命名为 `systolic_ws_*`。
- TOGSim 默认 core type 为 `WS_MESH`。
- `PyTorchSimFrontend/mlir/mlir_template.py` 的 GEMM mapping heuristic 注释强调 reuse weight。
- 搜索未发现 `output_stationary` / `input_stationary` / dense systolic dataflow YAML selection。

STONNE sparse path 中存在 dataflow enum，但它属于 vendored STONNE path，不是 PyTorchSim dense `WS_MESH` systolic path 的无源码配置轴。

因此 dataflow 当前是 `BLOCKED without source modification`。若未来要把 dataflow 纳入 Route 4，需要新增 YAML field、extension config plumbing、GEMM/BMM/SDPA loop order 和 tile descriptor generation，还要检查 TOG generation / gem5 cycle sampling 是否含 weight-stationary assumption。修改成本估计为 300-800 LOC 加验证。

## 5. SPAD partition 维度

Task 1 discovery 只做只读 grep，没有运行 TOGSim sweep。

检索命令覆盖：

```bash
grep -rn "spad_partition\|SPAD_partition\|input_buffer\|weight_buffer\|output_buffer" PyTorchSimFrontend/mlir/ TOGSim/ configs/
grep -rn "buffer\|allocator" PyTorchSimFrontend/mlir/
grep -rn "spad_size\|vpu_spad" configs/
```

发现：

- configs 中存在 `vpu_spad_size_kb_per_lane`，这是容量轴，不是 partition policy。
- Conv templates 明确创建 `input_buffer`、`weight_buffer`、`output_buffer`，例如 `PyTorchSimFrontend/mlir/mlir_conv_template.py` 中分别设置 `X_tile_desc.set_name("input_buffer")`、`W_tile_desc.set_name("weight_buffer")`、`Y_tile_desc.set_name("output_buffer")`。
- BMM/GEMM 类 template 也采用固定的 SRAM buffer 命名，例如 `X_buffer`、`W_buffer`、`Y_buffer`。
- `PyTorchSimFrontend/mlir/mlir_codegen_backend.py` 维护 `spad_buffer_dict`，并在 load/store 时调用 `get_scratchpad_buffer()`，但没有发现 YAML/CLI 层面的 partition strategy selector。

判定：`SPAD_PARTITION_HARDCODED`。

含义：当前可以 sweep SPAD 容量，例如 v1 的 `vpu_spad_size_kb_per_lane`，但不能无源码地表达“给 input/weight/output/intermediate 分配不同 SPAD fraction”或替换 allocation policy。若要做 SPAD partition co-design，需要修改 template tile descriptors、scratchpad allocation reuse policy，以及可能的 TOG address/tag generation。

## 6. DMA schedule 维度

Task 1 discovery 同样只做只读 grep。

检索命令覆盖：

```bash
rg -n "prefetch|double_buffer|async_wait|dma_schedule|dma_pipeline" PyTorchSimFrontend/mlir TOGSim/src TOGSim/include configs
rg -n "DMA|dma" configs PyTorchSimFrontend/mlir/mlir_codegen_backend.py PyTorchSimFrontend/mlir/mlir_template.py PyTorchSimFrontend/mlir/mlir_common.py
```

发现：

- 未发现 `prefetch`、`double_buffer`、`async_wait`、`dma_schedule`、`dma_pipeline` 等 YAML/CLI 控制字段。
- `PyTorchSimFrontend/mlir/mlir_common.py` 的 `format_dma_op_attributes()` 支持给 `memref.dma_start` 加 `async = ...` attribute，默认 async 行为由 codegen 传参决定。
- `PyTorchSimFrontend/mlir/mlir_template.py` 和 `mlir_codegen_backend.py` 内部有 `dma_loads` / `dma_stores` buffers，模板会在固定位置 splice DMA load/store。例如 template 中有“Do dma store first to overlap epilogue nodes”的硬编码调度意图。
- `mlir_codegen_backend.py` 中 `get_dma_code()` 生成 `memref.dma_start`，DMA tag/counter/cache 也在 backend 内部维护。

判定：`DMA_SCHEDULE_HARDCODED`。

含义：当前 DMA schedule 是 codegen/template 内部策略，不是无源码可 sweep 的 compiler knob。若未来要把它变成 Route 4 轴，至少要暴露 prefetch/double-buffer/wait policy，改 DMA insertion order，并验证 raw TOG / ONNX TOG 的 wait dependency 是否与新 schedule 一致。

## 7. 未来 Route 4 扩展建议

优先级建议如下：

1. Fusion x HW at multiple mappings：当前 Conv 4-HW 只有 single-mapping proxy。若要形式化 Gate-1 / Gate-3，应在 Conv-heavy workload 上加入多个 mapping，观察 champion migration 和 ranking correlation。
2. Fusion x HW on additional Conv-heavy workloads：用 ResNet-50 stage、MobileNet depthwise/pointwise block 或 Conv-BN-ReLU stack 验证 Conv 3x3 POSITIVE 是否泛化。
3. SPAD partition axis or DMA schedule axis：当前 discovery 判定二者都不是 YAML controlled。若愿意接受源码修改，优先选择改动范围更清楚的 DMA schedule，再考虑更复杂的 SPAD partition。
4. Dataflow axis：只有在接受较高源码修改成本后再做。当前 dense systolic path 是 weight-stationary，不能把 OS/IS 当作现有配置轴。

最终建议：短期 Route 4 应继续围绕 fusion axis 做 workload 和 mapping 扩展；不要把 Conv POSITIVE 直接泛化到 GPT-2 或所有 NPU workload。
