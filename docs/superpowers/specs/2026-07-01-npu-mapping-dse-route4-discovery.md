# Route 4 discovery report

## 1. Motivation and scope

v1 在 GPT-2 single block prefill seq=128、tiling-only search space 上得到 `NEGATIVE` verdict。之后的 re-analysis 在 10% Gate-1 threshold 下仍保持 `NEGATIVE`，而 5-repeat canary 诊断显示当前 measurement path 的 noise floor 约为 7%。因此，下一步问题不是继续微调 tiling threshold，而是判断更宽的 compilation search space 是否值得投入。

本文只做 Route 4 discovery，不写 Route 4 spec/plan/code，不重跑 TOGSim sweep，不修改 PyTorchSim / TOGSim / gem5 / ramulator2 源码，也不修改 measured cycles。证据来自代码搜索、关键文件阅读，以及一个不进入 NPU/TOGSim 的单 kernel 前端 smoke test。

## 2. Q1 answer: fusion capability

### Evidence

PyTorchSim 的 fusion 决策位于 TorchInductor scheduler 层。`PyTorchSimFrontend/mlir/mlir_scheduling.py:23-35` 定义 `MLIRScheduling`，并在初始化时 monkey patch `scheduler.can_fuse`；`mlir_scheduling.py:84-89` 先检查 `CONFIG_FUSION` 和 `max_fusion_size = 5`；`mlir_scheduling.py:99-101` 明确禁止 template-template fusion。

已实现的 fusion case 包括：

- reduction + reduction：`mlir_scheduling.py:110-124`。
- template + pointwise epilogue：`mlir_scheduling.py:126-170`。
- template + reduction epilogue：`mlir_scheduling.py:172-193`。
- pointwise prologue + template：`mlir_scheduling.py:37-65`。

template 能力由各 template 声明：

- GEMM 支持 epilogue / prologue / reduction fusion：`PyTorchSimFrontend/mlir/mlir_gemm_template.py:105-110`。
- BMM 支持 epilogue / prologue / reduction fusion：`PyTorchSimFrontend/mlir/mlir_bmm_template.py:154-159`。
- Conv 支持 epilogue fusion，不支持 prologue / reduction fusion：`PyTorchSimFrontend/mlir/mlir_conv_common.py:11-17`。
- Flash SDPA 作为 fused attention template 接入：`PyTorchSimFrontend/mlir/mlir_lowering.py:395-405` 将 `aten._scaled_dot_product_fused_attention_overrideable` lowering 到 `_mlir_tuned_flash_sdpa`，对应 template 为 `MLIRFlashSDPATemplate`。

配置暴露机制在 `PyTorchSimFrontend/extension_config.py:94-127`。`codegen_compiler_optimization` 支持：

```text
all
none
[fusion, reduction_epilogue, reduction_reduction, prologue, single_batch_conv, multi_tile_conv, subtile]
```

当前 configs 默认几乎都为 `codegen_compiler_optimization: all`，例如 `configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml:24-28`。因此 fusion 可以通过临时 YAML 变体切换为 `all`、`none` 或 subset list，而不需要修改源码。

mapping 与 fusion 的关系是：scheduler 先决定 fused group，template render 时再根据 `epilogue_nodes` / `prologue_nodes` 选择 tile。证据是 `mlir_gemm_template.py:298-310` 统计 `n_epilogue_node` / `n_prologue_node`，`mlir_template.py:208-278` 的 `gemm_combination_mapping()` 用这些数量计算 SPAD 使用量，`mlir_template.py:609-615` 再选择 autotune 或第一个 tile candidate。注意：如果使用 `external` mapping，`mlir_gemm_template.py:315-327` 直接读取 shape 对应 tile，fusion 维度本身并没有出现在 external mapping key 中。

### 结论

Fusion patterns available:

- `addmm/mm + bias`：GEMM template 原生支持 bias input。
- `GEMM/BMM/Conv + pointwise epilogue`，例如 ReLU / sigmoid / residual add 等。
- `GEMM/BMM + pointwise prologue`，包括 input-side / weight-side elementwise。
- `GEMM/BMM + reduction epilogue`。
- `reduction + reduction`。
- Flash SDPA fused attention template。

Fusion level selection mechanism:

- YAML field: `codegen_compiler_optimization`。
- Syntax: `all`、`none` 或 list subset。
- Existing configs use `all` by default.

Limitations:

- 不能通过现有 YAML 精确选择“只融合某一个具体 pattern”，只能粗粒度选择 fusion family。
- template-template fusion 被显式禁止。
- MaxPool template 作为 epilogue fusion 前置 template 被显式排除。
- 如果 Route 4 只比较 `fusion=all` vs `fusion=none`，不需要源码修改；如果要枚举 attention 内部拆分、LayerNorm+Matmul 等新 pattern，预计需要修改 `mlir_scheduling.py`、对应 template 和 lowering，粗略 150-400 LOC。

## 3. Q2 answer: dataflow capability

### Evidence

主路径 dense systolic array 是 weight-stationary。证据包括：

- config 命名统一为 `systolic_ws_*`，例如 `configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml`。
- TOGSim 默认 core type 是 `WS_MESH`：`TOGSim/src/Common.cc:49-62` 中 `core_type == "ws_mesh"` 映射到 `CoreType::WS_MESH`，没有 `core_type` 字段时也默认填充 `WS_MESH`。
- `PyTorchSimFrontend/mlir/mlir_sdpa_template.py:386-389` 明确写到当前 Systolic Array 使用 weight-stationary approach。
- `PyTorchSimFrontend/mlir/mlir_template.py:229` 的 GEMM mapping heuristic 注释为 `maximize_i_j = 1 # reuse weight`。

代码库中确实存在 STONNE sparse/dense dataflow enum，例如 `TOGSim/extern/stonneCore/include/types.h` 包含 `CNN_DATAFLOW, MK_STA_KN_STR, MK_STR_KN_STA, SPARSE_DENSE_DATAFLOW`，`TOGSim/extern/stonneCore/src/SparseSDMemory.cpp` 也实现 `MK_STA_KN_STR` / `MK_STR_KN_STA` 分支。但这些属于 vendored STONNE sparse path，不是当前 PyTorchSim dense `ws_mesh` systolic path 的 YAML 可选 dataflow。

搜索 `dataflow` / `output_stationary` / `input_stationary` 没有发现 PyTorchSimFrontend 主路径中的 YAML 选择字段。`configs/*.yml` 暴露的是 `core_type`、`codegen_mapping_strategy`、`codegen_compiler_optimization` 等字段，不暴露 `weight_stationary` / `output_stationary` / `input_stationary` 选择。

### 结论

Dataflow strategies implemented:

- PyTorchSim dense systolic main path: weight-stationary / `WS_MESH`。
- STONNE vendored sparse path: has separate sparse dataflow enum, but not directly usable as dense systolic Route 4 axis.

Selection mechanism:

- Dense systolic dataflow: hardcoded/default by config family and TOGSim core type.
- No CLI/YAML field found for `output_stationary` or `input_stationary`.

Default strategy in current `codesign_v1_2x2` and `codesign_v2_2x2` configs:

- weight-stationary / `WS_MESH`。

If Route 4 wants dataflow as a real axis, source modification is required. Bounded scope is roughly:

- Add YAML field and config plumbing in `PyTorchSimFrontend/extension_config.py` and TOGSim config parsing.
- Add dataflow-aware loop order / tile descriptor generation in `mlir_gemm_template.py`, `mlir_bmm_template.py`, and possibly `mlir_sdpa_template.py`。
- Update TOG generation / cycle sampling assumptions if the custom RISC-V/gem5 path encodes WS-specific SA semantics.

Rough cost: 300-800 LOC plus validation tests. This is bounded, but not a no-source-change experiment.

## 4. Q3 answer: silent-error risk assessment

### Evidence

Value-level correctness checks exist:

- `tests/Fusion/test_matmul_activation.py:51-65` compares compiled NPU output with CPU output via `torch.allclose`。
- `tests/Fusion/test_prologue_fusion.py:91-125` and `127-155` cover prologue-style matmul/BMM elementwise fusion and compare with CPU。
- `tests/Fusion/test_attention_fusion.py:195-230` implements an MHA-style fused attention path and compares with CPU。
- `tests/test_sdpa.py:46-67` defines CPU reference SDPA and compares compiled NPU SDPA output with CPU output; defaults include token lengths up to 1024 at `tests/test_sdpa.py:15-20`。
- `tests/Llama/test_llama.py:144-224` builds `LlamaDecoderLayer` with `hidden_size=4096` and checks compiled output against CPU。

There is also functional validation plumbing in the backend:

- `PyTorchSimFrontend/extension_codecache.py:176-209` builds a validation binary when `pytorchsim_functional_mode` is enabled。
- `extension_codecache.py:293-298` runs `FunctionalSimulator(...).run_spike(...)` before timing when functional mode is enabled and not autotune。

However, the current DSE timing harness disables this path:

- `scripts/mapping_dse_minimal.py:798-811` writes a per-run config and forces `pytorchsim_functional_mode = 0` and `pytorchsim_timing_mode = 1`。

Therefore the existing v1/v2 measured cycles are timing-only. They validate that TOGSim can consume the generated TOG and produce counters, but they do not prove that two Route 4 compilation strategies compute numerically equivalent values.

### 结论

Value-level correctness check present: yes。

Coverage:

- Present: matmul/addmm, BMM, Conv, common pointwise/reduction, Fusion tests, SDPA/GQA, Llama decoder/model style paths。
- Not guaranteed: pairwise correctness across Route 4 strategy variants, especially `fusion=off` vs `fusion=on` vs future dataflow/SPAD/DMA schedule variants in timing-only mode。
- TOGSim itself is not a numerical oracle in the current timing flow; the output comparison happens via compiled PyTorch/NPU functional path or Spike validation, not via TOGSim cycle-only execution.

Estimated cost to mitigate for GPT-2 attention block:

- Add a small Route 4 correctness harness that runs each candidate strategy with `pytorchsim_functional_mode=1`, compares compiled NPU output to CPU reference, and records `allclose`, max diff, dtype, shapes, and compiler option tuple。
- Reuse `pipeline_probe.py` style metadata and `tests/test_sdpa.py` / `tests/Fusion/test_attention_fusion.py` style CPU reference。
- Rough cost: 150-300 LOC for a standalone script or test, plus temporary YAML generation for compiler-option variants。

Risk assessment:

Silent-error risk is medium-high if Route 4 uses timing-only TOGSim outputs as the sole evidence. The repository has enough correctness machinery to reduce the risk, but Route 4 must make it an explicit gate. A fusion-positive or dataflow-positive result without functional cross-check would be weak.

## 5. Q4 answer: workload upper bound

### Evidence

`scripts/mapping_dse_minimal.py` can parameterize GPT-2 sequence length:

- CLI has `--seq` with default 128: `mapping_dse_minimal.py:1084-1094`。
- GPT-2 config sets `n_positions=max(args.seq,128)` and `n_ctx=max(args.seq,128)` at `mapping_dse_minimal.py:913-923`。
- Input tensor shape is `[1, args.seq, 768]` at `mapping_dse_minimal.py:925`。
- Metadata records `attention_scores_per_head = [args.seq, args.seq]` and `mlp_intermediate = [1, args.seq, 3072]` at `mapping_dse_minimal.py:1001-1015`。

Thus GPT-2 seq=1024 and seq=2048 are supported by the script interface, but this discovery did not run a full block or TOGSim sweep. That remains unproven at runtime.

Llama single-block support exists as tests, not as the current mapping DSE harness:

- `tests/Llama/test_llama.py:144-224` builds and compiles `LlamaDecoderLayer` with `hidden_size=4096` and `intermediate_size=11008`。
- CLI exposes `--seq_len` at `tests/Llama/test_llama.py:362-368`。
- This is not integrated into `mapping_dse_minimal.py` or the codesign sweep driver.

Single-kernel smoke:

```text
source /home/dyf/.venv/bin/activate
timeout 120s python scripts/pipeline_probe.py \
  --m 1024 --k 4096 --n 4096 \
  --dtype float32 \
  --output-dir /tmp/route4_probe_large \
  --skip-npu
```

Outcome:

- Exit status: `0`。
- Wall time observed by tool polling: about 79 seconds。
- `/tmp/route4_probe_large/02_aten_ops.json` reports `aten.addmm.default: 1` and `aten.relu.default: 1`。
- `/tmp/route4_probe_large/00_metadata.json` reports `fx_capture.ok = true`。
- `npu_compile` was skipped by `--skip-npu`。
- No MLIR / raw TOG / ONNX TOG was produced, as expected for `--skip-npu`。

### 结论

GPT-2 seq=1024 supported by current `mapping_dse_minimal.py` or `pipeline_probe`:

- Interface-level: yes, via `--seq 1024`。
- Timing-runtime proven in this discovery: no。

GPT-2 seq=2048 supported:

- Interface-level: yes, via `--seq 2048`。
- Timing-runtime proven in this discovery: no。

Llama single block supported:

- Test-level: yes, via `tests/Llama/test_llama.py`。
- Integrated Route 4 / mapping DSE harness: no。

Workload upper bound implication:

- GPT-2 seq=1024 gives QK^T per head of `1024*1024*4 = 4 MB`。
- GPT-2 seq=2048 gives QK^T per head of `2048*2048*4 = 16 MB`。
- Llama 3.2-style hidden sizes make MLP intermediate much larger than GPT-2, but current DSE harness does not expose Llama as a first-class workload.

## 6. Route 4 feasibility verdict

结论：`ROUTE_4_FEASIBLE_WITH_MODIFICATION`。

原因：

- Q1 is workable without source modification for coarse fusion axes (`all` / `none` / selected compiler optimization families) using temporary YAML variants。
- Q2 blocks no-source Route 4 for dataflow: dense systolic dataflow is effectively weight-stationary only, and OS/IS are not exposed as current main-path YAML choices。
- Q3 is workable but must become a required gate: existing value-level checks and functional validation exist, but the DSE timing path currently forces `pytorchsim_functional_mode=0`。
- Q4 is partially workable: GPT-2 larger seq is script-reachable, and large addmm front-end capture works, but full large-block timing reachability remains unproven and should not be assumed。

下一步最小动作：

1. Do not start with dataflow. Start with a no-source fusion-only Route 4 pilot。
2. Generate temporary configs for `codegen_compiler_optimization: none` and `all`。
3. Use a single GPT-2 block or single addmm/SDPA kernel with `pytorchsim_functional_mode=1` first, requiring CPU/NPU `allclose` before any timing claim。
4. Only after that, decide whether dataflow source modification is worth the cost。

## 7. Recommended next handoff scope

下一份 handoff 应该要求做 fusion-only pilot，而不是完整 Route 4 实现。它应生成 `codegen_compiler_optimization: none` 和 `all` 两个临时 YAML 变体，先运行一个带 correctness gate 的小 kernel 或 GPT-2 block 路径，再写一份短报告比较生成的 ATen / MLIR / TOG 结构。在 fusion-only 路径证明“策略变体可以生成、可以做数值验证、可以测量且没有 silent wrong-answer 风险”之前，不应进入 dataflow / SPAD / DMA schedule 源码修改。
