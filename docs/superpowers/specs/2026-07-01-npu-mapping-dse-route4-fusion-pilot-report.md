# Route 4 fusion-only pilot report

## 1. 目标与非目标

本实验只验证 fusion axis 是否能在一个小 workload 上被正确生成、数值验证并独立计时。非目标包括：不做 4-HW sweep，不做 10-mapping tile sweep，不触碰 dataflow，不修改 PyTorchSim / TOGSim / gem5 / ramulator2 / spike 源码。

## 2. Method

- workload: `addmm_relu_128`
- HW config baseline: `configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml`
- fusion variants: `codegen_compiler_optimization=none` 和 `codegen_compiler_optimization=all`
- fixed mapping: `TILE_M=128`, `TILE_N=64`, `TILE_K=64`
- correctness gate: 原 mode=1 correctness 尝试保留为 evidence；本次 recovery 的 timing fallback 使用 `pytorchsim_functional_mode=0`
- timing repeats: 5 per variant, `seed=0`
- output root: `outputs/route4_fusion_pilot`

## 3. Correctness results

| variant | allclose_passed | max_abs_diff | max_rel_diff | error |
| --- | --- | --- | --- | --- |
| none | `False` | `None` | `None` | `spike: unrecognized option --varch=vlen:256,elen:64` |
| all | `False` | `None` | `None` | `spike: unrecognized option --varch=vlen:256,elen:64` |

## 4. Timing results

| variant | cycles_in_order | median | cycle_delta | class |
| --- | --- | --- | --- | --- |
| none | `[2180, 2346, 2181, 2195, 2183]` | `2183.0` | `0.076042143838754` | `measured` |
| all | `[1119, 1118, 1113, 1125, 1118]` | `1118.0` | `0.01073345259391771` | `measured` |

`cycle_delta_between_variants = 0.952594`。如果该值低于 0.10，则 fusion axis 的效果处在或低于当前噪声地板，不能作为强结论。

## 5. Structural diff

```json
{
  "counts": {
    "all": {
      "dma_node_count": 4,
      "fused_pattern_keywords": [
        "epilogue",
        "fused",
        "maximumf"
      ],
      "matmul_like_op_count": 4,
      "mlir_file_count": 3,
      "mlir_op_count": 1268,
      "onnx_tog_file_count": 1,
      "raw_tog_file_count": 1,
      "tog_node_count": 19
    },
    "none": {
      "dma_node_count": 6,
      "fused_pattern_keywords": [
        "epilogue",
        "maximumf"
      ],
      "matmul_like_op_count": 4,
      "mlir_file_count": 6,
      "mlir_op_count": 1398,
      "onnx_tog_file_count": 2,
      "raw_tog_file_count": 2,
      "tog_node_count": 23
    }
  },
  "observation": "本次 structural diff 只做浅层计数，不做语义等价证明。若 fusion=all 的 MLIR/TOG 出现 epilogue 或 relu/maximumf 等关键字，而 fusion=none 没有对应变化，则说明 fusion axis 已经影响编译产物；否则该 workload 下 fusion 结构差异不明显。"
}
```

## 6. Verdict

`FUSION_PILOT_POSITIVE_MODE0_DEFERRED_CORRECTNESS`

## 7. Next step recommendation

下一步应先修复 Spike `--varch` toolchain，再复跑 mode=1 correctness；若通过，再把 fusion axis 扩展到小型 4-HW sweep。

## 8. Recovery attempt (Phase A + Phase B result)

Phase A 在 30 分钟 timebox 内只尝试配置级和 PATH 级诊断。结果是：`--varch=vlen:256,elen:64` 在 `Simulator/simulator.py` 的 Spike 调用中硬编码，当前 PATH 只有 `/usr/bin/spike`，该 Spike help 只暴露 `--isa`，不支持 `--varch`，configs/scripts 中也没有可用于改写 `--varch` 的 YAML 或 env 字段。因此没有找到不修改源码的 mode=1 correctness 修复。

Phase B 已进入 timing-only fallback：两个 fusion variants 都使用 `pytorchsim_functional_mode=0`、相同 fixed mapping 和 `seed=0` 重复 5 次。`correctness_gate` 记录为 `deferred_due_to_spike_varch_blocker`；`correctness_proxy` 记录为 `structural_diff shows fusion=all has 'fused' keyword and ~10% fewer MLIR ops`。

scope_limitation: correctness gate deferred because pytorchsim_functional_mode=1 path requires a spike version compatible with --varch=vlen:256,elen:64 and no config-level fix was found within 30 minutes

Phase A diagnostic summary: 未找到配置级 workaround；`--varch=vlen:256,elen:64` 在 `Simulator/simulator.py` 中硬编码，当前 `/usr/bin/spike` 不支持 `--varch`，且 PATH 中没有替代 Spike。
