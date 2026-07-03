# Route 4 fusion-only pilot report

## 1. 目标与非目标

本实验只验证 fusion axis 是否能在一个小 workload 上被正确生成、数值验证并独立计时。非目标包括：不做 4-HW sweep，不做 10-mapping tile sweep，不触碰 dataflow，不修改 PyTorchSim / TOGSim / gem5 / ramulator2 源码。

## 2. Method

- workload: `addmm_relu_128`
- HW config baseline: `configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml`
- fusion variants: `codegen_compiler_optimization=none` 和 `codegen_compiler_optimization=all`
- fixed mapping: `TILE_M=128`, `TILE_N=64`, `TILE_K=64`
- correctness gate: `pytorchsim_functional_mode=1`, `rtol=0.001`, `atol=0.0001`
- timing repeats: 5 per variant
- output root: `outputs/route4_fusion_pilot`

## 3. Correctness results

| variant | allclose_passed | max_abs_diff | max_rel_diff | error |
| --- | --- | --- | --- | --- |
| none | `False` | `None` | `None` | `spike: unrecognized option --varch=vlen:256,elen:64` |
| all | `False` | `None` | `None` | `spike: unrecognized option --varch=vlen:256,elen:64` |

## 4. Timing results

| variant | cycles_in_order | median | cycle_delta | class |
| --- | --- | --- | --- | --- |
| none | `[]` | `0` | `0` | `blocked` |
| all | `[]` | `0` | `0` | `blocked` |

`cycle_delta_between_variants = null`。如果该值低于 0.10，则 fusion axis 的效果处在或低于当前噪声地板，不能作为强结论。

## 5. Structural diff

```json
{
  "counts": {
    "none": {
      "mlir_file_count": 6,
      "onnx_tog_file_count": 2,
      "raw_tog_file_count": 2,
      "mlir_op_count": 1398,
      "tog_node_count": 23,
      "dma_node_count": 6,
      "matmul_like_op_count": 4,
      "fused_pattern_keywords": [
        "epilogue",
        "maximumf"
      ]
    },
    "all": {
      "mlir_file_count": 3,
      "onnx_tog_file_count": 1,
      "raw_tog_file_count": 1,
      "mlir_op_count": 1268,
      "tog_node_count": 19,
      "dma_node_count": 4,
      "matmul_like_op_count": 4,
      "fused_pattern_keywords": [
        "epilogue",
        "fused",
        "maximumf"
      ]
    }
  },
  "observation": "本次 structural diff 只做浅层计数，不做语义等价证明。若 fusion=all 的 MLIR/TOG 出现 epilogue 或 relu/maximumf 等关键字，而 fusion=none 没有对应变化，则说明 fusion axis 已经影响编译产物；否则该 workload 下 fusion 结构差异不明显。"
}
```

## 6. Verdict

`FUSION_PILOT_BLOCKED`

## 7. Next step recommendation

下一步应优先修复报告中记录的 correctness 或 timing 阻塞点，暂时不要扩展 HW sweep 或 dataflow 轴。
