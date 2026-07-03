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

`FUSION_CONV_CODESIGN_POSITIVE`

## 7. Next step recommendation

下一步应先修复 Spike `--varch` toolchain，再复跑 mode=1 correctness；若通过，再把 fusion axis 扩展到小型 4-HW sweep。

## 8. Recovery attempt (Phase A + Phase B result)

Phase A 在 30 分钟 timebox 内只尝试配置级和 PATH 级诊断。结果是：`--varch=vlen:256,elen:64` 在 `Simulator/simulator.py` 的 Spike 调用中硬编码，当前 PATH 只有 `/usr/bin/spike`，该 Spike help 只暴露 `--isa`，不支持 `--varch`，configs/scripts 中也没有可用于改写 `--varch` 的 YAML 或 env 字段。因此没有找到不修改源码的 mode=1 correctness 修复。

Phase B 已进入 timing-only fallback：两个 fusion variants 都使用 `pytorchsim_functional_mode=0`、相同 fixed mapping 和 `seed=0` 重复 5 次。`correctness_gate` 记录为 `deferred_due_to_spike_varch_blocker`；`correctness_proxy` 记录为 `structural_diff shows fusion=all has 'fused' keyword and ~10% fewer MLIR ops`。

scope_limitation: correctness gate deferred because pytorchsim_functional_mode=1 path requires a spike version compatible with --varch=vlen:256,elen:64 and no config-level fix was found within 30 minutes

Phase A diagnostic summary: 未找到配置级 workaround；`--varch=vlen:256,elen:64` 在 `Simulator/simulator.py` 中硬编码，当前 `/usr/bin/spike` 不支持 `--varch`，且 PATH 中没有替代 Spike。

## 9. Phase A workload generalization result

Phase A 使用 `HW-A codesign_v1_2x2`、`TILE_M=128 TILE_N=64 TILE_K=64`、`GPT-2 single transformer block prefill seq=128`，在 `pytorchsim_functional_mode=0` 下比较 `fusion=none` 与 `fusion=all`。结果如下：

```json
{
  "cycle_delta_between_variants": 0.01893965380602823,
  "generalizes": false,
  "hw_config": "HW-A codesign_v1_2x2",
  "mapping": "006",
  "note": "fusion generalizes if cycle_delta_between_variants >= 0.10; correctness remains deferred in mode=0",
  "variants": {
    "all": {
      "class": "measured",
      "cycle_delta": 0.07613239305341002,
      "cycles_in_order": [
        411299,
        422711,
        426208,
        441478,
        409296
      ],
      "failures": [],
      "median": 422711.0
    },
    "none": {
      "class": "measured",
      "cycle_delta": 0.06763837972497022,
      "cycles_in_order": [
        430717,
        416023,
        428783,
        445156,
        441708
      ],
      "failures": [],
      "median": 430717.0
    }
  },
  "workload": "gpt2_block_prefill_s128"
}
```

## 10. Phase B HW x fusion matrix

| HW | none cycles | none median | all cycles | all median | fusion_speedup |
| --- | --- | --- | --- | --- | --- |
| N/A | `[]` | `0` | `[]` | `0` | `0` |

完整 JSON：

```json
{}
```

## 11. Co-design analytics

| HW | fusion_speedup | champion_fusion |
| --- | --- | --- |
| N/A | `0` | `N/A` |

```json
{}
```

## 12. New final verdict + implication for next step

`FUSION_ROUTE_CRITICAL_PATH_LIMITED`

fusion axis 是真实存在的，但主要影响 Conv/epilogue 类结构；下一步应把 Route 4 workload 换成 ResNet、MobileNet 或 GEMM-stack，再设计后续 sweep。

## 13. Task 1 structural diff on GPT-2 block

Task 1 在 `HW-A codesign_v1_2x2`、`mapping=006`、`pytorchsim_functional_mode=0` 下各运行一个 `fusion=none` 与 `fusion=all` 的 GPT-2 block subprocess，并只从 fresh root-cause run 目录抽取 MLIR/TOG 结构计数。

结构 run summary：

```json
{
  "none": {
    "cycles_in_order": [
      416834
    ],
    "median": 416834.0,
    "cycle_delta": 0.0,
    "class": "measured",
    "failures": []
  },
  "all": {
    "cycles_in_order": [
      449967
    ],
    "median": 449967.0,
    "cycle_delta": 0.0,
    "class": "measured",
    "failures": []
  }
}
```

结构 diff：

```json
{
  "workload": "gpt2_block_prefill_s128",
  "counts": {
    "none": {
      "mlir_op_count": 5293,
      "tog_node_count": 239,
      "dma_node_count": 79,
      "matmul_like_op_count": 24,
      "fused_pattern_keywords": [
        "epilogue",
        "maximumf"
      ],
      "mlir_file_count": 42,
      "raw_tog_file_count": 22
    },
    "all": {
      "mlir_op_count": 4879,
      "tog_node_count": 227,
      "dma_node_count": 71,
      "matmul_like_op_count": 26,
      "fused_pattern_keywords": [
        "epilogue",
        "maximumf"
      ],
      "mlir_file_count": 36,
      "raw_tog_file_count": 19
    }
  },
  "observation": "GPT-2 structural diff is extracted from one fresh mode=0 timing subprocess per fusion variant under HW-A and mapping 006. Counts aggregate only the emitted MLIR and raw TOG files in the fresh root-cause run directories."
}
```

判读：`mlir_op_count` delta is 0.0782 and keyword sets are none=['epilogue', 'maximumf'], all=['epilogue', 'maximumf']; `fusion=all` changes the generated structure, so the weak GPT-2 timing result is more likely critical-path limited.

## 14. Task 2 Conv probe result

Conv probe 完成，`cycle_delta_between_variants` = `5.841693975296193`，`generalizes_to_conv` = `True`。

```json
{
  "workload": "conv3x3_probe",
  "variants": {
    "none": {
      "cycles_in_order": [
        189987,
        189882,
        190231
      ],
      "median": 189987.0,
      "cycle_delta": 0.0018369677925331733,
      "class": "measured",
      "failures": []
    },
    "all": {
      "cycles_in_order": [
        27660,
        27843,
        27769
      ],
      "median": 27769.0,
      "cycle_delta": 0.006590082466059275,
      "class": "measured",
      "failures": []
    }
  },
  "cycle_delta_between_variants": 5.841693975296193,
  "generalizes_to_conv": true
}
```

## 15. Root cause classification and recommended next step

`FUSION_ROUTE_CRITICAL_PATH_LIMITED`

root cause classification: `structural_active`

fusion axis 是真实存在的，但主要影响 Conv/epilogue 类结构；下一步应把 Route 4 workload 换成 ResNet、MobileNet 或 GEMM-stack，再设计后续 sweep。

## 16. Conv sweep 4x2 matrix table

本节使用 `conv3x3_probe`，即 Conv 3x3 `[1, 64, 56, 56] -> [1, 64, 56, 56]`，在 `codesign_v1_2x2` 的 `HW-A/B/C/D` 上分别比较 `codegen_compiler_optimization=none` 与 `all`。所有 run 均为 `pytorchsim_functional_mode=0`，每个 `(HW, fusion)` cell 使用 5 次 independent subprocess，`seed=0`，并保留原始 `cycles_in_order`。

| HW | none cycles | none median | all cycles | all median | fusion_speedup |
| --- | --- | --- | --- | --- | --- |
| HW-A | `[189837, 189861, 189927, 189322, 189816]` | `189837.0` | `[27934, 27315, 27181, 27533, 27708]` | `27533.0` | `6.894889768641267` |
| HW-B | `[240564, 240914, 240692, 241137, 240923]` | `240914.0` | `[49575, 50066, 49760, 49877, 50141]` | `49877.0` | `4.830162199009563` |
| HW-C | `[163105, 162017, 164981, 165155, 163275]` | `163275.0` | `[25310, 26083, 26459, 26074, 25620]` | `26074.0` | `6.261985119275907` |
| HW-D | `[298246, 292531, 291110, 294609, 299029]` | `294609.0` | `[56020, 55971, 56088, 56080, 56097]` | `56080.0` | `5.253370185449358` |

完整 matrix JSON：

```json
{
  "workload": "conv3x3_probe",
  "hw_config_set": "codesign_v1_2x2",
  "mapping": "harness_default",
  "pytorchsim_functional_mode": 0,
  "repeats_per_cell": 5,
  "seed": 0,
  "matrix": {
    "HW-A": {
      "none": {
        "cycles_in_order": [
          189837,
          189861,
          189927,
          189322,
          189816
        ],
        "median": 189837.0,
        "cycle_delta": 0.003186944589305562,
        "class": "measured",
        "failures": []
      },
      "all": {
        "cycles_in_order": [
          27934,
          27315,
          27181,
          27533,
          27708
        ],
        "median": 27533.0,
        "cycle_delta": 0.02734899938255911,
        "class": "measured",
        "failures": []
      },
      "fusion_speedup": 6.894889768641267
    },
    "HW-B": {
      "none": {
        "cycles_in_order": [
          240564,
          240914,
          240692,
          241137,
          240923
        ],
        "median": 240914.0,
        "cycle_delta": 0.002378442099670422,
        "class": "measured",
        "failures": []
      },
      "all": {
        "cycles_in_order": [
          49575,
          50066,
          49760,
          49877,
          50141
        ],
        "median": 49877.0,
        "cycle_delta": 0.011347915873047697,
        "class": "measured",
        "failures": []
      },
      "fusion_speedup": 4.830162199009563
    },
    "HW-C": {
      "none": {
        "cycles_in_order": [
          163105,
          162017,
          164981,
          165155,
          163275
        ],
        "median": 163275.0,
        "cycle_delta": 0.019219108865411116,
        "class": "measured",
        "failures": []
      },
      "all": {
        "cycles_in_order": [
          25310,
          26083,
          26459,
          26074,
          25620
        ],
        "median": 26074.0,
        "cycle_delta": 0.04406688655365498,
        "class": "measured",
        "failures": []
      },
      "fusion_speedup": 6.261985119275907
    },
    "HW-D": {
      "none": {
        "cycles_in_order": [
          298246,
          292531,
          291110,
          294609,
          299029
        ],
        "median": 294609.0,
        "cycle_delta": 0.026879694781897362,
        "class": "measured",
        "failures": []
      },
      "all": {
        "cycles_in_order": [
          56020,
          55971,
          56088,
          56080,
          56097
        ],
        "median": 56080.0,
        "cycle_delta": 0.0022467902995720397,
        "class": "measured",
        "failures": []
      },
      "fusion_speedup": 5.253370185449358
    }
  }
}
```

## 17. Co-design gate analytics

| HW | champion_fusion | fusion_speedup | per_hw_ratio |
| --- | --- | --- | --- |
| HW-A | `all` | `6.894889768641267` | `1.0` |
| HW-B | `all` | `4.830162199009563` | `1.0` |
| HW-C | `all` | `6.261985119275907` | `1.0` |
| HW-D | `all` | `5.253370185449358` | `1.0` |

关键 gate 结果：

- `hw_x_fusion_interaction_significant`: `True`
- `fusion_speedup_range.relative_range`: `0.427465473117048`
- `gate1_analog_champion_migrates`: `False`
- `gate2a_ratio_mean`: `1.0`
- `gate2b_ratio_fusion`: `0.522766004370752`
- `gate2b_passed`: `True`

完整 analysis JSON：

```json
{
  "champion_fusion_per_hw": {
    "HW-A": "all",
    "HW-B": "all",
    "HW-C": "all",
    "HW-D": "all"
  },
  "same_champion_across_all_hw": true,
  "fusion_speedup_per_hw": {
    "HW-A": 6.894889768641267,
    "HW-B": 4.830162199009563,
    "HW-C": 6.261985119275907,
    "HW-D": 5.253370185449358
  },
  "fusion_speedup_range": {
    "max": 6.894889768641267,
    "min": 4.830162199009563,
    "relative_range": 0.427465473117048
  },
  "hw_x_fusion_interaction_significant": true,
  "gate1_analog_champion_migrates": false,
  "global_champion_fusion": "all",
  "mean_cycles_by_fusion": {
    "none": 222158.75,
    "all": 39891.0
  },
  "per_hw_ratio": {
    "HW-A": 1.0,
    "HW-B": 1.0,
    "HW-C": 1.0,
    "HW-D": 1.0
  },
  "mean_per_hw": 39891.0,
  "mean_single_fusion": 39891.0,
  "gate2a_ratio_mean": 1.0,
  "gate2a_num_hw_meeting_threshold": 0,
  "gate2a_passed": false,
  "gate2b_ratio_fusion": 0.522766004370752,
  "gate2b_passed": true,
  "blocked_cells": []
}
```

## 18. Final co-design verdict + implication for next step

`FUSION_CONV_CODESIGN_POSITIVE`

Conv workload 上 fusion axis 不只是固定的软件优化；其收益会随 HW 配置变化，因此可以作为 Route 4 后续 HW/SW co-design sweep 的正向证据。

当前可编译 compiler-modification directions landscape：

```json
{
  "fusion": "proven strong on Conv (6.84x per Conv probe); proven weak on GPT-2 (1.9% - critical-path limited)",
  "dataflow": "BLOCKED without source modification (per discovery)",
  "SPAD partition": "NOT INVESTIGATED",
  "DMA schedule": "NOT INVESTIGATED"
}
```
