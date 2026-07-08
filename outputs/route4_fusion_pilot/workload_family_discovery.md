# Workload family discovery

Phase A time-boxed discovery checked the existing `route4_fusion_pilot.py` CLI and then used a temporary no-source-modification harness for single-kernel smoke tests.

## Grep evidence

```bash
grep -rn "workload.*conv\|conv3x3\|conv1x1\|depthwise\|pointwise" scripts/ 2>/dev/null | head -30
```
Status: `returncode=0`

```text
scripts/route4_fusion_pilot.py:325:def run_conv3x3_probe_child(args: argparse.Namespace) -> int:
scripts/route4_fusion_pilot.py:342:        "workload": "conv3x3_probe",
scripts/route4_fusion_pilot.py:384:    if args.workload == "conv3x3_probe":
scripts/route4_fusion_pilot.py:385:        return run_conv3x3_probe_child(args)
scripts/route4_fusion_pilot.py:393:        if args.workload == "conv3x3_probe":
scripts/route4_fusion_pilot.py:394:            return run_conv3x3_probe_child(args)
scripts/route4_fusion_pilot.py:1894:                    workload='conv3x3_probe',
scripts/route4_fusion_pilot.py:2342:- Workload: Conv 3x3 `[1, 64, 56, 56] -> [1, 64, 56, 56]`，使用既有 `conv3x3_probe`。
scripts/route4_fusion_pilot.py:2595:                    workload="conv3x3_probe",
scripts/route4_fusion_pilot.py:2623:        "workload": "conv3x3_probe",
scripts/route4_fusion_pilot.py:2740:            workload="conv3x3_probe",
scripts/route4_fusion_pilot.py:2754:        "workload": "conv3x3_probe",
scripts/route4_fusion_pilot.py:3078:本节使用 `conv3x3_probe`，即 Conv 3x3 `[1, 64, 56, 56] -> [1, 64, 56, 56]`，在 `codesign_v1_2x2` 的 `HW-A/B/C/D` 上分别比较 `codegen_compiler_optimization=none` 与 `all`。所有 run 均为 `pytorchsim_functional_mode=0`，每个 `(HW, fusion)` cell 使用 5 次 independent subprocess，`seed=0`，并保留原始 `cycles_in_order`。
scripts/route4_fusion_pilot.py:3144:<p>本节使用 <code>conv3x3_probe</code>，即 Conv 3x3 <code>[1, 64, 56, 56] -&gt; [1, 64, 56, 56]</code>，在 <code>codesign_v1_2x2</code> 的 <code>HW-A/B/C/D</code> 上分别比较 <code>codegen_compiler_optimization=none</code> 与 <code>all</code>。所有 run 均为 <code>pytorchsim_functional_mode=0</code>，每个 <code>(HW, fusion)</code> cell 使用 5 次 independent subprocess，<code>seed=0</code>。</p>
scripts/route4_fusion_pilot.py:3192:                workload="conv3x3_probe",
scripts/route4_fusion_pilot.py:3204:        "workload": "conv3x3_probe",
scripts/route4_fusion_pilot.py:3219:            "workload": "conv3x3_probe",
scripts/route4_fusion_pilot.py:3246:    parser.add_argument("--workload", choices=["gpt2_block_prefill_s128", "addmm_relu_128", "conv3x3_probe"], help=argparse.SUPPRESS)
```

```bash
grep -rn "workload_name\|workload_type\|--workload" scripts/route4_fusion_pilot*.py 2>/dev/null | head
```
Status: `returncode=0`

```text
430:        "--workload",
3246:    parser.add_argument("--workload", choices=["gpt2_block_prefill_s128", "addmm_relu_128", "conv3x3_probe"], help=argparse.SUPPRESS)
```

## Candidate table

| Candidate | Reachable | Selected | Smoke cycles | Tile schema / reason |
|---|---:|---:|---:|---|
| `conv3x3_large` | `True` | `True` | 6433 | conv2d_1_128_128_3_3_28_28 -> 7 Conv tile params |
| `conv1x1` | `True` | `True` | 1019 | conv2d_1_256_512_1_1_14_14 -> 7 Conv tile params |
| `depthwise_conv3x3` | `True` | `False` | 189982 | same conv2d key shape as 3x3, but grouped lowering emits multiple kernels; not selected due 2-additional-workload cap and less clean tile schema |
| `resnet50_stage1_residual` | `False` | `False` | - | composite multi-conv graph; Phase A allows only single-kernel smoke tests |
| `mobilenetv2_inverted_residual` | `False` | `False` | - | composite PW+DW+PW graph; Phase A allows only single-kernel smoke tests |

## Tile-span limitation

The two selected workloads are smaller than the inherited Conv3x3 `[1,64,56,56]` case. Their largest valid single-batch Conv tiles do not exceed the smallest-SPAD double-buffer budget, so Phase B keeps 8 ordered tiles but records that no selected additional workload reaches the 85-110% fit-boundary target.
