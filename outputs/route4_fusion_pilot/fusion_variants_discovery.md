# Fusion variants discovery

本轮目标是检查 `codegen_compiler_optimization` 除 `none/all` 外是否存在可作为第三个 fusion family 的稳定取值。

## Command

```bash
grep -rn "codegen_compiler_optimization\|compiler_optimization" PyTorchSimFrontend/mlir/ TOGSim/ configs/ scripts/ 2>/dev/null | head -30
```

Status: `returncode=0`

```text
configs/heterogeneous_c2_simple_noc.yml:35:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_booksim_tpuv2.yml:24:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_booksim_tpuv3.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_simple_noc_tpuv2.yml:27:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_simple_noc_tpuv3_half.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_simple_noc_tpuv3_timing_only.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_simple_noc_tpuv4.yml:30:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_booksim_tpuv3.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_booksim_tpuv3_bw_quarter.yml:36:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_chiplet_tpuv3.yml:30:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_chiplet_tpuv3_xnuma.yml:29:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_simple_noc_tpuv2.yml:27:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_simple_noc_tpuv3.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_simple_noc_tpuv3_ils.yml:31:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_simple_noc_tpuv3_partition.yml:32:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_simple_noc_tpuv4.yml:30:codegen_compiler_optimization: all
configs/systolic_ws_8x8_c1_booksim.yml:25:codegen_compiler_optimization: all
configs/systolic_ws_8x8_c1_simple_noc.yml:26:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_simple_noc_tpuv3_probe_timing_only.yml:28:codegen_compiler_optimization: all
Binary file scripts/__pycache__/route4_fusion_pilot.cpython-311.pyc matches
scripts/route4_fusion_pilot.py:71:        patched["codegen_compiler_optimization"] = variant
scripts/route4_fusion_pilot.py:80:        "only_changed_field": "codegen_compiler_optimization",
scripts/route4_fusion_pilot.py:840:- fusion variants: `codegen_compiler_optimization=none` 和 `codegen_compiler_optimization=all`
scripts/route4_fusion_pilot.py:963:- fusion variants: `codegen_compiler_optimization=none` 和 `codegen_compiler_optimization=all`
scripts/route4_fusion_pilot.py:1297:            patched["codegen_compiler_optimization"] = variant
scripts/route4_fusion_pilot.py:2043:        'grep -rn "codegen_compiler_optimization\\|compiler_optimization" PyTorchSimFrontend/mlir/ TOGSim/ configs/ scripts/ 2>/dev/null | head -30',
scripts/route4_fusion_pilot.py:2045:        "rg -n \"codegen_compiler_optimization|compiler_optimization\" PyTorchSimFrontend/mlir TOGSim configs scripts -g '!TOGSim/extern/**' | head -40",
scripts/route4_fusion_pilot.py:2057:        "本轮目标是检查 `codegen_compiler_optimization` 除 `none/all` 外是否存在可作为第三个 fusion family 的稳定取值。",
scripts/route4_fusion_pilot.py:2084:            "`PyTorchSimFrontend/extension_config.py` accepts `codegen_compiler_optimization` as `all`, `none`, or a list drawn from:",
```

## Command

```bash
grep -rn '"none"\|"all"\|"fusion_only"\|"epilogue"\|"elementwise"' PyTorchSimFrontend/mlir/ TOGSim/ configs/ 2>/dev/null | head -20
```

Status: `returncode=141`

```text
TOGSim/extern/booksim/src/booksim_config.cpp:55:  AddStrField( "routing_function", "none" );
TOGSim/extern/booksim/src/booksim_config.cpp:185:  AddStrField( "priority", "none" );  // message priorities
TOGSim/extern/booksim/src/trafficmanager.cpp:87:    } else if ( priority == "none" ) {
TOGSim/extern/booksim/src/vc.cpp:66:  } else if ( priority == "none" ) {
TOGSim/extern/onnx/docs/Changelog.md:14554:  If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
TOGSim/extern/onnx/docs/Changelog.md:14563:      // negative log likelihood loss, "none" reduction
TOGSim/extern/onnx/docs/Changelog.md:16794:  If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
TOGSim/extern/onnx/docs/Changelog.md:16810:      // negative log likelihood loss, "none" reduction
TOGSim/extern/onnx/docs/Changelog.md:20573:  In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
TOGSim/extern/onnx/docs/Changelog.md:20699:  In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
TOGSim/extern/onnx/docs/Operators.md:12935:  If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
TOGSim/extern/onnx/docs/Operators.md:12951:      // negative log likelihood loss, "none" reduction
TOGSim/extern/onnx/docs/Operators.md:19710:  In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
TOGSim/extern/onnx/docs/Operators.md:19946:  In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
TOGSim/extern/onnx/onnx/defs/math/defs.cc:2066:If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
TOGSim/extern/onnx/onnx/defs/math/defs.cc:2082:    // negative log likelihood loss, "none" reduction
TOGSim/extern/onnx/onnx/defs/math/defs.cc:2165:      if (reduction_attr == "none") {
TOGSim/extern/onnx/onnx/defs/math/defs.cc:2178:      if (reduction_attr == "none") {
TOGSim/extern/onnx/onnx/defs/math/defs.cc:2229:    if (reduction_attr == "none") {
TOGSim/extern/onnx/onnx/defs/math/defs.cc:2342:            if (getAttribute(ctx, "reduction", "mean") == "none") {
```

## Command

```bash
rg -n "codegen_compiler_optimization|compiler_optimization" PyTorchSimFrontend/mlir TOGSim configs scripts -g '!TOGSim/extern/**' | head -40
```

Status: `returncode=0`

```text
scripts/route4_fusion_pilot.py:71:        patched["codegen_compiler_optimization"] = variant
scripts/route4_fusion_pilot.py:80:        "only_changed_field": "codegen_compiler_optimization",
scripts/route4_fusion_pilot.py:840:- fusion variants: `codegen_compiler_optimization=none` 和 `codegen_compiler_optimization=all`
scripts/route4_fusion_pilot.py:963:- fusion variants: `codegen_compiler_optimization=none` 和 `codegen_compiler_optimization=all`
scripts/route4_fusion_pilot.py:1297:            patched["codegen_compiler_optimization"] = variant
scripts/route4_fusion_pilot.py:2043:        'grep -rn "codegen_compiler_optimization\\|compiler_optimization" PyTorchSimFrontend/mlir/ TOGSim/ configs/ scripts/ 2>/dev/null | head -30',
scripts/route4_fusion_pilot.py:2045:        "rg -n \"codegen_compiler_optimization|compiler_optimization\" PyTorchSimFrontend/mlir TOGSim configs scripts -g '!TOGSim/extern/**' | head -40",
scripts/route4_fusion_pilot.py:2057:        "本轮目标是检查 `codegen_compiler_optimization` 除 `none/all` 外是否存在可作为第三个 fusion family 的稳定取值。",
scripts/route4_fusion_pilot.py:2084:            "`PyTorchSimFrontend/extension_config.py` accepts `codegen_compiler_optimization` as `all`, `none`, or a list drawn from:",
scripts/route4_fusion_pilot.py:2127:            patched["codegen_compiler_optimization"] = opt_value
scripts/route4_fusion_pilot.py:2133:                "codegen_compiler_optimization": opt_value,
scripts/route4_fusion_pilot.py:3078:本节使用 `conv3x3_probe`，即 Conv 3x3 `[1, 64, 56, 56] -> [1, 64, 56, 56]`，在 `codesign_v1_2x2` 的 `HW-A/B/C/D` 上分别比较 `codegen_compiler_optimization=none` 与 `all`。所有 run 均为 `pytorchsim_functional_mode=0`，每个 `(HW, fusion)` cell 使用 5 次 independent subprocess，`seed=0`，并保留原始 `cycles_in_order`。
scripts/route4_fusion_pilot.py:3144:<p>本节使用 <code>conv3x3_probe</code>，即 Conv 3x3 <code>[1, 64, 56, 56] -&gt; [1, 64, 56, 56]</code>，在 <code>codesign_v1_2x2</code> 的 <code>HW-A/B/C/D</code> 上分别比较 <code>codegen_compiler_optimization=none</code> 与 <code>all</code>。所有 run 均为 <code>pytorchsim_functional_mode=0</code>，每个 <code>(HW, fusion)</code> cell 使用 5 次 independent subprocess，<code>seed=0</code>。</p>
configs/systolic_ws_128x128_c1_simple_noc_tpuv3_probe_timing_only.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_8x8_c1_simple_noc.yml:26:codegen_compiler_optimization: all
configs/systolic_ws_8x8_c1_booksim.yml:25:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_simple_noc_tpuv4.yml:30:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_simple_noc_tpuv3_partition.yml:32:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_simple_noc_tpuv3_ils.yml:31:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_simple_noc_tpuv3.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_simple_noc_tpuv2.yml:27:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_chiplet_tpuv3_xnuma.yml:29:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_chiplet_tpuv3.yml:30:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_booksim_tpuv3_bw_quarter.yml:36:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c2_booksim_tpuv3.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_simple_noc_tpuv4.yml:30:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_simple_noc_tpuv3_timing_only.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_simple_noc_tpuv3_half.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_simple_noc_tpuv2.yml:27:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_booksim_tpuv3.yml:28:codegen_compiler_optimization: all
configs/systolic_ws_128x128_c1_booksim_tpuv2.yml:24:codegen_compiler_optimization: all
configs/heterogeneous_c2_simple_noc.yml:35:codegen_compiler_optimization: all
```

## Command

```bash
sed -n '90,125p' PyTorchSimFrontend/extension_config.py
```

Status: `returncode=0`

```text
if name == "codegen_autotune_wall_slack_sec":
        v = float(config_yaml.get("codegen_autotune_wall_slack_sec", 15))
        return max(1.0, v)

    # Compiler Optimization
    if name == "codegen_compiler_optimization":
        opt_level = config_yaml["codegen_compiler_optimization"]
        valid_opts = {
            "fusion",
            "reduction_epilogue",
            "reduction_reduction",
            "prologue",
            "single_batch_conv",
            "multi_tile_conv",
            "subtile"
        }
        if opt_level == "all" or opt_level == "none":
            pass
        elif isinstance(opt_level, list):
            # Check if provided list contains only valid options
            invalids = set(opt_level) - valid_opts
            assert not invalids, f"Invalid optimization options found: {invalids}"
        else:
            assert False, "Invalid format: Must be 'all', none, or a list of options."
        return opt_level

    # Advanced fusion options
    is_opt_enabled = lambda key: (__getattr__("codegen_compiler_optimization") == "all") or \
                                 (isinstance(__getattr__("codegen_compiler_optimization"), list) and \
                                  key in __getattr__("codegen_compiler_optimization"))
    if name == "CONFIG_FUSION":
        return is_opt_enabled("fusion")
    if name == "CONFIG_FUSION_REDUCTION_EPILOGUE":
        return is_opt_enabled("reduction_epilogue") # Fixed typo here as well
    if name == "CONFIG_FUSION_REDUCTION_REDUCTION":
        return is_opt_enabled("reduction_reduction")
```

## Decision

`PyTorchSimFrontend/extension_config.py` accepts `codegen_compiler_optimization` as `all`, `none`, or a list drawn from:

- `fusion`
- `reduction_epilogue`
- `reduction_reduction`
- `prologue`
- `single_batch_conv`
- `multi_tile_conv`
- `subtile`

因此本轮使用第三个 meaningful variant：`fusion = ["fusion"]`。它只打开 `CONFIG_FUSION`，不同于 `all` 同时打开所有 compiler optimization。最终 sweep 的 fusion variants 为 `none`、`fusion`、`all`，即 `N=3`。
