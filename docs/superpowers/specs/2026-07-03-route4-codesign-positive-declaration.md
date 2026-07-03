# Route 4 HW/SW co-design POSITIVE 声明

## 1. 结论

在 Conv 3x3 workload 上，fusion 轴与 HW 配置存在 measured interaction：`fusion_speedup_range.relative_range=42.7%`，构成本项目首次 measured HW/SW co-design POSITIVE 证据。该结论来自 commit `79e6259` 的 Conv 4-HW x 2-fusion sweep，verdict 为 `FUSION_CONV_CODESIGN_POSITIVE`。

## 2. 证据链

完整证据链如下：

1. `92623de`：v1 baseline 在 GPT-2 single block prefill seq=128、tiling-only 2x2 SPAD/BW 空间上得到 NEGATIVE。
2. `09b9ac3`：在 10% Gate-1 threshold 下重算 v1，NEGATIVE 保持。
3. `42e57b6`：Route 4 discovery 证明 fusion 是当前 PyTorchSim 中可无源码切换的 compiler axis；dataflow 不是无源码配置轴。
4. `59f44ba`：fusion pilot recovery 后，addmm+relu 128 出现约 `2x` timing 差异，说明 `codegen_compiler_optimization` 轴确实能影响生成结构和周期。
5. `3489f94`：GPT-2 block seq=128 上 `fusion=all` vs `fusion=none` 只有 `1.9%` 差异，判定为 `FUSION_NOT_GENERALIZABLE_ON_GPT2`。
6. `ce5f0e3`：root-cause investigation 显示 GPT-2 中 fusion 结构上生效但不在 critical path；Conv 3x3 single-cell probe 显示 `fusion=none` median `189987`，`fusion=all` median `27769`，约 `6.84x`。
7. `79e6259`：Conv 4-HW x 2-fusion sweep 显示 fusion speedup 随 HW 显著变化，`relative_range=42.7%`，`gate2b_ratio_fusion=0.523`，得到 `FUSION_CONV_CODESIGN_POSITIVE`。

## 3. Measured 数据摘要

Conv workload 为 `conv3x3_probe`：Conv 3x3 `[1, 64, 56, 56] -> [1, 64, 56, 56]`，`pytorchsim_functional_mode=0`，每个 `(HW, fusion)` cell 5 repeats，`seed=0`，mapping 为 harness default。

| HW | fusion=none median cycles | fusion=all median cycles | fusion_speedup |
| --- | ---: | ---: | ---: |
| HW-A | 189837 | 27533 | 6.895 |
| HW-B | 240914 | 49877 | 4.830 |
| HW-C | 163275 | 26074 | 6.262 |
| HW-D | 294609 | 56080 | 5.253 |

Fusion speedup per HW：

| HW | fusion_speedup |
| --- | ---: |
| HW-A | 6.894889768641267 |
| HW-B | 4.830162199009563 |
| HW-C | 6.261985119275907 |
| HW-D | 5.253370185449358 |

关键 analytics：

- `fusion_speedup_range.max = 6.894889768641267`
- `fusion_speedup_range.min = 4.830162199009563`
- `fusion_speedup_range.relative_range = 0.427465473117048`
- `hw_x_fusion_interaction_significant = true`
- `gate2b_ratio_fusion = 0.522766004370752`
- `gate2b_passed = true`
- `champion_fusion_per_hw = all` for `HW-A/B/C/D`

## 4. Scope limitations

本 POSITIVE 声明必须和以下限制一起引用：

- Single workload：目前 POSITIVE 只在 Conv 3x3 kernel-level workload 上成立，不代表完整 ResNet、MobileNet、GPT-2 或所有 DNN。
- Single mapping：本次 Conv 4-HW sweep 使用 harness default mapping，没有做 multi-mapping ranking，因此不能声称正式 Gate-3 ranking interaction。
- Mode=0 timing：结果是 `pytorchsim_functional_mode=0` timing evidence；mode=1 correctness 因 Spike `--varch` source blocker 仍 deferred。
- Four HW configs only：HW 空间只有 `codesign_v1_2x2` 的 `HW-A/B/C/D` 四个 corner，没有覆盖内部连续空间。
- Fusion axis only：POSITIVE 只证明 fusion axis 与 HW 配置存在 interaction；dataflow / SPAD partition / DMA schedule 尚未 demonstrated POSITIVE。

## 5. 与 v1 NEGATIVE 的关系

v1 NEGATIVE 仍然有效。它的范围是 GPT-2 single block prefill seq=128、tiling-only search space、2x2 SPAD/BW HW corners。该 NEGATIVE 说明：在这个 workload 和 tiling axis 上，per-HW mapping tuning 没有形成 claim-bearing co-design value。

本 POSITIVE 是另一个 axis 和另一个 workload：fusion axis + Conv 3x3。二者不冲突。它们共同说明：HW/SW co-design 价值不是 universal property，而是 workload-dependent 和 axis-dependent。GPT-2 tiling-only 可以是 NEGATIVE，同时 Conv fusion x HW 可以是 POSITIVE。

## 6. 引用政策

任何外部引用本 POSITIVE 时，必须同时包含第 4 节的四类范围限制：single workload、single mapping、mode=0 timing / correctness deferred、four HW configs only，以及 fusion axis only。推荐引用格式：

> 在 PyTorchSim 的 Conv 3x3 kernel-level workload 上，Route 4 fusion axis 显示 measured HW/SW co-design POSITIVE：四个 HW corner 的 fusion speedup relative range 为 42.7%，`gate2b_ratio_fusion=0.523`。该结论限于 single Conv workload、single mapping、mode=0 timing、four HW corners 和 fusion axis。
