# NPU 软硬件协同验证 · 进度追踪

最后更新：2026-07-08 · PyTorchSim · master 分支 · 相关会话 `codex-019edfa6`

## 当前主线目的

在 PyTorchSim 上通过扩大编译搜索空间（Route 4：fusion / dataflow / SPAD partition / DMA schedule）判断 HW/SW co-design thesis 是否有 measured POSITIVE 支撑。

当前状态：Route 4 已经在 Conv 3x3 workload 上得到 measured POSITIVE，并完成 8-tile denser interior-optimum v2 probe。本次 v2 verdict 为 `TILE_FUSION_INTERIOR_OPTIMUM_CONFIRMED_V2`；global min 为 `HW-A/tile_C/all`，median `5880.0` cycles。

**最终停止条件（双必备）**：

- **条件 A：已满足。** 仓库中已有 Route 4 measured POSITIVE verdict：`outputs/route4_fusion_pilot/pilot_summary.json` 显示 `FUSION_CONV_CODESIGN_POSITIVE`，commit `79e6259`。
- **条件 B：已处理。** `docs/superpowers/specs/2026-07-03-npu-compiler-modification-landscape.md` / `.html` 已梳理 fusion / dataflow / SPAD partition / DMA schedule 四个 compiler-modification axes 的当前状态、修改成本和 measured evidence。

## 1. 各阶段命令、请求与状态

| 阶段 | 目的 | 命令 / handoff | Commit | 状态 | 主产出 |
|---|---|---|---|---|---|
| v1 baseline | GPT-2 单 block prefill seq=128 + 2x2 SPAD/BW 证伪 tiling-only co-design | `/tmp/codesign-goal-handoff.md`；4 HW x 8 tiling x 3 repeat = 96 measured runs | `92623de` | **NEGATIVE**（claim-bearing within scope） | v1 verdict.json，4 gate measured |
| v1 re-analysis | 用 10% Gate-1 threshold 重算 v1，规避 7% noise floor | `/tmp/codesign-v1-reanalysis-handoff.md`；不 rerun sweep，只重跑 analyzer | `09b9ac3` | **NEGATIVE 保持** | v1-reanalysis；tiling-only thesis 证伪范围更清楚 |
| v2 delta | HW 第 1 轴改为 systolic/vector lane size | `/tmp/codesign-v2-goal-handoff.md` | `a1e710b` + `b3db8c2` | **BLOCKED** | 8-lane path 触发 MLIR blocker：`Mismatched subtile K` |
| Determinism 诊断 | 解释 v1/v2 canary cycle_delta 漂移 | `/tmp/codesign-v2-recovery-handoff.md` | `8c5af69` | **完成** | measurement noise floor 约 7% |
| Route 4 discovery | 调查 fusion / dataflow / silent-bug / workload 可行性 | `/tmp/codesign-route4-discovery-handoff.md` | `42e57b6` | **FEASIBLE_WITH_MODIFICATION** | fusion 可无源码切换；dataflow 不可无源码切换；correctness gate 需要显式记录 |
| Fusion pilot recovery | addmm+relu 上验证 fusion 轴能产生结构和 timing 差异 | `/tmp/codesign-next-handoff.md` fusion pilot recovery | `59f44ba` | **POSITIVE within small kernel** | addmm+relu：`2183 -> 1118` cycles，约 2x |
| GPT-2 fusion x HW mini sweep | 检查 fusion 是否泛化到 GPT-2 block | `/tmp/codesign-next-handoff.md` fusion x HW mini co-design sweep | `3489f94` | **NOT_GENERALIZABLE_ON_GPT2** | GPT-2：`430717 -> 422711`，约 1.9%，低于 gate |
| Fusion root cause | 解释 GPT-2 no-op/critical path，并探测 Conv workload | `/tmp/codesign-next-handoff.md` fusion root cause investigation | `ce5f0e3` | **CRITICAL_PATH_LIMITED** | GPT-2 structural active but not critical path；Conv probe `189987 -> 27769`，约 6.84x |
| **Conv 4-HW fusion sweep** | 在 Conv 3x3 上检查 fusion speedup 是否随 HW 变化 | `/tmp/codesign-next-handoff.md` Conv 4-HW fusion sweep | `79e6259` | **FUSION_CONV_CODESIGN_POSITIVE** | `HW-A=6.895x`，`HW-B=4.830x`，`HW-C=6.262x`，`HW-D=5.253x`；relative range `42.7%`；`gate2b_ratio_fusion=0.523` |
| **Route 4 wrap-up** | 完成 compiler-modification landscape 与 POSITIVE declaration | `/tmp/codesign-next-handoff.md` Route 4 wrap-up | 当前提交 | **完成后归档** | landscape 文档 + POSITIVE declaration + tracker 更新 |
| **Tile x fusion x HW interior probe** | 在 Conv 3x3 上检查 tile fit boundary 是否使最佳软件选择随 HW 迁移，或产生 strict interior optimum | `/tmp/codesign-next-handoff.md` interior-optimum probe on tile x fusion x HW | `5f16398` | **CHAMPION_MIGRATION_ONLY** | 140 measured runs + 20 unavailable run slots；HW-A/B champion=`tile_D+all`，HW-C=`tile_A+all`，HW-D=`tile_B+all`；global min=`HW-C/tile_A/all` 但仍是 corner |
| **8-tile denser interior probe** | 在 Conv 3x3 上把 tile space 从 4 加密到 8，并加入 `fusion=["fusion"]` 第三 variant，检查 tile interior optimum | `/tmp/codesign-next-handoff.md` 8-tile denser sweep push interior optimum | 当前提交 | **TILE_FUSION_INTERIOR_OPTIMUM_CONFIRMED_V2** | 450 measured runs；30 unavailable slots；global min=`HW-A/tile_C/all` |

状态图例：**NEGATIVE**（实测证否，范围内有效）· **BLOCKED**（外部或源码约束）· **POSITIVE**（measured support，范围内有效）· **完成**。

## 2. 当前证据链摘要

v1 NEGATIVE 与 Route 4 POSITIVE 不冲突。

- v1 NEGATIVE 的范围：GPT-2 single block prefill seq=128、tiling-only search space、2x2 SPAD/BW HW corners。
- Route 4 POSITIVE 的范围：Conv 3x3 kernel-level workload、fusion axis、`codesign_v1_2x2` 的 4 个 HW corners、single mapping、mode=0 timing。

科学结论是：HW/SW co-design value 是 workload-dependent、axis-dependent，并且会受到 tile fit boundary 约束。GPT-2 tiling-only 可以没有收益，同时 Conv fusion x HW 可以有 measured interaction；在 Conv tile x fusion x HW 中，best software choice 会随 HW 迁移，但本轮尚未得到 global interior optimum。

## 3. Route 4 四个 compiler-modification axes 当前状态

| Axis | 当前状态 | 是否无需源码可 sweep | 是否有 measured POSITIVE |
|---|---|---:|---:|
| fusion | `codegen_compiler_optimization` YAML field 可控；Conv 上已显示强信号 | YES | YES，`79e6259` |
| dataflow | Dense systolic path 是 WS / `WS_MESH`；OS/IS 未暴露 | NO | NO |
| SPAD partition | 只有 `vpu_spad_size_kb_per_lane` 容量字段；未发现 partition strategy | NO | NO |
| DMA schedule | 未发现 `prefetch` / `double_buffer` / `dma_schedule` YAML/CLI 控制字段 | NO | NO |

完整文档见：

- `docs/superpowers/specs/2026-07-03-npu-compiler-modification-landscape.md`
- `docs/superpowers/specs/2026-07-03-route4-codesign-positive-declaration.md`

## 4. POSITIVE 声明的范围限制

任何引用 Route 4 POSITIVE 时必须同时说明：

- Single workload：目前只在 Conv 3x3 kernel-level workload 上证明。
- Single mapping：没有做 multi-mapping ranking，因此不能声称正式 Gate-3 ranking interaction。
- Mode=0 timing：correctness 由于 Spike `--varch` source blocker 仍 deferred。
- Four HW configs only：只覆盖 `codesign_v1_2x2` 的 `HW-A/B/C/D`。
- Fusion axis only：dataflow / SPAD partition / DMA schedule 尚未 demonstrated POSITIVE。

## 5. 下一步建议

优先级如下：

1. 基于 8-tile denser v2 结果，下一步换到 ResNet stage 或 GPT-2 block 做 generalization，验证 tile interior optimum 是否能跨 workload 保持。
2. 在 Conv-heavy workload 上做 fusion x HW x multi-mapping，补正式 Gate-1 / Gate-3 evidence。
3. 用 ResNet-50 stage 或 MobileNet block 验证 Conv-heavy generalization。
4. 若接受源码修改，优先考虑 DMA schedule axis；SPAD partition 次之。
5. Dataflow axis 只有在接受较高源码修改成本后再推进。
