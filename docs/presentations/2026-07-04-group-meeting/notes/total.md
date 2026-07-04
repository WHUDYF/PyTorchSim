# 01_cover

大家好，本次组会我汇报最近两条研究主线的进展。第一条是 PyTorchSim 上的硬件软件协同设计实验，我们拿到了第一份 measured POSITIVE 证据。第二条是 Lumina 论文 QualE 模块的最小复现，我们得到了两个方向相反的 NEGATIVE，同时也理清了论文原意与我们实现之间的四大错位。今天大约十五分钟。

---

# 02_outline

汇报分四个部分。首先花四页介绍 PyTorchSim 与 Lumina 各自的架构、以及为什么从 GPU 迁移到 NPU 这条研究路径和迁移中需要关注的新问题。然后用六页展开 PyTorchSim 主线的完整证据链，从 v1 NEGATIVE 到 fusion 与硬件配置形成 measured 交互作用的 POSITIVE。接着用四页讲 Lumina QualE 复现的两次失败模式和背后的 Gap Analysis。最后两页做方法论总结和下一步。

---

# 03_pytorchsim_arch

我们先建立共同语境。PyTorchSim 是 POSTECH 提出的一个 NPU 模拟框架，它把 PyTorch 模型经过 PyTorch2 编译器生成 NPU 机器码和 Tile-Operation Graph，然后 TOGSim 执行这个 TOG 做高速仿真，同时集成 BookSim 建 NoC、Ramulator2 建 DRAM，保证共享资源是周期精确的。核心创新叫 Tile-Level Simulation，本质是利用 tile 计算延迟的确定性。原论文相对 TPUv3 的 runtime 平均绝对误差是百分之十一点五。对我们的工作来说重要的是，它给出了一个可以无源码切换编译器优化、mapping、以及 L2 cache 的实验平台，这是后面所有协同设计实验的基础设施。

---

# 04_lumina_arch

Lumina 是 2026 年 arXiv 的一篇 LLM 辅助 GPU 架构探索的论文，它使用的仿真器是 LLMCompass，不是通用的 gpgpu-sim 或 Accel-Sim。这一点后面会成为核心 gap，请大家先记住。它的框架有四个模块：QualE 从仿真器源码提取架构参数到 PPA 指标的定性因果图，QuanE 跑 micro-benchmark 把定性边填上定量灵敏度系数，SE 从 critical path 挑选瓶颈选择改哪些 knob，EE 做闭环探索并返回一个 Pareto set。整个框架靠迭代式知识积累。QualE 是四个模块里最前端的一步，我们本次只复现它。

---

# 05_gpu_to_npu

看完两个 framework，我先交代一下为什么这条研究路径要从 GPU 换到 NPU。我们过去在 GPU 客体上做过 DSE 类工作，包括 gpgpu-sim 和 Accel-Sim。今天要复现的 Lumina QualE 客体依然是 GPU 仿真器，所以 GPU 这条线不是完全放下、而是延续。但主线换到了 NPU，具体有两个动机。第一，GPU 架构研究趋于饱和，主流微架构基本定型，DSE 和编译器方向的边际收益递减。第二，硬件软件协同设计在 NPU 上才真正可行，因为 NPU 的编译器栈是自研可控的、ISA 可以扩展、硬件参数是通过 YAML 暴露给我们的，所以我们能真正做架构选择、而不是在既定的 CUDA 与已固化的 GPU 微架构上小修小补。总结一句：迁移的核心逻辑是从被锁死的架构走到可控的架构。

---

# 06_tog_new_problems

迁到 NPU 之后，第一件需要重新思考的事情是抽象层次。GPU 侧我们习惯的抽象是 kernel、CTA、warp、SIMT lane，profiling 单位是 warp 利用率、divergence、L1/L2 hit rate，nsys 和 nsight 这一套工具积累了几十年。NPU 侧 PyTorchSim 的抽象是模型经过编译器生成 MLIR 或 TOG op，然后展开成 tile，最底层落到 RISC-V 向量指令。profiling 单位变成 tile 级 cycle 和 TOG 节点间的执行依赖，工具链还在演进中。核心差异是 GPU 是动态 SIMT 调度，NPU 是编译期静态 tile 图加确定性 latency。由此带来三个新问题：第一，critical path 定义变了，我们后面 P11 会看到 GPT-2 里 fusion 不在 critical path 就是这类问题的直接体现；第二，编译器暴露面有限，很多 axis 在源码里是 hardcode，这是 P09 会展开的 discovery；第三，measured 方法论要重新建立，GPU 时代的 baseline 和 verdict 判据不能直接搬到 tile-level 语境。这三个新问题正好对应 Part B 的 co-design axis 搜索和 Part C 的 validator 方法论迁移。

---

# 07_research_question

Part B 主线的核心问题是：在 PyTorchSim 上，给定 workload 和硬件配置空间，硬件软件协同设计是否显著优于在固定硬件上单独调 workload 或者反过来的组合。判据是 measured 数据，Gate 2b 相对 range 超过 15 个百分点才算 significant interaction。我们明确不接受占位、模型推导、待测量、或者空 telemetry 这四种数据来源。上一版 v1 已经在 tiling-only 空间给出 NEGATIVE，所以 Route 4 的目标是扩展到贴近真实编译器控制面的修改维度。

---

# 08_v1_negative

先来看 evidence chain 的起点。v1 的 setup 是 GPT-2 单个 block 的 prefill、序列长度 128、tiling-only 的二乘二 SPAD 与带宽搜索空间。commit 92623de 初次判决 NEGATIVE，commit 09b9ac3 在 10 个百分点的 Gate 1 阈值下重算，NEGATIVE 保持。verdict 名字叫 FUSION_TILING_ONLY_NEGATIVE，注意名字里明确带了 TILING_ONLY 前缀，这是 scope-aware 命名的关键。它告诉我们的下一步不是放弃 co-design，而是换 axis。这就自然导出 Route 4 探索四个编译器修改维度。

---

# 09_route4_pivot

Route 4 的 discovery 阶段全面盘点了四个候选轴。fusion 有 YAML 字段 codegen_compiler_optimization，可以设 none、all、或者部分子集，零源码修改，是唯一实测通过的轴。dataflow 主路径是固定的 weight-stationary WS_MESH，没有暴露 output-stationary 或 input-stationary 的 YAML 字段，改起来要动 300 到 800 行源码。SPAD partition 现在只有容量字段没有 partition strategy 字段，模板里 input/weight/output buffer 是硬编码的。DMA schedule 也没有 prefetch 或者 double buffer 字段。所以 Route 4 主线聚焦到 fusion 上。

---

# 10_fusion_positive

这是本次工作最关键的一张证据。commit 79e6259 的 Conv 3x3 4 硬件配置 × 2 fusion 状态的 sweep。workload 是 shape [1, 64, 56, 56] 的 Conv，functional mode 等于零，seed 等于零，每个 cell 跑 5 次取中位数，mapping 是 harness default。四个硬件配置上的 fusion 加速比分别是：HW-A 6.90×，HW-B 4.83×，HW-C 6.26×，HW-D 5.25×。fusion_speedup_range 的相对 range 是 42.7 个百分点，远超 15 个百分点阈值。gate2b_ratio_fusion 等于 0.523，Gate 2b pass。verdict 名叫 FUSION_CONV_CODESIGN_POSITIVE，这是本项目第一份 measured 协同设计 POSITIVE 证据。

---

# 11_workload_dependence

同一个 fusion 轴在不同 workload 上差异非常大。addmm+relu 128 大概是 2 倍的弱信号。GPT-2 block 序列长度 128 只有 1.02 倍，也就是 1.9 个百分点，被判定为 NOT_GENERALIZABLE_ON_GPT2。Conv 3x3 probe 是 6.84 倍的强信号。commit ce5f0e3 的根因调查发现，GPT-2 里 fusion 结构上生效，MLIR op count 减少了 7.8 个百分点，说明 fusion 不是 no-op，但是它变化的部分不在 critical path 上，所以端到端 timing 不变。Conv 3x3 是 compute-bound、序列短，fusion 直接落在 critical path 上，所以效果强。这个对比是本次工作最重要的一个 methodological 结论：协同设计的价值是 workload-dependent 加 axis-dependent，不是普适性质。

---

# 12_scope_limits

按学术诚实原则，POSITIVE 声明必须与五条 scope limit 一同引用，未来任何论文或外部引用都要带上。第一，只在 Conv 3x3 kernel-level 上成立，不代表端到端 ResNet 或 MobileNet。第二，只用了 harness 默认 mapping，没有做 multi-mapping ranking，所以不能声称正式的 Gate 3 ranking interaction。第三，只有 mode 等于零的 timing 证据，mode 等于一的 correctness 因为 Spike 的 --varch source blocker 仍然 deferred。第四，只覆盖了 codesign_v1_2x2 的 A、B、C、D 四个 corner，没有覆盖连续空间。第五，只有 fusion 轴 POSITIVE，dataflow、SPAD partition、DMA schedule 都还没 demonstrated。

---

# 13_quale_reproduction_goal

进入 Part C。QualE 复现的核心问题是：在严格 Spec Coding 约束下，原论文的 R1、R2、R3 三条机器验证规则是否足以从 GPU 仿真器源码提取真实可用的架构到 PPA 因果图。判据是至少 60 个百分点的 edge 通过人工语义抽检，低于 30 个百分点直接 NEGATIVE。工程约束我们卡得很死，前一版复现是 22 份 spec、5600 行代码的过度工程，这次总代码 671 行、6 个核心模块、3 个 CLI 脚本，明确禁止 provider registry、cache manifest、retry ledger 这些自证式 bookkeeping。整个流程组织成四个 Milestone Gate：A 是 control 用 gold-standard packet，B 是 real LLM 在 6 个 packet 上跑，C 是 validator 三条规则判定，D 是 verdict 加人工语义抽检。

---

# 14_v1_false_positive

v1 的结果是失败模式一：机器验证 86 分之 86，也就是百分之百通过，看起来非常漂亮，但人工语义抽检 pass_ratio 只有 0.20，硬触发 No-Go 阈值。判决是 NO_GO_NEGATIVE，属于假阳性。根因是三条规则各自太松。R1 的 symbol 匹配把 DDR 时序常量 tRAS、tRCD 这些误判成 arch knob，因为 gpgpu-sim 是老代码库，DRAM 常量和现代 m_ 前缀命名混在一起。R2 的 PPA 白名单太宽，几乎所有内部 counter 都能作为 target，让 LLM 可以给任何 knob 到任何 counter 编造一个 fake 因果。R3 只做 substring 匹配，只要 evidence_text 里出现 knob 名就 pass，注释、debug print、无关分支都能匹配。表面 pass，实际很多不是因果。

---

# 15_v2_false_negative

v2 我们做了一步补救：加了 R4 命名 pattern 分类器，只把符合 m_config_ 或 CONFIG_ 这类现代命名 pattern 的 symbol 判为 knob。结果 11 条 edge 语义抽检百分百通过，但覆盖率从 6 分之 6 崩到 6 分之 1，AC-C3 硬 Go 条件失败，判决还是 NO_GO_NEGATIVE，这次属于假阴性，方向和 v1 正好相反。副作用是 pattern 只匹配现代命名，gpgpu-sim 里许多真 knob 用的是老命名比如 n_sets、n_ways、line_size，R4 全部误伤剔除掉了。合起来的结论是：任何单一规则式 validator 都无法通吃异质命名规范的老代码库。我们必须放弃规则式，转向 AST 分析或者 LLM 两步 prompt。

---

# 16_gap_analysis

但是 v1 和 v2 的 NEGATIVE 不能作为对 Lumina QualE 方法论本身的否定，因为我们的实现与论文原意有四大深度错位。第一，仿真器不同：我们用 gpgpu-sim 或 Accel-Sim，论文用 LLMCompass。第二，knob 定义不同：我们抓 C++ 里 m_ 前缀成员变量，论文抓 LLMCompass 定义的架构参数。第三，PPA 指标不同：我们看仿真器内部 counter，论文看 LLMCompass 的输出指标加 critical path。第四，scope 不同：我们只做 QualE 单独实验没有下游，论文有 QualE 加 QuanE 加 SE 加 EE 的闭环。v3 方案是切换到 LLMCompass，直接闭合 Gap 1、2、3；Gap 4 是有意识的分阶段。v1、v2 的工程流程包括 spec-first、gate、measured artifact、TDD 全部 carry over。

---

# 17_summary_next

最后总结。两条主线的共同方法论贡献有三点：一是 Spec Coding，以 measured 数据为唯一 claim-bearing 源，不接受 modeled、placeholder、pending 冒充；二是 Gate-based verdict，四种诚实终态 POSITIVE、NEGATIVE、PARTIAL、BLOCKED；三是 scope limit 强制引用。PyTorchSim 主线的下一步是扩展 dataflow、SPAD partition、DMA schedule 三个尚未 demonstrated 的轴，同时扩大 workload 覆盖到 ResNet 和 MobileNet，然后做 multi-mapping ranking 来跑正式的 Gate 3。Lumina 主线下一步是切换到 LLMCompass 做 v3，放弃规则式 validator，分阶段接入 QuanE、SE、EE。总的一句话是：POSITIVE 和 NEGATIVE 都是诚实的科研结论，都要保留、都要引用 scope、都要指导下一步。

---

# 18_qa

汇报到这里，欢迎讨论。我特别想听大家对以下几个方向的意见：任一 gate 判决细节的合理性，如何在 dataflow、SPAD、DMA 三个 axis 上跑通 POSITIVE，LLMCompass v3 setup 可能的 pitfalls，以及方法论迁移到其它仿真器上的可行性。谢谢。
