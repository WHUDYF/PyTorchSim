# 01_从 Lumina 到 PyTorchSim
今天的讨论从 Lumina 开始，但目标不是简单复述一篇论文。我们想借 Lumina 和 A3C3 重新思考一个具体问题：在 PyTorchSim 中，怎样把昂贵的硬件 DSE 推进成可验证的 compiler–accelerator co-design。整场汇报会先拆解已有工作的机制，再落到我们下一步可以真正测量的实验闭环。

---

# 02_问题张力
PyTorchSim 已经把 PyTorch、MLIR、TOG 和周期级模拟器接起来了，但这也意味着每一个候选点可能触发完整的编译和模拟流程。硬件参数、tile、fusion、dataflow 和 DMA 之间还存在耦合，所以问题不是简单地增加 sweep 数量。我们真正需要的是一种能识别交互、控制模拟预算的实验方法。

---

# 03_Lumina 总览
Lumina 的核心贡献是把硬件探索变成瓶颈驱动的闭环。QualE 从代码提取结构关系，QuanE 用微扰实验估计影响强度，SE 根据 critical path 选择策略，EE 执行候选，Trajectory Memory 再把结果反馈回来。每次模拟都应该改变下一次选择。

---

# 04_QualE
QualE 先回答一个结构问题：哪些硬件参数可能影响哪些性能指标。它从 simulator 源码中抽取 influence map，先排除没有直接因果边的参数，再进入量化阶段。对 PyTorchSim，我们需要把这条边延伸到 compiler knob、MLIR、TOG 属性和 TOGSim 指标。

---

# 05_QuanE
QuanE 不一开始做大规模搜索，而是先做小的 sensitivity benchmark。只改变一个参数，观察 cycle、area 或 bandwidth 如何变化，就可以得到局部影响方向和强度。这种局部先验非常适合 PyTorchSim，因为完整 TOGSim 很贵，而小算子和短序列可以承担初筛任务。

---

# 06_SE–EE–TM
当 simulator 告诉我们当前瓶颈是 memory latency 或 interconnect congestion，Strategy Engine 就不应该让搜索器随意改所有参数。它只提出与当前瓶颈相关的受限候选，执行后把结果写进轨迹记忆。这个闭环是 Lumina 比静态规则更有价值的地方。

---

# 07_Lumina 的边界
Lumina 的优势是样本效率，而不是覆盖所有算法层次。它主要在固定 workload 下搜索硬件配置，并不会自动产生新的网络拓扑、compiler mapping 或 Medusa 式的算法重构。因此我们应该迁移它的搜索策略，同时自己补上编译器和 TOG 这一层。

---

# 08_A3C3 设计地图
A3C3 提供的是更宽的设计空间视角。算法、实现和硬件可以分别固定，也可以一起搜索；Bundle、联合目标和自动生成是它的共同语言。我们当前选择的是其中一个受控切片：固定算法拓扑，重点研究实现和硬件的交互。

---

# 09_SkyNet
SkyNet 解决的是“什么网络适合硬件”。它先把局部操作组合成 Bundle，在目标设备上预评估延迟、资源和准确率，再用 PSO 搜索通道数和池化位置。对 PyTorchSim，最值得借鉴的是先做 TOG motif 的预标定，再进入大 block 或完整模型。

---

# 10_EDD
EDD 把网络结构变量和实现变量放进同一个联合目标函数，并用可微的表示降低搜索成本。它告诉我们，co-design 的关键不是把两个结果最后拼起来，而是让硬件代价在搜索过程中直接影响算法或实现选择。固定拓扑以后，这个思想自然退化成 I 和 H 的联合 cost model。

---

# 11_HIDA 与固定拓扑
HIDA 代表另一条非常接近我们的路线：程序逻辑相对固定，重点优化 tiling、并行、buffer、streaming 和 dataflow。它通过功能层和结构层的 IR 管理跨层耦合。PyTorchSim 的 FX、MLIR 和 TOG 链路已经提供了类似的观察层。

---

# 12_Medusa
Medusa 说明协同设计还可以发生在算法执行方式上。它观察到自回归解码的串行依赖和重复读权重问题，于是增加多个 decoding heads，用 tree attention 一次验证多个候选。它对我们的启发是：当瓶颈来自执行组织时，算法图本身也可以成为未来的设计变量。

---

# 13_SnapKV
SnapKV 进一步把数据生命周期纳入协同设计。它根据 attention pattern 选择重要 KV 位置，减少 cache 容量和访问成本。对于 PyTorchSim，这提醒我们 workload 不能只有模型名和 tensor shape，还要描述 sequence phase、KV、DRAM 和 SPAD 行为。

---

# 14_综合比较
这些工作没有谁在所有维度上更高级。SkyNet 改网络结构，EDD 做算法和实现联合搜索，HIDA 优化固定程序的映射，Medusa 改变解码执行方式，SnapKV 改变内存保留策略，Lumina 则优化昂贵硬件实验的选择。它们共同回答的是：当前系统的真正瓶颈在哪里，最小的有效改动应该落在哪一层。

---

# 15_我们的形式化问题
我们的第一阶段可以明确写成固定拓扑问题。令算法结构为 A 等于 A 零，软件实现变量包括 tiling、mapping、fusion、dataflow 和 DMA，硬件变量包括 SPAD、DRAM、NoC 和 PE。我们要测的不是单纯硬件排名，而是换硬件后最优 mapping 是否迁移，以及 mapping 排名是否重排。

---

# 16_下一步设计
下一步不建议直接对所有 workload 和配置做全量 sweep。我们应该先建立 compiler knob 到 TOGSim 指标的 influence map，再用 GEMM、Conv、attention 和 KV motif 做 sensitivity probes，训练一个便宜的 surrogate，最后只把最有信息量的组合送进 TOGSim 做 measured gate。这样才能把 A3C3 的设计空间、Lumina 的搜索策略和 PyTorchSim 的跨层观测真正合在一起。
