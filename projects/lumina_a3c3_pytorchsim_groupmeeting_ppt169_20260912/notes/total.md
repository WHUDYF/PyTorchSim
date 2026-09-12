# 01_cover

报告主线是 Lumina、HIDA 和 PyTorchSim。我们从搜索闭环讲到固定算法下的联合实验。

---

# 02_problem

固定算法语义 A，搜索实现映射 I 和硬件 H。难点是组合多且评估慢。

---

# 03_lumina_overview

Lumina 用 QualE、QuanE、SE、EE、TM 把模拟反馈变成搜索依据。

---

# 04_quale

QualE 从模拟器代码提取影响关系，迁移后要继续追踪 MLIR、TOG 和指标。

---

# 05_quane

QuanE 用小幅微扰估计局部方向和资源代价。局部斜率不能外推整个空间。

---

# 06_se_ee_tm

SE 定位瓶颈，EE 执行模拟，TM 记录轨迹。迁移后还要检查合法性和正确性。

---

# 07_lumina_boundary

Lumina 外层重点搜索硬件，但 mapper 也可能随硬件调整映射。我们的增量是显式暴露 I。

---

# 08_a3c3_overview

SkyNet、EDD、Medusa、SnapKV 分别代表结构搜索、连续松弛、解码组织和 KV 压缩。HIDA 最贴近固定算法的实现问题。

---

# 09_hida_pipeline

HIDA 从 PyTorch 或 C++ 进入 MLIR，依次处理 Functional、Structural dataflow，最后生成 HLS C++ 和 RTL。

---

# 10_hida_ir

Functional 层表示任务层级，Structural 层表示存储、接口和调度细节。lowering 把上层选择变成约束。

---

# 11_hida_fusion

HIDA 按模式融合任务并检查关键路径。PyTorchSim 还要检查 TOG 的搬运和同步。

---

# 12_hida_buffers

HIDA 先保证多生产者等并发合法，再平衡路径。buffer 复制、soft FIFO 和 token flow 支持弹性流水。

---

# 13_hida_parallelism

IA 看计算强度，CA 看共享 buffer 连接和布局。并行因子受总预算与整除约束。

---

# 14_hida_code

论文 Algorithm 4 是局部 DSE 描述，公开仓库用多个 MLIR pass 实现 pipeline。二者不能逐行等同。

---

# 15_pytorchsim_overview

PyTorchSim 主链是 PyTorch、MLIR、TOG 和 TOGSim。Gem5、Spike、TOGSim 分工不同。

---

# 16_pytorchsim_frontend

torch_openreg 注册 npu 并接入 TorchInductor。fusion、tile、vector lane 和 scratchpad 是关键入口。

---

# 17_pytorchsim_codegen

MLIRCodeCache 生成验证二进制、周期样本和 TOG，缓存避免重复编译。

---

# 18_pytorchsim_tog

TOG 节点表达 MOVIN、COMP、MOVOUT 和 BAR，并携带循环、地址和 cycle 属性。

---

# 19_pytorchsim_runtime

TOGSim 每周期推进 Scheduler、Core、DMA、NoC 和 DRAM，输出 cycles、利用率、流量和 stall。

---

# 20_pytorchsim_config

YAML 字段分别影响编译和系统模拟。每个候选都要保存配置、TOG、属性和结果。

---

# 21_pytorchsim_cost

短片段适合预筛，完整 workload 才能确认跨 kernel 影响。microbenchmark 不能直接当最终答案。

---

# 22_compare_transfer

HIDA 提供表示和约束，Lumina 提供搜索顺序，PyTorchSim 提供执行证据。

---

# 23_research_loop

拟议闭环先固定 A，生成合法 I，再搜索 H 加 I。首个问题是相同预算下是否更有效。

---

# 24_next_design

第一阶段选 GEMM、attention、conv，暴露最小映射旋钮。随后比较 H-only 与 I 加 H，并解释每个结果。

---

