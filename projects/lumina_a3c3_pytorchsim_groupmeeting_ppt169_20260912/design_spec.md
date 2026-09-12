# Design Specification — HIDA / PyTorchSim revision

## I. Project Information
- 中文组会报告，24 页。沿用用户已确认的咨询风格与深色学术视觉。
- 主线：Lumina 搜索闭环 → A3C3 其他案例概览 → HIDA 实现 → PyTorchSim 架构 → 联合探索方案。
- 本次为用户授权的内容修订，不重新请求视觉确认。旧版保存在 backup/before_hida_architecture_revision/。
- 论文论述、仓库实况、拟议研究明确区分；没有新实验结果。

## II. Canvas
- PPT 16:9，1280 × 720；页边距 56；标题区 y=60–162；主体 y=200–625；来源/页码 y=680。

## III. Theme
- background #0B1220；panel #162235；blue #4DA3FF；mint #36D6B5；purple #A78BFA。
- warning #FFB454；failure #F26D85；text #F4F7FB；secondary #AAB8CC；tertiary #718096；border #2A3A52。
- 蓝色表示编译/映射，绿色表示执行/反馈，紫色表示表示层/约束，橙色表示边界/研究假设。

## IV. Typography
- title: Georgia, "Microsoft YaHei", "Noto Serif CJK SC", serif。
- body: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif。
- code: Consolas, "Courier New", monospace。
- 正文基准 18px；标题 34–38；小标题 24–28；图内辅助 16–18；脚注 12–14；封面 54–64。
- 字号比例：body .95–1.1；label .83–1.0；annotation .61–.83；subtitle 1.3–1.6；title 1.8–2.4；hero 3–4。
- Noto 为本机中文预览补充字体；PPT 保留微软雅黑为第一选择。

## V. Layout
- 自由设计，无版式模板；使用裸文本、横向流程、分层表示图、执行拓扑和比较表轮换。
- HIDA / PyTorchSim 页重点使用可编辑流程与结构图。
- 单页主张 + 一个主体图/表 + 一句结论；讲稿承载细节，不将论文正文压入页内。
- 用有向连线表达执行/转换，用虚线表达配置输入、反馈或可选分支。

## VI. Icon System
- chunk-filled；沿用原锁定 inventory；新增页面可不用图标，禁止字符冒充图标。

## VII. Visualization
- 全部手写原生 SVG，顺序逐页；无预设图表模板，无图像素材。
- P13 使用 HIDA Table 5 的 op count / parallel factor 说明；明确为论文例子。
- 研究示例均标“示意”或“拟议”，不编造实测曲线或加速比。

## VIII. Image Plan
- 不使用外部图片；来源 PDF 仅用于研究，不进入幻灯片背景。

## IX. Page Outline
| 页 | 文件 | 核心讲解 | 节奏 |
|---|---|---|---|
|01|01_cover|从搜索策略走向可解释的映射—硬件协同设计|anchor|
|02|02_problem|固定算法语义，搜索 I×H；工作负载选择问题|breathing|
|03|03_lumina_overview|QualE / QuanE / SE / EE / TM 总览|dense|
|04|04_quale|代码依赖变成可检验的影响关系|dense|
|05|05_quane|单参数微扰提供局部先验；成本边界|dense|
|06|06_se_ee_tm|策略、执行、记录和修正闭环|dense|
|07|07_lumina_boundary|外层硬件 DSE 与内层 mapper；贡献维度|breathing|
|08|08_a3c3_overview|SkyNet / EDD / Medusa / SnapKV 各一行，HIDA 是综述背景|dense|
|09|09_hida_pipeline|输入、五步优化、HLS C++、RTL 的完整链路|dense|
|10|10_hida_ir|Functional 的 dispatch/task；Structural 的 schedule/node/buffer/stream|dense|
|11|11_hida_fusion|模式融合、非关键任务再平衡、停止条件|dense|
|12|12_hida_buffers|多生产者合法化、残差路径平衡、soft FIFO 与 token|dense|
|13|13_hida_parallelism|IA 分资源、CA 约束布局、连接排序、数值示例|dense|
|14|14_hida_code|论文局部 DSE 与公开 pass 实现的对应及差别|dense|
|15|15_pytorchsim_overview|PyTorch→编译→TOG→TOGSim，Gem5/Spike 分支|breathing|
|16|16_pytorchsim_frontend|Dynamo/ATen/Inductor/NPU 注册；fusion 和模板|dense|
|17|17_pytorchsim_codegen|MLIRCodeCache 编译分支、采样、功能验证、TOG|dense|
|18|18_pytorchsim_tog|TOG 是 tile 程序：loop/load/compute/store/barrier 与地址|dense|
|19|19_pytorchsim_runtime|Scheduler、Core、DMA、NoC、DRAM 周期执行和反馈|dense|
|20|20_pytorchsim_config|配置影响编译端和运行端，字段消费者与重编译|dense|
|21|21_pytorchsim_cost|多阶段成本、缓存分层、代表片段与整图验证|dense|
|22|22_compare_transfer|HIDA 和 PyTorchSim 目标差别，Lumina/HIDA 各能迁移什么|dense|
|23|23_research_loop|固定 A 的约束驱动候选生成 + 证据驱动评估闭环|dense|
|24|24_next_design|最小可证伪实验与交付顺序|anchor|

## X. Speaker Notes
- 每页 3–5 句中文自然口述，定义缩写、解释图中的因果关系与适用边界。
- notes/total.md 的标题与 SVG 文件名一致，随后拆分写入 PPT notes。
- sources/references.md 单独记录论文链接、章节、代码入口和事实修正，不把文献清单读进讲稿。

## XI. Technical Constraints
- 逐页读取 spec_lock；主代理串行手写 SVG；禁用脚本批量生成页面。
- 显式 SVG 样式；不使用 style/class/foreignObject；命名逻辑 g 便于编辑/动画。
- 质量检查 → 讲稿拆分 → SVG finalize → 默认双路 PPTX 导出。
- 本次只修改演示文稿，不修改编译器/模拟器代码，不声称重新运行过性能实验。
