# groupmeeting-pytorchsim-lumina-20260704 - Design Spec

> Human-readable design narrative for the 16-slide academic group meeting deck covering (A) PyTorchSim + Lumina background, (B) PyTorchSim HW/SW co-design POSITIVE evidence chain, (C) Lumina QualE reproduction double-negative + gap analysis, (D) methodology summary.

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | groupmeeting-pytorchsim-lumina-20260704 |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 16 |
| **Design Style** | B) General Consulting + academic-defense |
| **Target Audience** | Advisor + lab peers (GPU/NPU architecture research group) |
| **Use Case** | Weekly group meeting report (approx. 15 minutes speaking time) |
| **Created Date** | 2026-07-04 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280×720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 60px, top/bottom 50px |
| **Content Area** | 1160×620 |

---

## III. Visual Theme

### Theme Style

- **Style**: B) General Consulting + academic-defense
- **Theme**: Light theme
- **Tone**: Academic, evidence-driven, restrained, data-clarity-first

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FFFFFF` | Page background |
| **Secondary bg** | `#F5F7FA` | Card background, evidence blocks |
| **Primary** | `#1E3A5F` | Deep navy title, structural decoration |
| **Accent** | `#0076A8` | Deloitte Blue — data highlight, key numbers |
| **Secondary accent** | `#4A6FA5` | Mid blue — gradient bridge, secondary emphasis |
| **Body text** | `#2C3E50` | Main body copy |
| **Secondary text** | `#5D6D7E` | Captions, annotations, footer info |
| **Tertiary text** | `#95A5A6` | Page number, footnote |
| **Border/divider** | `#D5DBDB` | Card borders, divider lines |
| **Success (POSITIVE)** | `#27AE60` | POSITIVE verdict markers, gate-pass |
| **Warning (NEGATIVE)** | `#C0392B` | NEGATIVE verdict markers, gate-fail |

### Gradient Scheme

```xml
<linearGradient id="titleGradient" x1="0%" y1="0%" x2="100%" y2="0%">
  <stop offset="0%" stop-color="#1E3A5F"/>
  <stop offset="100%" stop-color="#0076A8"/>
</linearGradient>

<radialGradient id="bgDecor" cx="90%" cy="10%" r="60%">
  <stop offset="0%" stop-color="#1E3A5F" stop-opacity="0.08"/>
  <stop offset="100%" stop-color="#1E3A5F" stop-opacity="0"/>
</radialGradient>
```

---

## IV. Typography System

### Font Plan

**Typography direction**: academic serif × modern CJK sans — English serif titles pair with CJK sans body for academic-defense feel.

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | `"Microsoft YaHei"` | `Georgia` | `serif` |
| **Body** | `"Microsoft YaHei", "PingFang SC"` | `Arial` | `sans-serif` |
| **Emphasis** | `SimSun` | `Georgia` | `serif` |
| **Code** | — | `Consolas, "Courier New"` | `monospace` |

**Per-role font stacks**:

- Title: `Georgia, "Microsoft YaHei", serif`
- Body: `"Microsoft YaHei", "PingFang SC", sans-serif`
- Emphasis: `Georgia, SimSun, serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = **18px** (dense — evidence chain + data tables dominate).

| Purpose | Ratio to body | Size (body=18) | Weight |
| ------- | ------------- | -------------- | ------ |
| Cover title | 3x | 54px | Bold |
| Chapter opener | 2.2x | 40px | Bold |
| Page title | 1.7x | 30px | Bold |
| Hero number (KPI) | 1.7x | 30px | Bold |
| Subtitle | 1.3x | 24px | SemiBold |
| **Body content** | **1x** | **18px** | Regular |
| Annotation | 0.78x | 14px | Regular |
| Page number / footnote | 0.61x | 11px | Regular |

---

## V. Layout Principles

### Page Structure

- **Header area**: Height 80px — page title + optional pill tag (Part A/B/C/D indicator)
- **Content area**: Height 560px — main body region, uses layout patterns below
- **Footer area**: Height 40px — page number + project tag + date on the right

### Layout Pattern Library

| Pattern | Applied To |
| ------- | ---------- |
| **Single column centered** | P01 Cover, P02 Outline, P16 Q&A |
| **Symmetric split (5:5)** | P03 PyTorchSim architecture (diagram + spec table), P04 Lumina architecture |
| **Asymmetric split (3:7)** | P06, P12 — status pill + narrative dominant |
| **Three/four column cards** | P07 (4-column pillars), P10 (5-item grid) |
| **Full-width chart + takeaway box** | P08 grouped bar, P09 workload comparison |
| **Matrix 2×2** | P14 Gap Analysis 4 gaps |
| **Vertical step chain** | P11 four-gate architecture |

### Spacing Specification

**Universal**:

| Element | Range | This deck |
| ------- | ----- | --------- |
| Safe margin from canvas edge | 40-60px | 50px |
| Content block gap | 24-40px | 32px |
| Icon-text gap | 8-16px | 12px |

**Card-based**:

| Element | Range | This deck |
| ------- | ----- | --------- |
| Card gap | 20-32px | 24px |
| Card padding | 20-32px | 24px |
| Card border radius | 8-16px | 10px |
| Three-column card width | 360-380px | 370px |
| Four-column card width | 270-290px | 275px |

---

## VI. Icon Usage Specification

### Source

- **Built-in icon library**: `templates/icons/chunk-filled/` — filled, straight-line geometry, sharp right angles; carries academic gravitas
- **Usage method**: SVG placeholder `<use data-icon="chunk-filled/<name>" .../>`
- **One deck = one stylistic library**: no mixing with tabler / phosphor

### Recommended Icon List

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| Chart / data | `chunk-filled/chart-bar` | P05, P08, P09 |
| Circle checkmark (POSITIVE) | `chunk-filled/circle-checkmark` | P08, P15 |
| Circle X (NEGATIVE) | `chunk-filled/circle-x` | P06, P12, P13 |
| Target (goal / question) | `chunk-filled/target` | P05, P11 |
| Warning (scope limit) | `chunk-filled/triangle-exclamation` | P10 |
| Lightbulb (insight / method) | `chunk-filled/lightbulb` | P07, P14, P15 |
| Gear (compiler / mechanism) | `chunk-filled/microchip` | P03, P07 |
| Cpu (hardware) | `chunk-filled/microchip` | P03, P07, P08 |
| Book (paper / reference) | `chunk-filled/book` | P04, P11 |
| Search (analysis / discovery) | `chunk-filled/magnifying-glass` | P07, P14 |
| Arrow-right (pivot / transition) | `chunk-filled/arrow-right` | P07, P15 |
| Info | `chunk-filled/circle-info` | P02, P10 |

---

## VII. Visualization Reference List

Catalog read: 71 templates

| Page | Template | Path | Summary-quote (verbatim from `charts_index.json`) | Usage |
| ---- | -------- | ---- | ------------------------------------------------- | ----- |
| P03 | process_flow | `templates/charts/process_flow.svg` | "Pick for 3-8 sequential steps connected by simple arrows — approval workflows, customer onboarding, request handling, lifecycle stages. Skip if cyclical (use circular_stages) or stages produce named o" | PyTorch Model → Compiler → TOG → TOGSim → cycle output pipeline |
| P04 | process_flow | `templates/charts/process_flow.svg` | "Pick for 3-8 sequential steps connected by simple arrows — approval workflows, customer onboarding, request handling, lifecycle stages. Skip if cyclical (use circular_stages) or stages produce named o" | Lumina: QualE → QuanE → AHK → SE/EE iterative loop |
| P07 | vertical_pillars | `templates/charts/vertical_pillars.svg` | "Pick for 1×3 / 1×4 / 1×5 vertical column layout where each pillar = one independent category with title + bullets — PEST (Political/Economic/Social/Technological), four-pillar strategy overview, side-" | 4 compiler modification dimensions (fusion / dataflow / SPAD / DMA) with status |
| P08 | grouped_bar_chart | `templates/charts/grouped_bar_chart.svg` | "Pick for 2-4 series side-by-side across the same categories (e.g. YoY/QoQ). Skip if showing composition within each category (use stacked_bar_chart)." | 4 HW × (fusion=none vs fusion=all) median cycles |
| P09 | bar_chart | `templates/charts/bar_chart.svg` | "Pick for single-series category value comparison, 3-8 categories. Skip for >12 long-label items (use horizontal_bar_chart) or multi-series (use grouped_bar_chart)." | Fusion speedup across 3 workloads (addmm+relu / GPT-2 / Conv 3x3) |
| P10 | icon_grid | `templates/charts/icon_grid.svg` | "Pick for 4-9 parallel features/capabilities/services as icon cards — feature grid, service lineup, benefits matrix, brand values, product highlights. Skip for sequential ordering (use numbered_steps) " | 5 scope limits of the POSITIVE claim |
| P11 | numbered_steps | `templates/charts/numbered_steps.svg` | "Pick for 3-6 horizontal sequential steps with numeric emphasis — how-it-works section, getting-started guide, methodology overview, implementation phases. Skip if steps need connector arrows (use proc" | 4 Milestone Gates A → B → C → D |
| P14 | matrix_2x2 | `templates/charts/matrix_2x2.svg` | "Pick for items plotted on Impact x Effort or similar generic 2-axis prioritization. Skip for named-quadrant text frameworks (use quadrant_text_bullets) or bubble-sized portfolios (use quadrant_bubble_" | 4 Gap Analysis quadrants (our impl vs Lumina paper) |

**Runners-up considered**:

- `stacked_bar_chart` | rejected for P08: composition within category is not the story — the story is Conv fusion speedup swings with HW, requiring two series side-by-side.
- `dumbbell_chart` | rejected for P09: fusion=none vs fusion=all could look like before/after, but the semantic focus is the aggregate speedup magnitude per workload, not the delta between two points.
- `comparison_table` | rejected for P14: table is dense but flattens the "we did X, paper does Y" pairing; matrix_2x2 pins each gap to a spatial position that reads faster.
- `process_flow` | rejected for P11: Gates are stricter than a loose sequential flow — each Gate has a hard AC. Numbered steps with strong ordinal emphasis reads more accurately.
- `quadrant_text_bullets` | rejected for P14: no named quadrants — the 4 gaps are enumerated, not classified by two axes.

---

## VIII. Image Resource List

Deck uses `h. A) No images` — the design draws every diagram via SVG native shapes (rect / circle / path). No `images/` files required.

---

## IX. Content Outline

### Part 1: Cover & Outline

#### Slide 01 - Cover

- **Layout**: Single column centered; muted decorative gradient at top-right; navy footer stripe with date
- **Title**: 从 fusion POSITIVE 到 QualE Double Negative
- **Subtitle**: PyTorchSim 上的 HW/SW Co-design 证据 · Lumina QualE 复现失败模式分析
- **Info**: dyf | 2026-07-04 | 周组会汇报

#### Slide 02 - Outline

- **Layout**: Single column centered; four numbered blocks corresponding to Parts A-D
- **Title**: 汇报大纲
- **Content**:
  - Part A · 背景介绍（PyTorchSim + Lumina 是什么）
  - Part B · PyTorchSim 上的 HW/SW Co-design：从 v1 NEGATIVE 到 fusion POSITIVE
  - Part C · Lumina QualE 复现：v1 假阳性、v2 假阴性、Gap Analysis
  - Part D · 方法论总结与下一步

---

### Part 2: 背景介绍

#### Slide 03 - PyTorchSim 架构与效果

- **Layout**: Symmetric split (5:5); left side = flow diagram (process_flow), right side = feature bullet list + KPI callout
- **Title**: PyTorchSim：面向 NPU 的快速、周期精确模拟框架
- **Visualization**: process_flow — PyTorch Model → PyTorch2 Compiler → NPU machine code + TOG → TOGSim (BookSim + Ramulator2) → cycle-accurate output
- **Content**:
  - 出处：POSTECH PSAL Lab，PSAL-POSTECH/PyTorchSim
  - 两大组件：Compiler（生成 TOG）+ TOGSim（执行 TOG，集成 BookSim NoC + Ramulator2 DRAM）
  - NPU 架构：基于 RISC-V 向量扩展，通用可扩展
  - 核心创新：Tile-Level Simulation（TLS），利用 tile 计算延迟确定性
  - 支持模型：ResNet / MobileNet / YOLO / BERT / GPT-2 / ViT / Mistral / SD-v1 / Llama 2-3 / DeepSeek-V3
  - **效果 KPI**：相对 TPUv3 的 runtime MAE = 11.5%

#### Slide 04 - Lumina 框架与效果

- **Layout**: Symmetric split (5:5); left side = 4-module iterative diagram (process_flow), right side = module description list + note
- **Title**: Lumina：LLM 辅助 GPU 架构探索框架
- **Visualization**: process_flow — QualE → QuanE → AHK (Architectural Heuristic Knowledge) → SE + EE → 反馈 Perf Metrics + Critical Path
- **Content**:
  - 出处：arXiv 2603.05904, 2026 — LLM-Guided GPU Architecture Exploration
  - Simulator：**LLMCompass**（不是 gpgpu-sim / Accel-Sim）
  - 四大模块：
    - **QualE (Qualitative Extraction)**：从 simulator 源码提取 arch → PPA 定性因果图
    - **QuanE (Quantitative Extraction)**：跑 simulator 做 micro-benchmark，填 sensitivity 系数
    - **SE (Strategy Engine)**：根据 critical path 选 knob
    - **EE (Exploration Engine)**：闭环探索 Pareto set
  - **效果**：迭代式知识积累 + refinement，产出 Pareto set

---

### Part 3: PyTorchSim Co-design 主线

#### Slide 05 - 研究问题

- **Layout**: Asymmetric split (3:7) — 左边一个大 target 图标 + 大问句，右边说明背景与限定
- **Title**: 研究问题：PyTorchSim 上是否存在 measured HW/SW Co-design POSITIVE？
- **Content**:
  - **核心问题**：给定 workload 和 HW 配置空间，co-design 是否显著优于 fixed HW × workload-tuned 或 fixed workload × HW-tuned？
  - **判据**：measured 数据 + gate2b_ratio ≥ 0.15（相对 range 超过 15% 视为 significant interaction）
  - **不接受**的证据类型：placeholder、modeled、pending_measurement
  - **Route 4 目标**：从 tiling-only search space 扩展到更贴近真实编译器控制面的修改维度

#### Slide 06 - v1 NEGATIVE：evidence chain 开端

- **Layout**: Asymmetric split (3:7); 左边红色 NEGATIVE pill，右边 v1 setup 与结论
- **Title**: v1 Baseline：GPT-2 + tiling-only 空间上 NEGATIVE
- **Content**:
  - **v1 setup**：GPT-2 single block prefill seq=128；tiling-only 2×2 SPAD/BW 空间
  - **commit 92623de**：初次判决 NEGATIVE
  - **commit 09b9ac3**：在 10% Gate-1 threshold 下重算，NEGATIVE 保持
  - **判决**：`FUSION_TILING_ONLY_NEGATIVE`
  - **含义**：并不是 co-design 不存在，而是这个 workload / 这个 axis 下不成立 → 需要换 axis

#### Slide 07 - Route 4 Pivot：四个编译器修改维度全景

- **Layout**: Four-column pillars (vertical_pillars); 每列 = 一个维度 + 状态徽章
- **Title**: 四个编译器修改维度：只有 fusion 无源码可切换
- **Visualization**: vertical_pillars — 4 columns
- **Content**:
  - **fusion**：`codegen_compiler_optimization` YAML；修改成本 0；已实测 → **POSITIVE**
  - **dataflow**：主路径固定 WS_MESH，未暴露 OS/IS；修改成本 HIGH；未实测
  - **SPAD partition**：`SPAD_PARTITION_HARDCODED`；只有容量字段；HIGH；未实测
  - **DMA schedule**：`DMA_SCHEDULE_HARDCODED`；未见 prefetch/double_buffer；MEDIUM-HIGH；未实测

#### Slide 08 - Fusion Axis POSITIVE：Conv 4-HW × 2-fusion sweep

- **Layout**: Full-width chart + right-side takeaway box
- **Title**: Conv 3×3 上 fusion × HW 存在 measured interaction
- **Visualization**: grouped_bar_chart — 4 HW categories × 2 series (fusion=none / fusion=all) median cycles
- **Content**:
  - **Workload**：Conv 3×3, shape [1, 64, 56, 56] → [1, 64, 56, 56]
  - **Config**：`pytorchsim_functional_mode=0`, seed=0, 每 cell 5 repeats, harness default mapping
  - **HW-A**：189,837 → 27,533 cycles（speedup **6.895×**）
  - **HW-B**：240,914 → 49,877 cycles（speedup **4.830×**）
  - **HW-C**：163,275 → 26,074 cycles（speedup **6.262×**）
  - **HW-D**：294,609 → 56,080 cycles（speedup **5.253×**）
  - **fusion_speedup_range.relative_range = 42.7%**（超过 15% threshold）
  - **gate2b_ratio_fusion = 0.523** → **`FUSION_CONV_CODESIGN_POSITIVE`**（commit 79e6259）

#### Slide 09 - Conv vs GPT-2：workload dependence

- **Layout**: Full-width bar chart + bottom takeaway box
- **Title**: 同一个 fusion axis，跨 workload 差异悬殊
- **Visualization**: bar_chart — fusion speedup by workload
- **Content**:
  - **addmm+relu 128**：fusion=all vs none = 2.0× 弱信号
  - **GPT-2 block seq=128**：fusion=all vs none = 1.019×（1.9%）→ `NOT_GENERALIZABLE_ON_GPT2`
  - **Conv 3×3 probe**：fusion=all vs none = **6.84×** 强信号
  - **根因（commit ce5f0e3）**：GPT-2 中 fusion 结构上生效（MLIR op count 减 7.8%）但不在 critical path；Conv 3×3 是 compute-bound + 短序列，fusion 直接影响 critical path
  - **结论**：HW/SW co-design 价值不是 universal property，而是 **workload-dependent + axis-dependent**

#### Slide 10 - Scope Limitations

- **Layout**: Icon grid (5 items) — 每个 limit 一张卡片带 warning icon
- **Title**: POSITIVE 声明必须与 5 条 scope limit 一同引用
- **Visualization**: icon_grid
- **Content**:
  - **Single workload**：只在 Conv 3×3 kernel-level workload 上成立
  - **Single mapping**：使用 harness default mapping，无 multi-mapping ranking
  - **Mode=0 timing**：mode=1 correctness 因 Spike `--varch` blocker deferred
  - **Four HW configs**：`codesign_v1_2x2` 四个 corner，未覆盖连续空间
  - **Fusion axis only**：仅 fusion 轴 POSITIVE；其余轴未 demonstrated

---

### Part 4: Lumina QualE 复现主线

#### Slide 11 - QualE 复现目标与 Spec Coding 约束

- **Layout**: Numbered steps (Gate A → B → C → D) 横向流程 + 顶部一句核心问题
- **Title**: 严格 Spec Coding 下复现 QualE 原版 3 规则 validator
- **Visualization**: numbered_steps — 4 Milestone Gates
- **Content**:
  - **核心问题**：在严格 Spec Coding 约束下，R1/R2/R3 是否足以从 GPU simulator 源码提取真实可用的 arch → PPA 因果图？
  - **判据**：≥ 60% edge 通过人工语义抽检（spec §3 Go 阈值）
  - **强制约束**：总代码上限 1000 行（实际 671）、单文件上限 200 行、6 个核心模块 + 3 个 CLI；禁止 provider registry / cache manifest / retry ledger
  - **四 Gate 架构**：Gate A（control）→ Gate B（real LLM）→ Gate C（validator）→ Gate D（verdict）

#### Slide 12 - v1 结果：R1/R2/R3 太松 → 假阳性

- **Layout**: Asymmetric split (3:7); 左边大红 NEGATIVE + KPI（100% vs 20%），右边失败原因
- **Title**: v1：机器 100% pass，人工语义 20%
- **Content**:
  - **机器验证**：86/86 syntactic edges 通过（100%）
  - **人工语义抽检**：pass_ratio = **0.20** — 硬触发 No-Go 阈值（< 0.30）
  - **判决**：`NO_GO_NEGATIVE`（v1 假阳性模式）
  - **根因**：
    - R1（symbol 匹配）太松，DDR 时序常量被误认作 arch knob
    - R2（PPA 白名单）过宽
    - R3（evidence substring）无法区分"提及"与"因果"
  - `m_*` 现代命名规则和老代码库的混合命名让 validator 无法区分

#### Slide 13 - v2 结果：R4 命名 pattern 太紧 → 假阴性

- **Layout**: Asymmetric split (3:7); 左边大红 NEGATIVE + KPI（100% 语义 / 1-of-6 覆盖），右边失败原因
- **Title**: v2：语义 100% pass，覆盖只剩 1/6 packet
- **Content**:
  - **新增 R4**：命名 pattern knob classifier
  - **人工抽检**：11 条 edge 全部 plausible（pass_ratio = **1.00**）
  - **覆盖度**：从 6/6 packet 崩到 **1/6 packet**（AC-C3 硬 Go 条件 ≥ 3 packet 失败）
  - **判决**：`NO_GO_NEGATIVE`（v2 假阴性模式，与 v1 相反）
  - **Double negative 核心结论**：任何单一规则式 validator 都无法通吃异质命名规范的老代码库 → 必须放弃规则式 validator，转向 AST 分析或 LLM 两步 prompt

#### Slide 14 - Gap Analysis：4 大深度错位 & v3 方向

- **Layout**: Matrix 2×2 (matrix_2x2 style) — 4 gaps 分布在 2×2 网格，右下角小 callout 指向 LLMCompass
- **Title**: v1/v2 的 NEGATIVE 不能否定 Lumina QualE — 因为 4 大错位
- **Visualization**: matrix_2x2
- **Content**:
  - **Gap 1 — Simulator**：我们用 gpgpu-sim / Accel-Sim；论文用 **LLMCompass**
  - **Gap 2 — Knob 定义**：我们抓 `m_*` C++ 成员变量；论文抓 LLMCompass 定义的架构参数
  - **Gap 3 — PPA metric**：我们看 simulator 内部 counter；论文看 LLMCompass 输出
  - **Gap 4 — Scope**：我们只做 QualE 单独实验；论文 QualE + QuanE + SE + EE 闭环
  - **v3 方向**：切换到 **LLMCompass** 直接闭合 Gap 1/2/3；Gap 4 是有意识的分阶段 scope

---

### Part 5: 总结

#### Slide 15 - 方法论共同贡献与下一步

- **Layout**: Symmetric split (5:5); 左边"共同方法论"三条，右边"两条主线下一步"
- **Title**: 共同方法论 & 下一步工作
- **Content**:
  - **共同方法论贡献**：
    - Spec Coding：measured 数据为准，禁止 modeled / placeholder 冒充
    - Gate-based verdict：POSITIVE / NEGATIVE / PARTIAL / BLOCKED 四种诚实终态
    - Scope limit 强制引用：POSITIVE 声明必须与限制一同引用
  - **PyTorchSim 下一步**：
    - 扩展 dataflow (300-800 LOC) / SPAD partition / DMA schedule
    - 扩大 workload 覆盖：ResNet / MobileNet Conv workload sweep
    - Multi-mapping ranking 做 Gate-3 interaction
  - **Lumina 复现下一步**：
    - 切换 LLMCompass 作 v3 基础
    - 分阶段扩展到 QuanE → SE / EE 闭环
    - Carry-over v1/v2 的工程流程（spec-first + gate + measured artifact + TDD）

#### Slide 16 - Q&A

- **Layout**: Single column centered; large "Q&A" + 下方三行提示
- **Title**: Q&A
- **Content**:
  - 欢迎讨论 evidence chain 或 methodology 中的任何一个 gate 判决细节
  - 期待建议：如何在 dataflow / SPAD / DMA 轴上跑通 POSITIVE
  - 期待建议：LLMCompass v3 setup 的关键 pitfalls
- **Info**: dyf | 2026-07-04

---

## X. Speaker Notes Requirements

- **Filename**: match SVG name (e.g., `01_cover.md` for `svg_output/01_cover.svg`)
- **Master notes**: `notes/total.md` (with `#` heading per page)
- **Total duration**: ~15 minutes speaking time; each page averages 55-60 seconds
- **Style**: formal, academic-defense tone; conclusion-first per page (Pyramid Principle); include cue lines for evidence chain transitions
- **Purpose**: inform + report (evidence-driven presentation, not persuasion)

---

## XI. Technical Constraints Reminder

### SVG Generation

1. viewBox: `0 0 1280 720`
2. Background uses `<rect>` elements
3. Text wrapping uses `<tspan>` (`<foreignObject>` FORBIDDEN)
4. Transparency uses `fill-opacity` / `stroke-opacity`; `rgba()` FORBIDDEN
5. FORBIDDEN: `mask`, `<style>`, `class`, `foreignObject`, `textPath`, `animate*`, `script`, `@font-face`
6. Text characters: raw Unicode only (— – → © etc.); HTML named entities forbidden
7. Icon uses `<use data-icon="chunk-filled/<name>" .../>` placeholder syntax
8. Every top-level content group has `<g id="...">` to enable per-element animation on export
9. All `<g opacity>` FORBIDDEN — set opacity on each child element individually

### PPT Compatibility

- Font family must resolve to installed families listed in §IV; converter writes only first CJK + first Latin into PPTX
- Chart pages read chart template SVG for coordinate/geometry vocabulary; adapt (not verbatim copy) to fit this deck's content and colors
- POSITIVE / NEGATIVE color badges use success `#27AE60` / warning `#C0392B` for immediate readability
