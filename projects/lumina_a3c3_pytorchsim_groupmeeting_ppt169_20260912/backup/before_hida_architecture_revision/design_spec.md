# Lumina × A3C3 × PyTorchSim：从昂贵 DSE 到可验证协同设计

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | lumina_a3c3_pytorchsim_groupmeeting |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 16 pages |
| **Design Style** | General Consulting + academic defense + dark technical editorial |
| **Target Audience** | AI systems / architecture research group members |
| **Use Case** | Group meeting presentation and research direction discussion |
| **Created Date** | 2026-09-12 |

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280×720 px |
| **viewBox** | `0 0 1280 720` |
| **Margins** | 56 px left/right; 44 px top/bottom |
| **Content Area** | x=56..1224, y=44..676 |

## III. Visual Theme

### Theme Style

- **Style**: dark technical editorial, research briefing, causal diagrams and controlled comparison
- **Theme**: dark
- **Tone**: analytical, rigorous, forward-looking, practical

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#0B1220` | Main page background |
| **Secondary bg** | `#162235` | Cards, panels, code blocks |
| **Primary** | `#4DA3FF` | Structure, titles, compiler and hardware paths |
| **Accent** | `#36D6B5` | Measured evidence, positive signals, next-step actions |
| **Secondary accent** | `#A78BFA` | Algorithm and A3C3 layer |
| **Warning** | `#FFB454` | Bottlenecks, limitations, open risks |
| **Body text** | `#F4F7FB` | Main text |
| **Secondary text** | `#AAB8CC` | Explanations and labels |
| **Tertiary text** | `#718096` | Footnotes and source labels |
| **Border/divider** | `#2A3A52` | Panel borders and separators |
| **Success** | `#36D6B5` | Positive / measured / verified |
| **Failure** | `#F26D85` | Negative / blocked / caution |

### Gradient Scheme

```xml
<linearGradient id="heroGradient" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#4DA3FF" stop-opacity="0.90"/>
  <stop offset="100%" stop-color="#A78BFA" stop-opacity="0.75"/>
</linearGradient>
```

## IV. Typography System

### Font Plan

Typography direction: CJK-primary technical briefing with a restrained serif accent for English paper names and key formulas.

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | Microsoft YaHei | Georgia | serif |
| **Body** | Microsoft YaHei | Arial | sans-serif |
| **Emphasis** | Microsoft YaHei | Georgia | serif |
| **Code** | — | Consolas | monospace |

**Per-role font stacks**:

- Title: `Georgia, "Microsoft YaHei", serif`
- Body: `"Microsoft YaHei", Arial, sans-serif`
- Emphasis: `Georgia, "Microsoft YaHei", serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 18px.

| Purpose | Ratio to body | Size |
| ------- | ------------- | ---- |
| Cover title | 4x | 72px |
| Chapter opener | 2.4x | 43px |
| Page title | 1.9x | 34px |
| Hero number | 2.4x | 43px |
| Subtitle | 1.35x | 24px |
| Body content | 1x | 18px |
| Annotation / caption | 0.78x | 14px |
| Page number / footnote | 0.6x | 11px |

## V. Layout Principles

### Page Structure

- **Header area**: y=44..118; page title, section label, page number.
- **Content area**: y=134..640; diagrams, comparison tables, code-like blocks.
- **Footer area**: y=650..676; source labels and short takeaway.

### Layout Pattern Library

- Cover: negative-space-driven with a right-side causal network.
- Chapter openers: asymmetric split with large section number.
- Technical pages: asymmetric split, process flow, center-radiating causal map, and editorial table.
- Comparison pages: 2-column or 3-column matrix with one highlighted conclusion strip.
- Roadmap pages: left-to-right pipeline with measured gates.

### Spacing Specification

| Element | Current Project |
| ------- | --------------- |
| Safe margin | 56px horizontal, 44px vertical |
| Content block gap | 24px |
| Icon-text gap | 12px |
| Card gap | 18px |
| Card padding | 20px |
| Card border radius | 12px |
| Divider width | 1px |

## VI. Icon Usage Specification

- **Source**: built-in icon library
- **Library**: `chunk-filled`
- **Usage method**: `<use data-icon="chunk-filled/<name>" .../>`

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| Search / exploration | `chunk-filled/compass` | P03, P16 |
| Bottleneck / target | `chunk-filled/target` | P04, P06, P15 |
| Code / simulator | `chunk-filled/code` | P04, P15 |
| Sensitivity / chart | `chunk-filled/chart-line` | P05, P06 |
| Hardware / compute | `chunk-filled/bolt` | P08, P11, P15 |
| Memory / data | `chunk-filled/database` | P06, P13, P15 |
| Algorithm / idea | `chunk-filled/lightbulb` | P09, P12 |
| Workflow / layers | `chunk-filled/layers` | P02, P08, P14 |
| Direction / next step | `chunk-filled/arrow-right` | P07, P14, P16 |
| Positive result | `chunk-filled/checkmark` | P07, P16 |
| Warning / limitation | `chunk-filled/triangle-exclamation` | P07, P14 |
| Network / graph | `chunk-filled/git-branch` | P02, P10, P15 |
| Efficiency | `chunk-filled/bolt` | P03, P06, P16 |
| Design / compass | `chunk-filled/compass-drafting` | P01, P16 |

## VII. Visualization Reference List

This deck uses custom SVG diagrams rather than adapting a catalog chart template. There are fewer than three data-visualization pages; all quantitative bars and matrices are schematic or directly labeled from the cited papers and are not intended as a new benchmark claim.

| Page | Template | Path | Summary-quote | Usage |
| ---- | -------- | ---- | ------------- | ----- |
| P05 | no-template-match | — | — | Custom sensitivity ladder for QualE → QuanE |
| P14 | no-template-match | — | — | Custom comparison matrix |
| P16 | no-template-match | — | — | Custom staged roadmap |

## VIII. Image Resource List

No external or AI-generated images. All visuals are native SVG shapes, icons, arrows, matrices, and code-like text blocks.

## IX. Content Outline

### Part 1: Why the problem matters

#### Slide 01 - Cover

- **Layout**: Negative-space-driven cover with causal graph on right.
- **Title**: 从 Lumina 到 PyTorchSim
- **Subtitle**: 如何把昂贵的硬件 DSE 推进为可验证的 compiler–accelerator co-design
- **Info**: 组会讨论 · 2026-09-12

#### Slide 02 - The research tension

- **Layout**: Left problem statement, right three-layer stack.
- **Title**: 我们真正缺的不是更多配置，而是更聪明的实验闭环
- **Content**: Large DSE space, expensive TOGSim, compiler mapping hidden inside hardware results.

### Part 2: Lumina

#### Slide 03 - Lumina in one picture

- **Layout**: Center-radiating loop.
- **Title**: Lumina：用瓶颈分析决定下一次模拟
- **Content**: QualE, QuanE, SE, EE, Trajectory Memory, simulator feedback.

#### Slide 04 - QualE

- **Layout**: Source-code to influence-map flow.
- **Title**: QualE：先从 simulator 源码建立因果边界
- **Content**: Which parameter can affect which metric; structural pruning before sampling.

#### Slide 05 - QuanE

- **Layout**: Sensitivity ladder with measured perturbation examples.
- **Title**: QuanE：用小实验把“可能相关”变成“影响强度”
- **Content**: ± perturbation, microbenchmarks, local influence priors, low-cost area/power fallback.

#### Slide 06 - SE / EE / TM

- **Layout**: Left-to-right closed loop.
- **Title**: SE–EE–TM：每一次模拟都应该改变下一次选择
- **Content**: Critical path → constrained adjustment → simulator → trajectory memory → refined knowledge.

#### Slide 07 - Lumina’s boundary

- **Layout**: Two-column “strength / boundary” page.
- **Title**: Lumina 的强项是样本效率，边界是搜索对象较窄
- **Content**: Fixed workload and hardware DSE; does not invent new network topology or compiler mapping by itself.

### Part 3: A3C3 case studies

#### Slide 08 - A3C3 map

- **Layout**: Three-axis design-space map.
- **Title**: A3C3：把算法、实现、硬件放进同一张设计地图
- **Content**: A / I / H axes; bundle library → co-search → co-generation.

#### Slide 09 - SkyNet

- **Layout**: Bundle library to PSO search flow.
- **Title**: SkyNet：搜索“什么网络适合硬件”
- **Content**: Bundle pre-evaluation, channel expansion, pooling position, accuracy–latency fitness.

#### Slide 10 - EDD

- **Layout**: Unified differentiable space.
- **Title**: EDD：把网络变量和实现变量联合连续化
- **Content**: architecture θ, implementation φ, quantization q, hardware-aware loss.

#### Slide 11 - HIDA / fixed-topology implementation

- **Layout**: Hierarchical IR stack.
- **Title**: HIDA：固定程序时，核心问题变成“如何映射得更好”
- **Content**: Functional IR, structural IR, tiling, parallelism, buffers, streams.

#### Slide 12 - Medusa

- **Layout**: Sequential decoding versus tree verification.
- **Title**: Medusa：不是换芯片，而是重构算法的执行方式
- **Content**: Multiple heads, speculative candidates, tree attention, fewer decoding steps.

#### Slide 13 - SnapKV

- **Layout**: Full KV cache versus selected/compressed cache.
- **Title**: SnapKV：把内存保留策略也纳入协同设计
- **Content**: Observation window, head-specific important positions, memory and latency reduction.

#### Slide 14 - Compare the paradigms

- **Layout**: Comparison matrix.
- **Title**: 这些工作优化的对象不同，但都在回答“瓶颈在哪里”
- **Content**: SkyNet / EDD / HIDA / Medusa / SnapKV / Lumina comparison.

### Part 4: PyTorchSim proposal

#### Slide 15 - Our formulation

- **Layout**: Fixed topology plus compiler/hardware axes.
- **Title**: 我们的切入点：固定算法拓扑，联合搜索 I × H
- **Content**: A=A₀; I={tiling, fusion, dataflow, DMA}; H={SPAD, DRAM, NoC, PE}; TOG as bridge.

#### Slide 16 - Next design

- **Layout**: Staged roadmap with gates.
- **Title**: 下一步：从全量 sweep 转向 TOG-aware、bottleneck-driven DSE
- **Content**: Influence Map → sensitivity motifs → surrogate ranking → measured validation → Pareto / co-design verdict.

## X. Speaker Notes Requirements

Speaker notes will be generated as one spoken Chinese narrative per page, with natural transitions and without meta labels. Each page will have 2–5 sentences and explain the methodological distinction before introducing the PyTorchSim implication.

## XI. Technical Constraints Reminder

- SVG viewBox: `0 0 1280 720`.
- Use background `<rect>` and inline attributes only.
- Use raw Unicode punctuation; escape XML reserved characters.
- No `mask`, `<style>`, `class`, `<foreignObject>`, `textPath`, `<script>`, `<animate*>`, `rgba()`, or group opacity.
- Icons use only `chunk-filled` inventory listed above.
- No external images are referenced.
