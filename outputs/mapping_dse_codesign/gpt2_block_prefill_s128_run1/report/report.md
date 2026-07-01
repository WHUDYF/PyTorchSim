# NPU mapping DSE co-design 实验报告

结论状态: `NEGATIVE`
数据标签: `measured`

## 4x8 cycles 表
| HW | 000 | 001 | 002 | 003 | 004 | 005 | 006 | 007 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HW-A | 2075876 | 1086632 | 766314 | 649730 | 542981 | 646994 | 408102 | 494088 |
| HW-B | 2940911 | 2104776 | 2037595 | 1539974 | 1402284 | 1564848 | 967906 | 1327220 |
| HW-C | 2076508 | 1089905 | 756508 | 660096 | 559079 | 652191 | 404330 | 494035 |
| HW-D | 2913219 | 2106675 | 2014717 | 1475786 | 1383767 | 1578949 | 973626 | 1320905 |

## 4x8 状态表
| 状态 | 数量 |
| --- | --- |
| measured | 32 |
| retry_exhausted | 0 |
| unavailable | 0 |

## migrating_pair_count/6 摘要
migrating_pair_count = 0/6；Gate-1 passed = False。
| 硬件对 | 状态 | champion | swap_dominance | 通过 |
| --- | --- | --- | --- | --- |
| HW-A/HW-B | same_champion | HW-A:006, HW-B:006 | 0 |  |
| HW-A/HW-C | same_champion | HW-A:006, HW-C:006 | 0 |  |
| HW-A/HW-D | same_champion | HW-A:006, HW-D:006 | 0 |  |
| HW-B/HW-C | same_champion | HW-B:006, HW-C:006 | 0 |  |
| HW-B/HW-D | same_champion | HW-B:006, HW-D:006 | 0 |  |
| HW-C/HW-D | same_champion | HW-C:006, HW-D:006 | 0 |  |

## per-HW ratio 表
Gate-2a ratio mean = 1；meeting threshold = 0。
| HW | per-HW ratio |
| --- | --- |
| HW-A | 1 |
| HW-B | 1 |
| HW-C | 1 |
| HW-D | 1 |

## unconstrained vs constrained oracle 对比
fit-all-HW mapping set = ['000', '001', '002', '003', '004', '005', '006', '007']；single-best-avg mapping = 006。
| HW | constrained mapping | constrained cycles | unconstrained mapping | unconstrained cycles |
| --- | --- | --- | --- | --- |
| HW-A | 006 | 408102 | 006 | 408102 |
| HW-B | 006 | 967906 | 006 | 967906 |
| HW-C | 006 | 404330 | 006 | 404330 |
| HW-D | 006 | 973626 | 006 | 973626 |

## Gate-3 与 Gate-1 联合解读
Gate-3 passed = False；mean Spearman = 0.984127；numeric pair count = 6。
Gate-1 观察 champion 是否迁移，Gate-3 观察 mapping 排序在不同硬件之间是否保持一致。若二者同时成立，说明硬件参数变化不仅改变最优点，也改变了搜索空间的相对排序。

## cost-matched pair 分析
pair = ['HW-B', 'HW-C']；ratio = 0.417737；Gate-2b passed = True。
framing = ex-post cost-tier allocation sensitivity upper bound。

## 反 baseline (HW-D) 参考值
| mapping | cycles |
| --- | --- |
| 000 | 2913219 |
| 001 | 2106675 |
| 002 | 2014717 |
| 003 | 1475786 |
| 004 | 1383767 |
| 005 | 1578949 |
| 006 | 973626 |
| 007 | 1320905 |

## 实验边界
- 当前结论限定在单一 GPT-2 block prefill seq=128 类工作负载。
- 搜索空间只覆盖 tile/mapping 选择，不覆盖 fusion、dataflow、SPAD partition 或 DMA schedule。
- 硬件空间只覆盖 2x2 corner 组合，不能证明内部参数空间单调或最优。
- DRAM channel 与 ICNT port 在本实验中绑定变化，不能拆分解释二者的独立贡献。
- 当前指标是 cycle-only，不包含面积、功耗、能耗、带宽成本或可布线性。
