# NPU mapping DSE co-design 实验报告

结论状态: `NEGATIVE`
数据标签: `measured`

## cycles 表
| HW | 000 | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HW-A | 1000 | 1010 | 1020 | 1030 | 1040 | 1050 | 1060 | 1070 | 1080 | 1090 |
| HW-B | 1100 | 1110 | 1120 | 1130 | 1140 | 1150 | 1160 | 1170 | 1180 | 1190 |
| HW-C | 1200 | 1210 | 1220 | 1230 | 1240 | 1250 | 1260 | 1270 | 1280 | 1290 |
| HW-D | 1300 | 1310 | 1320 | 1330 | 1340 | 1350 | 1360 | 1370 | 1380 | 1390 |

## 状态表
| 状态 | 数量 |
| --- | --- |
| measured | 40 |
| retry_exhausted | 0 |
| unavailable | 0 |

## migrating_pair_count/6 摘要
migrating_pair_count = 0/6；Gate-1 passed = False。
| 硬件对 | 状态 | champion | swap_dominance | 通过 |
| --- | --- | --- | --- | --- |
| HW-A/HW-B | same_champion | HW-A:000, HW-B:000 | 0 |  |
| HW-A/HW-C | same_champion | HW-A:000, HW-C:000 | 0 |  |
| HW-A/HW-D | same_champion | HW-A:000, HW-D:000 | 0 |  |
| HW-B/HW-C | same_champion | HW-B:000, HW-C:000 | 0 |  |
| HW-B/HW-D | same_champion | HW-B:000, HW-D:000 | 0 |  |
| HW-C/HW-D | same_champion | HW-C:000, HW-D:000 | 0 |  |

## per-HW ratio 表
Gate-2a ratio mean = 1；meeting threshold = 0。
| HW | per-HW ratio |
| --- | --- |
| HW-A | 1 |
| HW-B | 1 |
| HW-C | 1 |
| HW-D | 1 |

## unconstrained vs constrained oracle 对比
fit-all-HW mapping set = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009']；single-best-avg mapping = 000。
| HW | constrained mapping | constrained cycles | unconstrained mapping | unconstrained cycles |
| --- | --- | --- | --- | --- |
| HW-A | 000 | 1000 | 000 | 1000 |
| HW-B | 000 | 1100 | 000 | 1100 |
| HW-C | 000 | 1200 | 000 | 1200 |
| HW-D | 000 | 1300 | 000 | 1300 |

## Gate-3 与 Gate-1 联合解读
Gate-3 passed = False；mean Spearman = 1；numeric pair count = 6。
Gate-1 观察 champion 是否迁移，Gate-3 观察 mapping 排序在不同硬件之间是否保持一致。若二者同时成立，说明硬件参数变化不仅改变最优点，也改变了搜索空间的相对排序。

## cost-matched pair 分析
pair = ['HW-B', 'HW-C']；ratio = 0.916667；Gate-2b passed = False。
framing = ex-post cost-tier allocation sensitivity upper bound。

## v1-v2 cross check
cross_check_status = PASS；max_allowed_delta = 0.02。
| cell | v1 median | v2 median | cycle_delta |
| --- | --- | --- | --- |
| HW-A/000 | 1000 | 1000 | 0 |
| HW-A/001 | 1010 | 1010 | 0 |
| HW-A/002 | 1020 | 1020 | 0 |
| HW-A/003 | 1030 | 1030 | 0 |
| HW-A/004 | 1040 | 1040 | 0 |
| HW-A/005 | 1050 | 1050 | 0 |
| HW-A/006 | 1060 | 1060 | 0 |
| HW-A/007 | 1070 | 1070 | 0 |
| HW-B/000 | 1100 | 1100 | 0 |
| HW-B/001 | 1110 | 1110 | 0 |
| HW-B/002 | 1120 | 1120 | 0 |
| HW-B/003 | 1130 | 1130 | 0 |
| HW-B/004 | 1140 | 1140 | 0 |
| HW-B/005 | 1150 | 1150 | 0 |
| HW-B/006 | 1160 | 1160 | 0 |
| HW-B/007 | 1170 | 1170 | 0 |

## 反 baseline (HW-D) 参考值
| mapping | cycles |
| --- | --- |
| 000 | 1300 |
| 001 | 1310 |
| 002 | 1320 |
| 003 | 1330 |
| 004 | 1340 |
| 005 | 1350 |
| 006 | 1360 |
| 007 | 1370 |
| 008 | 1380 |
| 009 | 1390 |

## 实验边界
- 当前结论限定在单一 GPT-2 block prefill seq=128 类工作负载。
- 搜索空间只覆盖 tile/mapping 选择，不覆盖 fusion、dataflow、SPAD partition 或 DMA schedule。
- 硬件空间只覆盖 2x2 corner 组合，不能证明内部参数空间单调或最优。
- DRAM channel 与 ICNT port 在本实验中绑定变化，不能拆分解释二者的独立贡献。
- 当前指标是 cycle-only，不包含面积、功耗、能耗、带宽成本或可布线性。
