# Related Papers

This directory contains publicly available papers related to PyTorchSim and the simulator/modeling tools referenced by the project README.

## PyTorchSim paper

The main PyTorchSim paper is:

- **PyTorchSim: A Comprehensive, Fast, and Accurate NPU Simulation Framework**. MICRO 2025. DOI: <https://doi.org/10.1145/3725843.3756045>

The ACM DOI page lists the paper, but direct PDF download from `dl.acm.org` returned HTTP 403 in this environment. Use the DOI link above to access the publisher page.

## Downloaded PDFs

### Hardware/software co-design and compiler search

The files below are publicly accessible author-hosted or arXiv versions. A venue
name describes the published or accepted work; it does not imply that the local
copy is the publisher's version of record. Helios is a 2026 preprint.

| File | Paper | Relevance | Source |
|---|---|---|---|
| [hasco-arxiv2021.pdf](hasco-arxiv2021.pdf) | HASCO: Towards Agile HArdware and Software CO-design for Tensor Computation | Joint hardware architecture and tensor-program mapping search | <https://arxiv.org/abs/2105.01585> |
| [tlm-osdi2024.pdf](tlm-osdi2024.pdf) | Enabling Tensor Language Model to Assist in Generating High-Performance Tensor Programs for Deep Learning | Learned tensor-program schedule generation for fixed hardware | <https://www.usenix.org/conference/osdi24/presentation/zhai> |
| [autocomp-arxiv2025.pdf](autocomp-arxiv2025.pdf) | Autocomp: A Powerful and Portable Code Optimizer for Tensor Accelerators | LLM-guided low-level accelerator-code optimization with correctness and hardware feedback | <https://arxiv.org/abs/2505.18574> |
| [senna-tecs2025.pdf](senna-tecs2025.pdf) | SENNA: Unified Hardware/Software Space Exploration for Parametrizable Neural Network Accelerators | Joint network-wide search over accelerator configurations and software execution plans | <https://csap.snu.ac.kr/sites/default/files/papers/2025.TECS.Kwon.SENNA.Unified%20Hardware-Software%20Space%20Exploration%20for%20Parametrizable%20Neural%20Network%20Accelerators.pdf> |
| [looptree-tcasai2024.pdf](looptree-tcasai2024.pdf) | LoopTree: Exploring the Fused-Layer Dataflow Accelerator Design Space | Fused-layer tiling, scheduling, retention, recomputation, and buffer-capacity exploration | <https://people.csail.mit.edu/emer/media/papers/2024.09.tcas-ai.looptree.pdf> |
| [mas-attention-mlsys2025.pdf](mas-attention-mlsys2025.pdf) | MAS-Attention: Memory-Aware Stream Processing for Attention Acceleration on Resource-Constrained Edge Devices | MAC/VEC pipelining, multi-tier tiling, and on-chip memory management | <https://arxiv.org/abs/2411.17720> |
| [fusemax-micro2024.pdf](fusemax-micro2024.pdf) | FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design | Deep attention fusion and fine-grained binding across 1D and 2D PE arrays | <https://people.csail.mit.edu/emer/media/papers/2024.10.micro.fusemax.pdf> |
| [softex-jetcas2025.pdf](softex-jetcas2025.pdf) | A Flexible Template for Edge Generative AI with High-Accuracy Accelerated Softmax & GELU | Co-design of nonlinear approximations, systolic MatMul, and dedicated Softmax/GELU hardware | <https://arxiv.org/abs/2412.06321> |
| [edgellm-tcasi2025.pdf](edgellm-tcasi2025.pdf) | EdgeLLM: A Highly Efficient CPU-FPGA Heterogeneous Edge Accelerator for Large Language Models | Mixed-precision and structured-sparsity hardware paired with end-to-end compiler mapping | <https://arxiv.org/abs/2407.21325> |
| [facil-hpca2025.pdf](facil-hpca2025.pdf) | FACIL: Flexible DRAM Address Mapping for SoC-PIM Cooperative On-device LLM Inference | Memory-controller address mappings paired with software allocation and tensor layout | <https://99dhl.github.io/assets/pdf/hpca25_facil.pdf> |
| [helios-arxiv2026.pdf](helios-arxiv2026.pdf) | Hardware-Software Co-design for 3D-DRAM-based LLM Serving Accelerator (Helios) | Distributed NMP execution and communication paired with spatially aware KV-cache allocation | <https://arxiv.org/abs/2603.04797> |

| File | Paper | Relevance | Source |
|---|---|---|---|
| [accel-sim-isca2020.pdf](accel-sim-isca2020.pdf) | Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling | GPU simulator baseline referenced by PyTorchSim | <https://people.ece.ubc.ca/aamodt/publications/papers/accelsim.isca2020.pdf> |
| [aladdin-isca2014.pdf](aladdin-isca2014.pdf) | Aladdin: A Pre-RTL, Power-Performance Accelerator Simulator | Accelerator simulation foundation | <https://people.eecs.berkeley.edu/~ysshao/assets/papers/shao2014-isca.pdf> |
| [astra-sim2-arxiv2023.pdf](astra-sim2-arxiv2023.pdf) | ASTRA-sim 2.0 | Distributed deep learning system simulation | <https://arxiv.org/pdf/2303.14006> |
| [booksim-ispass2013.pdf](booksim-ispass2013.pdf) | BookSim 2.0 | Network-on-chip simulator used by TOGSim | <https://icn.kaist.ac.kr/~jjk12/papers/2013ISPASS.pdf> |
| [eyeriss-isca2016.pdf](eyeriss-isca2016.pdf) | Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for CNNs | DNN accelerator dataflow and architecture | <https://eems.mit.edu/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf> |
| [gem5-arxiv2020.pdf](gem5-arxiv2020.pdf) | The gem5 Simulator: Version 20.0+ | Full-system simulation foundation used by PyTorchSim | <https://arxiv.org/pdf/2007.03152> |
| [maeri-asplos2018.pdf](maeri-asplos2018.pdf) | MAERI: Enabling Flexible Dataflow Mapping over DNN Accelerators | Flexible dataflow mapping for DNN accelerators | <https://anands09.github.io/papers/maeri_asplos2018.pdf> |
| [maestro-micro2019.pdf](maestro-micro2019.pdf) | MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings | Mapping/performance model baseline referenced by PyTorchSim | <https://d1qx31qr3h6wln.cloudfront.net/publications/MICRO_2019_Maestro.pdf> |
| [mlir-arxiv2020.pdf](mlir-arxiv2020.pdf) | MLIR: A Compiler Infrastructure for the End of Moore's Law | Compiler infrastructure related to PyTorchSim code generation | <https://arxiv.org/pdf/2002.11054> |
| [mnpusim-iiswc2023.pdf](mnpusim-iiswc2023.pdf) | mNPUsim: Evaluating the Effect of Sharing Resources in Multi-Core NPUs | Multi-core NPU simulator baseline referenced by PyTorchSim | <https://jaehyuk-huh.github.io/papers/iiswc2023_multi_npu_sim.pdf> |
| [pytorch2-asplos2024.pdf](pytorch2-asplos2024.pdf) | PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation | PyTorch compiler stack used by PyTorchSim | <https://docs.pytorch.org/assets/pytorch2-2.pdf> |
| [ramulator2-arxiv2023.pdf](ramulator2-arxiv2023.pdf) | Ramulator 2.0 | DRAM simulator used by TOGSim | <https://arxiv.org/pdf/2308.11030> |
| [scale-sim-v3-ispass2025.pdf](scale-sim-v3-ispass2025.pdf) | SCALE-Sim v3 | Systolic-array simulator baseline referenced by PyTorchSim | <https://arxiv.org/pdf/2504.15377> |
| [sigma-hpca2020.pdf](sigma-hpca2020.pdf) | SIGMA: A Sparse and Irregular GEMM Accelerator with Flexible Interconnects | Sparse/irregular GEMM accelerator design | <https://anands09.github.io/papers/sigma_hpca2020.pdf> |
| [smaug-taco2020.pdf](smaug-taco2020.pdf) | SMAUG: End-to-End Full-Stack Simulation Infrastructure for Deep Learning Workloads | Full-stack DL workload simulation | <https://www.samxi.org/papers/xi_smaug_taco2020.pdf> |
| [stonne-iiswc2021.pdf](stonne-iiswc2021.pdf) | STONNE: Enabling Cycle-Level Microarchitectural Simulation for DNN Inference Accelerators | Cycle-level DNN accelerator simulation | <https://digitum.um.es/bitstreams/656ddc04-8d2b-4de6-addb-1eeac0d6d5b0/download> |
| [timeloop-ispass2019.pdf](timeloop-ispass2019.pdf) | Timeloop: A Systematic Approach to DNN Accelerator Evaluation | Mapping/performance model baseline referenced by PyTorchSim | <https://accelergy.mit.edu/timeloop.pdf> |
| [tpu-isca2017.pdf](tpu-isca2017.pdf) | In-Datacenter Performance Analysis of a Tensor Processing Unit | TPU baseline architecture modeled by PyTorchSim | <https://arxiv.org/pdf/1704.04760> |
| [tpuv4-isca2023.pdf](tpuv4-isca2023.pdf) | TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings | TPU generation related to PyTorchSim L2/CMEM examples | <https://arxiv.org/pdf/2304.01433> |
