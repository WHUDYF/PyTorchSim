# Spike --varch diagnostic

## Phase A timebox

本诊断遵守 handoff 的 30 分钟 wall-time 限制，只尝试配置级、环境变量级、PATH 级 workaround；没有修改 PyTorchSim / TOGSim / gem5 / ramulator2 / spike 源码。

## 1. Spike binary and version

命令：

```bash
which spike
spike --version
spike --help 2>&1 | grep -iE "varch|isa" | head -20
type -a spike
```

观察：

```text
which spike -> /usr/bin/spike
spike --version -> spike: unrecognized option --version
spike --help -> Spike RISC-V ISA Simulator 1.1.1-dev
spike --help -> --isa=<name> RISC-V ISA string [default rv64imafdc_zicntr_zihpm]
type -a spike -> spike is /usr/bin/spike
```

结论：当前 PATH 上只有 `/usr/bin/spike`，该版本的 help 输出没有 `--varch` 选项。

## 2. Where --varch is produced

命令：

```bash
rg -n "varch|vlen:256|elen:64" PyTorchSimDevice TOGSim PyTorchSimFrontend Simulator --glob '!**/build/**'
nl -ba Simulator/simulator.py | sed -n '120,180p'
```

关键命中：

```text
Simulator/simulator.py:147:
run = f'spike --isa rv64gcv_zfh --varch=vlen:256,elen:64 {vectorlane_option} ...'
```

结论：`--varch=vlen:256,elen:64` 是 `Simulator/simulator.py` 的 hardcoded Spike command，不是 YAML 字段。

## 3. Config/env controllability

命令：

```bash
rg -n "vlen|elen|varch" configs scripts --glob '!**/__pycache__/**'
rg -n "SPIKE|spike|RISCV|pk|varch|vlen" Simulator PyTorchSimFrontend PyTorchSimDevice configs scripts --glob '!**/build/**'
```

观察：

```text
configs/ 中没有控制 varch/elen 的字段。
scripts/ 中没有用于替换 Spike varch string 的参数。
PyTorchSimFrontend/extension_codecache.py 使用 vlen 生成 MLIR/LLVM lowering 参数，但不控制 Spike --varch。
```

结论：现有 YAML/env 只能影响 `vpu_vector_length_bits`、MLIR lowering 或 PATH；不能在不改源码的前提下删除或改写 Spike `--varch`。

## 4. Alternative Spike binary check

命令：

```bash
find /home/dyf/opt/pytorchsim-riscv-gcc-compat/bin /home/dyf/miniconda3/envs/pytorchsim-build/bin /home/dyf/src /home/dyf/local/bin /usr/local/bin /usr/bin -maxdepth 4 -type f -name spike -executable
```

观察：

```text
/usr/bin/spike
```

结论：没有找到可通过 PATH 调整切换的兼容 Spike binary。

## Phase A conclusion

未找到配置级 workaround。阻塞点是 functional correctness path 需要支持 `--varch=vlen:256,elen:64` 的 Spike，但当前 `/usr/bin/spike` 不支持该选项。根据 handoff，进入 Phase B：`pytorchsim_functional_mode=0` timing-only fallback，并明确记录 correctness deferred。
