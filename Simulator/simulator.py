import os
import shlex
import ctypes
import subprocess
import re
import sys
import yaml
import time
import datetime
import threading
from pathlib import Path
import uuid

import torch
import numpy as np

from PyTorchSimFrontend.mlir.mlir_common import MLIRKernelArgs
from PyTorchSimFrontend import extension_config

print_lock = threading.Lock()

TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.uint8,
    torch.bfloat16: np.float16,
}

class FunctionalSimulator():
    def __init__(self, path, key):
        self.path = path
        self.key = key

    def load_tensor(self, arg, arg_name, arg_attribute, path):
        # path = os.path.join(dump_path, arg_name, f'{n_call}.raw')
        with open(path, 'rb') as f:
            np_array = np.fromfile(f, dtype=TORCH_TO_NUMPY[arg.dtype])
            src_tensor = torch.as_strided(torch.from_numpy(np_array), arg.size(), arg.stride())
            arg.copy_(src_tensor.to(dtype=arg.dtype))

    def get_biggest_filename(self, path):
        return len(os.listdir(path))

    def write_arg(self, arg, path, name):
        dump_path = os.path.join(path, name)
        os.makedirs(dump_path, exist_ok=True)
        index = self.get_biggest_filename(dump_path)

        if (isinstance(arg, torch.Tensor)):
            data_path = os.path.join(dump_path, f'{index}.raw')
            tensor = arg.cpu().detach()
            buffer_size = tensor.untyped_storage().size()
            buffer = (ctypes.c_char * buffer_size).from_address(tensor.data_ptr())
            t_arr = np.frombuffer(buffer, dtype=TORCH_TO_NUMPY[tensor.dtype], count=buffer_size // tensor.element_size())
            t_arr.tofile(data_path)
        else:
            assert(0)
        return index

    def dump_args(self, args, arg_attributes, load_path, dump_path):
        array_size = []
        file_path = []
        for (arg_name, arg_attribute), arg in zip(arg_attributes, args):
            size = arg_attribute[2] if arg_attribute[1] != torch.bool else (arg_attribute[2] + 7) // 8
            array_size.append(size)
            if MLIRKernelArgs.is_mlir_arg_in(arg_attribute[0]):
                index = self.write_arg(arg, load_path, arg_name)
                file_path.append(os.path.join(load_path, arg_name, f'{index}.raw'))
            elif MLIRKernelArgs.is_mlir_arg_out(arg_attribute[0]):
                path = os.path.join(dump_path, arg_name)
                os.makedirs(path, exist_ok=True)
                file_path.append(os.path.join(path, f'{self.get_biggest_filename(path)}.raw'))

        return array_size, file_path

    def run_spike(self, args, arg_attributes, runtime_path, binary, vectorlane_size=4, spad_info=None, cleanup=False, silent_mode=False):
        load_path = runtime_path
        dump_path = runtime_path

        target_binary = os.path.join(self.path, binary)
        objdump = f"riscv64-unknown-elf-objdump -d {target_binary} > {os.path.join(self.path, 'binary.dump')}"
        kernel_start = f"nm {target_binary} | grep 'kernel' | awk 'NR==1 {{print $1}}'"
        kernel_end = f"nm {target_binary} | grep 'kernel' | awk 'NR==1 {{print $1}}' | xargs -I {{}} awk '/{{}}/,0' {os.path.join(self.path, 'binary.dump')} | grep ret | awk 'NR==1 {{print $1}}' | awk '{{gsub(/:$/, \"\"); print}}'"

        subprocess.run(objdump, shell=True)
        kernel_start_addr = subprocess.run(kernel_start, shell=True, stdout=subprocess.PIPE).stdout.strip().decode('utf-8')
        kernel_end_addr = subprocess.run(kernel_end, shell=True, stdout=subprocess.PIPE).stdout.strip().decode('utf-8')

        _, file_path = self.dump_args(args, arg_attributes, load_path, dump_path)
        file_path_str = ' '.join(file_path)

        # Set hardware information
        spad_option = f"-m0x{0x80000000:x}:0x{100<<30:x},0x{spad_info['spad_paddr']:x}:0x{spad_info['spad_size']*vectorlane_size:x} " + \
            f"--scratchpad-base-paddr={spad_info['spad_paddr']} " + \
            f"--scratchpad-base-vaddr={spad_info['spad_vaddr']} " + \
            f"--scratchpad-size={spad_info['spad_size']} "
        vectorlane_option = f"--vectorlane-size={vectorlane_size}"
        kernel_address = f"--kernel-addr={kernel_start_addr}:{kernel_end_addr}"
        base_path= f"--base-path={runtime_path}"
        os.makedirs(os.path.join(runtime_path, "indirect_access"), exist_ok=True)
        os.makedirs(os.path.join(runtime_path, "dma_access"), exist_ok=True)
        run = f'spike --isa rv64gcv --varch=vlen:256,elen:64 {vectorlane_option} {spad_option} {kernel_address} {base_path} /workspace/riscv-pk/build/pk {target_binary} {file_path_str}'
        if not silent_mode and extension_config.CONFIG_DEBUG_MODE:
            print("[Spike] cmd> ", run)
        print("[Spike] Running Spike simulator")
        run_cmd = shlex.split(run)
        try:
            stdout_setting = subprocess.DEVNULL if silent_mode else None
            stderr_setting = subprocess.DEVNULL if silent_mode else None
            subprocess.check_call(run_cmd, stdout=stdout_setting, stderr=stderr_setting)
        except subprocess.CalledProcessError as e:
            if not silent_mode:
                print("[Spike] Command failed with exit code", e.returncode)
            error_msg = ""
            if e.returncode == 200:
                error_msg = "INVALID_SPAD_ACCESS"
            elif e.returncode == 201:
                error_msg = "STACK_OVERFLOW"
            else:
                error_msg = "UNKNOWN_ERROR"
            raise RuntimeError(f"{error_msg}")

        for (arg_name, arg_attribute), arg, path in zip(arg_attributes, args, file_path):
            if MLIRKernelArgs.is_mlir_arg_out(arg_attribute[0]):
                self.load_tensor(arg, arg_name, arg_attribute, path)

        if cleanup:
            for path in file_path:
                if os.path.exists(path):
                    os.remove(path)

    @staticmethod
    def get_runtime_dump_path(base_path, prefix="runtime", zfill=4):
        indices = [
            int(match.group(1))
            for d in os.listdir(base_path)
            if (match := re.fullmatch(rf"{prefix}_(\d{{{zfill}}})", d))
        ]

        max_index = max(indices, default=-1)
        next_index = max_index + 1
        folder_name = f"{prefix}_{str(next_index).zfill(zfill)}"
        full_path = os.path.join(base_path, folder_name)

        os.makedirs(full_path)
        return full_path

class CycleSimulator():
    def __init__(self) -> None:
        pass

    def compile_and_simulate(self, target_binary, array_size, vectorlane_size, silent_mode=False):
        def show_progress():
            i = 0
            while not finished:
                i = (i + 1) % 3
                tail = "." * i + " " * (3-i)
                with print_lock:
                    sys.stdout.write("\r[Gem5] Gem5 is running." + tail)
                    sys.stdout.flush()
                time.sleep(1)
            with print_lock:
                print("")

        dir_path = os.path.join(os.path.dirname(target_binary), "m5out")
        gem5_script_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "gem5_script/script_systolic.py")
        gem5_cmd = [extension_config.CONFIG_GEM5_PATH, "-r", "--stdout-file=sto.log", "-d", dir_path, gem5_script_path, "-c", target_binary, "--vlane", str(vectorlane_size)]
        try:
            # Create progress thread
            is_dryrun = int(os.environ.get('TOGSIM_EAGER_MODE', default=False)) or silent_mode
            if not is_dryrun:
                if extension_config.CONFIG_DEBUG_MODE:
                    print("[Gem5] cmd> ", " ".join(gem5_cmd))
                finished = False
                progress_thread = threading.Thread(target=show_progress)
                progress_thread.start()
                output = subprocess.check_output(gem5_cmd, stderr=subprocess.DEVNULL)
                finished = True
                progress_thread.join()
            else:
                output = subprocess.check_output(gem5_cmd, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"[Gem5] Gem5 simulation failed with error: \"{e.output.decode()}\"")
            if not is_dryrun:
                finished = True
                progress_thread.join()
            raise RuntimeError(f"Gem5 Simulation Failed: \"{e.output.decode()}\"")

        with open(f"{dir_path}/stats.txt", "r") as stat_file:
            raw_list = stat_file.readlines()
            cycle_per_tick = [int(line.split()[1]) for line in raw_list if "system.clk_domain.clock" in line][0]
            cycle_list = [int(line.split()[1]) for line in raw_list if "system.cpu.numCycles" in line]
        cycle_list = cycle_list[:-1]
        return cycle_list

class TOGSimulator():
    TOGSIM_RESULT_PATH_KEY = "TOGSIM_RESULT_PATH"
    FINISH_STR = "Simulation finished"
    ALLOC_POOL = dict() # For eagermode buffer plan
    def __init__(self, togsim_path, config_path, vectorlane_size=-1) -> None:
        self.base_dir = togsim_path
        self.config_path = config_path
        self.config_yaml = self.load_yaml(self.config_path)
        self.process = None
        self.vectorlane_size = vectorlane_size

    def get_togsim_command(self):
        bin = os.path.join(self.base_dir, "build/bin/Simulator")
        config = os.path.join(self.base_dir, self.config_path)
        cmd = f"{bin} --config {config}"
        return cmd

    def simulation(self, model_path, attribute_path="", silent_mode=False, autotune_mode=False):
        def show_progress():
            i = 0
            while not finished:
                i = (i + 1) % 3
                tail = "." * i + " " * (3-i)
                sys.stdout.write("\r[TOGSim] TOGSim is running." + tail)
                time.sleep(1)
            print("")
        cmd = f"{self.get_togsim_command()} --models_list {model_path}"
        if extension_config.CONFIG_TOGSIM_DEBUG_LEVEL:
            cmd += f" --log_level {extension_config.CONFIG_TOGSIM_DEBUG_LEVEL}"
        if attribute_path:
            cmd = f"{cmd} --attributes_list {attribute_path}"
        if not silent_mode and extension_config.CONFIG_DEBUG_MODE:
            print("[TOGSim] cmd> ", cmd)

        # Create progress thread
        if not silent_mode:
            finished = False
            progress_thread = threading.Thread(target=show_progress)
            progress_thread.start()
        try:
            result = subprocess.check_output(shlex.split(cmd))
            if not silent_mode:
                finished = True
                progress_thread.join()
        except subprocess.CalledProcessError as e:
            if not silent_mode:
                finished = True
                progress_thread.join()
                with print_lock:
                    print("[TOGSim] Command failed with exit code", e.returncode)
                    print("[TOGSim] Error output:", e.output)
            assert 0

        # Separate Autotune logs
        if autotune_mode:
            base_dir = Path(model_path).parent / "togsim_result"
            base_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{len(list(base_dir.iterdir()))}.log"
        else:
            base_dir = Path(extension_config.CONFIG_TORCHSIM_LOG_PATH)
            unique_id = uuid.uuid4().hex[:8]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{unique_id}_{timestamp}.log"

        base_dir.mkdir(parents=True, exist_ok=True)
        result_path = base_dir / file_name

        # Prevent race condition
        with open(result_path, "w") as f:
            f.write(result.decode())
            f.flush()
            os.fsync(f.fileno())

        if not silent_mode or extension_config.CONFIG_DEBUG_MODE:
            model_path_log = f' of "{model_path}" ' if extension_config.CONFIG_DEBUG_MODE else " "
            with print_lock:
                print(f'[TOGSim] Simulation log{model_path_log}is stored to "{result_path}"')
        return result_path

    def interactive_simulation(self):
        cmd = f"{self.get_togsim_command()} --mode interactive"
        if extension_config.CONFIG_TOGSIM_DEBUG_LEVEL:
            cmd += f" --log_level {extension_config.CONFIG_TOGSIM_DEBUG_LEVEL}"

        if extension_config.CONFIG_DEBUG_MODE:
            print("[TOGSim] cmd> ", cmd)
        if self.process is None:
            self.process = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
        else:
            print("[TOGSim] Simulator is already running.")

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            print("[TOGSim] Simulator stopped.")

    def wait(self):
        if self.process:
            print("[TOGSim] Waiting for simulation to complete...")
            self.quit()
            self.process.wait()
            self.process = None
            print("[TOGSim] Simulation completed.")

    def send_command(self, command):
        if self.process:
            try:
                if extension_config.CONFIG_TORCHSIM_DEBUG_MODE:
                    print(command, flush=True)
                self.process.stdin.write(command + '\n')
                self.process.stdin.flush()
                ret = self.process.stderr.readline().strip()
                return ret
            except BrokenPipeError:
                err = self.process.stderr.readlines()
                for line in err:
                    print(line)
                self.process = None
                exit(1)
        else:
            print("Simulator is not running.")
            return None

    def launch(self, onnx_path, attribute_path, arrival_time=0, partion_id=0):
        command = f"launch {self.config_path} {onnx_path} {attribute_path} {arrival_time} {partion_id}"
        ret = self.send_command(command)
        return 0

    def cycle(self):
        ret = self.send_command("cycle")
        return int(ret.split(" ")[-1])

    def until(self, until_cycle):
        command = f"until {until_cycle}"
        ret = self.send_command(command)
        bitmap = int(ret.split(" ")[-1])
        indices = []
        for i in range(64):
            if (bitmap >> i) & 1:
                indices.append(i)
        return indices

    def quit(self):
        command = "quit"
        ret = self.send_command(command)
        return

    @classmethod
    def sram_alloc(cls, buf_name, addr_range):
        cls.ALLOC_POOL[buf_name] = addr_range

    @classmethod
    def sram_dealloc(cls, buf_name, addr_range):
        if buf_name in cls.ALLOC_POOL:
            del cls.ALLOC_POOL[buf_name]

    def create_attribute_file(self, attribute_path, inputs, **kwargs):
        address_info = {}
        sram_buffer = {}
        yaml_content = {}

        os.makedirs(attribute_path, exist_ok=True)
        index = str(len(os.listdir(attribute_path)))
        attribute_path = os.path.join(attribute_path, index)

        for idx, tensor in enumerate(inputs):
            address_info[f"arg{idx}"] = tensor.data_ptr()
        yaml_content["address_info"] = address_info

        for buf_name, range in self.ALLOC_POOL.items():
            sram_buffer[buf_name] = range
        yaml_content["sram_alloc"] = sram_buffer

        with open(attribute_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            f.flush()
            os.fsync(f.fileno()) # There could be a race condition.
        return attribute_path

    def load_yaml(self, config_path):
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"YAML file not found: {config_path}")

        try:
            with open(config_path, "r") as file:
                data = yaml.safe_load(file)
                return data
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")

    def get_core_freq(self):
        if "core_freq_mhz" in self.config_yaml:
            return self.config_yaml["core_freq_mhz"] * 1000 * 1000 # MHz
        else:
            raise KeyError("Key 'core_freq' not found in JSON.")

    def find_zero_sub_tensors(self, tensor):
        x, y = self.vectorlane_size, self.vectorlane_size
        zero_positions = {}

        # Need to set vectorlane size
        if self.vectorlane_size == -1:
            return zero_positions

        for i in range(0, tensor.shape[0], y):
            for j in range(0, tensor.shape[1], x):
                sub_tensor = tensor[i:i + y, j:j + x]
                if np.all(sub_tensor == 0):
                    if i not in zero_positions:
                        zero_positions[i] = {}
                    zero_positions[i][j] = 0 # i pos : j pos : 0
        return zero_positions

    @staticmethod
    def get_result_from_file(result_path):
        core_metrics = {}
        dram_channel_bw = {}
        avg_dram_bw = 0.0
        simulation_time = float("inf")
        total_cycle = float("inf")

        # Read and find total stat position
        with open(result_path, "r") as f:
            lines = f.readlines()

        simulation_finished_idx = -1
        simulation_finished = False
        for idx, line in enumerate(lines):
            if TOGSimulator.FINISH_STR in line:
                simulation_finished = True
                simulation_finished_idx = idx
                break

        if simulation_finished_idx == -1:
            print(f"[TOGSim] Warning: Unable to parse the output file ({result_path}). The file may be improperly formatted.")
            return core_metrics, dram_channel_bw, avg_dram_bw, simulation_time

        total_stat_lines = lines[simulation_finished_idx:]

        for line in total_stat_lines:
            # Parse core metrics (MatMul active cycle, Vector active cycle, etc.)
            if 'Core' in line:
                if 'MatMul active cycle' in line:
                    matmul_cycle = re.search(r'MatMul active cycle (\d+)', line).group(1)
                    vector_cycle = re.search(r'Vector active cycle (\d+)', line).group(1)
                    core_metrics['MatMul_active_cycle'] = int(matmul_cycle)
                    core_metrics['Vector_active_cycle'] = int(vector_cycle)
                elif 'Systolic Array Utilization' in line:
                    systolic_util = re.search(r'Systolic Array Utilization\(%\) (\d+\.?\d*)', line).group(1)
                    vector_util = re.search(r'Vector Unit Utilization\(%\) (\d+\.?\d*)', line).group(1)
                    total_cycle = re.search(r'Total cycle: (\d+)', line).group(1)
                    core_metrics['Systolic_Array_Utilization'] = float(systolic_util)
                    core_metrics['Vector_Unit_Utilization'] = float(vector_util)
                    core_metrics['Total_cycle'] = int(total_cycle)

            # Parse DRAM channel bandwidth utilization
            if 'DRAM CH' in line:
                channel = re.search(r'DRAM CH\[(\d+)\]', line).group(1)
                bw_util = re.search(r'AVG BW Util (\d+\.?\d*)%', line).group(1)
                dram_channel_bw[f'CH[{channel}]'] = float(bw_util)

            # Parse average DRAM bandwidth
            if 'DRAM: AVG BW Util' in line:
                avg_dram_bw = float(re.search(r'AVG BW Util (\d+\.?\d*)%', line).group(1))

            if 'Total execution cycles' in line:
                total_cycle = int(re.search(r'Total execution cycles: (\d+)', line).group(1))

            # Parse total simulation time
            if 'Wall-clock time for simulation' in line:
                simulation_time = float(re.search(r'Wall-clock time for simulation: (\d+\.?\d*) seconds', line).group(1))
        return core_metrics, dram_channel_bw, avg_dram_bw, simulation_time, total_cycle

if __name__ == "__main__":
    sim = TOGSimulator("/workspace/PyTorchSim/TOGSim", "/workspace/PyTorchSim/configs/systolic_ws_128x128_c2_simple_noc_tpuv3_partition.yml")
    sim.interactive_simulation()
    sim.until(4000)