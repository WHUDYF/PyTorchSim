"""
Generate machine-readable ramulator2 v2.1 config files for PyTorchSim.

Usage:
    python gen_configs.py

Each function generates a JSON config that C++ can load directly via
Config::parse_config_file(). No preset resolution happens in C++ anymore.
"""

import json
import sys
import os

# Add ramulator2 Python DSL to path
RAMULATOR_PYTHON = os.path.join(os.path.dirname(__file__),
                                "../../TOGSim/extern/ramulator2/python")
sys.path.insert(0, RAMULATOR_PYTHON)

import ramulator
import ramulator.dram
import ramulator.controller
import ramulator.scheduler
import ramulator.refresh_manager
import ramulator.row_policy
import ramulator.addr_mapper
import ramulator.channel_mapper
import ramulator.memory_system


def make_config(dram_obj, clock_ratio=1, refresh_scope="Rank"):
    """Wrap a DRAM object in a single-channel GenericDRAM config for PyTorchSim.

    PyTorchSim creates one Ramulator2 instance per channel, so each config
    always has exactly one controller (channel=1 in org is enforced by v2.1).
    The wrapper overrides 'frontend' to ExternalFrontEnd automatically.

    refresh_scope: level name for AllBank refresh.
      - DDR4 / LPDDR5 / LPDDR5X → "Rank"
      - HBM2 / HBM3              → "PseudoChannel"
    """
    ctrl = ramulator.controller.GenericDDR(
        dram=dram_obj,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(scope=refresh_scope),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
    )
    ms = ramulator.memory_system.GenericDRAM(
        clock_ratio=clock_ratio,
        controllers=[ctrl],
        # Single-channel per Ramulator2 instance — passthrough maps everything to ch 0
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
    )
    return {
        "frontend": {"impl": "External", "clock_ratio": 1},
        "memory_system": ms.to_config(),
    }


def gen_hbm2():
    # Available timing presets: HBM2_1600Mbps, HBM2_2000Mbps, HBM2_2400Mbps
    # HBM2 has no Rank level — AllBank refresh scope must be PseudoChannel
    dram = ramulator.dram.HBM2(org_preset="HBM2_8Gb", timing_preset="HBM2_2000Mbps")
    return make_config(dram, clock_ratio=1, refresh_scope="PseudoChannel")


def gen_hbm2_tpuv3():
    # TPUv3 HBM2: 900MHz → ~1.8 Gbps. Closest available preset: HBM2_2000Mbps
    dram = ramulator.dram.HBM2(org_preset="HBM2_8Gb", timing_preset="HBM2_2000Mbps")
    return make_config(dram, clock_ratio=1, refresh_scope="PseudoChannel")


def gen_ddr4():
    # Available timing presets — check python/ramulator/dram/ddr4.py
    dram = ramulator.dram.DDR4(org_preset="DDR4_8Gb_x8", timing_preset="DDR4_3200AA")
    return make_config(dram, clock_ratio=1)


def gen_lpddr5():
    dram = ramulator.dram.LPDDR5(org_preset="LPDDR5_8Gb_x16", timing_preset="LPDDR5_6400")
    return make_config(dram, clock_ratio=1)


def gen_lpddr5x():
    # LPDDR5X_8533: 8533 MT/s, tCK=938ps, CK=1066MHz
    dram = ramulator.dram.LPDDR5(org_preset="LPDDR5_8Gb_x16", timing_preset="LPDDR5X_8533")
    return make_config(dram, clock_ratio=1)


CONFIGS = {
    "HBM2.yaml":        gen_hbm2,
    "HBM2_TPUv3.yaml":  gen_hbm2_tpuv3,
    "DDR4.yaml":        gen_ddr4,
    "LPDDR5.yaml":      gen_lpddr5,
    "LPDDR5X.yaml":     gen_lpddr5x,
}


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for filename, gen_fn in CONFIGS.items():
        cfg = gen_fn()
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w") as f:
            # json is valid yaml — C++ parse_config_file reads either
            json.dump(cfg, f, indent=2)
        print(f"Generated {out_path}")

