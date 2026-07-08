# Conv tile axis discovery

- Conv 3x3 uses a 7-parameter heuristic tile axis: `TILE_K_H`, `TILE_K_W`, `TILE_O_H`, `TILE_O_W`, `TILE_M`, `TILE_N`, `TILE_K`.
- Derived input extents are `TILE_I_H = 1 + (TILE_O_H - 1) * stride_h + (TILE_K_H - 1) * dilation_h` and `TILE_I_W = 1 + (TILE_O_W - 1) * stride_w + (TILE_K_W - 1) * dilation_w`.
- Existing `external_mapping.json` originally exposed only GEMM `TILE_M/N/K`; this probe adds a Conv-specific key `conv2d_1_64_64_3_3_56_56` carrying the 7 Conv tile parameters so Route 4 can sweep tiles without touching simulator sources.
- Working-set model from `conv_combination_mapping`: `weight = K_H * K_W * TILE_K * TILE_N`, `input = TILE_I_H * TILE_I_W * TILE_M * TILE_K`, `output = TILE_O_H * TILE_O_W * TILE_M * TILE_N`, `bytes = 4 * (weight + input + output)` for this unfused Conv core.
- Smallest-SPAD HW usable double-buffer budget: `2097152` bytes (4 MiB total physical SPAD / 2 for double buffering).

## Selected tiles

- `tile_A`: {"TILE_K": 64, "TILE_K_H": 3, "TILE_K_W": 3, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 56, "TILE_O_W": 4, "fit_fraction_largest_spad": 0.0594482421875, "fit_fraction_smallest_spad": 0.23779296875, "predicted_fit_all_hw": true, "predicted_working_set_bytes": 498688, "regime": "tiny"}
- `tile_B`: {"TILE_K": 64, "TILE_K_H": 3, "TILE_K_W": 3, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 56, "TILE_O_W": 14, "fit_fraction_largest_spad": 0.111328125, "fit_fraction_smallest_spad": 0.4453125, "predicted_fit_all_hw": true, "predicted_working_set_bytes": 933888, "regime": "small"}
- `tile_C`: {"TILE_K": 64, "TILE_K_H": 3, "TILE_K_W": 3, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 56, "TILE_O_W": 28, "fit_fraction_largest_spad": 0.1839599609375, "fit_fraction_smallest_spad": 0.73583984375, "predicted_fit_all_hw": true, "predicted_working_set_bytes": 1543168, "regime": "medium"}
- `tile_D`: {"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 56, "TILE_O_W": 56, "fit_fraction_largest_spad": 0.291015625, "fit_fraction_smallest_spad": 1.1640625, "predicted_fit_all_hw": false, "predicted_working_set_bytes": 2441216, "regime": "large"}

## Candidate sample (smallest to largest working set)

- `{"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 1, "TILE_O_W": 1, "total_bytes": 33536}`
- `{"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 1, "TILE_O_W": 2, "total_bytes": 34304}`
- `{"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 2, "TILE_O_W": 1, "total_bytes": 34304}`
- `{"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 1, "TILE_O_W": 4, "total_bytes": 35840}`
- `{"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 2, "TILE_O_W": 2, "total_bytes": 35840}`
- `{"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 4, "TILE_O_W": 1, "total_bytes": 35840}`
- `{"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 1, "TILE_O_W": 7, "total_bytes": 38144}`
- `{"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 7, "TILE_O_W": 1, "total_bytes": 38144}`
- `{"TILE_K": 64, "TILE_K_H": 3, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 28, "TILE_O_W": 56, "total_bytes": 1331200}`
- `{"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 3, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 56, "TILE_O_W": 28, "total_bytes": 1331200}`
- `{"TILE_K": 64, "TILE_K_H": 3, "TILE_K_W": 3, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 28, "TILE_O_W": 56, "total_bytes": 1543168}`
- `{"TILE_K": 64, "TILE_K_H": 3, "TILE_K_W": 3, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 56, "TILE_O_W": 28, "total_bytes": 1543168}`
- `{"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 56, "TILE_O_W": 56, "total_bytes": 2441216}`
- `{"TILE_K": 64, "TILE_K_H": 1, "TILE_K_W": 3, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 56, "TILE_O_W": 56, "total_bytes": 2535424}`
- `{"TILE_K": 64, "TILE_K_H": 3, "TILE_K_W": 1, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 56, "TILE_O_W": 56, "total_bytes": 2535424}`
- `{"TILE_K": 64, "TILE_K_H": 3, "TILE_K_W": 3, "TILE_M": 1, "TILE_N": 128, "TILE_O_H": 56, "TILE_O_W": 56, "total_bytes": 2761728}`
