#!/bin/bash
LOG_DIR=$TORCHSIM_DIR/experiments/artifact/logs
CONFIG_DIR="$TORCHSIM_DIR/configs"
SIMULATOR_BIN="$TORCHSIM_DIR/TOGSim/build/bin/Simulator"

configs=(
    "systolic_ws_128x128_c2_simple_noc_tpuv3.yml"
    "systolic_ws_128x128_c2_booksim_tpuv3.yml"
)

target_list=(
  "gemm_512x512x512"
  "gemm_1024x1024x1024"
  "gemm_2048x2048x2048"
  "conv_1x56x56x64x64x3x1x1"
  "conv_1x28x28x128x128x3x1x1"
  "conv_1x14x14x256x256x3x1x1"
  "conv_1x7x7x512x512x3x1x1"
  "resnet50"
  "bert_large"
)

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
output_dir="$TORCHSIM_DIR/experiments/artifact/speedup/results"
mkdir -p "$output_dir"

echo "[*] Scanning log files in: $LOG_DIR"
echo ""

for log_file in "$LOG_DIR"/*.log; do
  filename=$(basename "$log_file")
  workload="${filename%.log}"

  if [[ ! " ${target_list[@]} " =~ " ${workload} " ]]; then
    continue
  fi
  echo "==> Workload: $workload"

  declare -a ONNX_ATTR_PAIRS=()

  # === Grep launch line ===
  while IFS= read -r line; do
    if [[ "$line" == launch* ]]; then
      read -r _ onnx_path attr_path _ <<< "$line"
      ONNX_ATTR_PAIRS+=("$onnx_path|$attr_path")
    fi
  done < "$log_file"

  # Normal configs
  for config in "${configs[@]}"; do
    output_file="$output_dir/${workload}_${config}.txt" 
    echo "Running with config=$config"
    echo "===== config=$config | model=$workload =====" >> "$output_file"
    sum_all_iters=0.0
    iter_count=0

     # === Run 5 iterations ===
    for iter in {1..5}; do
      echo "[Iter $iter] Running simulation for workload=$workload config=$config"
      cmd=""
      for pair in "${ONNX_ATTR_PAIRS[@]}"; do
        IFS="|" read -r onnx_path attr_path <<< "$pair"
        cmd+=" $SIMULATOR_BIN --config $CONFIG_DIR/$config --models_list $onnx_path --attributes_list $attr_path;"
      done

      output=$(bash -c "$cmd")
      sim_times=$(echo "$output" | grep "Simulation time:" | sed -E 's/.*Simulation time: ([0-9]+\.[0-9]+).*/\1/')

      if [[ -n "$sim_times" ]]; then
        sum_per_iter=0.0
        while IFS= read -r sim_time; do
          echo "Iteration $iter: simulation_time = $sim_time" >> "$output_file"
          sum_per_iter=$(awk -v a="$sum_per_iter" -v b="$sim_time" 'BEGIN {printf "%.6f", a + b}')
        done <<< "$sim_times"

        echo "Iteration $iter: total_simulation_time = $sum_per_iter" >> "$output_file"
        sum_all_iters=$(awk -v a="$sum_all_iters" -v b="$sum_per_iter" 'BEGIN {printf "%.6f", a + b}')
        iter_count=$((iter_count + 1))
      else
        echo "Iteration $iter: No simulation time found." >> "$output_file"
      fi
    done

    # === Final average ===
    if [[ $iter_count -gt 0 ]]; then
      avg=$(awk -v total="$sum_all_iters" -v n="$iter_count" 'BEGIN {printf "%.6f", total / n}')
      echo "Average simulation time for $workload with config $config: $avg seconds"
      echo "Average simulation time = $avg" >> "$output_file"
    else
      echo "No valid simulation times found for config $config"
      echo "Average simulation time = NA" >> "$output_file"
    fi
  done
done

# ILS mode should be run separately
$TORCHSIM_DIR/experiments/artifact/speedup/scripts/run_speed_ils_matmul.sh
$TORCHSIM_DIR/experiments/artifact/speedup/scripts/run_speed_ils_conv.sh
$TORCHSIM_DIR/experiments/artifact/speedup/scripts/run_speed_ils_bert.sh
$TORCHSIM_DIR/experiments/artifact/speedup/scripts/run_speed_ils_resnet.sh

python3 $TORCHSIM_DIR/experiments/artifact/speedup/summary_speedup.py | tee "$TORCHSIM_DIR/experiments/artifact/speedup/summary_speedup.log"