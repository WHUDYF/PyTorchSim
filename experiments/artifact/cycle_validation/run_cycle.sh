#!/bin/bash
set -e

export TOGSIM_CONFIG=$TORCHSIM_DIR/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml
LOG_DIR=$TORCHSIM_DIR/experiments/artifact/logs
mkdir -p $LOG_DIR

# Matmul
for sz in "256 256 256" "512 512 512" "1024 1024 1024" "2048 2048 2048"; do
  name="gemm_${sz// /x}"
  echo ""
  echo "==================================================="
  echo "[*] Running Matmul size=$sz"
  echo "==================================================="
  python3 $TORCHSIM_DIR/experiments/gemm.py --size $sz | tee $LOG_DIR/${name}.log
done

# Conv
for sz in \
  "1 56 56 64 64 3 1 1" \
  "1 28 28 128 128 3 1 1" \
  "1 14 14 256 256 3 1 1" \
  "1 7 7 512 512 3 1 1" \
  "64 56 56 64 64 3 1 1" \
  "64 28 28 128 128 3 1 1" \
  "64 14 14 256 256 3 1 1" \
  "64 7 7 512 512 3 1 1"; do
  name="conv_${sz// /x}"
  echo ""
  echo "==================================================="
  echo "[*] Running Conv size=$sz"
  echo "==================================================="
  python3 $TORCHSIM_DIR/experiments/conv.py --size $sz | tee $LOG_DIR/${name}.log
done

# Attention
for sz in "12 512 64" "16 512 64" "32 512 64"; do
  name="attention_${sz// /x}"
  echo ""
  echo "==================================================="
  echo "[*] Running Attention size=$sz"
  echo "==================================================="
  python3 $TORCHSIM_DIR/experiments/attention.py --size $sz | tee $LOG_DIR/${name}.log
done

# LayerNorm
for sz in "512 768" "2048 768" "8192 768"; do
  name="layernorm_${sz// /x}"
  echo ""
  echo "==================================================="
  echo "[*] Running LayerNorm size=$sz"
  echo "==================================================="
  python3 $TORCHSIM_DIR/experiments/layernorm.py --size $sz | tee $LOG_DIR/${name}.log
done

# Softmax
for sz in "512 512" "2048 2048" "8192 8192"; do
  name="softmax_${sz// /x}"
  echo ""
  echo "==================================================="
  echo "[*] Running Softmax size=$sz"
  echo "==================================================="
  python3 $TORCHSIM_DIR/experiments/softmax.py --size $sz | tee $LOG_DIR/${name}.log
done

# ResNet
for model in "resnet18" "resnet50"; do
  echo ""
  echo "==================================================="
  echo "[*] Running $model"
  echo "==================================================="
  python3 $TORCHSIM_DIR/experiments/${model}.py | tee $LOG_DIR/${model}.log
done

# BERT
for model in "base" "large" "xlarge"; do
  echo ""
  echo "==================================================="
  echo "[*] Running BERT size=$model"
  echo "==================================================="
  python3 $TORCHSIM_DIR/experiments/BERT.py --size $model | tee $LOG_DIR/bert_${model}.log
done

# Cycle Summary
python3 $TORCHSIM_DIR/experiments/artifact/cycle_validation/summary_cycle.py | tee "$TORCHSIM_DIR/experiments/artifact/cycle_validation/summary_cycle.out"