#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="data/mendeley_iqa_701515_grouped900"
CKPT_PATH="checkpoints/best_resnet50_imagenet.pth"
mkdir -p logs runs_binary_iqa

for SEED in {0..29}; do
  EXTRA=""
  if [ "$SEED" -eq 0 ]; then
    EXTRA="--save_qualitative --profile_batch_sizes 8,16,32,64 --profile_iters 30"
  fi

  python tools/proposed_method/train_dhtl.py \
    --data_root "${DATA_ROOT}" \
    --ckpt_path "${CKPT_PATH}" \
    --out_dir "runs_binary_iqa/DAF_seed${SEED}" \
    --epochs 25 \
    --warmup_epochs 3 \
    --batch_size 32 \
    --lr 1e-4 \
    --ft_mode l34 \
    --use_class_weights \
    --qualitative_size 512 \
    --qualitative_dpi 600 \
    --seed "${SEED}" \
    ${EXTRA} \
    2>&1 | tee "logs/DAF_seed${SEED}.log"
done

