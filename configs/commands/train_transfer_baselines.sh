#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="data/mendeley_iqa_701515_grouped900"
CKPT_PATH="checkpoints/best_resnet50_imagenet.pth"
mkdir -p logs runs_binary_iqa

for MODE in frozen all l4 l34 l234; do
  for SEED in {0..29}; do
    EXTRA=""
    if [ "$SEED" -eq 0 ]; then
      EXTRA="--save_qualitative --profile_batch_sizes 8,16,32,64 --profile_iters 30"
    fi

    python tools/baselines/train_transfer_baselines_minimal.py \
      --data_root "${DATA_ROOT}" \
      --ckpt_path "${CKPT_PATH}" \
      --out_dir "runs_binary_iqa/transfer_${MODE}_seed${SEED}" \
      --mode "${MODE}" \
      --epochs 25 \
      --warmup_epochs 3 \
      --batch_size 32 \
      --lr 1e-4 \
      --qualitative_size 512 \
      --qualitative_dpi 600 \
      --seed "${SEED}" \
      ${EXTRA} \
      2>&1 | tee "logs/transfer_${MODE}_seed${SEED}.log"
  done
done

