#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="data/mendeley_iqa_701515_grouped900"
CKPT_PATH="checkpoints/best_resnet50_imagenet.pth"
mkdir -p logs runs_binary_iqa

for FUSION in residual fixed_interp learnable_interp gating adapter; do
  for SEED in {0..29}; do
    EXTRA=""
    if [ "$SEED" -eq 0 ]; then
      EXTRA="--save_ckpt --save_qualitative --profile_batch_sizes 8,16,32,64 --profile_iters 30"
    fi

    python tools/baselines/train_fusion_baselines_minimal.py \
      --data_root "${DATA_ROOT}" \
      --ckpt_path "${CKPT_PATH}" \
      --out_dir "runs_binary_iqa/fusion_${FUSION}_seed${SEED}" \
      --fusion_mode "${FUSION}" \
      --ft_mode l34 \
      --epochs 25 \
      --warmup_epochs 3 \
      --batch_size 32 \
      --lr 1e-4 \
      --alpha_values "[0.85,0.75,0.55,0.30]" \
      --qualitative_size 512 \
      --qualitative_dpi 600 \
      --seed "${SEED}" \
      ${EXTRA} \
      2>&1 | tee "logs/fusion_${FUSION}_seed${SEED}.log"
  done
done

