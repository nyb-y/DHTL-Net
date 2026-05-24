#!/usr/bin/env bash
set -euo pipefail

python tools/recommendation/recommend_top1.py \
  --archive1_root data \
  --code_root . \
  --runs_root runs_binary_iqa \
  --out_dir docs/results/recommendation \
  --split test
