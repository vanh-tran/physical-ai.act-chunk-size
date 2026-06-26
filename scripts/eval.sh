#!/bin/bash
# Standalone eval on a saved checkpoint.
# Usage: bash scripts/eval.sh 16   (evaluates chunk_size=16)

CHUNK=${1:?Usage: bash scripts/eval.sh <chunk_size>}

lerobot-eval \
  --policy.path=/workspace/outputs/chunk${CHUNK}/checkpoints/last/pretrained_model \
  --env.type=pusht \
  --eval.n_episodes=50 \
  --policy.device=cuda \
  --seed=1000 \
  --output_dir=/workspace/outputs/chunk${CHUNK}/eval_standalone
