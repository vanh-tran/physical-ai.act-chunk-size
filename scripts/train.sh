#!/bin/bash
# Train ACT on PushT — chunk_size variant.
# Usage: bash scripts/train.sh 16   (runs chunk_size=16)

CHUNK=${1:?Usage: bash scripts/train.sh <chunk_size>}

lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --policy.type=act \
  --env.type=pusht \
  --policy.chunk_size=${CHUNK} \
  --policy.n_action_steps=${CHUNK} \
  --output_dir=/workspace/outputs/chunk${CHUNK} \
  --job_name=act_pusht_chunk${CHUNK} \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.video_backend=torchcodec \
  --batch_size=64 \
  --num_workers=8 \
  --seed=1000 \
  --steps=20000 \
  --log_freq=100 \
  --save_freq=10000 \
  --env_eval_freq=10000 \
  --eval.n_episodes=50 \
  --wandb.enable=true \
  --wandb.project=lerobot-chunksize
