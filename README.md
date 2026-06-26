# ACT chunk_size Experiment — Module 1 Artifact

> **Physical AI Study Plan, Module 1, Gap 4.** What does ACT's `chunk_size` actually control, and how does it affect task success on PushT?

**Result:** pc_success drops **14% → 8% → 0%** as chunk_size increases 16 → 32 → 100. The bottleneck is open-loop staleness, not training horizon.

[Full write-up →](#) *(link your long-form post when published)*

---

## Quick results

| chunk_size | pc_success | avg_max_reward | avg_sum_reward |
|---|---|---|---|
| 16 | **14.0%** (7/50) | 0.686 | 71.29 |
| 32 | **8.0%** (4/50) | 0.706 | 79.15 |
| 100 | **0.0%** (0/50) | 0.326 | 28.28 |

- **`pc_success`** — % of 50 eval episodes where the T-block sustained coverage on the goal. Strict: a brief graze doesn't count.
- **`avg_max_reward`** — average best single-frame T→goal overlap across episodes. 1.0 = perfect coverage, even if only for an instant.
- **`avg_sum_reward`** — average total overlap across all ~300 timesteps. Captures both peak AND duration.

Full per-episode data: `results/chunk{16,32,100}_eval.json`.

## The finding

pc_success falls monotonically, but `avg_max_reward` peaks at chunk32 (0.706). chunk32 approaches the goal more smoothly but overshoots — the matched `n_action_steps` commits to 32 open-loop actions, and drift accumulates before the next re-plan. chunk100 re-plans only ~3 times per episode and never gets close.

The confound: matching `n_action_steps = chunk_size` varies both training horizon AND re-plan frequency. This experiment measures the combined effect.

---

## Reproduce

### Prerequisites

- RunPod pod with image `huggingface/lerobot-gpu@sha256:b0318f57aaab7d8204f5bd3c61bf98b4660c4240b546e60d6205e1d65035b9d2`
- RTX 4090 / RTX PRO 4000 (24 GB) or equivalent
- Network volume mounted at `/workspace`

### One-time setup

```bash
# Run once on the network volume — survives stops and terminates
bash /workspace/lerobot-setup.sh
```

Or run the setup script from this repo:

```bash
bash lerobot-setup.sh
```

### Train

```bash
# Clone LeRobot (separately — not included in this repo)
git clone https://github.com/huggingface/lerobot
cd lerobot
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))

bash scripts/train.sh 16    # chunk_size=16, ~1:15 on RTX 4090
bash scripts/train.sh 32    # chunk_size=32
bash scripts/train.sh 100   # chunk_size=100
```

### Evaluate

```bash
bash scripts/eval.sh 16     # standalone eval on saved checkpoint
```

### Locked configuration

All runs: seed=1000, batch_size=64, num_workers=8, steps=20000, dataset=lerobot/pusht, video_backend=torchcodec.

Per-run configs: `configs/chunk{16,32,100}.yaml`.

---

## Repo structure

```
├── README.md
├── lerobot-setup.sh           # one-time pod setup
├── .gitignore
├── configs/
│   ├── chunk16.yaml
│   ├── chunk32.yaml
│   └── chunk100.yaml
├── scripts/
│   ├── train.sh               # lerobot-train wrapper
│   └── eval.sh                # standalone lerobot-eval
└── results/
    ├── chunk16_eval.json      # per-episode eval data
    ├── chunk32_eval.json
    └── chunk100_eval.json
```

Model checkpoints are not included (too large). To reproduce from scratch, run `scripts/train.sh`. To verify these exact results, use the pinned image digest and seed=1000.

---

## Environment

| Component | Version |
|---|---|
| Container image | `huggingface/lerobot-gpu@sha256:b0318f57aaab7d8204f5bd3c61bf98b4660c4240b546e60d6205e1d65035b9d2` |
| torch | 2.11.0+cu128 |
| LeRobot | 0.5.2 (editable, between v0.5.1 and main) |
| torchcodec | 0.11.1+cpu |
| gym-pusht | 0.1.x |

A `.pth` autoload shim registers `gym_pusht` + backfills `task_description`/`task` on `PushTEnv` — required because LeRobot 0.5.2 relies on gymnasium autoload and reads eval attrs unguarded. Included in `lerobot-setup.sh`.

---

*Part of the [Physical AI Study Plan](https://github.com/huggingface/lerobot) — Module 1.*
