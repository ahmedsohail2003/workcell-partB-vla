# TalkToTheCell — language-conditioned manipulation with SmolVLA

**Project B of the Sim2Cell portfolio:** fine-tune **SmolVLA (450M)**, a
vision-language-action model, on the author's own
[`so101-sim-pickplace`](https://huggingface.co/datasets/ahmedsohail2003/so101-sim-pickplace)
dataset — 100 language-labeled MuJoCo demonstrations of an SO-ARM100 arm
executing *"Pick up the red block and place it in the blue tray."* — and
compare it against the ACT baseline trained on the same data (65%, 75% with
temporal ensembling).

Where [sim2cell](../sim2cell) asks *"can a policy copy the expert from
pixels?"*, this project asks *"can a **foundation model** be steered to the
same task with natural language?"* — the paradigm industrial robotics is
converging on for taskable, reconfigurable cells.

## Status

| Step | State |
|---|---|
| Dataset published to the Hub (with card) | ✅ [`so101-sim-pickplace`](https://huggingface.co/datasets/ahmedsohail2003/so101-sim-pickplace) |
| Kaggle fine-tune notebook | ✅ [`notebooks/kaggle_smolvla_finetune.ipynb`](notebooks/kaggle_smolvla_finetune.ipynb) |
| Training run (Kaggle T4, 20k steps) | ⏳ pending (needs phone-verified Kaggle account) |
| Model on the Hub + model card | ⏳ `ahmedsohail2003/smolvla-so101-pickplace` |
| Local sim evaluation vs ACT baseline | ⏳ |
| Before/after demo video + VLA writeup | ⏳ |

## Free-tier engineering

SmolVLA full fine-tuning wants ~24 GB VRAM; the free-tier recipe that fits a
16 GB T4:

- **Frozen VLM backbone** — LeRobot's SmolVLA defaults (`freeze_vision_encoder=True`,
  `train_expert_only=True`) train only the ~100M-parameter action expert
- **AMP (fp16)** + batch 8 (vs the paper's batch 64 on A100)
- 20k steps with checkpoints every 5k → survives Kaggle's 12 h session cap,
  resumable via `--resume=true`

## License

Apache-2.0
