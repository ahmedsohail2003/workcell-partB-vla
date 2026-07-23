"""Evaluate the fine-tuned SmolVLA policy in the sim2cell PickPlaceEnv.

Same protocol as sim2cell's eval_policy.py (ACT baseline: 13/20 = 65%, 15/20 =
75% with ensembling): N rollouts, seed 123, nominal scene, 240-step cap. The
policy is language-conditioned -- the task instruction is tokenized by the
saved preprocessor pipeline.

Camera mapping (matches the training rename_map): the fine-tuned policy expects
the pretrained base's camera names, so front->camera1, wrist->camera2.

Usage:
    python scripts/eval_smolvla.py [n_episodes] [--gif] [--seed 123]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\sohai\robotics\sim2cell\src")

import imageio.v3 as iio
import numpy as np
import torch

from sim2cell.env import PickPlaceEnv

REPO_ID = "ahmedsohail2003/smolvla-so101-pickplace"
TASK = "Pick up the red block and place it in the blue tray."
CAM_RENAME = {"front": "camera1", "wrist": "camera2"}
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

MAX_STEPS = 240  # 16 s at 15 Hz, same cap as the ACT eval


def obs_to_batch(obs: dict, device: torch.device) -> dict:
    batch = {
        "observation.state": torch.from_numpy(obs["state"]).unsqueeze(0).to(device),
        "task": TASK,
    }
    for cam, img in obs["images"].items():
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        batch[f"observation.images.{CAM_RENAME[cam]}"] = t.unsqueeze(0).to(device)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_episodes", type=int, nargs="?", default=20)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--seed", type=int, default=123)   # same protocol as ACT eval
    ap.add_argument("--randomize", type=str, default="none",
                    choices=["none", "visual", "physics", "full"])
    args = ap.parse_args()

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained(REPO_ID)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=REPO_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(f"loaded {REPO_ID} on {device}")
    print(f'task: "{TASK}"')

    env = PickPlaceEnv(seed=args.seed, randomize=args.randomize)
    successes, frames = 0, []
    t0 = time.perf_counter()
    for ep in range(args.n_episodes):
        obs = env.reset()
        policy.reset()
        success = False
        for step in range(MAX_STEPS):
            with torch.no_grad():
                batch = preprocessor(obs_to_batch(obs, device))
                action = postprocessor(policy.select_action(batch))
            obs = env.step(action.squeeze(0).cpu().numpy())
            if args.gif and ep == 0:
                frames.append(obs["images"]["front"])
            if env.is_success():
                success = True
                break
        successes += success
        print(f"ep {ep:02d}: success={success}  steps={step + 1}"
              f"  block_final=({env.block_pos[0]:+.3f},{env.block_pos[1]:+.3f},{env.block_pos[2]:+.3f})",
              flush=True)

    dt = time.perf_counter() - t0
    print(f"\nSMOLVLA SUCCESS RATE: {successes}/{args.n_episodes}"
          f" ({100 * successes / args.n_episodes:.0f}%)  wall={dt:.1f}s"
          f"  [ACT baseline on same protocol: 13/20=65%, 15/20=75% w/ ensembling]")

    if args.gif and frames:
        iio.imwrite(OUT / "smolvla_episode.gif", np.stack(frames[::2]), duration=2 / env.control_hz, loop=0)
        print(f"wrote {OUT / 'smolvla_episode.gif'}")


if __name__ == "__main__":
    main()
