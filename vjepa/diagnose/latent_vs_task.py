"""
Is latent L1 a usable proxy for task-space distance?

Planning moves the arm decisively (~0.047 m/step, near the 0.05 ceiling) yet
ends ~0.11 m from the cube. So the planner is not stalling -- it is confidently
going somewhere else. That implicates the thing every goal-image planner
assumes: that minimising latent distance to a goal image also minimises
physical distance to the goal pose.

This measures that assumption directly. Park the gripper at many positions
around the workspace, and for each compare

    latent L1 (frame, goal_frame)      what the planner minimises
    ||tcp - goal_tcp||                 what we actually want

If these are strongly correlated, goal-image planning is sound and the fault is
elsewhere. If they are weakly correlated, the objective itself is broken: the
planner can drive latent distance down while moving away from the target, which
is exactly the observed behaviour.

    python3 latent_vs_task.py
"""

import numpy as np
import torch

from vjepa.core import loader
from vjepa.core import record


def servo(env, obs, target, steps=20):
    for _ in range(steps):
        tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
        a = np.zeros(7, np.float32)
        a[:3] = np.clip((target - tcp) / 0.1 * 0.6, -1, 1)
        a[6] = 1.0
        obs, *_ = env.step(a)
    return obs


def main():
    dev, dt = "cuda", torch.float16
    enc, _ = loader.load_ac_model(device=dev, dtype=dt)
    env = record.make_env("PickCube-v1", 256, camera="close")
    rng = np.random.default_rng(0)

    all_l1, all_d = [], []
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        cube = env.unwrapped.cube.pose.p[0].cpu().numpy()
        # goal pose: gripper right at the cube, as subgoal 2 intends
        obs = servo(env, obs, cube + np.array([0, 0, 0.01]), steps=30)
        goal_tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3].copy()
        z_goal = loader.encode(enc, record.obs_frame(obs)[None], dev, dt).float()

        l1s, ds = [], []
        for _ in range(18):
            obs, _ = env.reset(seed=seed)
            off = rng.normal(size=3) * np.array([0.10, 0.10, 0.06])
            obs = servo(env, obs, goal_tcp + off, steps=25)
            tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
            z = loader.encode(enc, record.obs_frame(obs)[None], dev, dt).float()
            l1s.append(float((z - z_goal).abs().mean()))
            ds.append(float(np.linalg.norm(tcp - goal_tcp)))
        r = np.corrcoef(l1s, ds)[0, 1]
        print(f"  seed {seed}: corr(latent L1, tcp distance) = {r:+.3f}  "
              f"over tcp range {min(ds):.3f}-{max(ds):.3f} m", flush=True)
        all_l1 += l1s
        all_d += ds
    env.close()

    r = np.corrcoef(all_l1, all_d)[0, 1]
    print(f"\npooled correlation = {r:+.3f}   (n={len(all_d)})")

    # how well does the latent-best position do in task space?
    order = np.argsort(all_l1)
    best_by_latent = np.array(all_d)[order[:5]].mean()
    print(f"mean tcp error of the 5 latent-closest poses: {best_by_latent:.4f} m")
    print(f"best achievable in this sample:               {min(all_d):.4f} m")
    if r < 0.5:
        print("\nWEAK: minimising latent distance does not reliably minimise")
        print("physical distance. Goal-image matching is the broken link.")
    else:
        print("\nSTRONG: latent distance tracks task distance, so the objective")
        print("is sound and the fault lies in the predictor or the search.")


if __name__ == "__main__":
    main()
