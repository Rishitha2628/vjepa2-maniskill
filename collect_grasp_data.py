"""
Collect a ManiSkill dataset that actually contains grasps.

The random-action dataset in collect_data.py never grasps anything: closing the
gripper in mid-air teaches the model nothing about contact. But pure scripted
demonstrations are also wrong for planning -- CEM has to score BAD actions too,
and a model that has only seen the correct trajectory has no idea what happens
off it.

So each episode interpolates between a scripted pick-and-place and random noise:

    action = (1 - eps) * scripted + eps * noise

with eps drawn per episode. eps=0 gives a clean demonstration, eps=1 gives pure
flailing, and the middle is where the useful contrastive signal lives -- near
misses, premature closes, knocked-over cubes.

Records `grasped` per frame so the coverage can be checked rather than assumed.

    python3 collect_grasp_data.py --episodes 250 --steps 30
"""

import argparse
import os

import numpy as np
import torch

import adapter
import loader
import record


def scripted_action(t, steps, tcp, cube, goal, phase_jitter=0):
    """Proportional pick-and-place in pd_ee_delta_pose space. Solves 8/8 seeds."""
    p1 = int(steps * 0.20) + phase_jitter      # hover above cube
    p2 = int(steps * 0.40) + phase_jitter      # descend
    p3 = int(steps * 0.55) + phase_jitter      # close gripper
    if t < p1:
        tgt, g = cube + np.array([0, 0, 0.09]), 1.0
    elif t < p2:
        tgt, g = cube + np.array([0, 0, 0.005]), 1.0
    elif t < p3:
        tgt, g = tcp, -1.0
    else:
        tgt, g = goal, -1.0
    a = np.zeros(7, np.float32)
    a[:3] = np.clip((tgt - tcp) / adapter.POS_SCALE * 0.6, -1, 1)
    a[6] = g
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--camera", default="close", choices=list(record.CAMERAS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=20_000)
    ap.add_argument("--out", default="data_grasp")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device, dtype = "cuda", torch.float16
    N = a.episodes * a.steps

    latents = np.lib.format.open_memmap(
        os.path.join(a.out, "latents.npy"), mode="w+", dtype=np.float16,
        shape=(N, loader.TOKENS_PER_FRAME, loader.EMBED_DIM))
    print(f"allocating {latents.nbytes/1e9:.2f} GB for {N} frames")

    encoder, _ = loader.load_ac_model(device=device, dtype=dtype)

    A = np.zeros((N, 7), np.float32)
    S = np.zeros((N, 7), np.float32)
    EP = np.zeros(N, np.int32)
    GR = np.zeros(N, bool)
    EPS = np.zeros(N, np.float32)

    env = record.make_env("PickCube-v1", 256, camera=a.camera)
    w, n_grasp_ep = 0, 0
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=a.seed0 + ep)
        e = env.unwrapped
        rng = np.random.default_rng(a.seed0 + ep)
        # 20% pure random, rest scripted with varying corruption
        eps = 1.0 if rng.random() < 0.2 else float(rng.uniform(0.0, 0.55))
        jit = int(rng.integers(-2, 3))
        cube = e.cube.pose.p[0].cpu().numpy()
        goal = record.to_np(obs["extra"]["goal_pos"][0])

        frames, ms, states, grasped = [], [], [], []
        noise = np.zeros(7, np.float32)
        for t in range(a.steps):
            frames.append(record.obs_frame(obs))
            states.append(record.obs_state(obs, env))
            grasped.append(bool(e.agent.is_grasping(e.cube)[0]))

            tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
            sc = scripted_action(t, a.steps, tcp, cube, goal, jit)
            noise = 0.7 * noise + 0.3 * rng.normal(size=7).astype(np.float32)
            nz = np.clip(noise, -1, 1)
            nz[3:6] = 0.0
            # gripper noise is a coin flip, not a nudge: it is near-binary
            nz[6] = 1.0 if rng.random() < 0.5 else -1.0

            cmd = (1 - eps) * sc + eps * nz
            cmd = np.clip(cmd, -1, 1).astype(np.float32)
            cmd[3:6] = 0.0
            ms.append(cmd)
            obs, _, _, _, _ = env.step(cmd)

        states = np.asarray(states, np.float32)
        actions = adapter.maniskill_action_to_metric(np.asarray(ms, np.float32))
        actions[:, 6] = np.diff(np.concatenate([states[:1, 6], states[:, 6]]))

        for i in range(0, len(frames), a.batch):
            h = loader.encode(encoder, np.asarray(frames[i:i + a.batch], np.uint8),
                              device, dtype)
            latents[w + i: w + i + len(h)] = h.cpu().numpy()

        A[w:w + a.steps] = actions
        S[w:w + a.steps] = states
        EP[w:w + a.steps] = ep
        GR[w:w + a.steps] = grasped
        EPS[w:w + a.steps] = eps
        n_grasp_ep += int(any(grasped))
        w += a.steps

        if (ep + 1) % 25 == 0:
            print(f"  episode {ep+1}/{a.episodes}  grasped in "
                  f"{n_grasp_ep}/{ep+1} episodes", flush=True)

    env.close()
    latents.flush()
    np.savez(os.path.join(a.out, "meta.npz"), actions=A, states=S, episode=EP,
             grasped=GR, eps=EPS, steps=a.steps, camera=a.camera)
    print(f"done: {w} frames, {GR.sum()} grasped frames "
          f"({100*GR.mean():.1f}%), {n_grasp_ep}/{a.episodes} episodes with a grasp")


if __name__ == "__main__":
    main()
