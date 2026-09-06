"""
Collect a ManiSkill dataset for fine-tuning the AC predictor.

The encoder stays frozen, so its output is computed ONCE here and cached. That
is the whole reason fine-tuning fits on a 6 GB card: training never loads
ViT-giant at all, only the 305M predictor.

Latents are written to a float16 memmap (256 tokens x 1408 dims = 720 KB per
frame), so the dataset never has to fit in RAM.

Outputs:
    data/latents.npy   (N, 256, 1408) float16 memmap, layer-normed
    data/meta.npz      actions, states, episode ids, frame index

    python3 collect_data.py --episodes 200 --steps 20
"""

import argparse
import os

import numpy as np
import torch

import adapter
import loader
import record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--camera", default="close", choices=list(record.CAMERAS))
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--smooth", type=float, default=0.7)
    ap.add_argument("--grip-prob", type=float, default=0.25,
                    help="fraction of episodes that also actuate the gripper")
    ap.add_argument("--batch", type=int, default=16, help="encoder batch size")
    ap.add_argument("--seed0", type=int, default=10_000)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device, dtype = "cuda", torch.float16
    N = a.episodes * a.steps

    lat_path = os.path.join(a.out, "latents.npy")
    latents = np.lib.format.open_memmap(
        lat_path, mode="w+", dtype=np.float16,
        shape=(N, loader.TOKENS_PER_FRAME, loader.EMBED_DIM))
    print(f"allocating {lat_path}: {N} frames, "
          f"{latents.nbytes/1e9:.2f} GB on disk")

    encoder, _ = loader.load_ac_model(device=device, dtype=dtype)

    all_actions = np.zeros((N, 7), np.float32)
    all_states = np.zeros((N, 7), np.float32)
    all_ep = np.zeros(N, np.int32)

    env = record.make_env("PickCube-v1", 256, camera=a.camera)
    w = 0
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=a.seed0 + ep)
        rng = np.random.default_rng(a.seed0 + ep)
        use_grip = rng.random() < a.grip_prob
        act = np.zeros(7, np.float32)

        frames, ms_actions, states = [], [], []
        for t in range(a.steps):
            frames.append(record.obs_frame(obs))
            states.append(record.obs_state(obs, env))
            act = a.smooth * act + (1 - a.smooth) * rng.normal(size=7).astype(np.float32)
            cmd = np.clip(act * a.scale, -1, 1).astype(np.float32)
            cmd[3:6] = 0.0
            # Gripper is an absolute target in ManiSkill; hold it for a while
            # rather than dithering, so open/close actually completes.
            cmd[6] = -1.0 if (use_grip and t >= a.steps // 3) else 1.0
            ms_actions.append(cmd)
            obs, _, _, _, _ = env.step(cmd)

        frames = np.asarray(frames, np.uint8)
        states = np.asarray(states, np.float32)
        actions = adapter.maniskill_action_to_metric(np.asarray(ms_actions, np.float32))
        # last channel must be a closedness DELTA, matching the predictor
        actions[:, 6] = np.diff(np.concatenate([states[:1, 6], states[:, 6]]))

        for i in range(0, len(frames), a.batch):
            h = loader.encode(encoder, frames[i:i + a.batch], device, dtype)
            latents[w + i: w + i + len(h)] = h.cpu().numpy()

        all_actions[w:w + a.steps] = actions
        all_states[w:w + a.steps] = states
        all_ep[w:w + a.steps] = ep
        w += a.steps

        if (ep + 1) % 20 == 0:
            print(f"  episode {ep+1}/{a.episodes}  ({w} frames)", flush=True)

    env.close()
    latents.flush()
    np.savez(os.path.join(a.out, "meta.npz"), actions=all_actions,
             states=all_states, episode=all_ep, steps=a.steps,
             camera=a.camera)
    print(f"done: {w} frames -> {a.out}/")


if __name__ == "__main__":
    main()
