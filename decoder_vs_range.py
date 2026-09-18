"""
Is the decoder accurate CLOSE to the cube, where grasping happens?

The stall radius sits at 3.6-4.0 cm no matter what changes in the planner. A
constant floor like that points at the estimate, not the search -- and there is
one property of the estimate nobody has checked.

fit_keypoint trained on random arm motion, which rarely brings the gripper near
the cube. Its headline 1.79 cm is therefore an average dominated by far poses.
If accuracy collapses inside 5 cm, the planner drives to where it believes the
target is and parks there, invariant to decoder quality elsewhere, step size,
bias correction, horizon or elite averaging -- exactly what we observe.

Bins decoder error by true gripper-to-cube distance.

    python3 decoder_vs_range.py
"""

import argparse

import numpy as np
import torch

import loader
import record
from fit_keypoint import KeypointDecoder

BINS = [(0.00, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.18), (0.18, 0.40)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="keypoint.pt")
    a = ap.parse_args()
    dev = "cuda"
    enc, _ = loader.load_ac_model(device=dev, dtype=torch.float16)
    dec = KeypointDecoder().to(dev)
    dec.load_state_dict(torch.load(a.ckpt, map_location=dev))
    print(f"evaluating {a.ckpt}")
    dec.eval()

    errs, dists = [], []
    for seed in range(8):
        env = record.make_env("PickCube-v1", 256, camera="close")
        obs, _ = env.reset(seed=seed)
        e = env.unwrapped
        cube = e.cube.pose.p[0].cpu().numpy()
        rng = np.random.default_rng(seed)

        # deliberately sample poses at many ranges, including very close
        for trial in range(16):
            r = float(rng.uniform(0.005, 0.25))
            u = rng.normal(size=3)
            u = u / np.linalg.norm(u)
            u[2] = abs(u[2])                       # stay above the table
            target = cube + u * r
            for _ in range(22):
                tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
                a = np.zeros(7, np.float32)
                a[:3] = np.clip((target - tcp) / 0.1 * 0.6, -1, 1)
                a[6] = 1.0
                obs, *_ = env.step(a)

            tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
            cube_now = e.cube.pose.p[0].cpu().numpy()
            true_off = cube_now - tcp
            z = loader.encode(enc, record.obs_frame(obs)[None], dev, torch.float16)
            pred = dec(z.float())[0].cpu().numpy()
            errs.append(float(np.linalg.norm(pred - true_off)))
            dists.append(float(np.linalg.norm(true_off)))
        env.close()

    errs, dists = np.array(errs), np.array(dists)
    print(f"\n{'true distance':>16} {'n':>5} {'decoder error':>15}")
    for lo, hi in BINS:
        m = (dists >= lo) & (dists < hi)
        if m.sum() == 0:
            continue
        print(f"{lo*100:5.0f}-{hi*100:3.0f} cm{'':>4} {m.sum():>5} "
              f"{errs[m].mean()*100:>13.2f} cm")
    near = errs[dists < 0.06]
    far = errs[dists >= 0.10]
    print(f"\nclose range (<6 cm):  {near.mean()*100:.2f} cm   (n={len(near)})")
    print(f"far  range (>10 cm):  {far.mean()*100:.2f} cm   (n={len(far)})")
    print(f"headline average was 1.79 cm; planner stalls at 3.7 cm")
    if len(near) and near.mean() > 0.03:
        print(f"\nFOUND IT: the decoder is worst exactly where grasping happens.")
        print(f"The planner parks where its estimate says target, i.e. ~{near.mean()*100:.1f} cm out.")
        print(f"Fix: retrain the decoder on poses sampled NEAR the cube.")
    else:
        print(f"\nThe decoder holds up at close range, so this is not the cause.")
    np.savez(f"range_{a.ckpt.replace('.pt','')}.npz", errs=errs, dists=dists)


if __name__ == "__main__":
    main()
