"""
What spatial precision can latent L1 actually resolve?

Every planning attempt stalls at 5-18 cm from the cube while a grasp needs
<2 cm. Cost re-weighting, longer horizons and subgoals all failed to close that
gap, which suggests a floor that is not about search at all.

The encoder tokenises a 256px image into 16x16 patches, so one patch covers
roughly 2-3 cm of table at this camera framing. Displacements below patch scale
may simply not move the representation much, in which case no planner can
servo to centimetre precision no matter how it searches.

This measures it directly: park the gripper at a reference pose, capture its
latent, then move it by a controlled offset and measure latent L1 back to the
reference. If the curve is flat below ~5 cm, that is the precision ceiling.

    python3 latent_precision.py
"""

import numpy as np
import torch

import loader
import record

OFFSETS = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
DIRS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


def servo_to(env, obs, target, steps=25):
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

    rows = {d: [] for d in OFFSETS}
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        cube = env.unwrapped.cube.pose.p[0].cpu().numpy()
        ref = cube + np.array([0.0, 0.0, 0.10])
        obs = servo_to(env, obs, ref)
        ref_actual = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
        z_ref = loader.encode(enc, record.obs_frame(obs)[None], dev, dt).float()

        for d in OFFSETS:
            vals = []
            for u in DIRS:
                obs, _ = env.reset(seed=seed)
                obs = servo_to(env, obs, ref_actual + u * d)
                got = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
                moved = float(np.linalg.norm(got - ref_actual))
                z = loader.encode(enc, record.obs_frame(obs)[None], dev, dt).float()
                l1 = float((z - z_ref).abs().mean())
                vals.append((l1, moved))
            rows[d].append(np.mean([v[0] for v in vals]))
            if d in (0.01, 0.16):
                print(f"  seed {seed} offset {d*100:.1f}cm -> actual move "
                      f"{np.mean([v[1] for v in vals])*100:.1f}cm", flush=True)
    env.close()

    print(f"\n{'commanded offset':>17} {'latent L1':>11} {'vs 16cm':>9}")
    big = np.mean(rows[0.16])
    for d in OFFSETS:
        m = float(np.mean(rows[d]))
        print(f"{d*100:>14.1f} cm {m:>11.5f} {100*m/big:>8.1f}%")
    small = float(np.mean(rows[0.02]))
    print(f"\nA 2 cm move -- the grasp tolerance -- shifts the latent by "
          f"{100*small/big:.1f}% of what a 16 cm move does.")
    print("If that is a few percent, latent L1 cannot servo at grasp precision.")
    np.savez("latent_precision.npz", offsets=np.array(OFFSETS),
             l1=np.array([np.mean(rows[d]) for d in OFFSETS]))


if __name__ == "__main__":
    main()
