"""
Fit position decoders from frozen-encoder latents.

latent_vs_task.py showed why every grasp attempt stalls: the argmin of latent
L1 sits ~6 cm from the true goal pose, so goal-image matching cannot servo to
the ~2 cm a grasp needs, no matter how good the model or the search is.

But probe_reps.py already showed the information is there -- the frozen encoder
decodes arm position at R^2 0.956 and cube position at R^2 0.962. So the fix is
not a better model, it is a better *readout*: decode positions from the latent
and plan in metres, where the optimum is where we actually want it.

This fits and saves those decoders. state_planner.py then uses them as the
planning cost.

    python3 fit_decoders.py --episodes 90 --steps 16
"""

import argparse

import numpy as np
import torch

from vjepa.core import loader
from vjepa.core import record


def collect(episodes, steps, camera, seed0):
    frames, tcps, cubes = [], [], []
    env = record.make_env("PickCube-v1", 256, camera=camera)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        e = env.unwrapped
        rng = np.random.default_rng(seed0 + ep)
        act = np.zeros(7, np.float32)
        for _ in range(steps):
            frames.append(record.obs_frame(obs))
            tcps.append(record.to_np(obs["extra"]["tcp_pose"][0])[:3])
            cubes.append(record.to_np(e.cube.pose.p[0]))
            act = 0.7 * act + 0.3 * rng.normal(size=7).astype(np.float32)
            cmd = np.clip(act * 0.7, -1, 1).astype(np.float32)
            cmd[3:6] = 0.0
            cmd[6] = 1.0 if rng.random() < 0.7 else -1.0
            obs, *_ = env.step(cmd)
    env.close()
    return (np.asarray(frames, np.uint8), np.asarray(tcps, np.float32),
            np.asarray(cubes, np.float32))


def ridge(X, Y, alpha, tr, te):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    A = np.concatenate([(X[tr] - mu) / sd, np.ones((len(tr), 1))], 1)
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ Y[tr])
    B = np.concatenate([(X[te] - mu) / sd, np.ones((len(te), 1))], 1)
    pred = B @ W
    err = np.linalg.norm(pred - Y[te], axis=1)
    ss = ((Y[te] - pred) ** 2).sum(0)
    tot = ((Y[te] - Y[tr].mean(0)) ** 2).sum(0)
    return W, mu, sd, float(np.mean(1 - ss / (tot + 1e-12))), float(err.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=90)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--camera", default="close")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=40_000)
    ap.add_argument("--out", default="checkpoints/decoders.npz")
    a = ap.parse_args()

    dev, dt = "cuda", torch.float16
    frames, tcp, cube = collect(a.episodes, a.steps, a.camera, a.seed0)
    print(f"collected {len(frames)} frames")

    enc, _ = loader.load_ac_model(device=dev, dtype=dt)
    feats = []
    for i in range(0, len(frames), a.batch):
        h = loader.encode(enc, frames[i:i + a.batch], dev, dt)
        feats.append(h.float().mean(1).cpu().numpy())
    X = np.concatenate(feats).astype(np.float64)

    n = len(X); idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]

    Wt, mut, sdt, r2t, errt = ridge(X, tcp.astype(np.float64), a.alpha, tr, te)
    Wc, muc, sdc, r2c, errc = ridge(X, cube.astype(np.float64), a.alpha, tr, te)
    # The quantity planning actually needs is the gripper->cube offset. Decoding
    # it directly beats differencing two independently-noisy position estimates,
    # because errors shared between them cancel.
    delta = (cube - tcp).astype(np.float64)
    Wd, mud, sdd, r2d, errd = ridge(X, delta, a.alpha, tr, te)

    print(f"tcp   decoder: mean position error {errt*100:.2f} cm  (R2 {r2t:+.3f})")
    print(f"cube  decoder: mean position error {errc*100:.2f} cm  (R2 {r2c:+.3f}, "
          f"z is constant so R2 is meaningless here)")
    print(f"DELTA decoder: mean error {errd*100:.2f} cm   <- what the planner uses")
    print(f"differencing tcp+cube would give ~{np.hypot(errt, errc)*100:.2f} cm")
    print(f"(latent matching bottoms out at ~6 cm; a grasp needs ~2 cm)")

    np.savez(a.out, Wt=Wt, mut=mut, sdt=sdt, Wc=Wc, muc=muc, sdc=sdc,
             Wd=Wd, mud=mud, sdd=sdd, r2t=r2t, r2c=r2c, errt=errt, errc=errc,
             errd=errd)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
