"""
Does the frozen ViT-g encoder actually see ManiSkill?

Before fine-tuning the predictor it is worth knowing whether the encoder's
representation contains the state at all. The predictor can only work with what
the encoder gives it, and the encoder is frozen and trained on real video --
ManiSkill renders are out of distribution for it.

This fits a ridge regression from the (mean-pooled) frame representation to
ground-truth positions and reports held-out R^2:

    tcp   end-effector xyz
    cube  cube xyz

R^2 near 1 -> the state is linearly decodable; the reps are fine and a
             predictor fine-tune is the right next step.
R^2 near 0 -> the encoder is the bottleneck, not the predictor. Fine-tuning the
             predictor on blind features cannot help.

A shuffled-label control is included, because R^2 on a small sample with 1408
features can look respectable purely by overfitting.

    python3 probe_reps.py --episodes 40 --steps 16
"""

import argparse

import numpy as np
import torch

import loader
import record


def collect(episodes, steps, camera, seed0=0):
    """Roll random actions across many seeds to vary arm and cube placement."""
    frames, tcps, cubes = [], [], []
    env = record.make_env("PickCube-v1", 256, camera=camera)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        rng = np.random.default_rng(1000 + ep)
        act = np.zeros(7, np.float32)
        for _ in range(steps):
            frames.append(record.obs_frame(obs))
            tcps.append(record.to_np(obs["extra"]["tcp_pose"][0])[:3])
            cubes.append(record.to_np(env.unwrapped.cube.pose.p[0]))
            act = 0.7 * act + 0.3 * rng.normal(size=7).astype(np.float32)
            cmd = np.clip(act * 0.6, -1, 1).astype(np.float32)
            cmd[3:6] = 0.0
            obs, _, _, _, _ = env.step(cmd)
    env.close()
    return (np.asarray(frames, np.uint8), np.asarray(tcps, np.float32),
            np.asarray(cubes, np.float32))


def ridge_r2(X, Y, alpha=1.0, split=0.75, seed=0, per_dim=False):
    """Held-out R^2. Dimensions with (near-)zero variance are skipped rather
    than averaged in: the cube's z never changes because it sits on the table,
    and dividing by its ~0 variance produces a meaningless huge negative R^2."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    k = int(len(X) * split)
    tr, te = idx[:k], idx[k:]
    Xtr, Xte = X[tr], X[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Y[tr])
    pred = Xte @ W
    ss_res = ((Y[te] - pred) ** 2).sum(0)
    ss_tot = ((Y[te] - Y[tr].mean(0)) ** 2).sum(0)
    var = Y[te].var(0)
    ok = var > 1e-8
    r2 = np.where(ok, 1 - ss_res / np.where(ok, ss_tot, 1.0), np.nan)
    if per_dim:
        return r2, ok
    return float(np.nanmean(r2)) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--camera", default="close", choices=list(record.CAMERAS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=10.0)
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    frames, tcp, cube = collect(a.episodes, a.steps, a.camera)
    print(f"collected {len(frames)} frames from {a.episodes} episodes "
          f"(camera={a.camera})")

    encoder, _ = loader.load_ac_model(device=device, dtype=dtype)
    feats = []
    for i in range(0, len(frames), a.batch):
        h = loader.encode(encoder, frames[i:i + a.batch], device, dtype)
        feats.append(h.float().mean(dim=1).cpu().numpy())   # mean-pool tokens
    X = np.concatenate(feats).astype(np.float64)
    print(f"features {X.shape}")

    np.savez("probe_feats.npz", X=X.astype(np.float32), tcp=tcp, cube=cube)
    rng = np.random.default_rng(0)
    for name, Y in (("tcp", tcp), ("cube", cube)):
        Y = Y.astype(np.float64)
        r2d, ok = ridge_r2(X, Y, alpha=a.alpha, per_dim=True)
        Ysh = Y[rng.permutation(len(Y))]
        r2s = ridge_r2(X, Ysh, alpha=a.alpha)
        axes = " ".join(
            f"{ax}={r2d[i]:+.3f}" if ok[i] else f"{ax}=n/a(const)"
            for i, ax in enumerate("xyz"))
        mean = float(np.nanmean(r2d)) if ok.any() else float("nan")
        print(f"  {name:5s} {axes}   mean R2 = {mean:+.3f}  "
              f"(shuffled control {r2s:+.3f})")
        print(f"        target std per axis: "
              f"{np.array2string(Y.std(0), precision=4)}")

    print("\nR2 near 1 -> encoder reps carry the state; fine-tune the predictor.")
    print("R2 near 0 -> the encoder is the bottleneck, not the predictor.")


if __name__ == "__main__":
    main()
