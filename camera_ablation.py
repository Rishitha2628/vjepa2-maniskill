"""
Does a DROID-like camera recover usable action signal on ManiSkill?

V-JEPA 2-AC was trained on DROID: a close third-person view where the gripper
and the object fill much of the frame. ManiSkill's default `base_camera` is a
wide-FOV view from 0.6 m up, leaving the cube a few pixels across. That is the
most obvious confound behind the weak transfer in energy_maniskill.py, and it
is cheap to test.

The comparison is paired: for each seed both cameras render the SAME executed
trajectory, so actions, states and dynamics are identical and only pixels differ.

Statistics are computed over SEEDS, not timesteps. Timesteps inside one
trajectory are strongly correlated, so treating them as independent samples
inflates significance -- an earlier version of this analysis did exactly that
and reported p=0.019 off a single trajectory.

Metric: rank_shell, the percentile of the true action's energy among candidate
actions of the same norm (0 = best, 0.5 = chance). Direction only; the
magnitude prior is factored out by construction.

    python3 camera_ablation.py --seeds 6 --steps 40
"""

import argparse

import numpy as np
import torch

import loader
import record
from energy_maniskill import energies, pad7, sphere_dirs


def trajectory(seed, steps, camera, res=256):
    """Re-render one seed under one camera. Identical actions across cameras."""
    import importlib

    import gymnasium as gym

    env = record.make_env("PickCube-v1", res, camera=camera)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    frames, ms_actions, states = [], [], []
    act = np.zeros(7, np.float32)
    for _ in range(steps):
        frames.append(record.obs_frame(obs))
        states.append(record.obs_state(obs, env))
        step = rng.normal(0.0, 1.0, size=7).astype(np.float32)
        act = 0.7 * act + 0.3 * step
        cmd = np.clip(act * 0.5, -1.0, 1.0).astype(np.float32)
        cmd[3:6] = 0.0
        cmd[6] = 1.0
        ms_actions.append(cmd)
        obs, _, _, _, _ = env.step(cmd)
    env.close()

    import adapter
    actions = adapter.maniskill_action_to_metric(np.asarray(ms_actions, np.float32))
    return (np.asarray(frames, np.uint8), actions,
            np.asarray(states, np.float32))


@torch.no_grad()
def mean_rank(encoder, predictor, frames, actions, states, dirs, stride,
              device, dtype, chunk):
    reps = loader.encode(encoder, frames, device, dtype)
    ranks = []
    for t in range(0, len(frames) - 1, stride):
        true = actions[t][:3].astype(np.float32)
        nt = float(np.linalg.norm(true))
        if nt < 1e-6:
            continue
        st = torch.as_tensor(states[t], device=device, dtype=dtype)[None, None]
        zc, zt = reps[t][None], reps[t + 1][None]
        shell = (dirs * nt).astype(np.float32)
        e = energies(predictor, zc, zt, st, pad7(shell), device, dtype, chunk)
        e_true = energies(predictor, zc, zt, st, pad7(true), device, dtype, chunk)[0]
        ranks.append(float((e < e_true).mean()))
    return float(np.mean(ranks)), len(ranks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--ndirs", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    dirs = sphere_dirs(a.ndirs)

    out = {"default": [], "close": []}
    print(f"{'seed':>5} {'default':>10} {'close':>10} {'delta':>10}")
    for seed in range(a.seeds):
        row = {}
        for cam in ("default", "close"):
            fr, ac, st = trajectory(seed, a.steps, cam)
            r, n = mean_rank(encoder, predictor, fr, ac, st, dirs, a.stride,
                             device, dtype, a.chunk)
            row[cam] = r
            out[cam].append(r)
        print(f"{seed:>5} {row['default']:>10.4f} {row['close']:>10.4f} "
              f"{row['close']-row['default']:>+10.4f}")

    d = np.array(out["default"])
    c = np.array(out["close"])
    from scipy import stats

    def se(x):
        return x.std(ddof=1) / len(x) ** 0.5

    print(f"\nper-seed mean rank_shell (0 = perfect, 0.5 = chance), n={len(d)} seeds")
    print(f"  default {d.mean():.4f} +- {se(d):.4f}   "
          f"vs chance p={stats.ttest_1samp(d, 0.5).pvalue:.4f}")
    print(f"  close   {c.mean():.4f} +- {se(c):.4f}   "
          f"vs chance p={stats.ttest_1samp(c, 0.5).pvalue:.4f}")
    pr = stats.ttest_rel(c, d)
    print(f"  paired close-vs-default: {(c-d).mean():+.4f}  p={pr.pvalue:.4f}")

    np.savez("camera_ablation.npz", default=d, close=c)
    print("\nwrote camera_ablation.npz")


if __name__ == "__main__":
    main()
