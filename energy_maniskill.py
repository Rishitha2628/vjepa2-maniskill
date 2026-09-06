"""
Does the world model carry usable action signal on ManiSkill frames?

openloop.py shows absolute prediction error in ManiSkill is high and saturates.
But absolute error is not what a planner consumes: CEM only compares candidate
actions against each other at the same timestep.

A naive grid sweep is misleading here, because the energy landscape has a strong
MAGNITUDE bias -- the lowest-energy action is almost always "don't move", which
swamps any directional preference. So this measures the two effects separately:

  magnitude bias : norm of the grid argmin vs norm of the true action
  direction      : among candidates of the SAME norm as the true action, does
                   the true direction score well?

Directional metrics (the ones that decide whether CEM can work):
  cos_shell   cosine between best-on-shell direction and true direction
  rank_shell  percentile of the true direction among same-norm candidates
              (0 = best, 0.5 = chance)

    python3 energy_maniskill.py --frames 8
"""

import argparse

import numpy as np
import torch

import loader


@torch.no_grad()
def energies(predictor, z_ctx, z_tgt, state, cands, device, dtype, chunk):
    out = []
    for i in range(0, len(cands), chunk):
        c = np.ascontiguousarray(cands[i:i + chunk])
        n = len(c)
        a = torch.as_tensor(c, device=device, dtype=dtype)[:, None]
        z = z_ctx.expand(n, -1, -1).contiguous()
        s = state.expand(n, -1, -1).contiguous()
        p = loader.predict_next(predictor, z, a, s)
        out.append((p.float() - z_tgt.float()).abs().mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def pad7(dxyz):
    dxyz = np.atleast_2d(dxyz).astype(np.float32)
    return np.concatenate([dxyz, np.zeros((len(dxyz), 4), np.float32)], axis=1)


def sphere_dirs(n, seed=0):
    """n roughly-uniform unit directions."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="traj.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--nsamples", type=int, default=5, help="grid points per axis")
    ap.add_argument("--grid", type=float, default=0.06)
    ap.add_argument("--ndirs", type=int, default=64, help="directions on the shell")
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    d = np.load(a.traj)
    frames, actions, states = d["frames"], d["actions"], d["states"]

    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    reps = loader.encode(encoder, frames, device, dtype)

    axis = np.linspace(-a.grid, a.grid, a.nsamples)
    grid = np.array([[x, y, z] for x in axis for y in axis for z in axis], np.float32)
    dirs = sphere_dirs(a.ndirs)

    idxs = np.linspace(0, len(frames) - 2, a.frames).astype(int)
    rows = []
    print(f"{'t':>4} {'spread':>8} {'|argmin|':>9} {'|true|':>8} "
          f"{'cos_shell':>10} {'rank_shell':>11}")
    for t in idxs:
        st = torch.as_tensor(states[t], device=device, dtype=dtype)[None, None]
        zc, zt = reps[t][None], reps[t + 1][None]
        true = actions[t][:3].astype(np.float32)
        nt = float(np.linalg.norm(true))

        e_grid = energies(predictor, zc, zt, st, pad7(grid), device, dtype, a.chunk)
        spread = (e_grid.max() - e_grid.min()) / (abs(e_grid.mean()) + 1e-9)
        n_argmin = float(np.linalg.norm(grid[e_grid.argmin()]))

        # Same-norm shell: isolates direction from the magnitude bias.
        shell = (dirs * nt).astype(np.float32)
        e_shell = energies(predictor, zc, zt, st, pad7(shell), device, dtype, a.chunk)
        best_dir = shell[e_shell.argmin()]
        cos = float(best_dir @ true / (nt * nt)) if nt > 1e-9 else np.nan
        e_true = energies(predictor, zc, zt, st, pad7(true), device, dtype, a.chunk)[0]
        rank = float((e_shell < e_true).mean())

        rows.append((spread, n_argmin, nt, cos, rank))
        print(f"{t:>4} {spread:>8.4f} {n_argmin:>9.4f} {nt:>8.4f} "
              f"{cos:>10.3f} {rank:>11.3f}")

    sp, na, ntv, co, rk = map(np.array, zip(*rows))
    print(f"\nmean spread {sp.mean():.4f}")
    print(f"magnitude bias : mean |argmin| {na.mean():.4f} vs mean |true| {ntv.mean():.4f}")
    print(f"direction      : mean cos_shell {co.mean():+.3f}  "
          f"mean rank_shell {rk.mean():.3f}   (chance: 0.000 / 0.500)")

    if sp.mean() < 0.01:
        print("\nFLAT: no action signal at all. Planning cannot work.")
    elif rk.mean() < 0.35 and co.mean() > 0.15:
        print("\nUSABLE: the true direction scores better than chance. "
              "Short-horizon CEM is worth running.")
    else:
        print("\nMARGINAL: energy varies with the action but does not track the "
              "true direction. Planning would be weak.")

    np.savez("energy_maniskill.npz", spread=sp, argmin_norm=na, true_norm=ntv,
             cos_shell=co, rank_shell=rk, idxs=idxs)


if __name__ == "__main__":
    main()
