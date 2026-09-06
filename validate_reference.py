"""
Validates the whole model stack against upstream's own ground-truth data,
before ManiSkill is involved at all.

Uses vjepa2_src/notebooks/franka_example_traj.npz -- two real Franka frames and
their two poses. The action that connects them is known, so we can sweep a grid
of candidate actions, ask the world model to imagine each one, and check that
the imagined future closest to the real next frame is the one produced by
(approximately) the true action.

If the argmin of this landscape sits near the ground-truth action, then
encode(), predict_next(), the normalization and the pose/action conventions are
all correct. If it does not, nothing downstream can be trusted.

    python3 validate_reference.py
"""

import os
import sys

import numpy as np
import torch

import loader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vjepa2_src"))
from notebooks.utils.mpc_utils import poses_to_diff  # noqa: E402

import argparse

GRID = 0.075
NSAMPLES = 5
CHUNK = 32


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=GRID,
                    help="half-width of the action sweep, metres")
    ap.add_argument("--nsamples", type=int, default=NSAMPLES,
                    help="grid points per axis; total forwards = nsamples**3")
    args = ap.parse_args()
    grid, nsamples = args.grid, args.nsamples
    device, dtype = "cuda", torch.float16
    traj = np.load(os.path.join("vjepa2_src", "notebooks", "franka_example_traj.npz"))
    frames, states = traj["observations"][0], traj["states"][0]   # (2,256,256,3), (2,7)
    gt_action = poses_to_diff(states[0], states[1]).numpy()
    print(f"ground-truth action dxyz = "
          f"({gt_action[0]:+.4f}, {gt_action[1]:+.4f}, {gt_action[2]:+.4f})")

    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    h = loader.encode(encoder, frames, device, dtype)             # (2, 256, D)
    z_ctx, z_tgt = h[0][None], h[1][None]

    # Grid of pure-translation candidate actions.
    axis = np.linspace(-grid, grid, nsamples)
    cands = np.array([[x, y, z, 0, 0, 0, 0] for x in axis for y in axis for z in axis],
                     dtype=np.float32)

    state0 = torch.as_tensor(states[0], device=device, dtype=dtype)[None, None]  # (1,1,7)

    errs = []
    for i in range(0, len(cands), CHUNK):
        c = cands[i:i + CHUNK]
        n = len(c)
        a = torch.as_tensor(c, device=device, dtype=dtype)[:, None]     # (n,1,7)
        z = z_ctx.expand(n, -1, -1).contiguous()
        s = state0.expand(n, -1, -1).contiguous()
        pred = loader.predict_next(predictor, z, a, s)                  # (n,256,D)
        e = (pred.float() - z_tgt.float()).abs().mean(dim=(1, 2))
        errs.append(e.cpu().numpy())
    errs = np.concatenate(errs)

    best = cands[errs.argmin()]
    print(f"argmin action     dxyz = ({best[0]:+.4f}, {best[1]:+.4f}, {best[2]:+.4f})")
    print(f"energy min {errs.min():.5f}  max {errs.max():.5f}  "
          f"spread {errs.max() - errs.min():.5f}")

    # A model that ignores actions produces a flat landscape; one that uses them
    # produces a minimum near the true displacement.
    dist = np.linalg.norm(best[:3] - gt_action[:3])
    rel = (errs.max() - errs.min()) / (abs(errs.mean()) + 1e-9)
    print(f"\ndistance(argmin, ground truth) = {dist:.4f} m  (grid step "
          f"{axis[1]-axis[0]:.4f} m)")
    print(f"relative energy spread         = {rel:.4f}")
    if rel < 0.01:
        print("FAIL: landscape is flat -- the predictor is ignoring the action input.")
    elif dist <= (axis[1] - axis[0]) * 1.5:
        print("PASS: energy minimum sits at the ground-truth action.")
    else:
        print("SUSPECT: landscape has structure but its minimum is off. Check "
              "action scaling and frame conventions.")

    np.savez("reference_energy.npz", cands=cands, errs=errs, gt_action=gt_action)


if __name__ == "__main__":
    main()
