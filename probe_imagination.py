"""
Can the predictor IMAGINE a grasp?

Planning with a grasp-aware cost still grasps 0/6. There is a failure mode that
would explain that completely, and it sits upstream of any cost function:

    if the predictor's imagined next latent never looks "grasped", then
    P(grasped | z_pred) is low for EVERY candidate action, the grasp term is
    constant, and re-weighting it changes nothing.

This tests exactly that, on cached data with known grasp onsets. For each
transition where the cube goes from not-held to held, compare the grasp probe's
score on:

    z_t        the real frame before the grasp          (should be low)
    z_{t+1}    the real frame after the grasp           (should be high)
    z_pred     the predictor's imagined next frame,
               given the TRUE action that caused it     (the question)

If z_pred tracks z_t instead of z_{t+1}, the predictor is not modelling the
contact event, and the cost was never the bottleneck.

A non-grasp control set is included so "scores rose" cannot be confused with
"scores drift upward generally".

    python3 probe_imagination.py
"""

import argparse

import numpy as np
import torch

import loader
from grasp_cost import GraspScorer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_grasp")
    ap.add_argument("--ckpt", default="predictor_grasp.pt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    device, dtype = "cuda", torch.float32
    lat = np.load(f"{a.data}/latents.npy", mmap_mode="r")
    meta = np.load(f"{a.data}/meta.npz")
    actions, states, grasped, ep = (meta["actions"], meta["states"],
                                    meta["grasped"], meta["episode"])

    same = ep[:-1] == ep[1:]
    onset = np.where(same & ~grasped[:-1] & grasped[1:])[0]
    hold = np.where(same & ~grasped[:-1] & ~grasped[1:])[0]
    rng = np.random.default_rng(0)
    onset = onset[:a.n] if len(onset) <= a.n else rng.choice(onset, a.n, False)
    hold = rng.choice(hold, min(a.n, len(hold)), replace=False)
    print(f"{len(onset)} grasp onsets, {len(hold)} non-grasp controls")

    predictor = loader.load_predictor(device=device, dtype=dtype)
    sd = torch.load(a.ckpt, map_location="cpu")
    predictor.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    predictor.eval()
    scorer = GraspScorer(device=device, dtype=dtype)

    def run(idx, label):
        s_cur, s_nxt, s_pred = [], [], []
        for i in range(0, len(idx), a.batch):
            j = np.sort(idx[i:i + a.batch])
            z = torch.as_tensor(np.asarray(lat[j]), device=device).float()
            zn = torch.as_tensor(np.asarray(lat[j + 1]), device=device).float()
            act = torch.as_tensor(actions[j], device=device).float()[:, None]
            st = torch.as_tensor(states[j], device=device).float()[:, None]
            zp = loader.predict_next(predictor, z, act, st)
            s_cur.append(scorer(z).cpu().numpy())
            s_nxt.append(scorer(zn).cpu().numpy())
            s_pred.append(scorer(zp).cpu().numpy())
        c, n, p = (np.concatenate(x) for x in (s_cur, s_nxt, s_pred))
        print(f"\n{label}")
        print(f"  real  z_t      P(grasped) = {c.mean():.3f}")
        print(f"  real  z_(t+1)  P(grasped) = {n.mean():.3f}")
        print(f"  IMAGINED z_pred          = {p.mean():.3f}")
        return c, n, p

    c1, n1, p1 = run(onset, "GRASP ONSET (not held -> held)")
    c0, n0, p0 = run(hold, "CONTROL (stays not held)")

    real_jump = (n1 - c1).mean() - (n0 - c0).mean()
    imag_jump = (p1 - c1).mean() - (p0 - c0).mean()
    print(f"\nreal     jump attributable to the grasp: {real_jump:+.3f}")
    print(f"imagined jump attributable to the grasp: {imag_jump:+.3f}")
    frac = imag_jump / real_jump if abs(real_jump) > 1e-6 else 0.0
    print(f"the predictor captures {100*frac:.0f}% of the real grasp signal")

    if frac < 0.15:
        print("\nThe predictor cannot imagine the grasp. Re-weighting the cost")
        print("cannot help: P(grasped|z_pred) is near-constant across actions,")
        print("so there is nothing for CEM to optimise. The predictor itself is")
        print("the bottleneck.")
    else:
        print("\nThe predictor does partly imagine grasps, so the cost/search")
        print("remains the place to look.")


if __name__ == "__main__":
    main()
