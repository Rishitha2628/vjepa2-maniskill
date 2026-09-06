"""
A grasp-aware planning cost.

probe_grasp.py showed the frozen encoder represents grasp state well
(is_grasped AUC 0.952), yet planning with plain L1 to a goal latent grasps 0/6.
Those are consistent: a linear probe may amplify one direction out of 1408,
while L1 weights every dimension equally, so "cube held" is negligible next to
gross arm and cube position.

The fix is to put the probe back into the cost:

    cost = L1(z_pred, z_goal) - lam * P(grasped | z_pred)

so the planner gets credit for imagined futures in which the cube is actually
held, at a scale we control rather than one the metric happens to give.

`fit` trains the probe on the cached latents and stores it; `GraspScorer`
applies it to predicted latents inside CEM.

    python3 grasp_cost.py --fit
"""

import argparse

import numpy as np
import torch


PROBE_FILE = "grasp_probe.npz"


def fit(data="data_grasp", alpha=10.0, max_n=4000, out=PROBE_FILE, seed=0):
    lat = np.load(f"{data}/latents.npy", mmap_mode="r")
    meta = np.load(f"{data}/meta.npz")
    grasped, ep = meta["grasped"], meta["episode"]

    rng = np.random.default_rng(seed)
    gi = np.where(grasped)[0]
    ni = rng.choice(np.where(~grasped)[0],
                    size=min(max_n - len(gi), int((~grasped).sum())), replace=False)
    idx = np.sort(np.concatenate([gi, ni]))

    X = np.empty((len(idx), lat.shape[2]), np.float64)
    for i in range(0, len(idx), 128):
        blk = np.asarray(lat[idx[i:i + 128]], dtype=np.float32)
        X[i:i + 128] = blk.mean(axis=1).astype(np.float64)
        del blk
    y = grasped[idx].astype(np.float64)

    eps = ep[idx]
    ue = np.unique(eps); rng.shuffle(ue)
    te_ep = set(ue[: max(1, len(ue) // 4)])
    te = np.array([i for i, e in enumerate(eps) if e in te_ep])
    tr = np.array([i for i, e in enumerate(eps) if e not in te_ep])

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    A = np.concatenate([(X[tr] - mu) / sd, np.ones((len(tr), 1))], 1)
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y[tr])
    w, b = W[:-1], W[-1]

    B = (X[te] - mu) / sd
    p = B @ w + b
    yt = y[te]
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(len(p))
    npos, nneg = (yt == 1).sum(), (yt == 0).sum()
    auc = float((ranks[yt == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg))
    print(f"fitted grasp probe on {len(idx)} frames; held-out AUC = {auc:.3f}")

    # scale so the score spans roughly [0,1] over observed data
    lo, hi = np.percentile(p, 2), np.percentile(p, 98)
    np.savez(out, w=w, b=b, mu=mu, sd=sd, lo=lo, hi=hi, auc=auc)
    print(f"wrote {out}  (score range {lo:.3f}..{hi:.3f})")
    return auc


class GraspScorer:
    """P(grasped) for predicted latents, as a torch op on the GPU."""

    def __init__(self, path=PROBE_FILE, device="cuda", dtype=torch.float32):
        d = np.load(path)
        self.w = torch.as_tensor(d["w"], device=device, dtype=dtype)
        self.b = float(d["b"])
        self.mu = torch.as_tensor(d["mu"], device=device, dtype=dtype)
        self.sd = torch.as_tensor(d["sd"], device=device, dtype=dtype)
        self.lo, self.hi = float(d["lo"]), float(d["hi"])
        self.auc = float(d["auc"])

    def __call__(self, z):
        """z: (B, tokens, D) predicted latents -> (B,) score in ~[0,1]."""
        f = z.float().mean(dim=1)
        s = ((f - self.mu) / self.sd) @ self.w + self.b
        return ((s - self.lo) / (self.hi - self.lo + 1e-9)).clamp(0.0, 1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--data", default="data_grasp")
    ap.add_argument("--alpha", type=float, default=10.0)
    a = ap.parse_args()
    if a.fit:
        fit(data=a.data, alpha=a.alpha)
    else:
        ap.print_help()
