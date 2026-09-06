"""
Why did planned grasping fail?

eval_grasp.py: the fine-tuned world model grasps 0/6, same as random, while a
scripted controller grasps 6/6. So the task is reachable and the planner is the
problem. There are two candidate explanations and they imply opposite fixes:

  A. The ENCODER cannot see grasp state. If "fingers around cube" and "fingers
     beside cube" map to nearly the same latent, then no predictor and no
     search can ever score a grasp, because the goal comparison is blind to it.
     Fix: not fine-tuning -- you would need a different representation.

  B. The encoder sees it but the PREDICTOR/planner cannot exploit it.
     Fix: more/better data, longer horizon, better search.

This distinguishes them with linear probes on the ALREADY-CACHED latents from
collect_grasp_data.py, so it costs no simulation and no encoding:

    gripper closedness   regression, R^2
    tcp-cube distance    regression, R^2
    is_grasped           classification, balanced accuracy + AUC

If is_grasped is not decodable, explanation A holds.

    python3 probe_grasp.py
"""

import argparse

import numpy as np


def ridge_fit(X, Y, alpha, tr, te):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    A = (X[tr] - mu) / sd
    B = (X[te] - mu) / sd
    A = np.concatenate([A, np.ones((len(A), 1))], 1)
    B = np.concatenate([B, np.ones((len(B), 1))], 1)
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ Y[tr])
    return B @ W


def r2(y, p):
    ss = ((y - p) ** 2).sum(0)
    tot = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss / (tot + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_grasp")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--max-n", type=int, default=4000)
    a = ap.parse_args()

    lat = np.load(f"{a.data}/latents.npy", mmap_mode="r")
    meta = np.load(f"{a.data}/meta.npz")
    states, grasped, ep = meta["states"], meta["grasped"], meta["episode"]

    rng = np.random.default_rng(0)
    # keep every grasped frame, subsample the rest: only 13% are grasps
    gi = np.where(grasped)[0]
    ni = rng.choice(np.where(~grasped)[0],
                    size=min(a.max_n - len(gi), int((~grasped).sum())),
                    replace=False)
    idx = np.sort(np.concatenate([gi, ni]))
    print(f"probing {len(idx)} frames ({len(gi)} grasped, {len(ni)} not)")

    # Pool in chunks: materialising all frames at once is (N,256,1408) float32,
    # which is ~5.8 GB for N=4000 and gets the process OOM-killed silently.
    X = np.empty((len(idx), lat.shape[2]), np.float64)
    CH = 128
    for i in range(0, len(idx), CH):
        blk = np.asarray(lat[idx[i:i + CH]], dtype=np.float32)
        X[i:i + CH] = blk.mean(axis=1).astype(np.float64)
        del blk
    g = grasped[idx].astype(np.float64)
    clos = states[idx, 6].astype(np.float64)

    # split by EPISODE so train and test never share a trajectory
    eps = ep[idx]
    ue = np.unique(eps)
    rng.shuffle(ue)
    te_ep = set(ue[: max(1, len(ue) // 4)])
    te = np.array([i for i, e in enumerate(eps) if e in te_ep])
    tr = np.array([i for i, e in enumerate(eps) if e not in te_ep])
    print(f"train {len(tr)} / test {len(te)} frames, split by episode")

    print("\n--- regression (held-out R^2) ---")
    for name, Y in (("gripper closedness", clos[:, None]),):
        p = ridge_fit(X, Y, a.alpha, tr, te)
        print(f"  {name:20s} R2 = {r2(Y[te], p):+.3f}")

    print("\n--- is_grasped (held-out classification) ---")
    p = ridge_fit(X, g[:, None], a.alpha, tr, te).ravel()
    yt = g[te]
    if yt.std() < 1e-9:
        print("  test split has one class only; rerun with another seed")
        return
    thr = np.median(p)
    pred = (p > thr).astype(float)
    tpr = float(((pred == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1))
    tnr = float(((pred == 0) & (yt == 0)).sum() / max((yt == 0).sum(), 1))
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(len(p))
    npos, nneg = (yt == 1).sum(), (yt == 0).sum()
    auc = float((ranks[yt == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg))
    print(f"  balanced accuracy = {(tpr+tnr)/2:.3f}   (chance 0.500)")
    print(f"  AUC               = {auc:.3f}   (chance 0.500)")
    print(f"  base rate         = {yt.mean():.3f}")

    print("\nInterpretation:")
    if auc < 0.65:
        print("  A: the encoder barely represents grasp state. Fine-tuning the")
        print("     predictor cannot fix this -- the goal comparison is blind to")
        print("     whether the cube is held.")
    else:
        print("  B: grasp state IS decodable from the frozen encoder. The")
        print("     bottleneck is the predictor/search, so more data, a longer")
        print("     horizon or a better objective are the things to try.")


if __name__ == "__main__":
    main()
