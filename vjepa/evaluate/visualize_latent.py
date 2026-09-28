"""Renders what the model predicts. V-JEPA has no pixel decoder, so each frame's
16x16 grid of 1408-dim tokens is projected to RGB with PCA. Shows the real next
frame, the imagined one, and the per-patch error between them.

    python3 -m vjepa.evaluate.visualize_latent
"""

import argparse

import numpy as np
import torch

from vjepa.core import loader

GRID = 16


def pca_rgb(tokens, mean, comps, lo, hi):
    """(N,1408) -> (16,16,3) uint8, using a PCA basis shared across all frames."""
    x = (tokens - mean) @ comps.T                      # (N,3)
    x = (x - lo) / (hi - lo + 1e-9)
    x = np.clip(x, 0, 1).reshape(GRID, GRID, 3)
    return (x * 255).astype(np.uint8)


def upscale(img, k):
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="datasets/traj_close.npz")
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 10, 16, 22])
    ap.add_argument("--ckpt", default="checkpoints/predictor_all.pt")
    ap.add_argument("--out", default="media/latent_view.png")
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    d = np.load(a.traj)
    frames, actions, states = d["frames"], d["actions"], d["states"]

    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    sd = torch.load(a.ckpt, map_location="cpu")
    predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    predictor.eval()

    reps = loader.encode(encoder, frames, device, dtype).float()      # (T,256,D)
    print(f"encoded {len(frames)} frames -> {tuple(reps.shape)}")

    # one PCA basis for everything, so colours are comparable across panels
    flat = reps.reshape(-1, reps.shape[-1]).cpu().numpy()
    mean = flat.mean(0)
    _, _, vt = np.linalg.svd(flat[::7] - mean, full_matrices=False)
    comps = vt[:3]
    proj = (flat - mean) @ comps.T
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)

    rows = []
    for t in a.steps:
        z = reps[t][None].to(dtype)
        act = torch.as_tensor(actions[t], device=device, dtype=dtype)[None, None]
        st = torch.as_tensor(states[t], device=device, dtype=dtype)[None, None]
        pred = loader.predict_next(predictor, z, act, st).float()[0].cpu().numpy()
        true = reps[t + 1].cpu().numpy()

        err = np.abs(pred - true).mean(-1).reshape(GRID, GRID)
        err = (err - err.min()) / (err.max() - err.min() + 1e-9)
        heat = np.stack([err, 1 - err, np.zeros_like(err)], -1)       # red = wrong

        panels = [frames[t], frames[t + 1],
                  upscale(pca_rgb(true, mean, comps, lo, hi), 16),
                  upscale(pca_rgb(pred, mean, comps, lo, hi), 16),
                  upscale((heat * 255).astype(np.uint8), 16)]
        sep = np.full((256, 4, 3), 255, np.uint8)
        rows.append(np.concatenate([p for pair in zip(panels, [sep] * 5)
                                    for p in pair][:-1], axis=1))
        print(f"  t={t}: mean |pred-true| = {np.abs(pred-true).mean():.4f}")

    gap = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    sheet = np.concatenate([r for pair in zip(rows, [gap] * len(rows))
                            for r in pair][:-1], axis=0)
    import imageio
    imageio.imwrite(a.out, sheet)
    print(f"\ncolumns: frame t | frame t+1 | latent t+1 (real) | "
          f"latent t+1 (PREDICTED) | error")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
