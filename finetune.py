"""
Fine-tune the V-JEPA 2-AC predictor on ManiSkill.

Motivated by probe_reps.py: the frozen ViT-g encoder decodes arm and cube
position at R^2 ~ 0.96 on ManiSkill renders, so the scene information is
present. What does not transfer is the action-conditioned dynamics on top of
it -- which is the predictor, and only the predictor.

Objective is the one the model was trained with: L1 between the predicted next
latent and the encoder's latent for the real next frame, both layer-normed.

Memory (6 GB card): the encoder is never loaded, latents are read from the
memmap collect_data.py wrote, and by default only the conditioning pathway plus
the top blocks are trained. `--trainable all` needs roughly 5 GB of optimizer
state and will likely OOM here; it is left available for a bigger card.

The metric that matters is not the loss but `rank_shell` -- the percentile of
the true action's energy among same-norm candidates, the same statistic
camera_ablation.py measured at chance (0.47) for the pretrained model. Loss can
fall while rank_shell stays at chance, which would mean the model learned the
scene's appearance and not its dynamics.

    python3 finetune.py --epochs 4
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import loader
from energy_maniskill import sphere_dirs


def trainable_params(predictor, mode, top_blocks):
    for p in predictor.parameters():
        p.requires_grad_(False)
    named = []
    if mode == "all":
        for n, p in predictor.named_parameters():
            p.requires_grad_(True)
            named.append((n, p))
        return named

    mods = [predictor.action_encoder, predictor.state_encoder]
    if mode == "top":
        mods += [predictor.predictor_norm, predictor.predictor_proj]
        mods += list(predictor.predictor_blocks[-top_blocks:])
    for m in mods:
        for p in m.parameters():
            p.requires_grad_(True)
    for n, p in predictor.named_parameters():
        if p.requires_grad:
            named.append((n, p))
    return named


def make_pairs(episode, steps):
    """Indices i such that i and i+1 are in the same episode."""
    i = np.arange(len(episode) - 1)
    return i[episode[i] == episode[i + 1]]


@torch.no_grad()
def eval_rank_shell(predictor, latents, actions, states, pairs, dirs,
                    device, n_eval, chunk, seed=0):
    """Percentile of the true action among same-norm candidates. 0.5 = chance."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(pairs, size=min(n_eval, len(pairs)), replace=False)
    ranks = []
    for i in sel:
        true = actions[i][:3]
        nt = float(np.linalg.norm(true))
        if nt < 1e-6:
            continue
        cand = np.concatenate([(dirs * nt).astype(np.float32), true[None]], 0)
        cand7 = np.concatenate(
            [cand, np.zeros((len(cand), 4), np.float32)], axis=1)
        z = torch.as_tensor(np.asarray(latents[i]), device=device).float()[None]
        tgt = torch.as_tensor(np.asarray(latents[i + 1]), device=device).float()[None]
        st = torch.as_tensor(states[i], device=device).float()[None, None]
        e = []
        for j in range(0, len(cand7), chunk):
            c = cand7[j:j + chunk]
            a = torch.as_tensor(c, device=device).float()[:, None]
            p = loader.predict_next(predictor, z.expand(len(c), -1, -1),
                                    a, st.expand(len(c), -1, -1))
            e.append((p.float() - tgt.float()).abs().mean(dim=(1, 2)).cpu().numpy())
        e = np.concatenate(e)
        ranks.append(float((e[:-1] < e[-1]).mean()))
    return float(np.mean(ranks)), len(ranks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--trainable", default="top", choices=["cond", "top", "all"])
    ap.add_argument("--top-blocks", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--n-eval", type=int, default=48)
    ap.add_argument("--eval-chunk", type=int, default=32)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--out", default="predictor_ft.pt")
    a = ap.parse_args()

    device = "cuda"
    latents = np.load(os.path.join(a.data, "latents.npy"), mmap_mode="r")
    meta = np.load(os.path.join(a.data, "meta.npz"))
    actions, states, episode = meta["actions"], meta["states"], meta["episode"]
    print(f"dataset: {latents.shape[0]} frames, {episode.max()+1} episodes, "
          f"camera={meta['camera']}")

    n_ep = int(episode.max()) + 1
    n_val = max(1, int(n_ep * a.val_frac))
    val_eps = set(range(n_ep - n_val, n_ep))
    pairs = make_pairs(episode, int(meta["steps"]))
    tr_pairs = np.array([i for i in pairs if episode[i] not in val_eps])
    va_pairs = np.array([i for i in pairs if episode[i] in val_eps])
    print(f"train pairs {len(tr_pairs)}, val pairs {len(va_pairs)} "
          f"({n_val} held-out episodes)")

    predictor = loader.load_predictor(device=device, dtype=torch.float32)
    if a.grad_checkpoint:
        predictor.use_activation_checkpointing = True
    named = trainable_params(predictor, a.trainable, a.top_blocks)
    n_tr = sum(p.numel() for _, p in named)
    print(f"trainable: {n_tr/1e6:.1f}M / "
          f"{sum(p.numel() for p in predictor.parameters())/1e6:.1f}M "
          f"({a.trainable})")

    opt = torch.optim.AdamW([p for _, p in named], lr=a.lr, weight_decay=a.wd)
    dirs = sphere_dirs(48)

    predictor.eval()
    r0, n0 = eval_rank_shell(predictor, latents, actions, states, va_pairs,
                             dirs, device, a.n_eval, a.eval_chunk)
    print(f"\nbefore fine-tuning: val rank_shell = {r0:.4f}  (chance 0.500, n={n0})")
    print(f"{'epoch':>6} {'train L1':>10} {'val L1':>10} {'rank_shell':>11}")

    rng = np.random.default_rng(0)
    best = None
    for ep in range(a.epochs):
        predictor.train()
        order = rng.permutation(tr_pairs)
        tot, nb = 0.0, 0
        for k in range(0, len(order) - a.batch + 1, a.batch):
            idx = np.sort(order[k:k + a.batch])
            z = torch.as_tensor(np.asarray(latents[idx]), device=device).float()
            tgt = torch.as_tensor(np.asarray(latents[idx + 1]), device=device).float()
            act = torch.as_tensor(actions[idx], device=device).float()[:, None]
            st = torch.as_tensor(states[idx], device=device).float()[:, None]

            out = predictor(z, act, st)[:, -loader.TOKENS_PER_FRAME:]
            out = F.layer_norm(out, (out.size(-1),))
            loss = (out - tgt).abs().mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
            opt.step()
            tot += loss.item()
            nb += 1

        predictor.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for k in range(0, len(va_pairs) - a.batch + 1, a.batch):
                idx = va_pairs[k:k + a.batch]
                z = torch.as_tensor(np.asarray(latents[idx]), device=device).float()
                tgt = torch.as_tensor(np.asarray(latents[idx + 1]), device=device).float()
                act = torch.as_tensor(actions[idx], device=device).float()[:, None]
                st = torch.as_tensor(states[idx], device=device).float()[:, None]
                out = loader.predict_next(predictor, z, act, st)
                vl += (out.float() - tgt).abs().mean().item()
                vn += 1
        r, _ = eval_rank_shell(predictor, latents, actions, states, va_pairs,
                               dirs, device, a.n_eval, a.eval_chunk)
        print(f"{ep+1:>6} {tot/max(nb,1):>10.5f} {vl/max(vn,1):>10.5f} {r:>11.4f}",
              flush=True)

        if best is None or r < best:
            best = r
            torch.save({n: p.detach().cpu() for n, p in named}, a.out)

    print(f"\nrank_shell {r0:.4f} -> {best:.4f}  (chance 0.500)")
    print("Lower is better. If this stays near 0.5 the model still has no "
          "usable action signal, whatever the loss did.")
    print(f"saved trainable weights to {a.out}")


if __name__ == "__main__":
    main()
