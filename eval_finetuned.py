"""
Did fine-tuning actually help, or did it just fit the collection run?

finetune.py reports rank_shell on held-out EPISODES, but those episodes come
from the same collection run, same camera and same action distribution as
training. That is a weak form of held-out.

This evaluates on entirely fresh seeds (training used 10000+, this uses 0..N)
and computes statistics over SEEDS, not timesteps -- the correction that made
the camera-ablation effect disappear. Pretrained and fine-tuned models are
scored on the IDENTICAL trajectories, so the comparison is paired.

    python3 eval_finetuned.py --seeds 8 --steps 40
"""

import argparse
import gc

import numpy as np
import torch

import adapter
import loader
import record
from energy_maniskill import energies, pad7, sphere_dirs


def rollout(seed, steps, camera):
    env = record.make_env("PickCube-v1", 256, camera=camera)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    frames, ms, states = [], [], []
    act = np.zeros(7, np.float32)
    for _ in range(steps):
        frames.append(record.obs_frame(obs))
        states.append(record.obs_state(obs, env))
        act = 0.7 * act + 0.3 * rng.normal(size=7).astype(np.float32)
        cmd = np.clip(act * 0.5, -1, 1).astype(np.float32)
        cmd[3:6] = 0.0
        cmd[6] = 1.0
        ms.append(cmd)
        obs, _, _, _, _ = env.step(cmd)
    env.close()
    actions = adapter.maniskill_action_to_metric(np.asarray(ms, np.float32))
    return (np.asarray(frames, np.uint8), actions,
            np.asarray(states, np.float32))


@torch.no_grad()
def seed_rank(predictor, reps, actions, states, dirs, stride, device, dtype, chunk):
    ranks = []
    for t in range(0, len(reps) - 1, stride):
        true = actions[t][:3].astype(np.float32)
        nt = float(np.linalg.norm(true))
        if nt < 1e-6:
            continue
        st = torch.as_tensor(states[t], device=device, dtype=dtype)[None, None]
        zc, zt = reps[t][None].to(dtype), reps[t + 1][None].to(dtype)
        shell = (dirs * nt).astype(np.float32)
        e = energies(predictor, zc, zt, st, pad7(shell), device, dtype, chunk)
        et = energies(predictor, zc, zt, st, pad7(true), device, dtype, chunk)[0]
        ranks.append(float((e < et).mean()))
    return float(np.mean(ranks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--camera", default="close")
    ap.add_argument("--ndirs", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--ckpt", default="predictor_ft.pt")
    a = ap.parse_args()

    device = "cuda"
    dirs = sphere_dirs(a.ndirs)

    # 1. roll fresh seeds and encode once with the frozen encoder
    print(f"rolling {a.seeds} fresh seeds (training used 10000+)...")
    data = [rollout(s, a.steps, a.camera) for s in range(a.seeds)]
    encoder, _ = loader.load_ac_model(device=device, dtype=torch.float16)
    reps = [loader.encode(encoder, f, device, torch.float16).cpu()
            for f, _, _ in data]
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    print("encoded; encoder released")

    results = {}
    for tag in ("pretrained", "finetuned"):
        predictor = loader.load_predictor(device=device, dtype=torch.float32)
        if tag == "finetuned":
            sd = torch.load(a.ckpt, map_location="cpu")
            missing = predictor.load_state_dict(
                {k: v.to(torch.float32) for k, v in sd.items()}, strict=False)
            print(f"loaded {len(sd)} fine-tuned tensors from {a.ckpt}")
        predictor.eval()
        rs = []
        for i, (_, act, st) in enumerate(data):
            r = seed_rank(predictor, reps[i].to(device), act, st, dirs,
                          a.stride, device, torch.float32, a.chunk)
            rs.append(r)
        results[tag] = np.array(rs)
        print(f"{tag:11s} per-seed rank_shell: "
              f"{np.array2string(results[tag], precision=3)}")
        del predictor
        gc.collect()
        torch.cuda.empty_cache()

    p, f = results["pretrained"], results["finetuned"]
    from scipy import stats

    def se(x):
        return x.std(ddof=1) / len(x) ** 0.5

    print(f"\nper-seed mean rank_shell (0 = perfect, 0.5 = chance), n={len(p)} seeds")
    print(f"  pretrained {p.mean():.4f} +- {se(p):.4f}  "
          f"vs chance p={stats.ttest_1samp(p, 0.5).pvalue:.4f}")
    print(f"  finetuned  {f.mean():.4f} +- {se(f):.4f}  "
          f"vs chance p={stats.ttest_1samp(f, 0.5).pvalue:.4f}")
    tt = stats.ttest_rel(f, p)
    print(f"  paired finetuned-vs-pretrained: {(f-p).mean():+.4f}  p={tt.pvalue:.4f}")
    np.savez("eval_finetuned.npz", pretrained=p, finetuned=f)
    print("\nwrote eval_finetuned.npz")


if __name__ == "__main__":
    main()
