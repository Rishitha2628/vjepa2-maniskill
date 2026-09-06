"""
Grasp planning with a grasp-aware cost and a multi-step horizon.

Two changes from cem_plan.py, each following directly from probe_grasp.py's
diagnosis that grasp state is well represented but invisible to the objective:

1. COST. Plain L1 to the goal latent is dominated by gross arm/cube position.
   Here the imagined rollout is also scored by a grasp classifier:

       cost = L1(z_final, z_goal) - lam * max_h P(grasped | z_h)

   The max over the horizon matters: it credits a sequence that achieves a
   grasp partway through, instead of only judging the end state.

2. HORIZON. rollout=1 is greedy and cannot represent "close the gripper now,
   which pays off three steps later". Grasping is exactly that shape, so the
   horizon has to be long enough to contain approach-close-lift.

    python3 plan_grasp.py --seeds 6 --lam 0.5 --rollout 3
"""

import argparse

import numpy as np
import torch

import adapter
import loader
import record
from eval_grasp import make_goal
from grasp_cost import GraspScorer


@torch.no_grad()
def score_seqs(predictor, z0, state0, seqs, goal_z, scorer, lam, device, dtype,
               chunk):
    """cost = L1(final, goal) - lam * max_h P(grasped at step h)."""
    out = []
    for i in range(0, len(seqs), chunk):
        s = np.ascontiguousarray(seqs[i:i + chunk])
        n, H = s.shape[0], s.shape[1]
        z = z0.expand(n, -1, -1).contiguous()
        pose = np.repeat(state0[None], n, axis=0)
        best_g = torch.zeros(n, device=device)
        for h in range(H):
            a = torch.as_tensor(s[:, h], device=device, dtype=dtype)[:, None]
            p = torch.as_tensor(pose, device=device, dtype=dtype)[:, None]
            z = loader.predict_next(predictor, z, a, p)
            if scorer is not None and lam > 0:
                best_g = torch.maximum(best_g, scorer(z))
            pose = adapter.compute_new_pose(pose, s[:, h])
        l1 = (z.float() - goal_z.float()).abs().mean(dim=(1, 2))
        out.append((l1 - lam * best_g).cpu().numpy())
    return np.concatenate(out)


def cem_grasp(predictor, z0, state0, goal_z, scorer, lam, device, dtype,
              rollout=3, samples=32, topk=8, iters=3, maxnorm=0.05,
              momentum=0.15, grip_momentum=0.15, chunk=16, rng=None):
    rng = rng or np.random.default_rng(0)
    mean = np.zeros((rollout, 4), np.float32)
    std = np.zeros((rollout, 4), np.float32)
    std[:, :3] = maxnorm
    std[:, 3] = 1.0

    hist = []
    for _ in range(iters):
        s = rng.normal(size=(samples, rollout, 4)).astype(np.float32) * std + mean
        s[:, :, :3] = np.clip(s[:, :, :3], -maxnorm, maxnorm)
        s[:, :, 3] = np.clip(s[:, :, 3], -1.0, 1.0)
        seqs = np.concatenate([s[:, :, :3],
                               np.zeros((samples, rollout, 3), np.float32),
                               s[:, :, 3:4]], axis=-1)
        sc = score_seqs(predictor, z0, state0, seqs, goal_z, scorer, lam,
                        device, dtype, chunk)
        elite = s[np.argsort(sc)[:topk]]
        m, sd = elite.mean(0), elite.std(0)
        mean[:, :3] = m[:, :3] * (1 - momentum) + mean[:, :3] * momentum
        std[:, :3] = sd[:, :3] * (1 - momentum) + std[:, :3] * momentum
        mean[:, 3] = m[:, 3] * (1 - grip_momentum) + mean[:, 3] * grip_momentum
        std[:, 3] = sd[:, 3] * (1 - grip_momentum) + std[:, 3] * grip_momentum
        hist.append(float(sc.min()))

    a = np.zeros(7, np.float32)
    a[:3] = mean[0, :3]
    a[6] = mean[0, 3]
    return a, hist


def episode(encoder, predictor, scorer, lam, seed, steps, camera, goal_z,
            device, dtype, rollout, samples, iters, topk, chunk):
    env = record.make_env("PickCube-v1", 256, camera=camera)
    obs, _ = env.reset(seed=seed)
    e = env.unwrapped
    rng = np.random.default_rng(seed)
    cube0 = e.cube.pose.p[0].cpu().numpy().copy()
    ever, lift, info = False, 0.0, {}
    for _ in range(steps):
        frame = record.obs_frame(obs)
        state = record.obs_state(obs, env)
        z0 = loader.encode(encoder, frame[None], device, dtype)
        act, _ = cem_grasp(predictor, z0, state, goal_z, scorer, lam, device,
                           dtype, rollout=rollout, samples=samples, topk=topk,
                           iters=iters, chunk=chunk, rng=rng)
        cmd = adapter.metric_action_to_maniskill(act)
        obs, _, term, trunc, info = env.step(cmd)
        if bool(e.agent.is_grasping(e.cube)[0]):
            ever = True
        lift = max(lift, float(e.cube.pose.p[0, 2].cpu()) - cube0[2])
        if bool(term) or bool(trunc):
            break
    succ = bool(info["success"][0]) if "success" in info else False
    env.close()
    return ever, lift, succ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.5, 2.0])
    ap.add_argument("--rollout", type=int, default=3)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--camera", default="close")
    ap.add_argument("--ckpt", default="predictor_grasp.pt")
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    goals = [make_goal(s, a.camera) for s in range(a.seeds)]
    print(f"goals built ({sum(g[1] for g in goals)}/{a.seeds} scripted successes)")

    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    sd = torch.load(a.ckpt, map_location="cpu")
    predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    predictor.eval()
    goal_zs = [loader.encode(encoder, g[0][None], device, dtype) for g in goals]
    scorer = GraspScorer(device=device, dtype=torch.float32)
    print(f"grasp probe loaded (AUC {scorer.auc:.3f})")

    print(f"\nrollout={a.rollout}  samples={a.samples}  iters={a.iters}")
    print(f"{'lam':>6} {'grasp':>8} {'lift':>10} {'success':>9}")
    rows = {}
    for lam in a.lam:
        res = [episode(encoder, predictor, scorer, lam, s, a.steps, a.camera,
                       goal_zs[s], device, dtype, a.rollout, a.samples,
                       a.iters, a.topk, a.chunk) for s in range(a.seeds)]
        g = np.mean([r[0] for r in res]); l = np.mean([r[1] for r in res])
        sc = np.mean([r[2] for r in res])
        rows[lam] = (g, l, sc)
        print(f"{lam:>6.2f} {g*100:>7.1f}% {l:>9.4f}m {sc*100:>8.1f}%", flush=True)

    np.savez("plan_grasp.npz", lams=np.array(a.lam),
             grasp=np.array([rows[l][0] for l in a.lam]),
             lift=np.array([rows[l][1] for l in a.lam]),
             succ=np.array([rows[l][2] for l in a.lam]))
    print("\nwrote plan_grasp.npz")


if __name__ == "__main__":
    main()
