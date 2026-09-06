"""
Can the world model plan an actual grasp?

Measures real task outcomes, not latent-space proxies:

    grasp rate   did the gripper ever actually hold the cube
    lift         how far the cube rose off the table
    success      ManiSkill's own PickCube success flag

Baselines matter more here than in the reaching test, because grasping has a
trap: a policy that drives the gripper down and closes it will sometimes grasp
the cube by luck. So we compare against random actions, and against the
scripted controller as an upper bound (it solves 8/8).

    python3 eval_grasp.py --seeds 6 --steps 30
"""

import argparse
import gc

import numpy as np
import torch

import adapter
import loader
import record
from cem_plan import cem
from collect_grasp_data import scripted_action


def make_goal(seed, camera, steps=90):
    """Scripted rollout to completion; its final frame is the goal image."""
    env = record.make_env("PickCube-v1", 256, camera=camera)
    obs, _ = env.reset(seed=seed)
    e = env.unwrapped
    cube = e.cube.pose.p[0].cpu().numpy()
    goal = record.to_np(obs["extra"]["goal_pos"][0])
    for t in range(steps):
        tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
        if t < 12:
            tgt, g = cube + np.array([0, 0, 0.09]), 1.0
        elif t < 24:
            tgt, g = cube + np.array([0, 0, 0.005]), 1.0
        elif t < 32:
            tgt, g = tcp, -1.0
        else:
            tgt, g = goal, -1.0
        a = np.zeros(7, np.float32)
        a[:3] = np.clip((tgt - tcp) / adapter.POS_SCALE * 0.6, -1, 1)
        a[6] = g
        obs, _, _, _, info = env.step(a)
    frame = record.obs_frame(obs)
    ok = bool(info["success"][0])
    env.close()
    return frame, ok


def episode(mode, seed, steps, camera, encoder=None, predictor=None,
            goal_z=None, device="cuda", dtype=torch.float16,
            samples=32, iters=3, topk=8, chunk=16):
    env = record.make_env("PickCube-v1", 256, camera=camera)
    obs, _ = env.reset(seed=seed)
    e = env.unwrapped
    rng = np.random.default_rng(seed)
    cube0 = e.cube.pose.p[0].cpu().numpy().copy()
    goal_pos = record.to_np(obs["extra"]["goal_pos"][0])
    ever, best_lift, info = False, 0.0, {}
    noise = np.zeros(7, np.float32)

    for t in range(steps):
        if mode == "random":
            noise = 0.7 * noise + 0.3 * rng.normal(size=7).astype(np.float32)
            cmd = np.clip(noise, -1, 1).astype(np.float32)
            cmd[3:6] = 0.0
            cmd[6] = 1.0 if rng.random() < 0.5 else -1.0
        elif mode == "scripted":
            tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
            cmd = scripted_action(t, steps, tcp, cube0, goal_pos)
        else:
            frame = record.obs_frame(obs)
            state = record.obs_state(obs, env)
            z0 = loader.encode(encoder, frame[None], device, dtype)
            action, _ = cem(predictor, z0, state, goal_z, device, dtype,
                            rollout=1, samples=samples, topk=topk, iters=iters,
                            maxnorm=0.05, chunk=chunk, rng=rng,
                            search_gripper=True)
            cmd = adapter.metric_action_to_maniskill(action)

        obs, _, term, trunc, info = env.step(cmd)
        if bool(e.agent.is_grasping(e.cube)[0]):
            ever = True
        best_lift = max(best_lift, float(e.cube.pose.p[0, 2].cpu()) - cube0[2])
        if bool(term) or bool(trunc):
            break

    succ = bool(info["success"][0]) if "success" in info else False
    env.close()
    return ever, best_lift, succ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--camera", default="close")
    ap.add_argument("--ckpt", default="predictor_grasp.pt")
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    goals = [make_goal(s, a.camera) for s in range(a.seeds)]
    print(f"goals built ({sum(g[1] for g in goals)}/{a.seeds} scripted successes)")

    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    base = {k: v.clone() for k, v in predictor.state_dict().items()}
    goal_zs = [loader.encode(encoder, g[0][None], device, dtype) for g in goals]

    rows = {}
    for mode in ("random", "scripted", "pretrained", "grasp-ft"):
        if mode in ("pretrained", "grasp-ft"):
            predictor.load_state_dict(base)
            if mode == "grasp-ft":
                sd = torch.load(a.ckpt, map_location="cpu")
                predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()},
                                          strict=False)
            predictor.eval()
        res = []
        for s in range(a.seeds):
            r = episode(mode if mode in ("random", "scripted") else "plan", s,
                        a.steps, a.camera, encoder, predictor, goal_zs[s],
                        device, dtype, a.samples, a.iters, 8, a.chunk)
            res.append(r)
        g = np.array([x[0] for x in res], float)
        l = np.array([x[1] for x in res], float)
        sc = np.array([x[2] for x in res], float)
        rows[mode] = (g, l, sc)
        print(f"  {mode:11s} grasp {g.mean()*100:5.1f}%  "
              f"lift {l.mean():.4f} m  success {sc.mean()*100:5.1f}%", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'mode':12s} {'grasp rate':>11} {'mean lift':>11} {'success':>9}  (n={a.seeds})")
    for m, (g, l, sc) in rows.items():
        print(f"{m:12s} {g.mean()*100:>10.1f}% {l.mean():>10.4f}m {sc.mean()*100:>8.1f}%")
    np.savez("eval_grasp.npz", **{f"{m}_{k}": v for m, (gg, ll, ss) in rows.items()
                                  for k, v in (("grasp", gg), ("lift", ll), ("succ", ss))})
    print("wrote eval_grasp.npz")


if __name__ == "__main__":
    main()
