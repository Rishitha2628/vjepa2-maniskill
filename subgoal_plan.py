"""
Grasping via subgoal images.

reach_precision showed the real failure: planning to a single FINAL goal image
leaves the gripper ~15 cm from the cube (a grasp needs <2 cm). That is not a bug
in the planner -- it is the planner working correctly on a bad objective. The
goal image shows the arm at the goal position holding the cube, and latent L1 to
that image is minimised by flying straight there. Picking the cube up requires a
DETOUR that temporarily increases distance to the goal, so greedy goal-matching
will never do it.

The fix is to stop asking for one shortcut-able goal and give the planner a
sequence of subgoal images, each of which IS a reaching problem:

    1  gripper above the cube
    2  gripper down at the cube
    3  gripper closed on the cube
    4  cube carried to the goal

Reaching is the thing that already works (0.4766 -> 0.1181 action ranking,
12/12 seeds closing ~2/3 of the distance), so this converts a task the planner
cannot express into four it can.

What this does and does not claim: the subgoal IMAGES come from a scripted
rollout, so the planner is given the phase structure of the task. It is not
given the actions -- it still has to work out how to achieve each subgoal by
imagining outcomes with the world model. That is standard hierarchical
goal-conditioned planning, and it is a weaker claim than end-to-end planning
from a single goal image, which result 8 shows does not work.

    python3 subgoal_plan.py --seeds 6
"""

import argparse

import numpy as np
import torch

import adapter
import loader
import record
from grasp_cost import GraspScorer
from plan_grasp import cem_grasp


def make_subgoals(seed, camera, capture=(14, 26, 34, 88)):
    """Scripted rollout; snapshot frames at the phase boundaries."""
    env = record.make_env("PickCube-v1", 256, camera=camera)
    obs, _ = env.reset(seed=seed)
    e = env.unwrapped
    cube = e.cube.pose.p[0].cpu().numpy()
    goal = record.to_np(obs["extra"]["goal_pos"][0])
    frames, info = {}, {}
    for t in range(max(capture) + 1):
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
        if t in capture:
            frames[t] = record.obs_frame(obs)
    ok = bool(info["success"][0])
    env.close()
    return [frames[t] for t in capture], ok


def run(encoder, predictor, scorer, lam, seed, camera, subgoal_z, budget,
        device, dtype, rollout, samples, iters, topk, chunk):
    env = record.make_env("PickCube-v1", 256, camera=camera)
    obs, _ = env.reset(seed=seed)
    e = env.unwrapped
    rng = np.random.default_rng(seed)
    cube0 = e.cube.pose.p[0].cpu().numpy().copy()
    ever, lift, info = False, 0.0, {}
    closest = 9.9

    for gi, gz in enumerate(subgoal_z):
        for _ in range(budget[gi]):
            frame = record.obs_frame(obs)
            state = record.obs_state(obs, env)
            z0 = loader.encode(encoder, frame[None], device, dtype)
            act, _ = cem_grasp(predictor, z0, state, gz, scorer, lam, device,
                               dtype, rollout=rollout, samples=samples,
                               topk=topk, iters=iters, chunk=chunk, rng=rng)
            obs, _, term, trunc, info = env.step(
                adapter.metric_action_to_maniskill(act))
            cube = e.cube.pose.p[0].cpu().numpy()
            tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
            closest = min(closest, float(np.linalg.norm(tcp - cube)))
            if bool(e.agent.is_grasping(e.cube)[0]):
                ever = True
            lift = max(lift, float(e.cube.pose.p[0, 2].cpu()) - cube0[2])
            if bool(term) or bool(trunc):
                break
    succ = bool(info["success"][0]) if "success" in info else False
    env.close()
    return ever, lift, succ, closest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--budget", type=int, nargs=4, default=[10, 10, 6, 12])
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--rollout", type=int, default=2)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--camera", default="close")
    ap.add_argument("--ckpt", default="predictor_grasp.pt")
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    subs = [make_subgoals(s, a.camera) for s in range(a.seeds)]
    print(f"subgoals built for {a.seeds} seeds "
          f"({sum(x[1] for x in subs)}/{a.seeds} scripted successes)", flush=True)

    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    sd = torch.load(a.ckpt, map_location="cpu")
    predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    predictor.eval()
    scorer = GraspScorer(device=device, dtype=torch.float32)
    zs = [[loader.encode(encoder, f[None], device, dtype) for f in frames]
          for frames, _ in subs]

    print(f"budget per subgoal {a.budget} (total {sum(a.budget)} steps)")
    print(f"\n{'seed':>5} {'grasp':>7} {'lift':>9} {'success':>8} {'closest':>9}",
          flush=True)
    rows = []
    for s in range(a.seeds):
        r = run(encoder, predictor, scorer, a.lam, s, a.camera, zs[s], a.budget,
                device, dtype, a.rollout, a.samples, a.iters, a.topk, a.chunk)
        rows.append(r)
        print(f"{s:>5} {str(r[0]):>7} {r[1]:>8.4f}m {str(r[2]):>8} {r[3]:>8.4f}m",
              flush=True)

    g = np.mean([r[0] for r in rows]); l = np.mean([r[1] for r in rows])
    sc = np.mean([r[2] for r in rows]); c = np.mean([r[3] for r in rows])
    print(f"\nSUBGOAL PLANNING (n={a.seeds})")
    print(f"  grasp rate       {g*100:.1f}%")
    print(f"  mean lift        {l:.4f} m")
    print(f"  success          {sc*100:.1f}%")
    print(f"  closest approach {c:.4f} m   (single-goal baseline was 0.1478 m)")
    np.savez("subgoal_plan.npz", grasp=np.array([r[0] for r in rows], float),
             lift=np.array([r[1] for r in rows]), succ=np.array([r[2] for r in rows], float),
             closest=np.array([r[3] for r in rows]))
    print("wrote subgoal_plan.npz")


if __name__ == "__main__":
    main()
