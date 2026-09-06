"""
Closed-loop test: does the better action ranking translate into planning that
actually moves toward the goal?

eval_finetuned.py shows the fine-tuned predictor ranks the true action in the
top ~12% of candidates. That is a one-step, open-loop property. Planning is the
thing we actually wanted, and it can still fail -- CEM compounds the model's
errors over a horizon and feeds its own output back in.

Protocol, per seed:
  1. roll a scripted "goal" trajectory; its final frame is the goal image and
     its final tcp position is the ground truth we measure against
  2. reset to the same seed, plan with CEM toward that goal image
  3. track ||tcp - goal_tcp|| over planning steps

Reported: distance improvement (start - end). Positive = moved toward the goal.
Pretrained and fine-tuned are run on identical goals and seeds.

    python3 eval_planning.py --seeds 4 --steps 15
"""

import argparse
import gc

import numpy as np
import torch

import adapter
import loader
import record
from cem_plan import cem


def goal_rollout(seed, steps, camera):
    """A deliberate reach so the goal image differs from the start."""
    env = record.make_env("PickCube-v1", 256, camera=camera)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 500)
    d = rng.normal(size=3).astype(np.float32)
    d /= np.linalg.norm(d) + 1e-9
    for _ in range(steps):
        cmd = np.zeros(7, np.float32)
        cmd[:3] = np.clip(d * 0.5, -1, 1)
        cmd[6] = 1.0
        obs, _, _, _, _ = env.step(cmd)
    frame = record.obs_frame(obs)
    tcp = record.obs_state(obs, env)[:3].copy()
    env.close()
    return frame, tcp


def run_episode(encoder, predictor, seed, steps, goal_frame, goal_tcp, camera,
                device, dtype, samples, iters, topk, chunk, min_step):
    goal_z = loader.encode(encoder, goal_frame[None], device, dtype)
    env = record.make_env("PickCube-v1", 256, camera=camera)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    dists = []
    for _ in range(steps):
        frame = record.obs_frame(obs)
        state = record.obs_state(obs, env)
        dists.append(float(np.linalg.norm(state[:3] - goal_tcp)))
        z0 = loader.encode(encoder, frame[None], device, dtype)
        action, _ = cem(predictor, z0, state, goal_z, device, dtype,
                        rollout=1, samples=samples, topk=topk, iters=iters,
                        maxnorm=0.05, chunk=chunk, rng=rng)
        n = float(np.linalg.norm(action[:3]))
        if min_step > 0 and 1e-9 < n < min_step:
            action[:3] *= min_step / n
        cmd = adapter.metric_action_to_maniskill(action)
        cmd[6] = 1.0
        obs, _, term, trunc, _ = env.step(cmd)
        if bool(term) or bool(trunc):
            break
    state = record.obs_state(obs, env)
    dists.append(float(np.linalg.norm(state[:3] - goal_tcp)))
    env.close()
    return np.array(dists)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--goal-steps", type=int, default=8)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--min-step", type=float, default=0.0)
    ap.add_argument("--camera", default="close")
    ap.add_argument("--ckpt", default="predictor_ft.pt")
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    goals = [goal_rollout(s, a.goal_steps, a.camera) for s in range(a.seeds)]
    print(f"built {len(goals)} goals (reach of ~{a.goal_steps} steps)")

    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    base_sd = {k: v.clone() for k, v in predictor.state_dict().items()}

    results = {}
    for tag in ("pretrained", "finetuned"):
        predictor.load_state_dict(base_sd)
        if tag == "finetuned":
            sd = torch.load(a.ckpt, map_location="cpu")
            predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()},
                                      strict=False)
        predictor.eval()
        imp = []
        for s, (gf, gt) in enumerate(goals):
            d = run_episode(encoder, predictor, s, a.steps, gf, gt, a.camera,
                            device, dtype, a.samples, a.iters, a.topk,
                            a.chunk, a.min_step)
            imp.append(d[0] - d[-1])
            print(f"  {tag:11s} seed {s}: {d[0]:.4f} -> {d[-1]:.4f} m "
                  f"(improvement {imp[-1]:+.4f})", flush=True)
        results[tag] = np.array(imp)
        gc.collect(); torch.cuda.empty_cache()

    p, f = results["pretrained"], results["finetuned"]
    from scipy import stats
    print(f"\nmean distance improvement (positive = moved toward goal), "
          f"n={len(p)} seeds")
    print(f"  pretrained {p.mean():+.4f} m")
    print(f"  finetuned  {f.mean():+.4f} m")
    if len(p) > 1:
        print(f"  paired difference {(f-p).mean():+.4f} m  "
              f"p={stats.ttest_rel(f, p).pvalue:.4f}")
    np.savez("eval_planning.npz", pretrained=p, finetuned=f)
    print("wrote eval_planning.npz")


if __name__ == "__main__":
    main()
