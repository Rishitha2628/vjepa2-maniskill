"""
Hierarchical planning: probe large, execute small.

Nine planner variants stalled at 3.6-4.0 cm. action_gain.py and
decoder_on_predicted.py explain why: CEM ranks candidates whose IMAGINED
positions differ by (action cap x 0.34 gain) ~= 1.7 cm, against 1.5-2.3 cm of
error in decoding an imagined state. Signal ~= noise, so near the cube the
ranking is meaningless.

The trap is that one number set both the evaluation scale and the execution
scale. Small steps give precision but indistinguishable candidates; large steps
distinguish them but overshoot. Nothing requires the two to be equal.

So: evaluate candidates as LARGE probe displacements, where imagined outcomes
separate well above the decoder's noise, then execute only the winning
DIRECTION as a small step, and re-plan. The world model still chooses -- it just
gets asked a question it can actually answer ("which way?" rather than "which of
these 48 nearly identical nudges?").

This is the coarse-to-fine idea from the literature (Coarse-to-Fine Q-attention,
Subgoal Diffuser, Hierarchical World Models) applied to our specific bottleneck.

    python3 hier_plan.py --seeds 6 --probe 0.12 --exec 0.015
"""

import argparse

import numpy as np
import torch

from vjepa.core import adapter
from vjepa.core import loader
from vjepa.core import record
from vjepa.plan.state_planner import KPDecoder

ACTION_SCALE = 0.40


@torch.no_grad()
def choose_direction(predictor, decoder, z0, state0, target_off, probe, device,
                     dtype, samples, topk, iters, chunk, rng, grip,
                     ablate=False):
    """CEM over large probe displacements; returns a unit direction.

    `ablate=True` replaces the world model with a trivial analytic stand-in:
    commanding action `a` is assumed to change the gripper->cube offset by
    exactly -a, with no prediction at all. Everything else -- the decoder, the
    cost, the CEM search, the phases, the budgets -- is identical.

    This is the control for "is the predictor doing real work, or is the decoded
    readout doing everything and the world model is decoration?" If the two
    score the same, the honest description of this planner is decoded-state
    servoing with extra steps.
    """
    mean = np.zeros(3, np.float32)
    std = np.full(3, probe, np.float32)
    tgt = torch.as_tensor(target_off, device=device, dtype=torch.float32)

    best_d, best_c = None, np.inf
    for _ in range(iters):
        s = rng.normal(size=(samples, 3)).astype(np.float32) * std + mean
        n = np.linalg.norm(s, axis=1, keepdims=True) + 1e-9
        s = s / n * probe                      # all candidates on a probe-radius shell
        if ablate:
            # analytic: offset_next = offset_now - action. No predictor call.
            off_now = decoder(z0).float()[0].cpu().numpy()
            costs = np.linalg.norm((off_now[None] - s) - target_off[None], axis=1)
            order = np.argsort(costs)
            if costs[order[0]] < best_c:
                best_c, best_d = costs[order[0]], s[order[0]].copy()
            elite = s[order[:topk]]
            mean = elite.mean(0) * 0.7 + mean * 0.3
            std = np.maximum(elite.std(0) * 0.7 + std * 0.3, probe * 0.35)
            continue
        costs = []
        for i in range(0, len(s), chunk):
            b = np.ascontiguousarray(s[i:i + chunk])
            m = len(b)
            a7 = np.concatenate([b, np.zeros((m, 3), np.float32),
                                 np.full((m, 1), grip, np.float32)], axis=1)
            act = torch.as_tensor(a7, device=device, dtype=dtype)[:, None]
            ps = torch.as_tensor(np.repeat(state0[None], m, 0),
                                 device=device, dtype=dtype)[:, None]
            z = loader.predict_next(predictor, z0.expand(m, -1, -1).contiguous(),
                                    act, ps)
            off = decoder(z)
            costs.append((off - tgt).norm(dim=-1).cpu().numpy())
        costs = np.concatenate(costs)
        order = np.argsort(costs)
        if costs[order[0]] < best_c:
            best_c, best_d = costs[order[0]], s[order[0]].copy()
        elite = s[order[:topk]]
        mean = elite.mean(0) * 0.7 + mean * 0.3
        std = np.maximum(elite.std(0) * 0.7 + std * 0.3, probe * 0.35)

    d = best_d / (np.linalg.norm(best_d) + 1e-9)
    return d, float(best_c)


def episode(encoder, predictor, decoder, seed, camera, device, dtype, budget,
            probe, step, samples, topk, iters, chunk, ablate=False):
    env = record.make_env("PickCube-v1", 256, camera=camera)
    obs, _ = env.reset(seed=seed)
    e = env.unwrapped
    rng = np.random.default_rng(seed)
    cube0 = e.cube.pose.p[0].cpu().numpy().copy()
    goal = record.to_np(obs["extra"]["goal_pos"][0])
    ever, lift, info, closest = False, 0.0, {}, 9.9
    diag = dict(grasp_first_step=-1, lost_after_grasp=0, cube_goal_end=9.9,
                cube_moved=0.0, grasped_at_lift_start=False, steps=0)

    phases = [(np.array([0, 0, -0.085], np.float32), 1.0, probe, step * 2.5),
              (np.array([0, 0, -0.004], np.float32), 1.0, probe * 0.8, step),
              (None, -1.0, 0, 0),    # close
              (None, -1.0, 0, 0)]    # lift / carry

    carry_left = budget[3]
    settle = max(3, budget[3] // 5)
    for pi, (target, grip, pr, st_size) in enumerate(phases):
        # Before lifting, confirm the cube is actually held. Seed 4 of the first
        # run grasped and then lifted nothing (lift 0.0007 m) -- the gripper had
        # closed but lost the cube, and the lift proceeded regardless.
        if pi == 3:
            # A single is_grasping==True is not a hold. Seed 4 reported a grasp
            # and then slipped 11 times, lifting nothing: contact flickered on
            # without ever bearing load. Require CONSECUTIVE grasped steps, and
            # if the hold will not stabilise, nudge down and re-close rather
            # than lifting air.
            for attempt in range(3):
                streak = 0
                for _ in range(8):
                    c = np.zeros(7, np.float32); c[6] = -1.0
                    obs, _, term, trunc, info = env.step(c)
                    if bool(e.agent.is_grasping(e.cube)[0]):
                        ever = True
                        streak += 1
                    else:
                        streak = 0
                    if streak >= 3 or bool(term) or bool(trunc):
                        break
                if streak >= 3:
                    break
                # not holding: open, drop 1 cm onto the cube, try again
                for _ in range(3):
                    c = np.zeros(7, np.float32); c[6] = 1.0
                    obs, _, term, trunc, info = env.step(c)
                for _ in range(3):
                    c = np.zeros(7, np.float32)
                    c[2] = -0.010 / adapter.POS_SCALE
                    c[6] = 1.0
                    obs, _, term, trunc, info = env.step(c)
        for _ in range(budget[pi]):
            frame = record.obs_frame(obs)
            state = record.obs_state(obs, env)
            if target is not None:
                z0 = loader.encode(encoder, frame[None], device, dtype)
                d, _ = choose_direction(predictor, decoder, z0, state, target,
                                        pr, device, dtype, samples, topk,
                                        iters, chunk, rng, grip, ablate)
                metric = np.zeros(7, np.float32)
                metric[:3] = d * st_size          # winning direction, small step
                cmd = adapter.metric_action_to_maniskill(metric)
            else:
                cmd = np.zeros(7, np.float32)
                if pi == 3:
                    tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
                    # Servo the CUBE to the goal, not the gripper. Success is
                    # scored on the cube, which hangs ~2.4 cm from the tool
                    # point, so aiming the gripper at the goal parks the cube
                    # ~2.4 cm away -- right on the 2.5 cm threshold. Four seeds
                    # were passing by about a millimetre and one was failing by
                    # nine. Compensating the offset aims the cube itself.
                    cube_now = e.cube.pose.p[0].cpu().numpy()
                    offset = cube_now - tcp          # where the cube sits
                    err = (goal - offset) - tcp      # put the CUBE on the goal
                    d = float(np.linalg.norm(err))
                    # PickCube needs the cube within 2.5 cm of the goal AND the
                    # robot static. Both matter, and they pull in opposite
                    # directions: an earlier attempt cut the command at 3 cm to
                    # settle the arm, which parked it OUTSIDE the success radius
                    # and dropped success from 66.7% to 16.7%.
                    # So: servo all the way in with a reducing gain, and buy the
                    # static condition from dedicated settle steps at the end
                    # instead of from an early cutoff.
                    if carry_left <= settle:
                        cmd[:3] = 0.0                       # settle, do not stop short
                    else:
                        # The controller executes only ~40% of each command
                        # (measured in servo_check.py), so a gain that DROPS as
                        # the error shrinks asymptotes short: every seed parked
                        # ~2.2 cm out regardless of where it was aimed. Hold the
                        # gain up so the residual actually closes.
                        gain = 0.8 if d > 0.04 else 0.6
                        cmd[:3] = np.clip(err / adapter.POS_SCALE * gain, -1, 1)
                    carry_left -= 1
            cmd[6] = grip
            obs, _, term, trunc, info = env.step(cmd)

            cube = e.cube.pose.p[0].cpu().numpy()
            tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
            closest = min(closest, float(np.linalg.norm(tcp - cube)))
            g_now = bool(e.agent.is_grasping(e.cube)[0])
            diag["steps"] += 1
            if g_now:
                ever = True
                if diag["grasp_first_step"] < 0:
                    diag["grasp_first_step"] = diag["steps"]
            elif ever:
                diag["lost_after_grasp"] += 1        # how often it slips back out
            if pi == 3:
                diag["grasped_at_lift_start"] = diag["grasped_at_lift_start"] or g_now
            lift = max(lift, float(e.cube.pose.p[0, 2].cpu()) - cube0[2])
            diag["cube_goal_end"] = float(np.linalg.norm(cube - goal))
            diag["cube_moved"] = float(np.linalg.norm(cube - cube0))
            if bool(term) or bool(trunc):
                break
    succ = bool(info["success"][0]) if "success" in info else False
    env.close()
    return ever, lift, succ, closest, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--budget", type=int, nargs=4, default=[10, 16, 8, 22],
                    help="hover, descend, close, carry. Close and carry are "
                         "longer than the original [10,16,5,15]: 5 close steps "
                         "left weak grasps, and 15 carry steps ran out before "
                         "the arm settled inside the 2.5 cm success threshold")
    ap.add_argument("--probe", type=float, default=0.12,
                    help="magnitude used to EVALUATE candidates (large)")
    ap.add_argument("--exec", dest="exec_step", type=float, default=0.015,
                    help="magnitude actually EXECUTED (small)")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--ablate-predictor", action="store_true",
                    help="replace the world model with an analytic stand-in, to "
                         "test whether the predictor is doing real work")
    ap.add_argument("--camera", default="close",
                    help="'dual' uses the panda_wristcam robot with the same "
                         "close base-camera pose -- a control for whether the "
                         "robot's changed appearance alone breaks the "
                         "base-camera models")
    ap.add_argument("--ckpt", default="checkpoints/predictor_all.pt")
    ap.add_argument("--kp-ckpt", default="checkpoints/keypoint_near.pt")
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    sd = torch.load(a.ckpt, map_location="cpu")
    predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    predictor.eval()
    decoder = KPDecoder(path=a.kp_ckpt, device=device, correct_bias=False,
                        err=0.0148)
    if a.ablate_predictor:
        print("*** ABLATION: predictor replaced by analytic offset_next = "
              "offset_now - action ***")
    print(f"probe {a.probe*100:.0f} cm -> imagined spread "
          f"~{a.probe*0.34*100:.1f} cm vs ~2 cm decode noise")
    print(f"execute {a.exec_step*100:.1f} cm per step\n")

    print(f"{'seed':>5} {'grasp':>7} {'lift':>10} {'success':>8} {'closest':>9}"
          f"  | diagnostics (success needs cube->goal < 0.025 m AND robot static)")
    rows = []
    for s in range(a.seeds):
        r = episode(encoder, predictor, decoder, s, a.camera, device, dtype,
                    a.budget, a.probe, a.exec_step, a.samples, a.topk,
                    a.iters, a.chunk, a.ablate_predictor)
        rows.append(r)
        d = r[4]
        print(f"{s:>5} {str(r[0]):>7} {r[1]:>9.4f}m {str(r[2]):>8} {r[3]:>8.4f}m"
              f"  | cube->goal {d['cube_goal_end']:.4f}m  cube moved "
              f"{d['cube_moved']:.4f}m  grasp@step {d['grasp_first_step']}"
              f"  slips {d['lost_after_grasp']}", flush=True)

    g = np.mean([r[0] for r in rows]); l = np.mean([r[1] for r in rows])
    sc = np.mean([r[2] for r in rows]); c = np.mean([r[3] for r in rows])
    print(f"\nHIERARCHICAL: probe large, execute small (n={a.seeds})")
    print(f"  grasp rate       {g*100:.1f}%      (flat CEM: 0%, servo: 100%)")
    print(f"  mean lift        {l:.4f} m")
    print(f"  success          {sc*100:.1f}%")
    print(f"  closest approach {c:.4f} m   (flat CEM stalled at 0.036-0.040 m)")
    cg = np.array([r[4]["cube_goal_end"] for r in rows])
    sl = np.array([r[4]["lost_after_grasp"] for r in rows])
    print(f"\n  final cube->goal: {np.array2string(cg, precision=4)}")
    print(f"  threshold is 0.025 m -- {int((cg < 0.025).sum())}/{len(cg)} inside")
    print(f"  slips after grasping: {np.array2string(sl)}")
    np.savez("results/hier_plan.npz", cube_goal=cg, slips=sl,
             grasp=np.array([r[0] for r in rows], float),
             lift=np.array([r[1] for r in rows]),
             succ=np.array([r[2] for r in rows], float),
             closest=np.array([r[3] for r in rows]))
    print("wrote hier_plan.npz")


if __name__ == "__main__":
    main()
