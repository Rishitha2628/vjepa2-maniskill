"""
Dual-camera grasping: plan on the base view, correct with the wrist view.

The wrist-camera ablation (result 13) swapped one camera for the other and lost:
better perception (1.11 cm vs 1.88 cm) could not compensate for twice the
prediction error, because CEM scores candidates by decoding IMAGINED futures.

But `panda_wristcam` exposes both cameras at once, so the trade is avoidable.
Each camera does what it is good at:

    base_camera  -> the PREDICTOR. Stable third-person view, val L1 0.203,
                    the dynamics CEM needs to rank candidates.
    hand_camera  -> the DECODER. Best close-range readout, 1.11 cm, used only
                    to MEASURE the current offset -- never to imagine one.

The wrist view never enters the planning loop, so its bad dynamics never matter.
It only supplies a short measured correction in the last steps before the
gripper closes, where 1.11 cm beats 1.88 cm and nothing has to be predicted.

Compared against hier_plan.py (base camera only): 100% grasp, 1.31 cm, 66.7%
PickCube success. The open question is whether a better-centred grasp survives
the lift -- 2 of 6 seeds there grasped the cube and then lost it.

    python3 dual_plan.py --seeds 6
"""

import argparse

import numpy as np
import torch

from vjepa.core import adapter
from vjepa.core import loader
from vjepa.core import record
from vjepa.plan.hier_plan import choose_direction
from vjepa.plan.state_planner import KPDecoder


@torch.no_grad()
def episode(encoder, predictor, dec_base, dec_wrist, seed, device, dtype,
            budget, probe, step, samples, topk, iters, chunk, fine_steps):
    env = record.make_env("PickCube-v1", 256, camera="dual")
    obs, _ = env.reset(seed=seed)
    e = env.unwrapped
    rng = np.random.default_rng(seed)
    cube0 = e.cube.pose.p[0].cpu().numpy().copy()
    goal = record.to_np(obs["extra"]["goal_pos"][0])
    ever, lift, info, closest = False, 0.0, {}, 9.9

    def step_env(cmd):
        nonlocal obs, ever, lift, closest, info
        obs, _, term, trunc, info = env.step(cmd)
        cube = e.cube.pose.p[0].cpu().numpy()
        tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
        closest = min(closest, float(np.linalg.norm(tcp - cube)))
        if bool(e.agent.is_grasping(e.cube)[0]):
            ever = True
        lift = max(lift, float(e.cube.pose.p[0, 2].cpu()) - cube0[2])
        return bool(term) or bool(trunc)

    # --- phases 1-2: world-model planning on the BASE camera ---
    for pi, (target, pr, st) in enumerate([
            (np.array([0, 0, -0.085], np.float32), probe, step * 2.5),
            (np.array([0, 0, -0.004], np.float32), probe * 0.8, step)]):
        for _ in range(budget[pi]):
            frame = record.obs_frame(obs, "base_camera")
            state = record.obs_state(obs, env)
            z0 = loader.encode(encoder, frame[None], device, dtype)
            d, _ = choose_direction(predictor, dec_base, z0, state, target, pr,
                                    device, dtype, samples, topk, iters, chunk,
                                    rng, 1.0)
            m = np.zeros(7, np.float32)
            m[:3] = d * st
            cmd = adapter.metric_action_to_maniskill(m)
            cmd[6] = 1.0
            if step_env(cmd):
                break

    # --- fine correction: MEASURED from the wrist camera, nothing imagined ---
    tgt = np.array([0, 0, -0.004], np.float32)
    for _ in range(fine_steps):
        wframe = record.obs_frame(obs, "hand_camera")
        zw = loader.encode(encoder, wframe[None], device, dtype)
        off = dec_wrist(zw)[0].cpu().numpy()        # decoded cube - tcp
        move = off - tgt
        n = float(np.linalg.norm(move))
        if n > 0.012:
            move = move / n * 0.012
        m = np.zeros(7, np.float32)
        m[:3] = move
        cmd = adapter.metric_action_to_maniskill(m)
        cmd[6] = 1.0
        if step_env(cmd):
            break

    # --- close, then lift/carry ---
    for _ in range(budget[2]):
        cmd = np.zeros(7, np.float32); cmd[6] = -1.0
        if step_env(cmd):
            break
    for _ in range(budget[3]):
        tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
        cmd = np.zeros(7, np.float32)
        cmd[:3] = np.clip((goal - tcp) / adapter.POS_SCALE * 0.6, -1, 1)
        cmd[6] = -1.0
        if step_env(cmd):
            break

    succ = bool(info["success"][0]) if "success" in info else False
    env.close()
    return ever, lift, succ, closest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--budget", type=int, nargs=4, default=[10, 14, 5, 15])
    ap.add_argument("--fine-steps", type=int, default=4)
    ap.add_argument("--probe", type=float, default=0.12)
    ap.add_argument("--exec", dest="exec_step", type=float, default=0.015)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--ckpt", default="checkpoints/predictor_all.pt")
    ap.add_argument("--kp-base", default="checkpoints/keypoint_near.pt")
    ap.add_argument("--kp-wrist", default="checkpoints/keypoint_wrist.pt")
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    sd = torch.load(a.ckpt, map_location="cpu")
    predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    predictor.eval()
    dec_base = KPDecoder(path=a.kp_base, device=device, correct_bias=False, err=0.0148)
    dec_wrist = KPDecoder(path=a.kp_wrist, device=device, correct_bias=False, err=0.0111)
    print(f"plan on base_camera (predictor val L1 0.203), "
          f"correct on hand_camera (decoder 1.11 cm)")
    print(f"{a.fine_steps} measured wrist correction steps before closing\n")

    print(f"{'seed':>5} {'grasp':>7} {'lift':>10} {'success':>8} {'closest':>9}")
    rows = []
    for s in range(a.seeds):
        r = episode(encoder, predictor, dec_base, dec_wrist, s, device, dtype,
                    a.budget, a.probe, a.exec_step, a.samples, a.topk, a.iters,
                    a.chunk, a.fine_steps)
        rows.append(r)
        print(f"{s:>5} {str(r[0]):>7} {r[1]:>9.4f}m {str(r[2]):>8} {r[3]:>8.4f}m",
              flush=True)

    g = np.mean([r[0] for r in rows]); l = np.mean([r[1] for r in rows])
    sc = np.mean([r[2] for r in rows]); c = np.mean([r[3] for r in rows])
    print(f"\nDUAL CAMERA (n={a.seeds})")
    print(f"  grasp rate       {g*100:.1f}%      (base-only: 100%, wrist-only: 16.7%)")
    print(f"  mean lift        {l:.4f} m")
    print(f"  success          {sc*100:.1f}%      (base-only: 66.7%)")
    print(f"  closest approach {c:.4f} m   (base-only: 0.0131 m)")
    np.savez("results/dual_plan.npz", grasp=np.array([r[0] for r in rows], float),
             lift=np.array([r[1] for r in rows]),
             succ=np.array([r[2] for r in rows], float),
             closest=np.array([r[3] for r in rows]))
    print("wrote dual_plan.npz")


if __name__ == "__main__":
    main()
