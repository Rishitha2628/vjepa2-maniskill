"""
Isolate which layer is failing: the decoded-state idea, or the planner on top.

Everything measurable has been eliminated -- objective, decoder accuracy (2.33 cm
even on predicted latents), step size, horizon, search, capacity, camera. Yet
planning stalls 3.7 cm from the cube.

So test the layer underneath. This drops the world model entirely and servos
straight to the decoder's estimate: look at the frame, decode where the cube is
relative to the gripper, move that way, repeat. If THIS grasps, the decoded-state
readout is sound and the world-model planning layer is what fails. If it does not,
the problem is below both -- the controller, the phase targets, or the geometry.

Also reports commanded vs achieved motion, to check the controller actually
executes small deltas rather than damping them away.

    python3 servo_check.py
"""

import numpy as np
import torch

from vjepa.core import adapter
from vjepa.core import loader
from vjepa.core import record
from vjepa.train.fit_keypoint import KeypointDecoder


@torch.no_grad()
def main():
    dev = "cuda"
    enc, _ = loader.load_ac_model(device=dev, dtype=torch.float16)
    dec = KeypointDecoder().to(dev)
    dec.load_state_dict(torch.load("checkpoints/keypoint.pt", map_location=dev))
    dec.eval()

    print(f"{'seed':>5} {'grasp':>7} {'lift':>10} {'closest':>9} {'cmd/actual':>12}")
    rows = []
    for seed in range(6):
        env = record.make_env("PickCube-v1", 256, camera="close")
        obs, _ = env.reset(seed=seed)
        e = env.unwrapped
        cube0 = e.cube.pose.p[0].cpu().numpy().copy()
        closest, ever, lift = 9.9, False, 0.0
        ratios = []

        # phase 1 hover, 2 descend, 3 close, 4 lift
        for phase, (tgt_off, grip, nsteps, cap) in enumerate([
                (np.array([0, 0, -0.09]), 1.0, 12, 0.04),
                (np.array([0, 0, -0.005]), 1.0, 14, 0.015),
                (None, -1.0, 6, 0.0),
                (None, -1.0, 12, 0.04)]):
            for _ in range(nsteps):
                frame = record.obs_frame(obs)
                z = loader.encode(enc, frame[None], dev, torch.float16)
                off = dec(z.float())[0].cpu().numpy()      # decoded cube - tcp
                tcp_before = record.to_np(obs["extra"]["tcp_pose"][0])[:3]

                cmd = np.zeros(7, np.float32)
                if tgt_off is not None:
                    move = off - tgt_off                    # metres to travel
                    n = np.linalg.norm(move)
                    if n > cap:
                        move = move / n * cap
                    cmd[:3] = np.clip(move / adapter.POS_SCALE, -1, 1)
                elif phase == 3:
                    goal = record.to_np(obs["extra"]["goal_pos"][0])
                    cmd[:3] = np.clip((goal - tcp_before) / adapter.POS_SCALE * 0.6, -1, 1)
                cmd[6] = grip
                obs, _, term, trunc, info = env.step(cmd)

                tcp_after = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
                commanded = np.linalg.norm(cmd[:3] * adapter.POS_SCALE)
                achieved = np.linalg.norm(tcp_after - tcp_before)
                if commanded > 1e-4:
                    ratios.append(achieved / commanded)

                cube = e.cube.pose.p[0].cpu().numpy()
                closest = min(closest, float(np.linalg.norm(tcp_after - cube)))
                if bool(e.agent.is_grasping(e.cube)[0]):
                    ever = True
                lift = max(lift, float(e.cube.pose.p[0, 2].cpu()) - cube0[2])
        succ = bool(info["success"][0]) if "success" in info else False
        env.close()
        rows.append((ever, lift, succ, closest))
        print(f"{seed:>5} {str(ever):>7} {lift:>9.4f}m {closest:>8.4f}m "
              f"{np.mean(ratios):>11.2f}", flush=True)

    g = np.mean([r[0] for r in rows]); l = np.mean([r[1] for r in rows])
    s = np.mean([r[2] for r in rows]); c = np.mean([r[3] for r in rows])
    print(f"\nDIRECT SERVO on decoded position (no world model)")
    print(f"  grasp rate       {g*100:.1f}%")
    print(f"  mean lift        {l:.4f} m")
    print(f"  success          {s*100:.1f}%")
    print(f"  closest approach {c:.4f} m   (world-model planning: 0.036 m)")
    print()
    if c < 0.02:
        print("The decoded readout is sufficient. The world-model PLANNING layer")
        print("is what fails -- CEM is not converting a good estimate into good actions.")
    else:
        print("Even direct servoing cannot close the gap, so the limit is below")
        print("the planner: controller fidelity, phase targets, or reachability.")
    np.savez("results/servo_check.npz", grasp=np.array([r[0] for r in rows], float),
             closest=np.array([r[3] for r in rows]))


if __name__ == "__main__":
    main()
