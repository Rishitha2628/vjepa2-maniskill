"""
Watch the planner solve PickCube.

Runs the configuration that actually works (result 14): coarse-to-fine CEM over
a decoded-position cost, with the predictor ABLATED -- 100% grasp, 100% task
success. Pass --predictor to watch the world-model version instead (83.3%).

    python3 visualize.py --seed 0            # live window + video
    python3 visualize.py --seed 0 --video-only
"""

import argparse

import numpy as np
import torch

from vjepa.core import adapter
from vjepa.core import loader
from vjepa.core import record
from vjepa.plan.hier_plan import choose_direction
from vjepa.plan.state_planner import KPDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--predictor", action="store_true",
                    help="use the world model (83.3%%) instead of the analytic "
                         "model (100%%)")
    ap.add_argument("--video-only", action="store_true")
    ap.add_argument("--out", default="media/grasp.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--budget", type=int, nargs=4, default=[10, 16, 8, 26])
    ap.add_argument("--probe", type=float, default=0.12)
    ap.add_argument("--exec", dest="step", type=float, default=0.015)
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    sd = torch.load("checkpoints/predictor_all.pt", map_location="cpu")
    predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    predictor.eval()
    decoder = KPDecoder(path="checkpoints/keypoint_near.pt", device=device,
                        correct_bias=False, err=0.0148)
    ablate = not a.predictor
    print(f"mode: {'world model' if a.predictor else 'ANALYTIC (the one that works)'}")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    record.set_camera("close")
    mode = "rgb_array" if a.video_only else "human"
    try:
        env = gym.make("PickCube-v1", obs_mode="rgb", render_mode=mode,
                       control_mode="pd_ee_delta_pose",
                       sensor_configs=dict(width=256, height=256, fov=1.0))
    except Exception as e:
        print(f"live window unavailable ({type(e).__name__}); recording only")
        mode = "rgb_array"
        env = gym.make("PickCube-v1", obs_mode="rgb", render_mode="rgb_array",
                       control_mode="pd_ee_delta_pose",
                       sensor_configs=dict(width=256, height=256, fov=1.0))

    obs, _ = env.reset(seed=a.seed)
    e = env.unwrapped
    rng = np.random.default_rng(a.seed)
    cube0 = e.cube.pose.p[0].cpu().numpy().copy()
    goal = record.to_np(obs["extra"]["goal_pos"][0])
    frames, info = [], {}

    def shoot():
        try:
            img = env.render()
            if img is None:
                return
            img = img[0] if hasattr(img, "ndim") and img.ndim == 4 else img
            frames.append(np.asarray(img.cpu() if hasattr(img, "cpu") else img,
                                     dtype=np.uint8))
        except Exception:
            pass

    def step(cmd):
        nonlocal obs, info
        obs, _, term, trunc, info = env.step(cmd)
        shoot()
        return bool(term) or bool(trunc)

    shoot()
    carry_left, settle = a.budget[3], max(3, a.budget[3] // 5)
    phases = [(np.array([0, 0, -0.085], np.float32), 1.0, a.probe, a.step * 2.5),
              (np.array([0, 0, -0.004], np.float32), 1.0, a.probe * .8, a.step),
              (None, -1.0, 0, 0), (None, -1.0, 0, 0)]

    for pi, (target, grip, pr, st) in enumerate(phases):
        if pi == 3:
            for _ in range(3):
                streak = 0
                for _ in range(8):
                    c = np.zeros(7, np.float32); c[6] = -1.0
                    if step(c):
                        break
                    streak = streak + 1 if bool(e.agent.is_grasping(e.cube)[0]) else 0
                    if streak >= 3:
                        break
                if streak >= 3:
                    break
                for _ in range(3):
                    c = np.zeros(7, np.float32); c[6] = 1.0; step(c)
                for _ in range(3):
                    c = np.zeros(7, np.float32)
                    c[2] = -0.010 / adapter.POS_SCALE; c[6] = 1.0; step(c)
        for _ in range(a.budget[pi]):
            if target is not None:
                z0 = loader.encode(encoder, record.obs_frame(obs)[None], device, dtype)
                d, _ = choose_direction(predictor, decoder, z0,
                                        record.obs_state(obs, env), target, pr,
                                        device, dtype, 48, 8, 3, 16, rng, grip, ablate)
                m = np.zeros(7, np.float32); m[:3] = d * st
                cmd = adapter.metric_action_to_maniskill(m)
            else:
                cmd = np.zeros(7, np.float32)
                if pi == 3:
                    tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
                    err = (goal - (e.cube.pose.p[0].cpu().numpy() - tcp)) - tcp
                    if carry_left <= settle:
                        cmd[:3] = 0.0
                    else:
                        g = 0.8 if np.linalg.norm(err) > 0.04 else 0.6
                        cmd[:3] = np.clip(err / adapter.POS_SCALE * g, -1, 1)
                    carry_left -= 1
            cmd[6] = grip
            if step(cmd):
                break

    cube = e.cube.pose.p[0].cpu().numpy()
    succ = bool(info["success"][0]) if "success" in info else False
    print(f"\nseed {a.seed}: success={succ}  "
          f"grasped={bool(e.agent.is_grasping(e.cube)[0])}  "
          f"lift={cube[2]-cube0[2]:.4f} m  cube->goal={np.linalg.norm(cube-goal):.4f} m")
    env.close()

    if frames:
        import imageio
        imageio.mimsave(a.out, frames, fps=a.fps, quality=8)
        print(f"wrote {a.out}  ({len(frames)} frames)")
        n = len(frames)
        sheet = np.concatenate([frames[int(i)] for i in
                                np.linspace(0, n - 1, 6)], axis=1)
        imageio.imwrite("media/grasp_sheet.png", sheet)
        print("wrote media/grasp_sheet.png")
    else:
        print("no frames captured")


if __name__ == "__main__":
    main()
