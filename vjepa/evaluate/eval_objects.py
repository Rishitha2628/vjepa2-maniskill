"""
Does the grasp pipeline generalise beyond the one red cube?

Every result so far is on a single 4 cm cube. The decoder was trained on that
cube's positions, and the phase targets were tuned to its geometry. So the
obvious question -- "does this work on anything else?" -- is currently
unanswerable, which is weak for a 6-seed, one-object evaluation.

This runs the identical pipeline on PickSingleYCB objects: banana, mustard
bottle, drill, cans, boxes. Nothing is retrained. The point is to measure
transfer, and degradation is a perfectly good result.

Two confounds are controlled deliberately:

  robot   PickSingleYCB defaults to panda_wristcam, whose gripper-mounted
          camera degrades the base view (decoder 3.62 cm vs 1.88 cm, measured
          in the wrist ablation). Forced back to plain `panda`.
  camera  PickSingleYCB hardcodes the far default viewpoint. Patched to the
          same `close` pose the decoder was trained on, so the only thing that
          changes between this and the cube result is the OBJECT.

    python3 -m vjepa.evaluate.eval_objects --objects 8 --seeds 2
"""

import argparse

import numpy as np
import torch

from vjepa.core import adapter, loader, record
from vjepa.plan.hier_plan import choose_direction
from vjepa.plan.state_planner import KPDecoder

CLOSE_EYE = [0.45, 0.30, 0.45]
CLOSE_TGT = [-0.05, 0.0, 0.12]
CLOSE_FOV = 1.0


def patch_camera():
    """Force PickSingleYCB onto the same viewpoint the decoder was trained on."""
    import numpy as _np
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    import mani_skill.envs.tasks.tabletop.pick_single_ycb as m

    def _cfg(self):
        pose = sapien_utils.look_at(eye=CLOSE_EYE, target=CLOSE_TGT)
        return [CameraConfig("base_camera", pose, 256, 256, CLOSE_FOV, 0.01, 100)]

    m.PickSingleYCBEnv._default_sensor_configs = property(_cfg)


def make_env(view=False):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    return gym.make("PickSingleYCB-v1",
                    robot_uids="panda",          # NOT the wristcam default
                    obs_mode="rgb",
                    render_mode="human" if view else "rgb_array",
                    control_mode="pd_ee_delta_pose",
                    sensor_configs=dict(width=256, height=256))


def object_name(env):
    """Which YCB object got loaded. The env merges objects into one Actor, so
    the id lives on the underlying per-scene actor rather than on the env."""
    objs = getattr(env.unwrapped, "_objs", None)
    return objs[0].name.rsplit("-", 1)[0] if objs else "?"


@torch.no_grad()
def episode(encoder, predictor, decoder, env, seed, device, dtype,
            budget, probe, step, samples, topk, iters, chunk, view=False):
    # reconfigure=True resamples which YCB object is loaded; the episode RNG
    # keys it off the seed, so seed i gives object i deterministically.
    obs, _ = env.reset(seed=seed, options=dict(reconfigure=True))
    e = env.unwrapped
    rng = np.random.default_rng(seed)
    obj0 = e.obj.pose.p[0].cpu().numpy().copy()
    goal = record.to_np(obs["extra"]["goal_pos"][0])
    ever, lift, info, closest = False, 0.0, {}, 9.9
    carry_left, settle = budget[3], max(3, budget[3] // 5)

    def step_env(cmd):
        nonlocal obs, ever, lift, closest, info
        obs, _, term, trunc, info = env.step(cmd)
        if view:
            try:
                env.render()
            except Exception:
                pass
        o = e.obj.pose.p[0].cpu().numpy()
        tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
        closest = min(closest, float(np.linalg.norm(tcp - o)))
        if bool(e.agent.is_grasping(e.obj)[0]):
            ever = True
        lift = max(lift, float(e.obj.pose.p[0, 2].cpu()) - obj0[2])
        return bool(term) or bool(trunc)

    phases = [(np.array([0, 0, -0.085], np.float32), 1.0, probe, step * 2.5),
              (np.array([0, 0, -0.004], np.float32), 1.0, probe * .8, step),
              (None, -1.0, 0, 0), (None, -1.0, 0, 0)]

    for pi, (target, grip, pr, st) in enumerate(phases):
        if pi == 3:
            for _ in range(3):
                streak = 0
                for _ in range(8):
                    c = np.zeros(7, np.float32); c[6] = -1.0
                    if step_env(c):
                        break
                    streak = streak + 1 if bool(e.agent.is_grasping(e.obj)[0]) else 0
                    if streak >= 3:
                        break
                if streak >= 3:
                    break
                for _ in range(3):
                    c = np.zeros(7, np.float32); c[6] = 1.0; step_env(c)
                for _ in range(3):
                    c = np.zeros(7, np.float32)
                    c[2] = -0.010 / adapter.POS_SCALE; c[6] = 1.0; step_env(c)
        for _ in range(budget[pi]):
            if target is not None:
                z0 = loader.encode(encoder, record.obs_frame(obs)[None], device, dtype)
                d, _ = choose_direction(predictor, decoder, z0,
                                        record.obs_state(obs, env), target, pr,
                                        device, dtype, samples, topk, iters,
                                        chunk, rng, grip, True)   # ablated
                m = np.zeros(7, np.float32); m[:3] = d * st
                cmd = adapter.metric_action_to_maniskill(m)
            else:
                cmd = np.zeros(7, np.float32)
                if pi == 3:
                    tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
                    err = (goal - (e.obj.pose.p[0].cpu().numpy() - tcp)) - tcp
                    if carry_left <= settle:
                        cmd[:3] = 0.0
                    else:
                        g = 0.8 if np.linalg.norm(err) > 0.04 else 0.6
                        cmd[:3] = np.clip(err / adapter.POS_SCALE * g, -1, 1)
                    carry_left -= 1
            cmd[6] = grip
            if step_env(cmd):
                break

    succ = bool(info["success"][0]) if "success" in info else False
    return ever, lift, succ, closest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true",
                    help="open a live SAPIEN window instead of running headless")
    ap.add_argument("--objects", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--budget", type=int, nargs=4, default=[10, 16, 8, 26])
    ap.add_argument("--probe", type=float, default=0.12)
    ap.add_argument("--exec", dest="step", type=float, default=0.015)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--ckpt", default="checkpoints/predictor_all.pt")
    ap.add_argument("--kp-ckpt", default="checkpoints/keypoint_near.pt")
    a = ap.parse_args()

    patch_camera()
    device, dtype = "cuda", torch.float16
    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    sd = torch.load(a.ckpt, map_location="cpu")
    predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    predictor.eval()
    decoder = KPDecoder(path=a.kp_ckpt, device=device, correct_bias=False, err=0.0148)

    env = make_env(a.view)
    n = a.objects * a.seeds
    print(f"testing {n} episodes on random YCB objects; decoder trained on a CUBE\n")
    print(f"{'seed':>4}  {'object':<28} {'grasp':>6} {'success':>8} {'closest':>9}")

    rows = []
    for s in range(n):
        r = episode(encoder, predictor, decoder, env, s, device, dtype,
                    a.budget, a.probe, a.step, a.samples, a.topk, a.iters,
                    a.chunk, a.view)
        name = object_name(env)
        rows.append((name, float(r[0]), float(r[2]), r[3]))
        print(f"{s:>4}  {name:<28} {str(r[0]):>6} {str(r[2]):>8} {r[3]:>8.4f}m",
              flush=True)
    env.close()

    G = np.array([r[1] for r in rows]); S = np.array([r[2] for r in rows])
    C = np.array([r[3] for r in rows])
    print(f"\nACROSS {len(set(r[0] for r in rows))} UNSEEN OBJECTS "
          f"({len(rows)} episodes)")
    print(f"  grasp rate       {G.mean()*100:.1f}%   (red cube: 100%)")
    print(f"  success          {S.mean()*100:.1f}%   (red cube: 100%)")
    print(f"  closest approach {C.mean():.4f} m   (red cube: 0.0131 m)")
    print(f"  episodes grasped {int((G>0).sum())}/{len(rows)}")
    np.savez("results/eval_objects.npz", ids=np.array([r[0] for r in rows]),
             grasp=G, succ=S, closest=C)
    print("wrote results/eval_objects.npz")


if __name__ == "__main__":
    main()
