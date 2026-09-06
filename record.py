"""
Records a fixture trajectory from ManiSkill in V-JEPA 2-AC's conventions.

Writes traj.npz with
    frames       (T, 256, 256, 3) uint8
    ms_actions   (T, 7)  what was passed to env.step, in [-1, 1]
    actions      (T, 7)  the same actions in metric DROID units
    states       (T, 7)  [x, y, z, roll, pitch, yaw, closedness]

The metric/euler conversion is the whole point -- see adapter.py. Storing the
raw ManiSkill tcp_pose would give a 7-vector that loads fine and means the
wrong thing.

Actions are a smoothed random walk rather than i.i.d. samples: independent
samples average out to near-zero net motion, which makes an action-conditioning
test vacuous because nothing moves.

    python3 record.py --steps 32
"""

import argparse

import numpy as np
import torch

import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers the environments)

import adapter

# Camera presets. The default ManiSkill `base_camera` is a wide-FOV view from
# 0.6 m up, which leaves the cube a few pixels across. V-JEPA 2-AC was trained
# on DROID: a close third-person view where the gripper and the manipulated
# object both fill a good part of the frame. "close" approximates that.
CAMERAS = {
    "default": None,
    "close": dict(eye=[0.45, 0.30, 0.45], target=[-0.05, 0.0, 0.12], fov=1.0),
}


def set_camera(preset):
    """Mutates ManiSkill's PickCube camera config. Must run before gym.make."""
    cfg = CAMERAS[preset]
    if cfg is None:
        return None
    from mani_skill.envs.tasks.tabletop.pick_cube import PICK_CUBE_CONFIGS

    PICK_CUBE_CONFIGS["panda"]["sensor_cam_eye_pos"] = cfg["eye"]
    PICK_CUBE_CONFIGS["panda"]["sensor_cam_target_pos"] = cfg["target"]
    return cfg


def to_np(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def make_env(task="PickCube-v1", res=256, seed=None, camera="default"):
    # pd_ee_delta_pose gives 7-dim end-effector actions, matching the DROID-style
    # conditioning V-JEPA 2-AC was trained with. The default pd_joint_delta_pos
    # would give 8-dim joint actions, which the predictor has no notion of.
    cfg = set_camera(camera)
    sensor_configs = dict(width=res, height=res)
    if cfg is not None:
        sensor_configs["fov"] = cfg["fov"]
    return gym.make(
        task,
        obs_mode="rgb",
        render_mode="rgb_array",
        control_mode="pd_ee_delta_pose",
        sensor_configs=sensor_configs,
    )


def obs_frame(obs, camera="base_camera"):
    return to_np(obs["sensor_data"][camera]["rgb"][0])


def obs_state(obs, env):
    """ManiSkill observation -> DROID-convention 7-vector."""
    tcp = to_np(obs["extra"]["tcp_pose"][0])
    qpos = to_np(env.unwrapped.agent.robot.get_qpos()[0])
    return adapter.maniskill_state_to_droid(tcp, qpos)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PickCube-v1")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=float, default=0.5,
                   help="action magnitude in [-1,1] units; 0.5 -> 0.05 m/step, "
                        "which is the range V-JEPA 2-AC saw on DROID")
    p.add_argument("--smooth", type=float, default=0.7,
                   help="random-walk momentum; 0 gives i.i.d. actions")
    p.add_argument("--camera", default="default", choices=list(CAMERAS),
                   help="'close' approximates DROID's close third-person view")
    p.add_argument("--out", default="traj.npz")
    a = p.parse_args()

    env = make_env(a.task, a.res, camera=a.camera)
    assert env.action_space.shape[0] == 7, (
        f"expected 7-dim end-effector actions, got {env.action_space.shape}")

    obs, _ = env.reset(seed=a.seed)
    rng = np.random.default_rng(a.seed)

    frames, ms_actions, states = [], [], []
    act = np.zeros(7, dtype=np.float32)
    for _ in range(a.steps):
        frames.append(obs_frame(obs))
        states.append(obs_state(obs, env))

        step = rng.normal(0.0, 1.0, size=7).astype(np.float32)
        act = a.smooth * act + (1.0 - a.smooth) * step
        cmd = np.clip(act * a.scale, -1.0, 1.0).astype(np.float32)
        cmd[3:6] = 0.0     # translation-only, matching how the reference CEM searches
        cmd[6] = 1.0       # keep the gripper open for this fixture
        ms_actions.append(cmd)
        obs, _, _, _, _ = env.step(cmd)

    frames = np.asarray(frames, dtype=np.uint8)
    ms_actions = np.asarray(ms_actions, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    actions = adapter.maniskill_action_to_metric(ms_actions)
    # The predictor's last action channel is a closedness delta, not a target.
    actions[:, 6] = np.diff(np.concatenate([states[:1, 6], states[:, 6]]))

    np.savez(a.out, frames=frames, ms_actions=ms_actions, actions=actions,
             states=states, camera=a.camera)

    disp = np.linalg.norm(states[-1, :3] - states[0, :3])
    print(f"saved {a.out}: frames {frames.shape}, actions {actions.shape}, "
          f"states {states.shape}")
    print(f"end-effector moved {disp:.3f} m over {a.steps} steps "
          f"(per-step |dxyz| mean {np.linalg.norm(actions[:, :3], axis=1).mean():.4f} m)")
    if disp < 0.02:
        print("WARNING: the arm barely moved. An action-conditioning test on "
              "this trajectory will not be informative -- raise --scale.")

    try:
        from PIL import Image
        tag = "" if a.camera == "default" else f"_{a.camera}"
        Image.fromarray(frames[0]).save(f"frame0{tag}.png")
        Image.fromarray(frames[-1]).save(f"frame_last{tag}.png")
        print(f"wrote frame0{tag}.png and frame_last{tag}.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
