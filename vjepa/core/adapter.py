"""
ManiSkill <-> V-JEPA 2-AC convention conversion.

This module exists because of a trap: ManiSkill's `extra/tcp_pose` and
V-JEPA 2-AC's `states` input are BOTH 7-dimensional, so they concatenate and
run without error -- and mean completely different things.

    ManiSkill tcp_pose : [x, y, z, qw, qx, qy, qz]        quaternion
    V-JEPA 2-AC state  : [x, y, z, roll, pitch, yaw, g]   euler + gripper

Verified against notebooks/franka_example_traj.npz in the upstream repo, whose
`states` are euler with a 0..1 gripper closedness in the last slot.

Actions have a second, independent mismatch:

    ManiSkill pd_ee_delta_pose : 7-dim, each in [-1, 1], scaled by the
                                 controller to +/-0.1 m and +/-0.1 rad.
    V-JEPA 2-AC action         : 7-dim METRIC deltas, [dx,dy,dz,dr,dp,dy,dg],
                                 trained on |dxyz| <~ 0.05 m (DROID).

Feeding raw ManiSkill actions to the predictor overstates every motion by ~10x
at full scale. Both conversions are applied here and nowhere else.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# ManiSkill PDEEPoseController defaults for pd_ee_delta_pose, read off
# env.unwrapped.agent.controller.controllers['arm'].config at runtime.
POS_SCALE = 0.1   # action of 1.0 -> 0.1 m
ROT_SCALE = 0.1   # action of 1.0 -> 0.1 rad
FINGER_OPEN = 0.04  # Panda finger joint value when fully open


def quat_wxyz_to_euler(q):
    """(..., 4) wxyz quaternion -> (..., 3) xyz euler radians."""
    q = np.asarray(q, dtype=np.float64)
    # scipy wants xyzw
    xyzw = np.concatenate([q[..., 1:], q[..., :1]], axis=-1)
    return Rotation.from_quat(xyzw).as_euler("xyz", degrees=False)


def gripper_closedness(qpos):
    """Panda finger joints -> DROID closedness in [0, 1] (0 open, 1 closed)."""
    finger = np.asarray(qpos, dtype=np.float64)[..., -1]
    return np.clip(1.0 - finger / FINGER_OPEN, 0.0, 1.0)


def maniskill_state_to_droid(tcp_pose, qpos):
    """ManiSkill observation -> V-JEPA 2-AC `states` row.

    tcp_pose : (..., 7) [x,y,z,qw,qx,qy,qz]
    qpos     : (..., 9) joint positions, last two are the fingers
    returns  : (..., 7) [x,y,z,roll,pitch,yaw,closedness]
    """
    tcp_pose = np.asarray(tcp_pose, dtype=np.float64)
    xyz = tcp_pose[..., :3]
    rpy = quat_wxyz_to_euler(tcp_pose[..., 3:7])
    g = gripper_closedness(qpos)[..., None]
    return np.concatenate([xyz, rpy, g], axis=-1).astype(np.float32)


def maniskill_action_to_metric(action):
    """ManiSkill pd_ee_delta_pose action -> metric action for the predictor.

    action  : (..., 7) each component in [-1, 1]
    returns : (..., 7) [dx,dy,dz,dr,dp,dy,d_closedness] in metres / radians
    """
    a = np.asarray(action, dtype=np.float64)
    dxyz = a[..., :3] * POS_SCALE
    drpy = a[..., 3:6] * ROT_SCALE
    # ManiSkill's gripper channel is an absolute target (+1 open, -1 closed),
    # not a delta. Express it as the closedness it commands; callers that need
    # a true delta subtract the current closedness.
    target_closedness = (1.0 - a[..., -1:]) / 2.0
    return np.concatenate([dxyz, drpy, target_closedness], axis=-1).astype(np.float32)


def metric_action_to_maniskill(action):
    """Inverse of maniskill_action_to_metric, for executing planned actions.

    action  : (..., 7) metric [dx,dy,dz,dr,dp,dy,closedness_target]
    returns : (..., 7) in [-1, 1], safe to pass to env.step
    """
    a = np.asarray(action, dtype=np.float64)
    dxyz = a[..., :3] / POS_SCALE
    drpy = a[..., 3:6] / ROT_SCALE
    grip = 1.0 - 2.0 * a[..., -1:]
    out = np.concatenate([dxyz, drpy, grip], axis=-1)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def compute_new_pose(pose, action):
    """Propagate a DROID pose by a metric action. Mirrors mpc_utils.compute_new_pose.

    pose   : (B, 7) [x,y,z,r,p,y,closedness]
    action : (B, 7) [dx,dy,dz,dr,dp,dy,d_closedness]
    """
    pose = np.asarray(pose, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)

    new_xyz = pose[:, :3] + action[:, :3]

    mats = Rotation.from_euler("xyz", pose[:, 3:6], degrees=False).as_matrix()
    dmats = Rotation.from_euler("xyz", action[:, 3:6], degrees=False).as_matrix()
    new_rpy = Rotation.from_matrix(
        np.einsum("bij,bjk->bik", dmats, mats)
    ).as_euler("xyz", degrees=False)

    new_g = np.clip(pose[:, -1:] + action[:, -1:], 0.0, 1.0)
    return np.concatenate([new_xyz, new_rpy, new_g], axis=-1).astype(np.float32)
