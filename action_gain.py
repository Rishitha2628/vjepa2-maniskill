"""
Does the predictor faithfully propagate SMALL actions?

Every fix aimed at the planner has failed: better decoder (2.60 -> 1.79 cm),
finer steps, action-scale correction, bias correction, shorter horizon, more
samples. It sits at ~3.6 cm while a servo using the same decoder reaches 2.4 cm
and grasps 6/6. So the difference is not the readout -- it is what the predictor
does with a candidate action.

CEM only works if imagining action `a` moves the imagined gripper by roughly what
`a` commands. If the predictor under-responds at small scales, then every
candidate decodes to nearly the same place, the scores are nearly equal, and CEM
has no gradient to follow -- it wanders and parks at a fixed radius. That would
explain a stall no fix to the cost or the search could ever touch.

This measures the gain directly: command a known displacement, decode where the
predictor thinks the gripper went, and compare.

    gain = (decoded movement) / (commanded movement)

gain ~1 means the imagination is faithful. gain ~0 means it ignores the action.

    python3 action_gain.py
"""

import numpy as np
import torch

import adapter
import loader
import record
from fit_keypoint import KeypointDecoder

STEPS_M = [0.005, 0.010, 0.020, 0.040, 0.080]
DIRS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], float)


@torch.no_grad()
def main():
    dev = "cuda"
    enc, pred = loader.load_ac_model(device=dev, dtype=torch.float16)
    sd = torch.load("predictor_all.pt", map_location="cpu")
    pred.load_state_dict({k: v.to(torch.float16) for k, v in sd.items()}, strict=False)
    pred.eval()
    dec = KeypointDecoder().to(dev)
    dec.load_state_dict(torch.load("keypoint.pt", map_location=dev))
    dec.eval()

    rows = {m: [] for m in STEPS_M}
    for seed in range(4):
        env = record.make_env("PickCube-v1", 256, camera="close")
        obs, _ = env.reset(seed=seed)
        e = env.unwrapped
        cube = e.cube.pose.p[0].cpu().numpy()
        # park the gripper near the cube, where grasp planning actually happens
        for _ in range(22):
            tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
            a = np.zeros(7, np.float32)
            a[:3] = np.clip((cube + np.array([0, 0, .06]) - tcp) / .1 * .6, -1, 1)
            a[6] = 1.0
            obs, *_ = env.step(a)

        frame = record.obs_frame(obs)
        st = record.obs_state(obs, env)
        z = loader.encode(enc, frame[None], dev, torch.float16)
        base = dec(z.float())[0].cpu().numpy()          # decoded offset now

        for m in STEPS_M:
            moved = []
            for u in DIRS:
                act = np.zeros(7, np.float32)
                act[:3] = u * m                          # metric action
                at = torch.as_tensor(act, device=dev, dtype=torch.float16)[None, None]
                stt = torch.as_tensor(st, device=dev, dtype=torch.float16)[None, None]
                zc = loader.predict_next(pred, z, at, stt)
                after = dec(zc.float())[0].cpu().numpy()
                # offset is cube - tcp, so moving the gripper by +u*m should
                # change the decoded offset by -u*m
                moved.append(-(after - base) @ u)
            rows[m].append(np.mean(moved))
        env.close()

    print(f"\n{'commanded':>11} {'decoded move':>14} {'gain':>8}")
    for m in STEPS_M:
        got = float(np.mean(rows[m]))
        print(f"{m*100:>9.1f}cm {got*100:>12.2f}cm {got/m:>8.2f}")

    small = np.mean([np.mean(rows[m]) / m for m in (0.005, 0.010, 0.020)])
    big = np.mean([np.mean(rows[m]) / m for m in (0.040, 0.080)])
    print(f"\ngain on small steps (0.5-2 cm): {small:.2f}")
    print(f"gain on large steps (4-8 cm):   {big:.2f}")
    print(f"\nservo works at 1-2 cm; planner stalls at 3.6 cm")
    if small < 0.4:
        print(f"\nFOUND IT: the predictor barely moves the imagined gripper for")
        print(f"small actions (gain {small:.2f}). Candidates separated by 1-2 cm look")
        print(f"nearly identical to CEM, so there is no gradient to follow at the")
        print(f"scale grasping needs. No cost or search fix can recover that.")
    else:
        print(f"\nThe predictor does respond at small scales, so the stall lies")
        print(f"elsewhere.")
    np.savez("action_gain.npz", steps=np.array(STEPS_M),
             gain=np.array([np.mean(rows[m]) / m for m in STEPS_M]))


if __name__ == "__main__":
    main()
