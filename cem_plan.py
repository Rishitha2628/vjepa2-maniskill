"""
Closed-loop CEM planning in ManiSkill using V-JEPA 2-AC as the dynamics model.

No reward function and no policy: actions are scored purely by how close the
imagined next latent lands to a goal latent.

Read energy_maniskill.py's output before trusting anything here. On ManiSkill
renders the energy landscape is dominated by a magnitude prior -- the
lowest-energy action is "don't move" -- so CEM will happily converge to zero
motion. `--min-step` exists for that reason: it projects the planned action out
to a minimum norm so the arm commits to a direction instead of stalling.

Defaults follow the upstream MPC config (notebooks/utils/mpc_utils.py):
rollout 2, maxnorm 0.05 m, momentum on both mean and std.

    python3 cem_plan.py --steps 20 --rollout 1 --samples 64
"""

import argparse

import numpy as np
import torch

import adapter
import loader
import record
from record import make_env, obs_frame, obs_state


@torch.no_grad()
def rollout_scores(predictor, z0, state0, seqs, goal_z, device, dtype, chunk):
    """Score each action sequence by the L1 distance of its final imagined
    latent to the goal latent."""
    out = []
    for i in range(0, len(seqs), chunk):
        s = np.ascontiguousarray(seqs[i:i + chunk])
        n, H = s.shape[0], s.shape[1]
        z = z0.expand(n, -1, -1).contiguous()
        pose = np.repeat(state0[None], n, axis=0)
        for h in range(H):
            a = torch.as_tensor(s[:, h], device=device, dtype=dtype)[:, None]
            p = torch.as_tensor(pose, device=device, dtype=dtype)[:, None]
            z = loader.predict_next(predictor, z, a, p)
            pose = adapter.compute_new_pose(pose, s[:, h])
        out.append((z.float() - goal_z.float()).abs().mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def cem(predictor, z0, state0, goal_z, device, dtype, rollout=2, samples=64,
        topk=8, iters=4, maxnorm=0.05, momentum=0.15, chunk=16, rng=None,
        search_gripper=False, grip_momentum=0.15):
    """Returns (best_action_7, per-iteration best score).

    The gripper is searched as a separate channel when `search_gripper` is set,
    following upstream's MPC: it is near-binary rather than a small delta, so it
    gets its own std (order 1, not `maxnorm`) and its own momentum. Searching it
    with the translation std would sample gripper deltas around 0.05, which is
    far too small to ever open or close the hand.
    """
    rng = rng or np.random.default_rng(0)
    dim = 4 if search_gripper else 3
    mean = np.zeros((rollout, dim), np.float32)
    std = np.zeros((rollout, dim), np.float32)
    std[:, :3] = maxnorm
    if search_gripper:
        std[:, 3] = 1.0

    history = []
    for _ in range(iters):
        s = rng.normal(size=(samples, rollout, dim)).astype(np.float32) * std + mean
        s[:, :, :3] = np.clip(s[:, :, :3], -maxnorm, maxnorm)
        if search_gripper:
            s[:, :, 3] = np.clip(s[:, :, 3], -1.0, 1.0)
            seqs = np.concatenate([
                s[:, :, :3],
                np.zeros((samples, rollout, 3), np.float32),
                s[:, :, 3:4],
            ], axis=-1)
        else:
            seqs = np.concatenate(
                [s, np.zeros((samples, rollout, 4), np.float32)], axis=-1)

        scores = rollout_scores(predictor, z0, state0, seqs, goal_z, device,
                                dtype, chunk)
        elite = s[np.argsort(scores)[:topk]]
        m_new, s_new = elite.mean(0), elite.std(0)
        mean[:, :3] = m_new[:, :3] * (1 - momentum) + mean[:, :3] * momentum
        std[:, :3] = s_new[:, :3] * (1 - momentum) + std[:, :3] * momentum
        if search_gripper:
            mean[:, 3] = m_new[:, 3] * (1 - grip_momentum) + mean[:, 3] * grip_momentum
            std[:, 3] = s_new[:, 3] * (1 - grip_momentum) + std[:, 3] * grip_momentum
        history.append(float(scores.min()))

    action = np.zeros(7, np.float32)
    action[:3] = mean[0, :3]
    if search_gripper:
        action[6] = mean[0, 3]
    return action, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--goal-traj", default="traj.npz",
                    help="goal image is the last frame of this trajectory")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--rollout", type=int, default=1)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--maxnorm", type=float, default=0.05)
    ap.add_argument("--min-step", type=float, default=0.0,
                    help="rescale the planned translation to at least this norm "
                         "(metres); counteracts the model's do-nothing bias")
    ap.add_argument("--chunk", type=int, default=16, help="lower this first if you OOM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None,
                    help="fine-tuned predictor weights from finetune.py")
    ap.add_argument("--camera", default="close", choices=list(record.CAMERAS))
    ap.add_argument("--search-gripper", action="store_true",
                    help="also plan the gripper channel (needed for grasping)")
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu")
        predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()},
                                  strict=False)
        print(f"loaded {len(sd)} fine-tuned tensors from {a.ckpt}")

    goal_frame = np.load(a.goal_traj)["frames"][-1]
    goal_z = loader.encode(encoder, goal_frame[None], device, dtype)

    env = make_env(a.task, camera=a.camera)
    obs, _ = env.reset(seed=a.seed)
    rng = np.random.default_rng(a.seed)

    dists = []
    for t in range(a.steps):
        frame = obs_frame(obs)
        state = obs_state(obs, env)
        z0 = loader.encode(encoder, frame[None], device, dtype)

        action, hist = cem(predictor, z0, state, goal_z, device, dtype,
                           rollout=a.rollout, samples=a.samples, topk=a.topk,
                           iters=a.iters, maxnorm=a.maxnorm, chunk=a.chunk, rng=rng,
                           search_gripper=a.search_gripper)

        n = float(np.linalg.norm(action[:3]))
        if a.min_step > 0 and 1e-9 < n < a.min_step:
            action[:3] *= a.min_step / n

        cmd = adapter.metric_action_to_maniskill(action)
        if not a.search_gripper:
            cmd[6] = 1.0                  # gripper open; translation-only search
        obs, _, term, trunc, _ = env.step(cmd)

        tcp = obs_state(obs, env)[:3]
        goal = obs["extra"]["goal_pos"][0].cpu().numpy()
        dists.append(float(np.linalg.norm(tcp - goal)))
        print(f"t={t:3d} |a|={n:.4f} score/iter={[round(h,4) for h in hist]} "
              f"tcp->goal={dists[-1]:.4f}")

        if bool(term) or bool(trunc):
            print(f"episode ended at t={t}")
            break

    print(f"\ntcp->goal start {dists[0]:.4f} -> end {dists[-1]:.4f} "
          f"(min {min(dists):.4f})")
    print("Planning is working if this decreases. Compare against "
          "`--samples 1 --iters 1` as a random-action control.")


if __name__ == "__main__":
    main()
