"""
Grasp planning with a decoded-state cost instead of latent L1.

The blocking result was latent_vs_task.py: the argmin of latent L1 sits ~6 cm
from the true goal pose, so goal-image matching cannot servo to the ~2 cm a
grasp needs. That is a property of the objective, which is why better models,
longer horizons, re-weighted costs and subgoals all failed to help.

fit_decoders.py gives a better readout of the same latents: gripper->cube offset
to ~2.6 cm, and gripper position to ~2.0 cm. So keep V-JEPA as the world model
-- it still imagines the future for each candidate action -- but score those
imagined futures in metres:

    cost = || decode_offset(z_predicted) - target_offset ||

whose optimum is where we actually want it. The phases (approach, descend,
close, lift) are scripted; the actions inside each phase are planned by the
world model.

Honest scope: this is not planning from a goal image. It is world-model planning
against a decoded state objective, which is what the measurements say is needed.

    python3 state_planner.py --seeds 6
"""

import argparse

import numpy as np
import torch

import adapter
import loader
import record

# The PD controller reaches only a fraction of each commanded delta within one
# control step -- measured at 0.40 in servo_check.py (commanded vs achieved tcp
# displacement). CEM's rollouts previously advanced the imagined arm by the FULL
# command, so over a 2-step rollout its idea of the gripper sat ~2.5x further
# along than the real one, and every candidate was scored at the wrong place.
ACTION_SCALE = 0.40


class KPDecoder:
    """Spatial soft-argmax decoder (fit_keypoint.py), ~1.79 cm vs the linear
    readout's 2.60 cm. Keeps the 16x16 token grid instead of mean-pooling it.

    Bias correction matters here in a way it does not for a servo. Decoding a
    PREDICTED latent carries a systematic offset of ~1.5 cm (decoder_bias.npz),
    three times the offset on a real latent. A servo re-measures and converges
    through it; a planner drives until decoded == target, so it parks exactly
    |bias| short of the cube, every time, regardless of search quality.
    """

    def __init__(self, path="keypoint.pt", device="cuda", bias_file="decoder_bias.npz",
                 correct_bias=True, err=0.0179):
        from fit_keypoint import KeypointDecoder
        self.m = KeypointDecoder().to(device)
        self.m.load_state_dict(torch.load(path, map_location=device))
        self.m.eval()
        self.err = err
        self.bias = torch.zeros(3, device=device)
        if correct_bias:
            try:
                b = np.load(bias_file)["bias_pred"]
                self.bias = torch.as_tensor(b, device=device, dtype=torch.float32)
                print(f"bias correction: [{b[0]*100:+.2f}, {b[1]*100:+.2f}, "
                      f"{b[2]*100:+.2f}] cm (|b| {np.linalg.norm(b)*100:.2f} cm)")
            except Exception as e:
                print(f"no bias correction ({type(e).__name__})")

    def __call__(self, z):
        return self.m(z.float()) - self.bias


class Decoder:
    """Latent -> gripper->cube offset, in metres, on the GPU."""

    def __init__(self, path="decoders.npz", device="cuda"):
        d = np.load(path)
        self.W = torch.as_tensor(d["Wd"], device=device, dtype=torch.float32)
        self.mu = torch.as_tensor(d["mud"], device=device, dtype=torch.float32)
        self.sd = torch.as_tensor(d["sdd"], device=device, dtype=torch.float32)
        self.err = float(d["errd"])

    def __call__(self, z):
        f = z.float().mean(dim=1)
        f = (f - self.mu) / self.sd
        f = torch.cat([f, torch.ones(len(f), 1, device=f.device)], dim=1)
        return f @ self.W          # (B, 3) offset in metres


@torch.no_grad()
def plan(predictor, decoder, z0, state0, target_off, device, dtype, rollout,
         samples, topk, iters, maxnorm, chunk, rng, grip, use_best=True):
    mean = np.zeros((rollout, 3), np.float32)
    std = np.full((rollout, 3), maxnorm, np.float32)
    tgt = torch.as_tensor(target_off, device=device, dtype=torch.float32)
    best_a, best_c = None, np.inf

    for _ in range(iters):
        s = rng.normal(size=(samples, rollout, 3)).astype(np.float32) * std + mean
        s = np.clip(s, -maxnorm, maxnorm)
        seqs = np.concatenate([s, np.zeros((samples, rollout, 3), np.float32),
                               np.full((samples, rollout, 1), grip, np.float32)],
                              axis=-1)
        costs = []
        for i in range(0, len(seqs), chunk):
            b = np.ascontiguousarray(seqs[i:i + chunk])
            n = len(b)
            z = z0.expand(n, -1, -1).contiguous()
            pose = np.repeat(state0[None], n, axis=0)
            for h in range(b.shape[1]):
                act = torch.as_tensor(b[:, h], device=device, dtype=dtype)[:, None]
                ps = torch.as_tensor(pose, device=device, dtype=dtype)[:, None]
                z = loader.predict_next(predictor, z, act, ps)
                pose = adapter.compute_new_pose(pose, b[:, h] * ACTION_SCALE)
            off = decoder(z)
            costs.append((off - tgt).norm(dim=-1).cpu().numpy())
        costs = np.concatenate(costs)
        order = np.argsort(costs)
        if costs[order[0]] < best_c:
            best_c, best_a = costs[order[0]], s[order[0]].copy()
        elite = s[order[:topk]]
        mean = elite.mean(0) * 0.85 + mean * 0.15
        # Keep exploring: elite std collapses near the target, and a collapsed
        # std means later iterations resample almost the same action.
        std = np.maximum(elite.std(0) * 0.85 + std * 0.15, maxnorm * 0.25)

    # Execute the BEST sampled action, not the elite mean. Averaging 8 elites
    # that disagree on direction produces a vector shorter than any of them --
    # harmless far from the target where elites agree, but near the target the
    # disagreement grows and the mean cancels itself, so the arm creeps to a
    # halt at a fixed radius. The servo, which executes its full computed
    # direction, has no such floor.
    a = np.zeros(7, np.float32)
    a[:3] = best_a[0] if (use_best and best_a is not None) else mean[0]
    a[6] = grip
    return a


def episode(encoder, predictor, decoder, seed, camera, device, dtype, budget,
            rollout, samples, topk, iters, maxnorm, chunk, fine=None,
            use_best=True):
    env = record.make_env("PickCube-v1", 256, camera=camera)
    obs, _ = env.reset(seed=seed)
    e = env.unwrapped
    rng = np.random.default_rng(seed)
    cube0 = e.cube.pose.p[0].cpu().numpy().copy()
    goal = record.to_np(obs["extra"]["goal_pos"][0])
    ever, lift, info, closest = False, 0.0, {}, 9.9

    # phase: (target gripper->cube offset, gripper command, step cap)
    # The step cap matters as much as the cost. Approach wants big strides, but
    # descending onto a 2 cm cube with a 5 cm step size cannot land inside the
    # tolerance -- the planner overshoots by construction, however good the
    # decoder is.
    fine = fine if fine is not None else maxnorm
    phases = [(np.array([0, 0, -0.08], np.float32), 1.0, maxnorm),   # hover
              (np.array([0, 0, -0.005], np.float32), 1.0, fine),     # descend
              (None, -1.0, fine),                                     # close
              (None, -1.0, maxnorm)]                                  # lift

    for pi, (target, grip, step_cap) in enumerate(phases):
        for _ in range(budget[pi]):
            frame = record.obs_frame(obs)
            st = record.obs_state(obs, env)
            z0 = loader.encode(encoder, frame[None], device, dtype)
            if target is not None:
                act = plan(predictor, decoder, z0, st, target, device, dtype,
                           rollout, samples, topk, iters, step_cap, chunk, rng,
                           grip, use_best)
                cmd = adapter.metric_action_to_maniskill(act)
            else:
                cmd = np.zeros(7, np.float32)
                if pi == 3:      # lift toward the goal position
                    tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
                    cmd[:3] = np.clip((goal - tcp) / adapter.POS_SCALE * 0.6, -1, 1)
                cmd[6] = grip
            obs, _, term, trunc, info = env.step(cmd)
            cube = e.cube.pose.p[0].cpu().numpy()
            tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
            closest = min(closest, float(np.linalg.norm(tcp - cube)))
            if bool(e.agent.is_grasping(e.cube)[0]):
                ever = True
            lift = max(lift, float(e.cube.pose.p[0, 2].cpu()) - cube0[2])
            if bool(term) or bool(trunc):
                break
    succ = bool(info["success"][0]) if "success" in info else False
    env.close()
    return ever, lift, succ, closest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--budget", type=int, nargs=4, default=[10, 10, 5, 15])
    ap.add_argument("--rollout", type=int, default=1,
                    help="1 is sound here: the decoded cost is a true distance, "
                         "so greedy descent on it is well-posed and avoids "
                         "compounding the predictor's error")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--maxnorm", type=float, default=0.05,
                    help="step cap for approach/lift phases, metres")
    ap.add_argument("--fine", type=float, default=0.015,
                    help="step cap for the descend/close phases -- must be well "
                         "under the 2 cm grasp tolerance")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--camera", default="close")
    ap.add_argument("--ckpt", default="predictor_all.pt")
    ap.add_argument("--decoder", default="keypoint", choices=["linear", "keypoint"])
    ap.add_argument("--no-bias-correct", action="store_true")
    ap.add_argument("--kp-ckpt", default="keypoint.pt",
                    help="keypoint_near.pt is trained on poses around the cube: "
                         "1.48 cm close-range error vs 3.69 cm for keypoint.pt")
    ap.add_argument("--elite-mean", action="store_true",
                    help="execute the elite mean (old behaviour) instead of the "
                         "single best sampled action")
    a = ap.parse_args()

    device, dtype = "cuda", torch.float16
    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    sd = torch.load(a.ckpt, map_location="cpu")
    predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    predictor.eval()
    decoder = (KPDecoder(path=a.kp_ckpt, device=device,
                         correct_bias=not a.no_bias_correct,
                         err=0.0148 if "near" in a.kp_ckpt else 0.0179)
               if a.decoder == "keypoint" else Decoder(device=device))
    print(f"decoder: {a.decoder}, offset error {decoder.err*100:.2f} cm; "
          f"ckpt {a.ckpt}")
    print(f"budget {a.budget}  step cap: {a.maxnorm} m coarse / {a.fine} m fine\n")

    print(f"{'seed':>5} {'grasp':>7} {'lift':>10} {'success':>8} {'closest':>9}")
    rows = []
    for s in range(a.seeds):
        r = episode(encoder, predictor, decoder, s, a.camera, device, dtype,
                    a.budget, a.rollout, a.samples, a.topk, a.iters,
                    a.maxnorm, a.chunk, a.fine, not a.elite_mean)
        rows.append(r)
        print(f"{s:>5} {str(r[0]):>7} {r[1]:>9.4f}m {str(r[2]):>8} {r[3]:>8.4f}m",
              flush=True)

    g = np.mean([r[0] for r in rows]); l = np.mean([r[1] for r in rows])
    sc = np.mean([r[2] for r in rows]); c = np.mean([r[3] for r in rows])
    print(f"\nDECODED-STATE PLANNING (n={a.seeds})")
    print(f"  grasp rate       {g*100:.1f}%      (latent-L1 planning: 0.0%)")
    print(f"  mean lift        {l:.4f} m")
    print(f"  success          {sc*100:.1f}%")
    print(f"  closest approach {c:.4f} m   (latent-L1: 0.148 m, subgoals: 0.106 m)")
    np.savez("state_planner.npz", grasp=np.array([r[0] for r in rows], float),
             lift=np.array([r[1] for r in rows]),
             succ=np.array([r[2] for r in rows], float),
             closest=np.array([r[3] for r in rows]))
    print("wrote state_planner.npz")


if __name__ == "__main__":
    main()
