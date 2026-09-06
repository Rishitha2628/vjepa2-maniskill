"""
Open-loop prediction error vs horizon. This is the real gate -- run it before
writing any planner.

Feeds the model frame 0 plus a sequence of actions, rolls forward in latent
space without ever seeing another real frame, and compares each imagined latent
against the encoded real frame.

Three curves are produced, and only the comparison between them means anything:

  true      rolled with the actions the robot actually executed
  shuffled  the same actions in a random order -- a control. If this matches
            `true`, the model is ignoring the actions and every planning result
            downstream would be noise.
  static    error of simply asserting nothing changes (rep of frame 0 vs rep of
            frame h). A world model that cannot beat this is not modelling
            dynamics, only appearance.

Poses are propagated through the actions rather than read from the simulator,
because reading them back would leak ground-truth state into an open-loop test.

    python3 openloop.py --horizon 12
"""

import argparse

import numpy as np
import torch

import adapter
import loader


@torch.no_grad()
def rollout(encoder, predictor, true_reps, actions, states0, horizon,
            device, dtype, max_context):
    """Roll the model forward `horizon` steps from frame 0. Returns L1 error per step."""
    z_ctx = true_reps[0][None]                       # (1, 256, D)
    pose = states0[None].copy()                      # (1, 7)
    a_hist, s_hist = [], []
    errs = []

    for h in range(horizon):
        a_hist.append(actions[h][None].copy())
        s_hist.append(pose.copy())

        # Keep the context window bounded: attention is quadratic in frames and
        # the predictor was built for at most NUM_FRAMES // TUBELET of them.
        keep = min(len(a_hist), max_context)
        z_in = z_ctx[:, -keep * loader.TOKENS_PER_FRAME:]
        a_in = torch.as_tensor(np.stack(a_hist[-keep:], 1), device=device, dtype=dtype)
        s_in = torch.as_tensor(np.stack(s_hist[-keep:], 1), device=device, dtype=dtype)

        z_next = loader.predict_next(predictor, z_in, a_in, s_in)      # (1,256,D)
        z_ctx = torch.cat([z_ctx, z_next], dim=1)

        tgt = true_reps[h + 1][None]
        errs.append((z_next.float() - tgt.float()).abs().mean().item())

        pose = adapter.compute_new_pose(pose, actions[h][None])

    return errs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", default="traj.npz")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--max-context", type=int, default=4,
                   help="frames of imagined history fed back in; the reference "
                        "MPC uses a short window and attention is quadratic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="openloop.npz")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    d = np.load(a.traj)
    frames, actions, states = d["frames"], d["actions"], d["states"]
    H = min(a.horizon, len(frames) - 1)

    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    true_reps = loader.encode(encoder, frames, device, dtype)     # (T,256,D)
    print(f"encoded {len(frames)} frames -> {tuple(true_reps.shape)}")

    true_err = rollout(encoder, predictor, true_reps, actions, states[0], H,
                       device, dtype, a.max_context)

    rng = np.random.default_rng(a.seed)
    shuf = actions.copy()
    rng.shuffle(shuf)
    shuf_err = rollout(encoder, predictor, true_reps, shuf, states[0], H,
                       device, dtype, a.max_context)

    z0 = true_reps[0][None].float()
    static_err = [(z0 - true_reps[h + 1][None].float()).abs().mean().item()
                  for h in range(H)]

    print(f"\n{'h':>3} {'true':>10} {'shuffled':>10} {'static':>10} "
          f"{'shuf-true':>10} {'static-true':>12}")
    for h in range(H):
        print(f"{h+1:>3} {true_err[h]:>10.5f} {shuf_err[h]:>10.5f} "
              f"{static_err[h]:>10.5f} {shuf_err[h]-true_err[h]:>10.5f} "
              f"{static_err[h]-true_err[h]:>12.5f}")

    t, s, st = map(np.array, (true_err, shuf_err, static_err))
    gap, sgap = float((s - t).mean()), float((st - t).mean())
    print(f"\nmean shuffled-true gap : {gap:+.5f}")
    print(f"mean static-true gap   : {sgap:+.5f}")

    if gap <= 0:
        print("FAIL: shuffling the actions does not hurt. The model is not "
              "conditioning on your actions -- fix that before planning.")
    elif sgap <= 0:
        print("WEAK: the model beats shuffled actions but not a do-nothing "
              "baseline. Conditioning works; dynamics are not yet useful.")
    else:
        print("PASS: actions matter and the model beats the static baseline.")

    np.savez(a.out, true_err=t, shuf_err=s, static_err=st)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
