"""
How accurate is the decoder on the latents it is ACTUALLY applied to?

state_planner scores imagined futures by decoding a gripper->cube offset from
the PREDICTOR's output. But every decoder here was fitted on the ENCODER's
output. Those are different distributions -- probe_imagination.py already showed
predicted latents score systematically lower on the grasp classifier than real
ones.

If the decoder degrades on predicted latents, the planner is aiming at a point
that is systematically wrong, and it would stall short of the target no matter
how good the search, the step size, or the decoder's accuracy on real frames.

Measures both on the same fresh rollouts:
    error on encoder latents   what fit_keypoint reported (1.79 cm)
    error on predicted latents what the planner actually relies on

    python3 decoder_on_predicted.py
"""

import numpy as np
import torch

import adapter
import loader
import record
from fit_keypoint import KeypointDecoder


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

    v_enc, v_pred = [], []
    for seed in range(6):
        env = record.make_env("PickCube-v1", 256, camera="close")
        obs, _ = env.reset(seed=seed)
        e = env.unwrapped
        rng = np.random.default_rng(seed)
        act = np.zeros(7, np.float32)

        prev = None
        for t in range(18):
            frame = record.obs_frame(obs)
            st = record.obs_state(obs, env)
            tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
            cube = e.cube.pose.p[0].cpu().numpy()
            true_off = torch.as_tensor(cube - tcp, device=dev).float()[None]

            z = loader.encode(enc, frame[None], dev, torch.float16)
            v_enc.append((dec(z.float()) - true_off)[0].cpu().numpy())

            # what the planner sees: offset decoded from an IMAGINED latent
            if prev is not None:
                z_prev, a_prev, s_prev = prev
                a = torch.as_tensor(a_prev, device=dev, dtype=torch.float16)[None, None]
                s = torch.as_tensor(s_prev, device=dev, dtype=torch.float16)[None, None]
                z_hat = loader.predict_next(pred, z_prev, a, s)
                v_pred.append((dec(z_hat.float()) - true_off)[0].cpu().numpy())

            act = 0.7 * act + 0.3 * rng.normal(size=7).astype(np.float32)
            cmd = np.clip(act * 0.5, -1, 1).astype(np.float32)
            cmd[3:6] = 0.0
            cmd[6] = 1.0
            metric = adapter.maniskill_action_to_metric(cmd[None])[0]
            prev = (z, metric, st)
            obs, *_ = env.step(cmd)
        env.close()

    A, B = np.array(v_enc), np.array(v_pred)

    def report(name, V):
        bias = V.mean(0)                       # systematic component
        spread = (V - bias)                    # random component
        print(f"\n{name}  (n={len(V)})")
        print(f"  mean |error|      {np.linalg.norm(V, axis=1).mean()*100:5.2f} cm")
        print(f"  BIAS vector       [{bias[0]*100:+.2f}, {bias[1]*100:+.2f}, "
              f"{bias[2]*100:+.2f}] cm   -> |bias| {np.linalg.norm(bias)*100:.2f} cm")
        print(f"  random scatter    {np.linalg.norm(spread, axis=1).mean()*100:5.2f} cm")
        return bias

    print("\nA BIAS is fatal for planning and harmless-ish for servoing: the")
    print("planner stops where decoded == target, i.e. |bias| from the truth.")
    b_enc = report("ENCODER latents  (what the servo decodes)", A)
    b_pred = report("PREDICTED latents (what the planner decodes)", B)

    print(f"\nplanner stalls at ~4.0 cm; servo reaches 2.4 cm and grasps")
    if np.linalg.norm(b_pred) > 0.02:
        print(f"\nEXPLAINS IT: predicted-latent decoding carries a systematic")
        print(f"{np.linalg.norm(b_pred)*100:.2f} cm offset. Subtract it and the planner")
        print(f"should reach the cube.")
    else:
        print("\nBias is small, so the stall is not a systematic offset.")
    np.savez("decoder_bias.npz", enc=A, pred=B, bias_enc=b_enc, bias_pred=b_pred)


if __name__ == "__main__":
    main()
