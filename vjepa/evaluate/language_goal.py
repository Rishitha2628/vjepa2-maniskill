"""Language-conditioned goals by CLIP retrieval over frames the simulator actually
produced, so the goal is reachable by construction. It does not work; see
RESULTS.md result 16.

    python3 -m vjepa.evaluate.language_goal --retrieve-only
"""

import argparse

import numpy as np
import torch

from vjepa.core import record


def build_bank(n_episodes, steps, camera, seed0=60_000):
    """Frames spanning a range of arm poses, so retrieval has choices."""
    frames, states = [], []
    env = record.make_env("PickCube-v1", 256, camera=camera)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        e = env.unwrapped
        rng = np.random.default_rng(seed0 + ep)
        cube = e.cube.pose.p[0].cpu().numpy()
        # half the episodes wander, half approach the cube -- gives the bank
        # both "far away" and "near the cube" frames to retrieve between
        approach = ep % 2 == 0
        act = np.zeros(7, np.float32)
        for t in range(steps):
            frames.append(record.obs_frame(obs))
            tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
            states.append(np.concatenate([tcp, cube - tcp]))
            if approach:
                tgt = cube + np.array([0, 0, max(0.02, 0.18 - 0.012 * t)])
                cmd = np.zeros(7, np.float32)
                cmd[:3] = np.clip((tgt - tcp) / 0.1 * 0.5, -1, 1)
            else:
                act = 0.7 * act + 0.3 * rng.normal(size=7).astype(np.float32)
                cmd = np.clip(act * 0.6, -1, 1).astype(np.float32)
            cmd[3:6] = 0.0
            cmd[6] = 1.0
            obs, *_ = env.step(cmd)
    env.close()
    return np.asarray(frames, np.uint8), np.asarray(states, np.float32)


def _emb(out):
    """transformers 5.x returns BaseModelOutputWithPooling from get_*_features;
    older versions return the tensor directly. Handle both."""
    if torch.is_tensor(out):
        return out
    return getattr(out, "pooler_output", None) if getattr(
        out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]


@torch.no_grad()
def clip_index(frames, device, batch=64):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    feats = []
    for i in range(0, len(frames), batch):
        inp = proc(images=list(frames[i:i + batch]), return_tensors="pt").to(device)
        f = _emb(model.get_image_features(**inp))
        feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    return model, proc, torch.cat(feats)


@torch.no_grad()
def retrieve(model, proc, bank_feats, query, device, k=1):
    inp = proc(text=[query], return_tensors="pt", padding=True).to(device)
    t = torch.nn.functional.normalize(
        _emb(model.get_text_features(**inp)), dim=-1).cpu()
    sims = (bank_feats @ t.T).squeeze(-1)
    idx = sims.topk(k).indices.numpy()
    return idx, sims[idx].numpy()


QUERIES = [
    "the robot gripper is high above the table",
    "the robot gripper is right next to the red cube",
    "the robot arm is far from the red cube",
    "the gripper is touching the red cube",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--camera", default="close")
    ap.add_argument("--retrieve-only", action="store_true")
    ap.add_argument("--out", default="media/language_retrieval.png")
    a = ap.parse_args()

    device = "cuda"
    frames, states = build_bank(a.episodes, a.steps, a.camera)
    print(f"bank: {len(frames)} frames")
    model, proc, feats = clip_index(frames, device)
    print(f"CLIP indexed -> {tuple(feats.shape)}\n")

    rows, K = [], 4
    print(f"{'query':<46} {'sim':>6}  {'gripper->cube of top hit':>26}")
    for q in QUERIES:
        idx, sims = retrieve(model, proc, feats, q, device, k=K)
        d = np.linalg.norm(states[idx[0]][3:])       # cube - tcp of best match
        print(f"{q:<46} {sims[0]:>6.3f}  {d:>24.3f} m")
        rows.append(np.concatenate([frames[i] for i in idx], axis=1))

    import imageio
    sep = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    imageio.imwrite(a.out, np.concatenate(
        [r for pair in zip(rows, [sep] * len(rows)) for r in pair][:-1], axis=0))
    print(f"\nwrote {a.out}  (one row per query, top-{K} matches left to right)")
    print("Check: do 'next to the cube' rows actually show the gripper near the cube,")
    print("and 'far from' rows show it away? If not, CLIP cannot read these renders")
    print("and retrieval is not a usable goal source.")
    np.savez("results/language_retrieval.npz",
             queries=np.array(QUERIES), states=states)


if __name__ == "__main__":
    main()
