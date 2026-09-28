"""Language-conditioned planning with a rule-based parser instead of a
vision-language model. The planner wants a target offset in metres, not an
image, so an instruction only has to become an offset. Nothing is retrained.
This demonstrates that the planner is instruction-addressable; it does not claim
language understanding.

    python3 -m vjepa.evaluate.language_plan --instruction "hover above the cube"
    python3 -m vjepa.evaluate.language_plan --demo
"""

import argparse
import re

import numpy as np
import torch

from vjepa.core import adapter, loader, record
from vjepa.plan.hier_plan import choose_direction
from vjepa.plan.state_planner import KPDecoder

# offsets are (cube - gripper), metres: negative z means the cube is BELOW
PRIMITIVES = {
    "hover":  np.array([0.00, 0.00, -0.085], np.float32),
    "touch":  np.array([0.00, 0.00, -0.004], np.float32),
    "left":   np.array([0.00, 0.08, -0.004], np.float32),
    "right":  np.array([0.00, -0.08, -0.004], np.float32),
    "front":  np.array([0.08, 0.00, -0.004], np.float32),
    "behind": np.array([-0.08, 0.00, -0.004], np.float32),
    "high":   np.array([0.00, 0.00, -0.18], np.float32),
}

# Order matters: the FIRST match wins, so more specific patterns come first.
# "lift the gripper high above the cube" contains both "high" and "above"; with
# hover listed first it silently parsed as hover and ran the wrong target.
RULES = [
    (r"\b(high|way up|far above|lift up)\b",        "high"),
    (r"\b(hover|above|over|on top)\b",              "hover"),
    (r"\b(down|touch|onto|reach|at the cube|grasp)\b", "touch"),
    (r"\b(left)\b",                                 "left"),
    (r"\b(right)\b",                                "right"),
    (r"\b(front|toward me|nearer)\b",               "front"),
    (r"\b(behind|back|away)\b",                     "behind"),
]


def parse(instruction):
    """Instruction -> target offset. Returns (name, offset) or (None, None)."""
    s = instruction.lower()
    hits = [name for pat, name in RULES if re.search(pat, s)]
    if not hits:
        return None, None
    # a lateral word plus a height word: keep the lateral, take height from it
    lateral = [h for h in hits if h in ("left", "right", "front", "behind")]
    name = lateral[0] if lateral else hits[0]
    off = PRIMITIVES[name].copy()
    if lateral and "hover" in hits:
        off[2] = PRIMITIVES["hover"][2]
        name += "+hover"
    m = re.search(r"(\d+)\s*cm", s)                 # "10 cm to the left"
    if m and lateral:
        d = float(m.group(1)) / 100.0
        off[:2] = np.sign(off[:2]) * d
    return name, off


class RangeDecoder:
    """Pick the decoder by how far the target is.

    keypoint_near.pt was retrained on poses AROUND the cube, which is what made
    grasping work -- 1.48 cm inside 6 cm. The cost is accuracy further out:
    3.0 cm at 10-18 cm and 10.4 cm beyond that (decoder_vs_range.py). A target
    like "18 cm above the cube" lands in exactly that band, and planning toward
    it made the error worse rather than better.

    keypoint.pt was trained on random arm motion, so it is the weaker decoder up
    close and the better one far away. Switching on the requested standoff gets
    both regimes right.
    """

    def __init__(self, near_path, far_path, device, switch=0.12):
        from vjepa.plan.state_planner import KPDecoder as _K
        self.near = _K(path=near_path, device=device, correct_bias=False, err=0.0148)
        self.far = _K(path=far_path, device=device, correct_bias=False, err=0.0179)
        self.switch = switch
        self.active = self.near

    def select(self, target):
        """Called once per instruction, on the requested offset magnitude."""
        far = float(np.linalg.norm(target)) > self.switch
        self.active = self.far if far else self.near
        return "far" if far else "near"

    def __call__(self, z):
        return self.active(z)


@torch.no_grad()
def run(encoder, predictor, decoder, seed, target, device, dtype,
        steps=24, probe=0.12, exec_step=0.015, samples=48, topk=8, iters=3,
        chunk=16, view=False):
    env = record.make_env("PickCube-v1", 256, camera="close")
    if view:
        env.close()
        import gymnasium as gym, mani_skill.envs  # noqa: F401
        record.set_camera("close")
        env = gym.make("PickCube-v1", obs_mode="rgb", render_mode="human",
                       control_mode="pd_ee_delta_pose",
                       sensor_configs=dict(width=256, height=256, fov=1.0))
    obs, _ = env.reset(seed=seed)
    e = env.unwrapped
    rng = np.random.default_rng(seed)

    def offset_now():
        tcp = record.to_np(obs["extra"]["tcp_pose"][0])[:3]
        return e.cube.pose.p[0].cpu().numpy() - tcp

    start_err = float(np.linalg.norm(offset_now() - target))
    for _ in range(steps):
        z0 = loader.encode(encoder, record.obs_frame(obs)[None], device, dtype)
        d, _ = choose_direction(predictor, decoder, z0,
                                record.obs_state(obs, env), target, probe,
                                device, dtype, samples, topk, iters, chunk,
                                rng, 1.0, True)
        m = np.zeros(7, np.float32)
        m[:3] = d * exec_step
        cmd = adapter.metric_action_to_maniskill(m)
        cmd[6] = 1.0
        obs, _, term, trunc, _ = env.step(cmd)
        if view:
            try:
                env.render()
            except Exception:
                pass
        if bool(term) or bool(trunc):
            break
    end_err = float(np.linalg.norm(offset_now() - target))
    env.close()
    return start_err, end_err


DEMO = ["hover above the cube",
        "go down and touch the cube",
        "move 10 cm to the left of the cube",
        "lift the gripper high above the cube"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/predictor_all.pt")
    ap.add_argument("--kp-ckpt", default="checkpoints/keypoint_near.pt",
                    help="decoder for close targets")
    ap.add_argument("--kp-far", default="checkpoints/keypoint.pt",
                    help="decoder for distant targets (trained on random poses)")
    ap.add_argument("--switch", type=float, default=0.12,
                    help="target magnitude above which the far decoder is used")
    a = ap.parse_args()

    instructions = DEMO if a.demo else [a.instruction or "hover above the cube"]
    device, dtype = "cuda", torch.float16
    encoder, predictor = loader.load_ac_model(device=device, dtype=dtype)
    sd = torch.load(a.ckpt, map_location="cpu")
    predictor.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    predictor.eval()
    decoder = RangeDecoder(a.kp_ckpt, a.kp_far, device, switch=a.switch)

    print(f"\n{'instruction':<40} {'parsed':<12} {'dec':>4} "
          f"{'start err':>10} {'end err':>9}")
    rows = []
    for ins in instructions:
        name, target = parse(ins)
        if target is None:
            print(f"{ins:<40} {'UNPARSED':<14}")
            continue
        which = decoder.select(target)
        s, en = run(encoder, predictor, decoder, a.seed, target, device, dtype,
                    steps=a.steps, view=a.view)
        rows.append((ins, name, s, en))
        print(f"{ins:<40} {name:<12} {which:>4} {s:>9.3f}m {en:>8.3f}m",
              flush=True)

    if rows:
        red = np.mean([(s - e) / max(s, 1e-6) for _, _, s, e in rows])
        print(f"\nmean error reduction: {red*100:.0f}%")
        print("(error = distance between the achieved and the requested offset)")
        np.savez("results/language_plan.npz",
                 instructions=np.array([r[0] for r in rows]),
                 parsed=np.array([r[1] for r in rows]),
                 start=np.array([r[2] for r in rows]),
                 end=np.array([r[3] for r in rows]))


if __name__ == "__main__":
    main()
