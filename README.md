# V-JEPA 2-AC + ManiSkill

Running Meta's action-conditioned world model as a planner in ManiSkill, on a
6 GB laptop GPU.

Status: **works after fine-tuning, end to end.** The pretrained model does not transfer to
ManiSkill zero-shot (action signal at chance), but fine-tuning the 305M AC
predictor on 4k ManiSkill frames — encoder frozen — moves the true action from
the 48th percentile to the 12th (p<0.0001, fresh seeds) and makes CEM planning
close ~2/3 of the distance to a goal image on 12/12 seeds (p=0.0003), where the
pretrained model diverges. All numbers here were
produced on this machine (RTX 3060 Laptop, 6 GB).

## Install

Already satisfied on this machine. For a fresh one:

```bash
pip install torch torchvision mani-skill transformers huggingface_hub timm einops scipy
sudo apt install libvulkan1 libegl1 libgl1-mesa-dri   # headless rendering
```

Then fetch the upstream model source (not committed here — it is Meta's code
under its own license):

```bash
./setup.sh          # clones facebookresearch/vjepa2 into vjepa2_src/
```

Weights are pulled from Hugging Face on first use (5.27 GB) and cached in
`~/.cache/huggingface`.

**Not in this repo** (regenerate, or ask for them): `data/latents.npy` (2.9 GB
of cached latents, rebuild with `collect_data.py`), `predictor_ft.pt` (294 MB
of fine-tuned weights, rebuild with `finetune.py`), and the recorded
`traj*.npz` trajectories. Everything needed to reproduce them is here; the
small `*.npz` result files from each experiment are committed so the numbers in
this README can be checked without rerunning anything.

## Why not torch.hub

Two independent upstream problems, both verified:

1. `src/hub/backbones.py` ships a debug value: `VJEPA_BASE_URL =
   "http://localhost:8300"`, with the real URL commented out. Nothing
   downloads.
2. The real checkpoint, `dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`, is
   **11.76 GB** — it carries optimizer state and will not fit here.

`loader.py` sidesteps both by building the model from the vendored source and
loading the stripped 5.27 GB inference checkpoint `vjepa2_ac_oss.pth.tar` from
the `facebook/jepa-wms` mirror.

## The convention trap (the important part)

ManiSkill's `extra/tcp_pose` and V-JEPA 2-AC's `states` are both 7-dimensional,
so they concatenate without error and mean completely different things:

| | layout |
|---|---|
| ManiSkill `tcp_pose` | `[x, y, z, qw, qx, qy, qz]` — quaternion |
| V-JEPA 2-AC `states` | `[x, y, z, roll, pitch, yaw, closedness]` — euler + gripper |

Actions have a second, independent mismatch: ManiSkill's `pd_ee_delta_pose`
emits values in `[-1, 1]` that the controller scales to ±0.1 m, while the
predictor wants **metric** deltas and was trained on DROID at |dxyz| ≲ 0.05 m.
Feeding raw ManiSkill actions overstates every motion by ~10× at full scale.

Both conversions live in `adapter.py` and nowhere else. Verified against
`vjepa2_src/notebooks/franka_example_traj.npz`, whose `states` are euler with a
0–1 gripper closedness.

## Run order

```bash
python3 validate_reference.py      # 1  does the stack work at all?
python3 record.py --steps 32       # 2  render + record a fixture trajectory
python3 openloop.py --horizon 12   # 3  prediction error vs horizon
python3 energy_maniskill.py        # 4  is there usable action signal?
python3 camera_ablation.py         # 5  is the viewpoint the problem?
python3 probe_reps.py              # 6  is the encoder the bottleneck? (no)
python3 collect_data.py            # 7  cache 4k encoded frames  (~25 min)
python3 finetune.py --epochs 4     # 8  fine-tune the predictor  (~30 min)
python3 eval_finetuned.py          # 9  fresh-seed paired test
python3 eval_planning.py           # 10  closed-loop goal reaching
python3 cem_plan.py --ckpt predictor_ft.pt --steps 20   # 11  interactive planning
```

`validate_reference.py` runs entirely on upstream's own ground-truth Franka
pair, so it isolates "is the model wired up correctly" from "does it transfer".
Run it first; if it fails, nothing downstream means anything.

## Results

**1. The stack is correct.** On upstream's reference Franka pair, sweeping a
7×7×7 grid of translation actions and scoring each imagined future against the
real next frame:

- true action `(+0.092, +0.031, +0.084)`, grid argmin `(+0.080, +0.080, +0.080)`
- x and z hit the nearest grid point exactly; y is one step off
- relative energy spread **0.163** — the landscape is far from flat
- predicted-vs-true error **0.426** beats the static baseline **0.556** by +0.130,
  and beats a zero-action prediction (0.480)

So `encode()`, `predict_next()`, the layer-norm and the conventions are right.

**2. Open-loop rollout on ManiSkill fails against a do-nothing baseline.**
`openloop.py`, 12 steps, L1 in latent space:

| | h=1 | h=6 | h=12 |
|---|---|---|---|
| true actions | 0.315 | 0.455 | 0.506 |
| shuffled actions | 0.337 | 0.446 | 0.499 |
| static (frame 0 vs frame h) | 0.126 | 0.302 | 0.387 |

The shuffled-action gap is +0.023 at h=1 and collapses to **+0.001 on average**,
going negative at several horizons. The static baseline wins everywhere by a
wide margin. Compare with the reference pair above, where the model beat static
by +0.130 — the yardstick is valid, so this is a genuine domain-transfer effect,
not a metric artifact.

**3. Why: a magnitude prior, not a direction signal.** A naive grid sweep on
ManiSkill frames is misleading, because the argmin is the **zero action at every
single timestep sampled** (`|argmin| = 0.0000` vs `|true| = 0.032`). The model's
preferred prediction is "nothing moves". Separating magnitude from direction by
scoring only candidates on the same-norm shell as the true action leaves a
landscape with real structure (mean spread 0.154) that does not track the true
direction.

**4. The camera is not the explanation.** The obvious confound is viewpoint:
DROID is a close third-person view, while ManiSkill's `base_camera` leaves the
cube a few pixels across. `camera_ablation.py` tests this properly — for each
seed both cameras render the *same* executed trajectory, so only pixels differ.

| camera | mean `rank_shell` (n=6 seeds) | vs chance |
|---|---|---|
| default | 0.4649 ± 0.0548 | p=0.55 |
| close (DROID-like) | 0.4836 ± 0.0320 | p=0.63 |
| paired difference | +0.0186 | p=0.61 |

**Neither camera beats chance, and the close view is very slightly worse.**
Per-seed values range 0.30–0.63: the variance between trajectories dwarfs any
camera effect.

> **A caution worth keeping.** An earlier version of this analysis pooled
> timesteps *within a single trajectory* and reported the close camera at
> `rank_shell` 0.386, p=0.019 — apparently significant. That was an artifact:
> timesteps inside one rollout are strongly correlated, so treating them as
> independent samples inflates n and manufactures significance. Computing the
> statistic over seeds instead makes the effect vanish (p=0.61). If you rerun
> any of this, keep seeds as the unit of analysis.

So the transfer failure is *not* explained by viewpoint, and the honest summary
is that V-JEPA 2-AC's energy landscape carries **no measurable action signal on
ManiSkill renders**, while carrying a clear one on DROID frames (result 1).

**5. Fine-tuning the predictor fixes it.** Two findings make this the right
intervention rather than a guess:

- `probe_reps.py`: a ridge probe from the **frozen** encoder's features decodes
  arm position at R² = **0.956** and cube position at R² = **0.962**
  (shuffled-label controls ≈ −1.0). The encoder is not the bottleneck — the
  scene state is fully present in its representation.
- So what fails is the action-conditioned dynamics on top of it, i.e. the
  predictor, and only the predictor.

`collect_data.py` caches 4,000 encoded ManiSkill frames (200 episodes), and
`finetune.py` trains the conditioning pathway plus the top 6 blocks (78M of
305M params) on the model's own objective — L1 to the encoder's next-frame
latent. The encoder is never loaded during training, which is what makes this
fit in 6 GB.

| epoch | train L1 | val L1 | rank_shell |
|---|---|---|---|
| before | — | — | 0.4427 |
| 1 | 0.21913 | 0.21293 | 0.3898 |
| 2 | 0.20583 | 0.20854 | 0.3537 |
| 3 | 0.20064 | 0.20382 | 0.1667 |
| 4 | 0.19581 | 0.20043 | **0.1159** |

`eval_finetuned.py` then re-tests on **fresh seeds** (training used 10000+,
evaluation uses 0–7), scoring both models on identical trajectories with
statistics over seeds:

| | mean `rank_shell` (n=8 seeds) | vs chance |
|---|---|---|
| pretrained | 0.4766 ± 0.0291 | p=0.45 — chance |
| fine-tuned | **0.1181 ± 0.0198** | **p<0.0001** |
| paired difference | **−0.3585** | **p<0.0001** |

All 8/8 seeds improved and the distributions do not overlap (worst fine-tuned
seed 0.228 beats best pretrained seed 0.341). Unlike the camera effect in
result 4, this survives the seed-level correction comfortably.

**6. Closed-loop planning works.** Ranking actions well one step ahead does not
guarantee planning works — CEM feeds the model its own predictions back over a
horizon, which is exactly where result 2 showed error compounding. So this is
tested directly. `eval_planning.py` builds a goal image by rolling a scripted
reach, resets, plans toward that image with CEM, and measures how much closer
the end-effector gets. Both models plan on identical goals and seeds.

| | mean distance improvement (n=12 seeds) | vs zero | closes distance on |
|---|---|---|---|
| pretrained | −0.0600 ± 0.0232 m | p=0.025 (moves *away*) | 3/12 seeds |
| fine-tuned | **+0.1021 ± 0.0102 m** | **p<0.0001** | **12/12 seeds** |
| paired difference | **+0.1621 m** | **p=0.0003** | favours FT on 11/12 |

Starting distance is ~0.15 m, so the fine-tuned planner closes roughly two
thirds of the gap on average, and the pretrained one actively diverges.

> An n=4 version of this run gave the same effect size (+0.1622 m) at p=0.12 —
> suggestive but underpowered. Same lesson as result 4: pick n before believing
> a paired comparison.

## What to try next

Roughly in order of expected value per unit effort:

1. ~~Close the visual gap.~~ **Tested and ruled out** — see result 4.
2. ~~Fine-tune the AC predictor.~~ **Done, and it worked** — see result 5.
   Natural extensions: train all 305M params (needs >6 GB or 8-bit Adam), more
   data, longer rollout horizons, and closed-loop task success.
3. **Try the simulation-trained checkpoints.** `facebook/jepa-wms` also carries
   `jepa_wm_metaworld.pth.tar` (0.21 GB) and `dino_wm_metaworld.pth.tar`
   (0.28 GB), trained on sim rather than real footage. Different architecture,
   so they need their own loading path, but they are small and the real-to-sim
   gap is exactly the hypothesis they test.
4. **Report the whole arc.** Items 2–3 may not work either. "V-JEPA
   2-AC does not zero-shot transfer to ManiSkill renders; the energy landscape
   is a magnitude prior with no measurable directional signal, and viewpoint is
   not the cause" is a complete finding. The magnitude/direction decomposition
   and the seed-level camera ablation are the parts nobody bothers to produce.

## Files

| file | role |
|---|---|
| `adapter.py` | ManiSkill ↔ DROID conventions. The trap above lives here. |
| `loader.py` | Build + load the model, `encode()`, `predict_next()`. |
| `validate_reference.py` | Ground-truth check against upstream's own data. |
| `record.py` | Record a fixture trajectory in DROID conventions. |
| `openloop.py` | Prediction error vs horizon, with shuffled + static controls. |
| `energy_maniskill.py` | Magnitude-vs-direction decomposition of the energy landscape. |
| `camera_ablation.py` | Paired default-vs-DROID-like camera test, statistics over seeds. |
| `probe_reps.py` | Linear probe: is the state decodable from frozen encoder feats? |
| `collect_data.py` | Cache encoded ManiSkill rollouts for training. |
| `finetune.py` | Fine-tune the AC predictor on cached latents. |
| `eval_finetuned.py` | Paired pretrained-vs-fine-tuned test on fresh seeds. |
| `eval_planning.py` | Closed-loop goal-reaching, pretrained vs fine-tuned. |
| `cem_plan.py` | Closed-loop CEM planning. |

## Practical notes

- **Disk is the binding constraint**, not VRAM: the checkpoint is 5.27 GB and
  this machine has ~2 GB free after it. Do not also download
  `vjepa2_ac_droid.pth.tar` without freeing space first.
- Weights load in fp16 (2.65 GB VRAM); peak during these experiments was 2.68 GB.
- The predictor's RoPE builds its angles in fp32 while values stay fp16, which
  makes `scaled_dot_product_attention` reject the mismatch. `loader._autocast`
  reconciles this; it is why every forward runs under `torch.autocast`.
- Building the encoder on the meta device fails — `VisionTransformer.__init__`
  calls `.item()`. `loader.py` builds directly in fp16 instead to keep host RAM
  near 5 GB rather than 10 GB.
- If you OOM in CEM, lower `--chunk` first (it does not change results), then
  `--samples`, then `--rollout`.
