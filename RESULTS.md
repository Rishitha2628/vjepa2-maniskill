# Detailed results

These are the full write-ups of every experiment, in the order I ran them. The
[README](README.md) has the summary and the final picture. This file keeps the
back and forth, including the parts where I was wrong, because the order things
happened in is most of what I learned.

All numbers came off my laptop, an RTX 3060 with 6 GB of VRAM.

## 1. The stack is wired up correctly

Before testing anything on ManiSkill I checked the model against Meta's own
ground-truth Franka pair. I swept a 7x7x7 grid of translation actions, imagined
each one, and scored the result against the real next frame.

- true action `(+0.092, +0.031, +0.084)`, grid argmin `(+0.080, +0.080, +0.080)`
- x and z land on the nearest grid point exactly, y is one step off
- relative energy spread 0.163, so the landscape is far from flat
- predicted-vs-true error 0.426, against 0.556 for a static baseline and 0.480
  for a zero action

So `encode()`, `predict_next()`, the layer norm and my conventions are right.
Anything that fails later is a transfer problem, not a wiring problem. I would
run this first again.

## 2. Open-loop rollout on ManiSkill loses to doing nothing

`openloop.py`, 12 steps, L1 in latent space:

| | h=1 | h=6 | h=12 |
|---|---|---|---|
| true actions | 0.315 | 0.455 | 0.506 |
| shuffled actions | 0.337 | 0.446 | 0.499 |
| static (frame 0 vs frame h) | 0.126 | 0.302 | 0.387 |

Feeding the true actions beats feeding shuffled ones by 0.023 at h=1, and that
gap collapses to 0.001 on average, going negative at several horizons. The
static baseline wins everywhere by a lot. On the reference pair in result 1 the
model beat static by 0.130, so the yardstick works and this is a real
domain-transfer effect rather than a broken metric.

## 3. What the model gives me is a magnitude prior, not a direction

A plain grid sweep on ManiSkill frames is misleading, because the argmin is the
zero action at every single timestep I sampled (`|argmin| = 0.0000` against
`|true| = 0.032`). The model's favourite prediction is that nothing moves.

Separating magnitude from direction, by scoring only candidates on the same-norm
shell as the true action, leaves a landscape with real structure (mean spread
0.154) that does not track the true direction. This is where `rank_shell` comes
from: the percentile of the true action among same-norm candidates, so 0.5 is
chance.

## 4. The camera is not the explanation

The obvious confound is viewpoint. DROID is a close third-person view, while
ManiSkill's `base_camera` leaves the cube a few pixels across. `camera_ablation.py`
tests it properly: for each seed both cameras render the same executed
trajectory, so only the pixels differ.

| camera | mean `rank_shell` (n=6 seeds) | vs chance |
|---|---|---|
| default | 0.4649 +/- 0.0548 | p=0.55 |
| close (DROID-like) | 0.4836 +/- 0.0320 | p=0.63 |
| paired difference | +0.0186 | p=0.61 |

Neither camera beats chance, and the close view is slightly worse. Per-seed
values range from 0.30 to 0.63, so the variance between trajectories is much
bigger than any camera effect.

An earlier version of this analysis pooled timesteps inside a single trajectory
and reported the close camera at `rank_shell` 0.386, p=0.019, which looks
significant. It was an artifact. Timesteps inside one rollout are strongly
correlated, so treating them as independent inflates n and manufactures
significance. Computing the statistic over seeds makes it vanish (p=0.61). If
you rerun any of this, keep seeds as the unit of analysis.

So the transfer failure is not about viewpoint. The honest summary is that
V-JEPA 2-AC's energy landscape carries no measurable action signal on ManiSkill
renders while carrying a clear one on DROID frames.

## 5. Fine-tuning the predictor fixes the action signal

Two things made this the right intervention rather than a guess:

- `probe_reps.py`: a ridge probe on the frozen encoder's features decodes arm
  position at R2 = 0.956 and cube position at R2 = 0.962, with shuffled-label
  controls near -1.0. The encoder is not the bottleneck, the scene state is
  already in its representation.
- So what fails is the action-conditioned dynamics on top of it, which is the
  predictor and only the predictor.

`collect_data.py` caches 4,000 encoded ManiSkill frames from 200 episodes, and
`finetune.py` trains the conditioning pathway plus the top 6 blocks (78M of 305M
params) on the model's own objective, L1 to the encoder's next-frame latent. The
encoder is never loaded during training, which is what makes this fit in 6 GB.

| epoch | train L1 | val L1 | rank_shell |
|---|---|---|---|
| before | | | 0.4427 |
| 1 | 0.21913 | 0.21293 | 0.3898 |
| 2 | 0.20583 | 0.20854 | 0.3537 |
| 3 | 0.20064 | 0.20382 | 0.1667 |
| 4 | 0.19581 | 0.20043 | 0.1159 |

`eval_finetuned.py` re-tests on fresh seeds (training used 10000+, evaluation
uses 0 to 7), scoring both models on identical trajectories with statistics over
seeds:

| | mean `rank_shell` (n=8 seeds) | vs chance |
|---|---|---|
| pretrained | 0.4766 +/- 0.0291 | p=0.45, chance |
| fine-tuned | 0.1181 +/- 0.0198 | p<0.0001 |
| paired difference | -0.3585 | p<0.0001 |

All 8 seeds improved and the distributions do not overlap: the worst fine-tuned
seed (0.228) beats the best pretrained one (0.341). Unlike the camera effect,
this survives the seed-level correction comfortably.

## 6. Closed-loop planning works for reaching

Ranking actions well one step ahead does not mean planning works, because CEM
feeds the model its own predictions back over a horizon, which is where result 2
showed error compounding. `eval_planning.py` builds a goal image by rolling a
scripted reach, resets, plans toward that image with CEM, and measures how much
closer the end-effector gets. Both models plan on identical goals and seeds.

| | mean distance improvement (n=12 seeds) | vs zero | closes distance on |
|---|---|---|---|
| pretrained | -0.0600 +/- 0.0232 m | p=0.025, moves away | 3/12 seeds |
| fine-tuned | +0.1021 +/- 0.0102 m | p<0.0001 | 12/12 seeds |
| paired difference | +0.1621 m | p=0.0003 | favours FT on 11/12 |

Starting distance is about 0.15 m, so the fine-tuned planner closes roughly two
thirds of the gap and the pretrained one actively diverges.

An n=4 version of this run gave the same effect size (+0.1622 m) at p=0.12,
suggestive but underpowered. Same lesson as result 4: pick n before believing a
paired comparison.

## 7. Grasping does not work, and the reason is not the representation

`collect_grasp_data.py` builds 7,500 frames that actually contain grasps
(113 of 250 episodes) by interpolating a scripted pick-and-place with noise. I
extended CEM to search the gripper channel and fine-tuned the predictor again.
Evaluated on real task outcomes:

| mode | grasp rate | mean lift | PickCube success |
|---|---|---|---|
| random | 0.0% | 0.0000 m | 0.0% |
| scripted (upper bound) | 100.0% | 0.1835 m | 83.3% |
| pretrained + CEM | 0.0% | 0.0000 m | 0.0% |
| fine-tuned + CEM | 0.0% | 0.0000 m | 0.0% |

Both planners are indistinguishable from random. The task is solvable in the 30
steps allowed, since the scripted controller grasps 6 out of 6, so this is a
planner failure and not an impossible task.

`probe_grasp.py` rules out the obvious explanation. Linear probes on the frozen
encoder's cached latents, split by episode:

| target | held-out |
|---|---|
| gripper closedness | R2 = +0.730 |
| `is_grasped` | AUC = 0.952, balanced accuracy 0.801 |

Grasp state is clearly represented, so the encoder is not blind to it.

I think the cause is the objective rather than the representation. Being
decodable by a probe and being salient in an L1 distance are different things. A
linear probe can find and amplify one small direction in 1408 dimensions, while
the planning cost weights all dimensions equally, so "cube held or not"
contributes a tiny fraction of a distance dominated by gross arm and cube
position. On top of that, planning ran at `rollout=1`, and a greedy one-step
planner cannot discover that closing the gripper now pays off several steps
later. It just drives the arm toward the goal image's arm position and never
closes.

That makes grasping a different problem from reaching rather than a harder dose
of the same one. Reaching worked precisely because arm position is what
dominates the latent distance.

## 8. The model's prediction error swamps the planning signal

I tried three fixes and all of them failed. The diagnosis only became clear by
ruling them out in order.

Fix 1, a grasp-aware cost and a longer horizon (`grasp_cost.py`,
`plan_grasp.py`). Score rollouts with
`L1(z_final, z_goal) - lam * max_h P(grasped | z_h)` at `rollout=3`:

| lam | grasp | lift | success |
|---|---|---|---|
| 0.0 | 0.0% | 0.0000 m | 0.0% |
| 0.5 | 0.0% | 0.0000 m | 0.0% |
| 2.0 | 0.0% | 0.0000 m | 0.0% |

Can the predictor imagine a grasp at all? (`probe_imagination.py`) Partly. It
reproduces 41% of the real grasp signal, so "it cannot imagine contact" is
refuted. This also exposed a calibration bug worth knowing about: the probe is
fitted on encoder latents but applied to predictor latents, which sit in a
different region, so predicted scores run about 0.15 low.

Where does the gripper actually end up? This was the decisive measurement. The
planner stalls 0.148 m from the cube and a grasp needs less than 0.02 m, so it
never approaches the object at all. The single goal image shows the arm at the
goal holding the cube, and latent L1 to that image is minimised by flying
straight to the goal position. Picking the cube up is a detour that temporarily
increases distance, so greedy goal-matching provably will not do it.

Fix 2, subgoal images (`subgoal_plan.py`): four waypoints, above the cube, at
the cube, closed, carried. This halves the gap, closest approach going from
0.148 m to 0.106 m with a best seed of 0.058 m, which confirms the shortcut
diagnosis, but it still grasps 0 out of 6.

Fix 3, is the latent too coarse to servo on? (`latent_precision.py`) No, and this
turned out to be the answer. Latent L1 against controlled displacements:

| offset | latent L1 | % of 16 cm |
|---|---|---|
| 0.5 cm | 0.193 | 42.8% |
| 2 cm | 0.281 | 62.5% |
| 4 cm | 0.337 | 75.0% |
| 16 cm | 0.450 | 100% |

The latent is very sensitive, 0.5 cm already gives 43% of a 16 cm response, but
the response saturates. Any displacement immediately costs about 0.19, after
which telling 2 cm from 4 cm is worth only 0.056. Set that against the model's
own one-step prediction error of 0.194 (val L1, result 5): the prediction error
is about 3.5 times larger than the difference the planner has to resolve.

This explains the whole arc. Reaching works because 10 to 15 cm motions produce
latent differences around 0.45, far above the error floor. Grasping fails
because centimetre positioning produces differences around 0.05, far below it.
The limit is not the representation, the cost function or the search, it is the
accuracy of the world model relative to the precision the task needs.

So to grasp, prediction error has to drop below about 0.05, roughly a 4x
improvement. More ManiSkill data, training all 305M params instead of 78M, or a
shorter effective horizon are the levers. Cost re-weighting and better search
are not.

## 9. Full-parameter training improves action ranking but not prediction error

Result 8 pointed at prediction error as the thing gating grasping, with a target
of about 0.05, so I threw capacity and data at it: all 305M params instead of
78M, both datasets merged (11,500 frames instead of 7,500), bitsandbytes Adam8bit
plus gradient checkpointing to fit in 6 GB, lr 5e-5.

| run | trainable | frames | val L1 | rank_shell |
|---|---|---|---|---|
| reach fine-tune | 78M | 4,000 | 0.2004 | 0.1159 |
| grasp fine-tune | 78M | 7,500 | 0.1939 | 0.1354 |
| full, epoch 1 | 305M | 11,500 | 0.2034 | 0.1098 |
| full, epoch 2 | 305M | 11,500 | 0.1969 | 0.0959 |
| full, epoch 3 | 305M | 11,500 | 0.1939 | 0.0833 |

The two quantities move in opposite directions, and that distinction is the
point. Action ranking improves a lot, 0.1354 to 0.0833, the best of any run:
more capacity and more data genuinely help the model tell good actions from bad
ones one step ahead. Prediction error does not move at all. Val L1 lands on
0.1939, within 0.0000 of where the 4x smaller model on 65% of the data
plateaued, and the per-epoch decrements halve (-0.0065, then -0.0030), so the
asymptote is around 0.19, not 0.05.

That means the 0.19 error floor is not a capacity limit and not a dataset-size
limit. Quadrupling parameters and nearly tripling the data left it unchanged to
four decimal places. It points at the data distribution (random and
scripted-with-noise rollouts) or at the one-step latent objective itself, so
"train a bigger model for longer" is not the route to centimetre-precision
planning with this recipe.

## 10. Grasping works with the encoder as a sensor, not the predictor as a planner

`state_planner.py` replaces latent-distance scoring with a decoded position cost.
The world model still imagines each candidate's outcome, but the outcomes are
scored in metres via a decoder read off the frozen encoder's features. That took
the gripper from 14.8 cm to 4.2 cm, and still grasped 0 out of 6.

`servo_check.py` then removes the world model entirely: decode the
gripper-to-cube offset from the current frame, move that way, re-measure, repeat.

| method | grasp rate | closest approach | task success |
|---|---|---|---|
| world-model CEM (9 configurations) | 0% | 3.6 to 4.0 cm | 0% |
| decoded-position servo | 100% (6/6) | 2.4 cm | 16.7% |

Same decoder, same phases, same controller. The only difference is that the
servo measures and the planner imagines. Note the gap between grasping and task
success: it grasps reliably and then usually fails to lift and carry, so full
PickCube is 16.7%.

## 11. Why the planner cannot close the last 4 cm

I tried nine configurations: a grasp-aware cost, horizon 3, subgoal images, a
keypoint decoder (2.60 cm to 1.79 cm), finer step caps, an action-scale
correction, bias correction, executing the best action instead of the elite
mean, and a decoder retrained on near-cube poses (1.48 cm close-range error).
Every one stalled at 3.6 to 4.0 cm.

Stall distance against close-range decoder error:

| decoder close-range error | 2.60 cm | 1.79 cm | 1.48 cm |
|---|---|---|---|
| planner stall distance | 4.23 cm | 3.62 cm | 3.82 cm |

Improving the estimate does not help, which rules out the readout. The mechanism
is a signal-to-noise problem specific to ranking:

- CEM scores candidates against each other. Candidates differ by at most 5 cm
  commanded, and `action_gain.py` measures the predictor's response gain at 0.34,
  which correctly matches the controller's own 0.40, so the predictor is faithful
  rather than broken. That means candidates' imagined positions differ by about
  1.7 cm.
- Scoring requires decoding an imagined latent, which carries 1.5 to 2.3 cm of
  error (`decoder_on_predicted.py`).
- Signal is about equal to noise, so near the target CEM cannot rank candidates
  and parks at a fixed radius.

The servo escapes this because its signal is the entire remaining distance, 5 to
10 cm, read from a real latent with 1.48 cm of error, roughly 5:1 instead of
1:1. Improving the decoder shrinks the noise while the signal stays pinned at
1.7 cm, which is why the stall never moved.

Generalising: latent world models can plan coarse motion, but at contact
precision the difference between candidate actions falls below the noise of
decoding imagined states. Closed-loop measurement wins not because the model is
wrong, it is measurably faithful, but because its signal-to-noise for ranking is
worse.

Six hypotheses died on the way here: a do-nothing magnitude bias (the planner
moves at 0.047 m/step, near its cap), patch resolution being too coarse (0.5 cm
displacement already moves the latent 43% of a 16 cm one), step size, predictor
action gain, elite-mean averaging, and close-range decoder error (whose 3.69 cm
matched the stall exactly, and was a coincidence). Each was killed by a
measurement. Several of my diagnostic scripts print a pass/fail verdict line
whose threshold turned out to be wrong for the case at hand, so read the numbers
and not the verdict.

## 12. Probe large, execute small

Result 11 diagnosed the stall as signal-to-noise specific to ranking. The trap
was that one number set both the evaluation scale and the execution scale: small
steps give precision but indistinguishable candidates, large steps distinguish
them but overshoot. Nothing requires the two to be equal.

`hier_plan.py` separates them. Candidates are evaluated as 12 cm probe
displacements, where imagined outcomes separate by about 4 cm, well clear of the
noise, and only the winning direction is kept, executed as a 1.5 cm step, then
re-planned.

| method | grasp rate | closest approach | PickCube success |
|---|---|---|---|
| flat CEM (9 configurations) | 0% | 3.6 to 4.0 cm | 0% |
| decoded-position servo (no world model) | 100% | 2.4 cm | 16.7% |
| coarse-to-fine, predictor in the loop | 100% | 1.31 cm | 83.3% |

The change was not to the model, the decoder or the controller, but to how the
question was posed. This is the coarse-to-fine idea from the literature
(Coarse-to-Fine Q-attention, Subgoal Diffuser, Hierarchical World Models), which
I arrived at from my own measurements rather than from reading.

Two caveats. The four phases (hover, descend, close, lift) are scripted, and the
final carry-to-goal phase servos on the simulator's tcp pose rather than a
decoded one. That is true of every method compared here so the comparison is
fair, but the 83.3% is not end-to-end learned. The grasp itself, approach and
descend, is planned from decoded positions.

The carry phase was also where I lost a lot of time to my own bug. I stopped
commanding at 3 cm when PickCube's threshold is 2.5 cm, which dropped success
from 83.3% to 16.7%. The cube hangs about 2.4 cm below the tool point, so I was
servoing the gripper when I should have been servoing the cube, and every earlier
"success" had been passing by about a millimetre.

## 13. A wrist camera makes planning worse, not better

The obvious objection to all of this is that a fixed third-person camera is the
wrong sensor for grasping. ManiSkill ships `panda_wristcam`, where the cube fills
much of the frame instead of spanning two or three patches. I re-ran the whole
pipeline on it: 7,500 wrist frames (114 of 250 episodes with grasps, matching
the base run's 113), predictor fine-tuned from scratch, decoder retrained, same
planner.

| | base camera | wrist camera |
|---|---|---|
| decoder error (perception) | 1.88 cm | 1.11 cm |
| val L1 (dynamics) | 0.203 | 0.407 |
| action ranking `rank_shell` | 0.135 | 0.209 |
| grasp rate | 100% | 16.7% |
| closest approach | 1.31 cm | 3.12 cm |
| PickCube success | 83.3% | 0% |

The wrist camera gave better position estimates, 1.11 cm against 1.88 cm, but
the predictor did much worse on it, val L1 0.407 against 0.203, and grasping
dropped from 100% to 16.7%. A 41% more accurate decoder could not compensate for
a 2x worse predictor, because CEM ranks candidates by decoding imagined futures,
so the two error sources compound instead of trading off.

I think the cause is that with a wrist camera the camera moves with the gripper,
so the entire image shifts whenever the arm moves, and egocentric dynamics are
much harder to predict than watching an arm cross a static scene. Training showed
this cleanly: val L1 was pinned at 0.407 across all four epochs while
`rank_shell` fell steadily from 0.432 to 0.209, the two quantities diverging
further than anywhere else in this project.

It also explains a design choice in the paper. V-JEPA 2-AC was trained on
third-person DROID footage, not wrist views, and a fixed viewpoint keeps the
prediction problem tractable.

## 14. Ablation: the predictor is not what makes grasping work

Results 12 and 13 claimed grasping as a world-model planning result. That claim
does not survive its own control, which I should have run first.

`hier_plan.py --ablate-predictor` replaces the world model with a trivial
analytic stand-in: commanding action `a` is assumed to change the
gripper-to-cube offset by exactly `-a`, with no prediction at all. Everything
else is identical, the same decoder, cost, CEM search, probe and execute scales,
phases, budgets and seeds.

| | grasp rate | closest approach | PickCube success |
|---|---|---|---|
| with the predictor | 100% | 1.31 cm | 83.3% |
| without the predictor | 100% | 0.97 cm | 100% (6/6) |

Removing the world model improves every metric. Seed 4, which failed with the
predictor, succeeds without it, and its approach error drops from 2.15 cm to
1.24 cm.

The mechanism is something I had already measured and failed to follow through
on. Scoring a candidate two ways:

    with predictor:   decode( predict(z, a) )   -> 2.33 cm error (imagined latent)
    analytic:         decode(z) - a             -> 1.11 cm error (real latent) plus exact arithmetic

`decoder_on_predicted.py` showed that decoding an imagined latent is strictly
noisier than decoding a real one. For pure translation in free space, "commanding
`a` moves you by `a`" is very nearly exact, so the predictor was injecting noise
into a problem that needed no prediction.

So grasping here is not world-model planning. It is V-JEPA's frozen encoder used
as a learned position sensor, plus CEM over a trivial analytic model, plus a
scripted phase controller. The predictor can be deleted with no loss.

Reaching is still a genuine world-model result (results 5 and 6). It scores
candidates by latent goal-matching with no decoder in the loop, so the predictor
is doing the work there: action ranking 0.4766 to 0.1181 (p<0.0001) and 12 of 12
seeds closing about two thirds of the distance to the goal.

I identified this ablation as necessary early and ran it late, after building
three more experiments on the unverified claim. The diagnostics in results 8 to
11 were correct and actually predicted this outcome, I just did not carry them to
their conclusion.

## 15. More seeds, and no transfer to new objects

Every grasping number above came from 6 seeds on one red cube, which is thin.

More seeds. Rerunning the working configuration (coarse-to-fine, predictor
ablated) on 30 seeds:

| | 6 seeds | 30 seeds |
|---|---|---|
| grasp rate | 100% | 96.7% |
| PickCube success | 100% | 100% |
| closest approach | 0.0097 m | 0.0112 m |

The headline holds at five times the evidence. The 96.7% grasp against 100%
success is a sampling artifact: on one seed `is_grasping` read false at the
moments it was polled while the cube still reached the goal.

New objects. `eval_objects.py` runs the identical pipeline on `PickSingleYCB-v1`,
16 episodes with 16 different YCB objects and nothing retrained. I controlled two
confounds: the task defaults to `panda_wristcam` (forced back to plain `panda`)
and hardcodes a far camera (patched to the same close pose the decoder was
trained on), so the only thing that varies is the object.

| | red cube | 16 YCB objects |
|---|---|---|
| grasp rate | 96.7% | 0.0% |
| success | 100% | 0.0% |
| closest approach | 0.011 m | 0.278 m |

Zero transfer, across foam brick, clamp, baseball, knife, banana, spatula, lemon,
lego, marbles and cups.

The magnitude localises the failure. 27 cm is not "mislocated the banana
slightly", the planner is driving confidently to a meaningless position, which
means the decoder emitted garbage. That is expected by construction, since it was
trained to regress the coordinates of a red cube. The best of the 16 was
`012_strawberry` at 10 cm closest approach, roughly 3x nearer than the rest, and
the only small red object in the set, so the decoder is keying on appearance
rather than on object-ness.

This shows the pipeline does not generalise, and that the bottleneck is the
position decoder in isolation. The planner and world model never received a
usable input, so they were not really tested. The fix follows directly: a
detector trained across many objects generalises by construction where a
cube-specific regressor cannot.

## 16. Language instructions, without retraining anything

The planner takes a target offset, not an image, so a language instruction only
has to become an offset. I tried two ways.

CLIP retrieval (`language_goal.py`) picks a goal frame from a library by
image-text similarity. It failed: the retrieved frames scored barely above chance
because all the candidate frames look nearly identical to CLIP, differing only in
where a small arm is.

A rule-based parser (`language_plan.py`) maps an instruction to one of seven
position primitives and hands the offset to the existing planner. Nothing is
retrained.

| instruction | parsed | decoder | start error | end error |
|---|---|---|---|---|
| hover above the cube | hover | near | 0.080 m | 0.005 m |
| go down and touch the cube | touch | near | 0.159 m | 0.023 m |
| move 10 cm to the left of the cube | left | near | 0.180 m | 0.043 m |
| lift the gripper high above the cube | high | far | 0.027 m | 0.016 m |

Mean error reduction 74%, 4 of 4.

The interesting part was the failure I had to fix. "High" asks for an 18 cm
standoff, and `keypoint_near.pt`, the decoder retrained on poses around the cube,
is accurate to 1.5 cm inside 6 cm but 3.0 cm at 10 to 18 cm and 10.4 cm beyond
(`decoder_vs_range.py`). So the planner was steering by a readout wrong by more
than the motion it was trying to make, and it moved away from the target rather
than toward it (0.027 m to 0.127 m). Loading the original `keypoint.pt`, trained
on random arm poses and better far away, and switching on the requested standoff
fixes it.

This is a two-point switch tuned on a range curve I measured, not a learned
selector. It is honest as "the decoder has a competent range and I respect it",
not as "the system handles arbitrary distances".
