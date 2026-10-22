# PRISM — Results

Committed runs, three seeds each, on an NVIDIA L4. Every number below is
reproducible from an artifact in [`results/`](results/). Earlier writeups:
[`archive/v0.1/RESULTS.md`](archive/v0.1/RESULTS.md) (the attribution pass),
[`archive/v0.0/RESULTS.md`](archive/v0.0/RESULTS.md) (before attribution).

## The runs

| # | run | artifact |
|---|---|---|
| A | Recipe as tuned (LR 5e-4) vs. baseline (LR 1e-3), 5,000 steps | [`recipe_20260718T002717Z.json`](results/recipe_20260718T002717Z.json) |
| B | **Attribution control**: recipe at the baseline's own LR, 5,000 steps | [`recipe_20260720T230405Z.json`](results/recipe_20260720T230405Z.json) |
| C | **Un-censored speed**: tuned recipe, dense eval (every 10), 1,500 steps | [`recipe_20260721T142104Z.json`](results/recipe_20260721T142104Z.json) |
| D | **Overlap probe**: 12 overlaps × 3 seeds, random blocks, matched LR, 100 steps | [`recipe_20260721T050203Z.json`](results/recipe_20260721T050203Z.json) |
| E | **Cross-domain**: far corpus (Sherlock), student scored on its own data | [`recipe_20260721T161208Z.json`](results/recipe_20260721T161208Z.json) |
| F | **Teacher-strength sweep**: teachers 100→2,000 steps, 300-step students | [`recipe_20260721T143246Z.json`](results/recipe_20260721T143246Z.json) |
| G | **Lever eval — grassmann alignment alone** (single variable vs. Run E's plain control) | [`recipe_20260721T162552Z.json`](results/recipe_20260721T162552Z.json) |
| H | **Lever eval — top-k=128 alone** (single variable vs. Run E's plain control) | [`recipe_20260721T164007Z.json`](results/recipe_20260721T164007Z.json) |
| I | **Teacher saturation**: teachers 2,000 / 4,000 / 8,000 — finds the plateau | [`recipe_20260721T172238Z.json`](results/recipe_20260721T172238Z.json) |
| — | Sliding-window overlap sweep — **record only, interpretation retired** (difficulty confound; superseded by D) | [`recipe_20260721T022342Z.json`](results/recipe_20260721T022342Z.json) |
| — | Far-corpus sweep scored on Shakespeare val — superseded by E (see scope note in [`results/README.md`](results/README.md)) | [`recipe_20260721T153218Z.json`](results/recipe_20260721T153218Z.json) |
| J | **Finetune-retention frontier** (§7): 11 arms × 3 seeds — raw anchor vs. low-LR vs. spectral vs. shuffled | [`finetune_20260721T215319Z.json`](results/finetune_20260721T215319Z.json) |
| K | Finetune-retention core pair (§7): plain vs. raw-0.01 anchor, 3 seeds — the 5.73× headline | [`finetune_20260721T201955Z.json`](results/finetune_20260721T201955Z.json) |
| — | Finetune go/no-go probe (1 seed) — confirms plain finetuning forgets | [`finetune_20260721T200415Z.json`](results/finetune_20260721T200415Z.json) |
| — | Finetune attribution probe (1 seed) — superseded by J | [`finetune_20260721T212408Z.json`](results/finetune_20260721T212408Z.json) |
| O | **Teacher-free init (§10)**: Sherlock-*only* teacher → Shakespeare student, matched LR, dense eval | [`recipe_20260725T164604Z.json`](results/recipe_20260725T164604Z.json) |
| P | **Teacher-free control (§10)**: identical rig, same-corpus teacher | [`recipe_20260725T164640Z.json`](results/recipe_20260725T164640Z.json) |
| Q | **Modern web — full recipe (§11)**: byte-level FineWeb-Edu, native teacher | [`recipe_20260727T145008Z.json`](results/recipe_20260727T145008Z.json) |
| R | **Modern web — dirs_only (§11)**: init-only transfer, no Mod Wheel — **the winner** | [`dirs_only_20260727T154359Z.json`](results/dirs_only_20260727T154359Z.json) |
| S | **Modern web — spectral_only ablation (§11)**: spectrum + wheel, no directions | [`spectral_only_20260727T145028Z.json`](results/spectral_only_20260727T145028Z.json) |
| T | **Modern web — GPT-2 fingerprint (§11)**: spectrum-only from OpenAI's public weights, no teacher | [`spectral_only_20260727T145039Z.json`](results/spectral_only_20260727T145039Z.json) |
| U | **Fact injection (§12)**: closed-book recall of novel facts — anchor vs low-LR vs plain, modernweb base | [`finetune_20260727T203023Z.json`](results/finetune_20260727T203023Z.json) |
| V | **Fact injection round 2 (§12)**: the weak-band anchor + selective (free-FFN / free-attn) anchors | [`finetune_20260727T230910Z.json`](results/finetune_20260727T230910Z.json) |
| W | **Catalog task (§13)**: scored structured-decision workflow — the dial on *judgment*, not facts | [`finetune_20260728T044511Z.json`](results/finetune_20260728T044511Z.json) |
| X | **enwik8 (§14)**: dirs_only on the canonical byte-LM bench — 2.90×, −0.152 bpb | [`dirs_only_20260728T054705Z.json`](results/dirs_only_20260728T054705Z.json) |
| Y | **Cold directions (§15)**: never-trained analytic DCT bases — refuted, 3/3 seeds | [`dirs_only_20260728T060256Z.json`](results/dirs_only_20260728T060256Z.json) |
| Z1 | **Energy P1 (§16)**: mined fingerprint crosses corpora (modernweb → enwik8) at 1.97× | [`dirs_only_20260728T065345Z.json`](results/dirs_only_20260728T065345Z.json) |
| Z2 | **Energy P2 (§16)**: baseline-1500 quality from a complete 600-step run — 42% of the GPU time | [`dirs_only_20260728T085214Z.json`](results/dirs_only_20260728T085214Z.json) |
| Z3 | **Energy P3 (§16)**: −0.072 bpb beyond the baseline's ceiling at 56% of its GPU time | [`dirs_only_20260728T102700Z.json`](results/dirs_only_20260728T102700Z.json) |
| AA | **Act 1 — the prior's floor on enwik8 (§17)**: NOT a PRISM result; the reach-at-init gate | [`ngram_floor_20260728T202028Z.json`](results/ngram_floor_20260728T202028Z.json) |
| AB | **W1 tuning, catalog (§18)**: 21 arms, seed 1336 — competitor settings by the pre-registered rule | [`finetune_20261022T024927Z.json`](results/finetune_20261022T024927Z.json) |
| AC | **W1 tuning, facts (§18)**: 21 arms, seed 1336 | [`finetune_20261022T024917Z.json`](results/finetune_20261022T024917Z.json) |
| AD | **W1 catalog head-to-head (§18)**: 23 arms × 3 seeds — SST / RP-SFT / EWC / LoRA / self-distillation vs the dial | [`finetune_20261022T043049Z.json`](results/finetune_20261022T043049Z.json) |
| — | W2a tuning sweeps (§19), seed 1336: baseline / dirs_only grids, Muon LR | [`sweep_20261022T*.json`](results/) |
| AF | **Cross-size case 1 (§20)**: width 768→384 — early head start, no endpoint gain | [`dirs_only_20261022T035754Z.json`](results/dirs_only_20261022T035754Z.json) |
| AG | **Cross-size case 2 (§20)**: heads 12→6 — 3.7×, −0.11 nats | [`dirs_only_20261022T040349Z.json`](results/dirs_only_20261022T040349Z.json) |
| AH | **Cross-size case 3 (§20)**: head-dim 64→32 — 1.28× in 3/3 | [`dirs_only_20261022T040448Z.json`](results/dirs_only_20261022T040448Z.json) |
| AI | **Cross-size case 4 (§20)**: GPT-2 124M → byte student — null | [`dirs_only_20261022T043832Z.json`](results/dirs_only_20261022T043832Z.json) |
| AJ | **FRONT from GPT-2 (§20)**: our re-implementation — far below scratch | [`init_state_20261022T041102Z.json`](results/init_state_20261022T041102Z.json) |
| AK | **Weight Selection, case 1 (§20)**: 768→384 by index selection — 4.0×, beats the projection | [`init_state_20261022T170647Z.json`](results/init_state_20261022T170647Z.json) |
| AL | **W2a controls (§19)**: dirs_only / dirs_pure / recipe / spectral_only / mimetic / statmatch vs the tuned baseline; Muon arms | [`*_20261022T15*`, `*_20261022T16*`, `*_20261022T17*`](results/) |
| — | **W5 run 1 (§21) — VOID** (gather off-by-one; kept as a record) | [`prior_20261022T035740Z.json`](results/prior_20261022T035740Z.json) |

## 1. Speed: ~12×, resolved (Run C)

The tuned recipe's speed was left-censored in Run A (it crossed the baseline's best
before the first eval at step 100, so "≥13–14×" was a floor with coarse
resolution). Run C was built to resolve it: eval every 10 steps, 1,500-step
horizon.

| seed | baseline best @step | recipe reaches it @step | Prism Score |
|---|---|---|---|
| 1337 | 1.7663 @1,020 | @100 | 10.2× |
| 1338 | 1.7784 @1,190 | @100 | 11.9× |
| 1339 | 1.7674 @1,300 | @110 | 11.8× |

**Median 11.8× (10.2–11.9×), `left_censored: false` on all seeds.** The earlier
"≥13–14×" language is retired — the two runs don't conflict (different eval
resolution and horizon place the baseline's best differently), but the resolved
number is the number: **~12× as tuned, 7× at matched LR (Run B, below)**.

## 2. Attribution: it's the method, not the schedule (Runs A + B)

| | Baseline (LR 1e-3) | Recipe (LR 5e-4) | Recipe (LR 1e-3, matched) |
|---|---|---|---|
| Best val loss (median of 3) | 1.782 | 1.656 | **1.671** |
| Val loss @ step 5,000 | ~2.31 (overfit) | ~1.66 (stable) | ~1.67 (stable) |
| Overfits within 5,000 steps | yes — 3/3 | no — 0/3 | no — 0/3 |
| Prism Score | — | 11.8× (from Run C) | **7.0× (6.5–7.0×)** |

Run B holds the learning rate and warmup identical to the baseline
(`schedule_matched: true`) and toggles only the spectral flags. The effect
survives on every seed: lower best loss, no overfitting, 7× to the baseline's
best. The "maybe it was just the lower learning rate" explanation is closed.

## 3. Structure, not content (Run D)

The teacher/student overlap dial, difficulty-controlled: the corpus is cut into
100 random blocks; each arm gets a random half (both spanning the whole corpus, so
neither arm gets an "easier" slice); only the *shared fraction* varies. 12 overlap
points from 1.0 (identical data) to 0.0 (fully disjoint), 3 seeds each, 100-step
students, matched LR.

**The early advantage is flat: Δloss 0.565–0.587 (~23% lower loss at step 100) at
overlap 1.0, at 0.0, and everywhere between.** Baseline flat at ~2.48 across the
sweep (the difficulty control working). Nothing the student gains depends on
sharing text with the teacher.

(An earlier sliding-window version of this sweep showed the advantage *declining*
with overlap — that was a confound: window position correlated with text
difficulty, the baseline drifted 1.88→1.55 across the sweep. The artifact stays
committed as a record; its interpretation is retired.)

## 4. Cross-domain: the head start survives a change of corpus (Run E)

The student's non-shared blocks come from **Sherlock Holmes** (`data/far.txt`,
Project Gutenberg #1661), char-encoded in Shakespeare's vocabulary, and — the part
that makes the test decisive — the student is **scored on a validation set drawn
from its own training mixture** (`far_val: true`): pure held-out Shakespeare at
overlap 1.0, pure held-out Sherlock at overlap 0.0. Matched LR, 3 seeds, 100-step
students. The overlap-1.0 row reproduces the base protocol exactly (the sanity
gate).

| overlap | token-JS | val set | baseline | recipe | Δloss |
|---|---|---|---|---|---|
| 1.00 | 0.0000 | 100% Shakespeare | 2.469 | 1.878 | 0.591 |
| 0.75 | 0.0008 | 76% Shakespeare | 2.487 | 1.898 | 0.589 |
| 0.50 | 0.0044 | 50/50 | 2.475 | 1.882 | 0.593 |
| 0.25 | 0.0121 | 32% Shakespeare | 2.464 | 1.851 | 0.613 |
| 0.00 | 0.0266 | **100% Sherlock** | 2.414 | 1.786 | **0.627** |

**A Shakespeare teacher's geometry accelerates learning of Sherlock at least as
much as it accelerates Shakespeare.** The gap does not shrink with distance — it
grows slightly. Prism Scores are ≥9–10× at every overlap (left-censored at this
probe's 10-step resolution — consistent with Run C's resolved ~12×). Combined with
Run D, this closes the content-leakage explanation twice over: the transfer is
structural, and the structure is portable across corpora.

An earlier version of this run scored all students on the *Shakespeare* val set —
which conflates accelerated learning of the new domain with retention of the
teacher's. That artifact ([`…T153218Z`](results/recipe_20260721T153218Z.json)) is
kept with a scope note; Run E is the corrected protocol.

## 5. The lever: teacher strength (Run F)

Same-data probe, 300-step students, one teacher size per arm:

| teacher steps | baseline best | recipe best | Δloss | score |
|---|---|---|---|---|
| 100 | 2.180 | 2.249 | **−0.069** | never reached |
| 250 | 2.180 | 2.092 | 0.088 | 2.3× |
| 500 | 2.180 | 1.932 | 0.248 | 7.5× |
| 1,000 | 2.180 | 1.804 | 0.377 | 7.5× |
| 2,000 | 2.180 | 1.729 | 0.451 | 7.5× |

And the saturation run (Run I, teachers 2,000 / 4,000 / 8,000, same rig):

| teacher steps | Δloss |
|---|---|
| 2,000 | +0.458 (anchor — reproduces Run F's +0.451) |
| 4,000 | +0.465 |
| 8,000 | +0.456 |

**Monotonic to ≈2,000 teacher steps, then a plateau at Δloss ≈ +0.46** — the
lever saturates right where the teacher itself converges (its best is ~step
1,350). The advantage tracks teacher-geometry convergence: once the geometry has
converged, more teacher training adds nothing. And the sharp edge: a
barely-trained teacher is *actively worse than random init* — its geometry is
noise being imprinted with authority. If you use `prism_accelerate.py`, use a
trained teacher.

This is also indirect evidence *for* the spectral mechanism: the effect tracks the
quality of the teacher's geometry, which a generic regularizer has no access to.

## 6. Lever evals: the plain blend wins at this distance (Runs G, H)

Two geometric-alignment refinements (contributed in PR #1) evaluated one variable
at a time against Run E's plain control — same rig, same seeds, same overlaps,
only the lever flag added:

- **Grassmann geodesic pairing** (`--align_mode=grassmann`): Δloss **−0.012 to
  −0.023** at every overlap — the recipe never reaches the baseline's best. The
  geodesic re-pairing eliminates the head start the plain 75% blend delivers
  (+0.59 to +0.63 on the identical rig).
- **Top-k=128 subspace transfer** (`--align_topk=128`): the head start survives
  (Δloss +0.54 to +0.57, scores ~5×) but is uniformly **+0.04 to +0.06 worse**
  than transferring all ~384 directions. The discarded low-energy tail carries
  useful geometry at this scale.

Verdict at Sherlock-distance: **the plain full-direction blend is the strongest
configuration measured** — there is no headroom for geometric refinement here,
because the plain recipe already transfers at full strength (Run E). The levers'
real test is a corpus far enough that the plain recipe degrades (see next
experiments).

## 7. Finetuning without forgetting (Runs J, K)

Runs A–H are all *from-scratch*. A separate study points the Mod Wheel the other
way: kept on during a **finetune** and self-anchored to a trained model's own
pre-finetune weights, does it prevent catastrophic forgetting? Setup: a plain
Shakespeare base (2,000 steps, one per seed) finetuned 1,000 steps on Sherlock; each
arm forks the same base and is scored every step on **both** Sherlock (adaptation)
and Shakespeare (retention). The control and every anchor arm share the identical
schedule — only the Mod Wheel differs.

**The technique works: up to ~10× less forgetting** (Run J, 3-seed frontier,
[`finetune_20260721T215319Z.json`](results/finetune_20260721T215319Z.json); base
Shakespeare val 1.488, from-scratch Sherlock ceiling 1.493):

| arm | forgetting (Shakespeare val climb) | Sherlock best | vs. plain |
|---|---|---|---|
| plain finetune | +0.428 | 1.252 | 1.0× |
| raw anchor 0.02 | **+0.043** | 1.368 | **9.9×** |
| raw anchor 0.01 | +0.067 | 1.337 | 6.6× |
| raw anchor 0.005 | +0.090 | 1.307 | 4.8× |
| low-LR 5e-5 | +0.227 | 1.297 | 1.9× |
| low-LR 1e-4 | +0.282 | 1.277 | 1.5× |
| spectral (spectrum only) | +0.399 | 1.254 | 1.07× |
| shuffled (wrong spectrum) | +1.085 | 1.413 | 0.39× |

Every anchor arm still beats the from-scratch Sherlock ceiling (1.493), so retention
is never "it didn't learn Sherlock." Run K is the same-schedule core pair (plain vs.
the raw-0.01 anchor) at the headline **5.73×**
([`finetune_20260721T201955Z.json`](results/finetune_20260721T201955Z.json)).

**The attribution is a negative on the obvious guess.** Anchoring only the
*spectrum* (freeing the directions) does essentially nothing for retention (1.07×, ≈
plain); a *wrong*-spectrum placebo actively harms (0.39×); and the raw anchor
**Pareto-dominates** the low-LR frontier — at comparable adaptation it retains ~2×
more (raw 0.090 @1.307 vs. low-LR 0.227 @1.297). The protection is a raw
directional/whole-weight anchor (L2-to-init / EWC-lite), **not** the spectral
geometry and **not** just a smaller learning rate.

This separates PRISM's two regimes cleanly: **from scratch the *spectrum* carries
transferable, data-independent structure (§3, §4); in finetuning the *directions*
carry retained content and must be pinned.** Method, full frontier, and bounds:
[`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md).

## 8. The arc: PRISM pretraining + finetuning compound (Runs L, M)

Does combining the two directions buy anything? The test: a PRISM-pretrained base vs.
a plain base as the thing you finetune-anchor, at **matched** Shakespeare quality
(`train.py --stop_val_target`). 3 seeds, all bases matched at ~1.79, finetuned on
Sherlock with the *identical* raw anchor. Medians:

| base (matched ~1.79) | forgets Shakespeare | learns Sherlock | run |
|---|---|---|---|
| plain (LR 1e-3) | +0.050 | 1.572 | [`…T172023Z`](results/arc_20260724T172023Z.json) (L) |
| plain_fastlr (LR 5e-4, no spectral) | +0.084 | 1.571 | [`…T210817Z`](results/arc_20260724T210817Z.json) (M) |
| **prism** (spectral) | **−0.001** | **1.438** | (L, M) |

**A PRISM base finetunes with ≈ zero forgetting** (vs +0.05 for plain) **and adapts
~8% better**, at matched quality. And it's the **spectral geometry**: the
`plain_fastlr` control (plain at PRISM's learning rate, no spectral) forgets *more*
than plain and adapts identically — so the schedule doesn't do it, only the spectral
machinery does. Pretraining and finetuning with PRISM reinforce each other. Full study
and bounds: [`docs/UNIFIED-ARC.md`](docs/UNIFIED-ARC.md).

## 9. Prior-Fused PRISM: a T9-style statistical prior compounds (Run N)

Structure comes in more than one flavour. Fuse a fixed shared **n-gram prior** into the
logits (product of experts, `final = model_logits + λ·log p_ngram(next | last C chars)`),
so the model learns only the residual, and stack it with PRISM's spectral init. A startling
input: a context-3 char n-gram already predicts Shakespeare val at **2.5722 bits/char** ≈
the neural baseline's 2.565 best. 1 seed, steps-to-baseline-best
([`…T011443Z`](results/prior_20260725T011443Z.json)):

| arm | init (bits/ch) | best (bits/ch) | speedup |
|---|---|---|---|
| baseline | 6.18 | 2.565 | 1.0× |
| prism | 4.47 | 2.245 | 15× |
| prior (n-gram only) | 2.70 | 2.504 | 3.8× |
| **prism_prior** | 2.68 | **2.235** | **30×** |

**The hybrid doubles PRISM (15× → 30×) and reaches the best loss of all four** — below
baseline, PRISM-alone, and the n-gram floor (2.50). The T9 statistical prior and PRISM's
geometry compound: the prior pre-loads local structure for free, PRISM pre-loads the
representational geometry, and PRISM's geometry is what breaks the residual below the
n-gram floor. The "30×" is the compounding of two free priors (the fused arms start near
baseline), not PRISM training 30× faster alone — the defensible claim is that the hybrid
beats PRISM alone *and* reaches a better loss. The literal 1000× (reach baseline *at
init*) is a knife-edge here — context-3 tops out *at* baseline, and a logit-gate meant to
enable it backfired ([`…T013656Z`](results/prior_20260725T013656Z.json)); a clean version
needs a context-4+ prior. Full study: [`docs/PRIOR-FUSED-PRISM.md`](docs/PRIOR-FUSED-PRISM.md).

## 10. Teacher-free init: half the speedup survives a teacher that never saw the corpus (Runs O, P)

The explicit test of the "the fingerprint is a modality constant" hypothesis
(experiment #1 in [NEXT-EXPERIMENTS](docs/NEXT-EXPERIMENTS.md)), run as two
identical rigs where the ONLY difference is the teacher's training data
(`--cross_teacher`): 3 seeds, matched schedules (`schedule_matched: true`),
2,000-step teachers (the saturation point from Runs F/I), 1,500-step students,
eval every 10 (all scores resolved, none left-censored).

- **Control (Run P):** teacher trained on the student's own Shakespeare split.
- **Cross (Run O):** teacher trained *exclusively* on Sherlock (`data/far.txt`,
  499K tokens, token-JS 0.028 from the student's data) — it never saw a single
  character of Shakespeare. Student trained and scored on the standard
  Shakespeare benchmark either way.

| arm (medians of 3) | baseline best | recipe best | Δbest | score |
|---|---|---|---|---|
| control — same-corpus teacher | 1.7694 | 1.6799 | 0.0895 | **9.9×** (9.6–10.4×) |
| **cross — Sherlock-only teacher** | 1.7603 | **1.6880** | 0.0723 | **4.6×** (3.9–7.6×) |

**Two readings, both honest.** The strong form of the hypothesis — cross ≈
control — is **refuted at this distance**: the foreign fingerprint keeps ~46% of
the matched teacher's speedup (per-seed 0.38–0.79), with more seed variance. But
the weak form lands cleanly: a ~128-byte fingerprint from a teacher that **never
saw the target corpus** still trains it **4.6× faster**, reaches a best loss the
baseline never touches (~81% of the matched teacher's quality gain), and does not
overfit where the baseline does. Teacher-free PRISM init works as a drop-in — it
just leaves the other half of the speedup on the table.

Read together with Run E (Shakespeare teacher → Sherlock student, where the
*early-window* advantage did not shrink at all), this bounds where the geometry
stops being free: the step-100 head start is fully portable; the *speed to
baseline-best quality* over a 1,500-step horizon is what pays a cross-corpus
price. Bounds: one corpus pair, one direction, one distance (token-JS 0.028), and
the cross teacher trained on a ~40% smaller corpus (499K vs 803K tokens) — a
teacher-corpus-size control would isolate that confound.

## 11. Modern web text: init-only transfer wins outright; the Mod Wheel is an overfitting brake (Runs Q–T)

The move off Shakespeare: a committed ~5MB slice of **FineWeb-Edu** (2024
CommonCrawl web text, [`data/modernweb/`](data/modernweb/), provenance + sha256
in its README), encoded **byte-level** (vocab 256 — nothing dropped, and the
vocab now covers code/other languages for future far-modality work). 4.5M-token
train pool — 4.5× Shakespeare's — so the baseline **no longer overfits** inside
the horizon. Four arms on identical matched-schedule rigs (3 seeds, 2,000-step
native teachers where used, 1,500-step students, eval every 10, all scores
resolved), decomposing the recipe one component at a time:

| arm (medians of 3) | components | best loss | vs. baseline best (1.4892) | reaches it |
|---|---|---|---|---|
| baseline | — | 1.4892 | — | — |
| **dirs_only (Run R)** | imprint + EigenTransfer at init, *nothing during training* | **1.3794** | **−0.110 (better), 3/3 seeds** | **3.3×** faster (2.9–3.9×) |
| recipe (Run Q) | + Mod Wheel | 1.4907 | +0.001 (tie), 1/3 seeds | 1.1× (1 seed) |
| spectral_only (Run S) | imprint + wheel, **no directions** | 1.7623 | +0.273 (worse) | never, 0/3 |
| GPT-2 fingerprint (Run T) | as S, spectra from **GPT-2's public weights**, no teacher | 1.7471 | +0.258 (worse) | never, 0/3 |

**The headline (Run R): hand a fresh model a trained teacher's geometry at init
— directions + spectrum, then leave it alone — and on 2024 web text it reaches
the baseline's best quality 3.3× faster and converges 0.11 nats below anything
the baseline ever reaches**, on every seed, without overfitting. The early
window is bigger still: the quality dirs_only has at step 100, the baseline
needs 600–650 steps to reach (~6×); its init loss is already 4.4 vs the
baseline's 5.6.

**The decomposition, and an honest revision of the story so far:**

- **The directions carry the from-scratch effect.** Remove them (Run S) and the
  arm is *below baseline at every step measured* — the spectrum alone is not
  the active ingredient, mirroring the finetune-retention attribution (§7)
  where the raw directions were the anchor and the spectrum did nothing.
- **The Mod Wheel is an overfitting brake, not a universal accelerant.** The
  full recipe (Run Q) = dirs_only + wheel, and the wheel costs the entire
  endpoint lead (1.379 → 1.491): its continuous pull toward the spectral
  target throttles late learning. On 1M-token Shakespeare, where every baseline
  overfit (§2), that brake *was* the endpoint win. On 4.5M tokens there is no
  overfitting to prevent, so the brake only binds. Small data → wheel on;
  adequate data → init-only.
- **The one clean positive for the spectrum: it is genuinely universal.**
  GPT-2's fingerprint — 40 numbers off OpenAI's public weights, crossing model
  size (124M→10.65M), tokenizer (BPE→bytes), and corpus — performs at **parity
  with a purpose-trained native teacher's spectrum** in the identical mode
  (1.747 vs 1.762, GPT-2's marginally *better*). The spectrum ports across
  everything; it just isn't the lever from scratch. The teacher-free/universal
  init that *should* work is directional — which needs the cross-size
  projection (future work).

Bounds: one modern corpus, one 1,500-step horizon; the wheel's decay schedule
(0.01 / 0.9999) was tuned on Shakespeare and never retuned here — at step 1,500
it still pulls at ~86% strength, so a faster decay might rescue the recipe arm
(untested); and the mirror cell — dirs_only on *Shakespeare*, where the brake
should still win — hasn't run, so "wheel helps on small data" is inferred from
Runs A–C, not yet isolated.

## 12. Fact injection: the directional anchor blocks it — and loss metrics can't see that (Run U)

The retention anchor (§7) meets the task it wasn't built for: injecting
**novel facts** into the modern-web base. 120 facts about invented people
(60 birth-years, 60 cities; [`data/facts/`](data/facts/)) trained as 4
paraphrase templates each; a 5th template is held out for both the adaptation
val and closed-book **exact-match recall** — `seen` (a trained phrasing: did
the facts go in) vs `unseen` (the held-out phrasing: do they generalize). The
base scores **0%** on both (the facts are novel by construction). 3 seeds, one
base per seed, every arm forks it with one flag changed, 1,000 finetune steps.

| arm (medians of 3) | forgets modernweb | facts-val loss | seen recall | unseen recall |
|---|---|---|---|---|
| plain | +3.31 | 1.59 | **100%** | 26% |
| low-LR 1e-4 | +2.19 | 1.36 | 98% | **59%** |
| raw anchor 0.01 | +0.49 | **1.14** | **3%** | 0% |
| raw anchor 0.02 | **+0.31** | 1.17 | 2% | 0% |

**Three findings, one reversal:**

- **The anchor blocks fact injection.** At the exact strengths that won the
  domain-adaptation study (§7, where the anchor Pareto-dominated low-LR), the
  facts simply never go in: 2–3% seen recall vs 98–100% unanchored. The §7
  conclusion is **task-dependent** — adaptation to a new domain reuses the
  base's structure and survives the anchor; storing *new bindings* requires
  moving the very directions the anchor pins. This is also the sharpest
  confirmation of "directions carry content" in the whole project: pin them
  and new content *cannot be written*, by construction.
- **LM loss is not an injection metric.** The anchored arms post the *best*
  facts-val loss (1.14 vs plain's 1.59 — they model the unseen templates'
  English scaffolding beautifully) while recalling *nothing*. Only the
  exact-match probe catches this. Any fact-injection study scored on loss
  alone would have ranked these arms exactly backwards.
- **Low-LR generalizes better than full-LR at equal injection.** 59% vs 26%
  median unseen recall (seed-noisy: 28–77% vs 19–41%) with less forgetting
  (+2.19 vs +3.31). For fact injection, the simple low-LR baseline survives;
  the anchor, as-is, is disqualified.

**Round 2 (Run V): the anchor is a dial, and the weak band is the recipe.**
Same rig (plain and low-LR reproduce round 1), three new cells — a weak full
anchor (s=0.0025) and two *selective* anchors at s=0.01
(`--prism_anchor_exclude`): free-FFN (attention+embeddings pinned) and
free-attention (FFNs+embeddings pinned).

| arm (medians of 3) | forgets modernweb | seen recall | unseen recall |
|---|---|---|---|
| plain | +3.31 | 100% | 26% |
| low-LR 1e-4 | +2.19 | 98% | 59% |
| **weak anchor, s=0.0025** | **+1.08** | **96%** | 25% |
| free-FFN anchor, s=0.01 | +2.55 | 100% | 21% |
| free-attn anchor, s=0.01 | +1.55 | 92% | 37% |

- **The weak-band anchor injects at parity with plain (96% seen) while
  forgetting 3× less than plain and 2× less than low-LR.** The blocking
  threshold sits between 0.0025 (injects) and 0.01 (blocks): the anchor is a
  continuous injection↔retention dial and s≈0.002–0.003 is the useful band
  for fact injection at this rig. One line, any optimizer, W₀ already frozen
  in adapter-style setups.
- **Fact storage is not FFN-exclusive at this scale.** Both selective cells
  inject (100% through FFNs alone, 92% through attention alone) — facts go
  wherever there is free capacity. And freeing the FFNs costs *more*
  retention (+2.55) than freeing attention (+1.55): the old domain leans
  harder on the FFNs than the FFN-centric editing literature would predict
  here. Selective anchoring works for injection but the weak *full* anchor
  dominates it on retention.
- Low-LR keeps the best *unseen*-phrasing generalization (59% median, noisy);
  the weak anchor's is plain-like (25%, range 23–90%). The untested combo —
  low-LR × weak anchor — is the natural next cell.

Bounds: probe scale (10.65M params, byte-level), a brutal ~200-epoch
memorization regime (19KB of facts for 1,000 steps), one corpus, and high
seed variance on unseen-phrasing recall everywhere. The weak band was probed
at a single strength (0.0025); the dial's shape between 0.0025 and 0.005 is
unmapped.

## 13. The specialist's dial: task-training with retention (Run W)

The fact-injection dial (§12) pointed at the deployment pattern the industry
is converging on — specializing small open models on scored company workflows
("intelligence ownership"). This probe runs that shape directly: a
**catalog-integrity task** ([`data/catalog/`](data/catalog/)) — listing line
in, category + policy verdict out — with ground truth fixed by deterministic
rules (banned product types; protected brands allowed in exactly one category,
with legitimate uses as hard negatives). Exact-match is scored on **novel
attribute combinations never seen in training**, so the metric is internalized
*judgment*, not memorization. Same rig as §12: modernweb base, 3 seeds, one
flag per arm, base recall floor 0%.

| arm (medians of 3) | forgets modernweb | seen | **novel cases** | task-val |
|---|---|---|---|---|
| plain | +2.91 | 100% | 98% | 0.141 |
| low-LR 1e-4 | +2.00 | 90% | 88% | 0.144 |
| **weak anchor, s=0.0025** | **+0.94** | 93% | **97%** | 0.145 |
| anchor, s=0.01 | +0.37 | 15% | 12% | 0.153 |

- **On judgment, the weak-band anchor Pareto-dominates low-LR on both axes**:
  better task accuracy on novel cases (97% vs 88%) *and* half the forgetting
  (+0.94 vs +2.00). Against plain it concedes ~1 point of task accuracy for
  **3.1× less** loss of general ability. This is stronger than the facts
  result (§12), where the anchor matched plain's generalization but didn't
  beat low-LR's: rule-shaped capability apparently coexists with the weak
  pull better than rote bindings do.
- **The blocking cliff is task-general.** s=0.01 blocks judgment acquisition
  (12–15%) just as it blocked facts (2–3%) — the dial's useful band and its
  cliff do not depend on what is being learned.
- Read as the industry playbook's missing axis: specialization pipelines
  report task score and cost; none report what the specialist *lost*. The
  dial makes that loss a chosen quantity instead of an accident.

Bounds: probe scale, one synthetic rule system (~30 product types, 6
protected brands), single-format prompts, and the same unmapped dial region
(0.0025–0.01) as §12. The natural production analog — the anchor as a
zero-forward-pass alternative to KL-to-reference in RL post-training — is
designed but unrun (see NEXT-EXPERIMENTS).

## 14. enwik8: the canonical bench, same verdict (Run X)

The §11 protocol on the benchmark with two decades of published numbers around
it: **enwik8** ([`data/enwik8/`](data/enwik8/), standard 90M/5M split, final
5M test tail untouched, fixed corpus sha256 in the README). 3 seeds, matched
schedules, 2,000-step native teachers, dense eval. The 90M pool means the
1,500-step student sees every byte **once at most** — zero epochs, the purest
adequate-data regime yet, and the least hospitable setting for any
memorization-flavored account of the head start.

| medians of 3 | best (nats/byte) | best (bits/byte) |
|---|---|---|
| baseline @1,500 steps | 1.3388 | 1.9315 |
| **dirs_only** | **1.2335** | **1.7796** |

**dirs_only reaches the baseline's best quality 2.90× faster (2.72–3.10, all
resolved) and converges 0.152 bpb below anything the baseline reaches** — it
passes the baseline's *final* quality at ~step 500 and leads by 1.40 bpb at
step 100. Third corpus, third win for geometry-at-init; first on a bench
where every number has published context. (These are probe-horizon figures —
1,500 steps, 10.65M params — not literature-competitive absolute bpb.)

## 15. Cold directions: never-trained geometry is refuted (Run Y)

The cold-assembly control. Runs R/X show *trained* directions carry the
transfer; Run S shows ~random directions don't. The missing cell: directions
that are **structured but never trained** — analytic orthonormal DCT-II bases
(pure trigonometry, low-frequency-first, `src/gen_cold_directions.py`) paired
with GPT-2's universal spectrum ([`fingerprints/cold-dct/`](fingerprints/cold-dct/)),
so *nothing* in the fingerprint was trained on anything relevant. Modern-web
bench, 3 seeds, no teacher stage at all.

**Refuted, 3/3 seeds**: a marginally better init (step-0 Δ +0.28) that falls
behind by step 10 and never reaches baseline-best (1.539 vs 1.489 median).
Smooth well-formedness is not the load-bearing property — **training is**.
The triad closes cleanly: trained directions win (R, X), random directions
lose (S), structured-but-untrained directions lose (Y). Geometry is *mined*,
not *drawn* — which makes trained checkpoints (the "geometry commons") the
irreducible asset in any cold-assembly or amortized-init scheme, and makes
fingerprint *reuse* (§10, and the energy passes below) the economically
interesting operation.

## 16. The energy ladder: a committed quality-per-joule frontier (Runs Z1–Z3)

Three passes on the enwik8 bench turning §14's speedup into an explicit
**performance-per-energy** frontier, with wall-clock seconds recorded inside
every artifact (single L4, `wall_sec` per stage — the energy proxy).

**The frontier (medians of 3; "beats baseline" = every seed's final loss below
every seed's 1,500-step baseline best):**

| arm | bits/byte | vs. baseline best | GPU wall | energy | beats baseline 3/3 |
|---|---|---|---|---|---|
| baseline, 1,500 steps | 1.9315 | — | 414s | 100% | — |
| **dirs_only, 600 steps (Z2)** | 1.9031 | **−0.028** | 172s | **42%** | yes |
| **dirs_only, 800 steps (Z3)** | 1.8595 | **−0.072** | 230s | **56%** | yes |
| dirs_only, 1,500 steps (X) | 1.7796 | −0.152 | 403s | 97% | yes |

Every PRISM point strictly dominates the baseline — better quality at equal or
less energy; the practitioner picks the point. These are *complete* schedules
(warmup + full LR decay to the shorter horizon), not truncated curves: the
600-step *baseline* control only reaches 1.75 nats, so the short schedule
alone does nothing of the sort.

**The teacher term is amortized, not free — and Z1 prices the amortization.**
A native teacher costs ~2,000 steps once (~560s/seed here, cache-hit for every
later run on the volume). Z1 shows the mined fingerprint **crosses corpora**:
a teacher trained only on the 4.5M-token modern-web corpus accelerates enwik8
students at **1.97× (1.82–2.01)**, converging 0.122 bpb below the baseline —
~68% of the native speedup, ~80% of its quality gain. One mined fingerprint
serves many corpora; marginal cost per new training run approaches
student-only.

Bounds: wall-clock on one GPU class as the energy proxy (no power metering);
probe scale and horizon; the frontier's low end is unmapped below 600 steps;
and the amortization constant (how many corpora one fingerprint usefully
serves, vs. fingerprint distance) has exactly one cross-corpus datapoint
here plus §10's Shakespeare pair.

## 17. Act 1: the prior's floor on enwik8 — not a PRISM result (Run AA)

The free CPU gate for the reach-at-init experiment, run under the binding
two-act framing spec in [NEXT-EXPERIMENTS](docs/NEXT-EXPERIMENTS.md). **This
number belongs to the n-gram**: a static sparse byte n-gram (orders 0–6,
Jelinek–Mercer interpolation, μ tuned only on a dev slice held out of the
counting data; `src/ngram_floor.py`) counted over enwik8's training bytes and
scored sequentially on the untouched val slice:

**1.9061 bits/byte** — below xz's whole-file figure (≈1.99, different
protocol), below zstd and gzip, and 0.025 below the final val quality of the
from-scratch 1,500-step baseline (1.9315, Run X's rig). Dev→val gap 0.014;
the order curve was still descending at 6 (2.05 → 1.89 from order 5 → 6);
11.4M distinct 7-grams at order 6.

Per the spec, no PRISM claim attaches here. The canonical sentence: *a sparse
byte n-gram prior already reaches 1.91 bpb on the held-out tail (Act 1);
adding the PRISM geometry and training produces a trajectory that continues
below that prior floor (Act 2)* — Act 2 is the part that is a PRISM claim,
and this gate result is what authorizes spending on it (needs the sparse
gather + block-edge backoff in `train.py`'s `_prior_logp`, then
`prism_prior_eval` arms on enwik8).

## 18. The injection-dial head-to-head (Runs AB–AE) — catalog rig landed; facts rig pending

*Alpha-moves W1 ([`docs/NEXT-EXPERIMENTS.md`](docs/NEXT-EXPERIMENTS.md) pre-registration
of 2026-10-21). Competitors are our re-implementations (`src/prism_baselines.py`,
deviations listed there), each tuned on seed 1336 only with three settings by the
rule written before looking (lowest forgetting with seen-recall ≥ 0.90, else highest
seen-recall). Tuning records: [`finetune_20261022T024927Z.json`](results/finetune_20261022T024927Z.json)
(catalog), [`finetune_20261022T024917Z.json`](results/finetune_20261022T024917Z.json) (facts).*

**Catalog rig, 3 seeds** ([`finetune_20261022T043049Z.json`](results/finetune_20261022T043049Z.json);
modernweb base, 2,000 + 1,000 steps, the §13 rig; medians of 3; forgetting = modernweb val climb):

| arm | forgetting ↓ | ×less than plain | seen ↑ | novel cases ↑ | step time vs plain |
|---|---|---|---|---|---|
| plain | 2.915 | 1.0× | 100% | 98% | 1.00 |
| low-LR 1e-4 | 2.003 | 1.5× | 90% | 88% | 1.00 |
| **weak anchor 0.0025** (`raw_weak`) | 0.944 | 3.1× | 93% | 97% | 1.06 |
| anchor 0.0035 | 0.797 | 3.6× | 92% | 93% | 1.06 |
| anchor 0.005 | 0.647 | 4.5× | 88% | 88% | 1.06 |
| anchor 0.01 (`raw_mid`) | 0.369 | 7.8× | 15% | 12% | 1.06 |
| low-LR × weak anchor | 0.784 | 3.7× | 58% | 62% | 1.05 |
| anisotropic anchor (k=32, 0.01 top / 0.0025 rest) | 0.624 | 4.6× | 92% | 93% | 1.12 |
| weak anchor, embedding excluded | 1.053 | 2.8× | 93% | 97% | 1.06 |
| weak anchor, absent-token rows only | 1.040 | 2.8× | 93% | 97% | 1.04 |
| weak anchor + Adam reset | 0.980 | 3.0× | 95% | 97% | 1.06 |
| weak anchor + Adam-moment lerp | 0.851 | 3.4× | 93% | 93% | 1.22 |
| Adam-ε 1e-4 on the tied embedding (`out_eps`) | 2.894 | 1.0× | 100% | 99% | 1.00 |
| plain + Adam reset | 3.070 | 0.95× | 100% | 99% | 1.00 |
| **self-distillation, α=1.0** (KL to the frozen base) | **0.399** | **7.6×** | **98%** | **99%** | 1.05 |
| SST, paper default r=0.99 (R100) | 0.946 | 3.1× | 3% | 3% | 1.04 |
| SST, tuned r=0.5 | 1.161 | 2.5× | 12% | 12% | 1.04 |
| RP-SFT, λ=0.1 (k=64) | 2.627 | 1.1× | 100% | 98% | 1.01 |
| EWC, λ=0.1 (normalised Fisher) | 0.412 | 7.4× | 12% | 9% | 1.02 |
| LoRA r=4 (lr 1e-3) | 2.232 | 1.3× | 12% | 8% | 1.36 |
| LoRA r=16 (lr 1e-3) | 3.396 | 0.87× | 95% | 97% | 1.42 |

**The pre-registered readouts.**

1. **Hard projection blocks injection like PRISM's strong band.** SST at its paper
   default stores 3% of the task (per seed 12/2/3%), within 10 points of the 0.01
   anchor (15%) and 97 points below plain. Even with half the spectrum free (r = 0.5,
   the tuned setting) it stores 12%. EWC (Fisher-weighted) and LoRA r=4 block too.
   The prediction the brief made from the literature holds: methods that keep the
   update off the principal directions cannot write a new task into them.
2. **One arm beats the weak anchor on both axes in 3 of 3 seeds: self-distillation.**
   α = 1.0 KL-to-the-frozen-base forgets 0.40 (vs 0.94) while storing 98% (vs 93%),
   per seed 0.399/0.407/0.373 vs 0.991/0.944/0.930 forgetting and 98/100/98% vs
   93/95/93% recall. It costs one base forward pass per micro-step — the very cost the
   anchor's pitch removes — and at this scale that shows as 1.05× step time. So on
   the catalog rig the single-scalar anchor does **not** own the frontier; a
   function-space anchor dominates it. The anisotropic anchor (0.62, 92%) is the best
   PRISM-family point and sits between the two.
3. **The retention is directional, not embedding-row or momentum repair.** Excluding
   the tied embedding changes forgetting by +12%, anchoring only absent-token rows
   by +10%, resetting Adam by +4%, lerping Adam moments by −10% — all under the 20%
   bar. The Adam-ε control alone (arXiv 2610.09835's fix) does nothing here (1.0×),
   with the caveat that on a 256-byte vocabulary almost no token is "absent" from
   the new corpus, so the mechanism that paper targets barely exists in this rig.

The other open cells: the dial's shape between 0.0025 and 0.01 is monotone
(0.0035 → 92% at 3.6×, 0.005 → 88% at 4.5×); the low-LR × weak-anchor combination
hurts injection (58%) and is dominated by the anchor alone; RP-SFT at the one
setting that injects forgets like plain; LoRA r=16 learns the task but forgets
*more* than full finetuning at this LR.

**Facts rig, 3 seeds:** running at the time of writing (`finetune_*_w1` on
volume `prism-alpha-a`); this section is completed when it lands. Verdicts above
are catalog-only until then.

## 19. Hardened controls for the core bet (W2a): the geometry survives a tuned baseline under AdamW; under Muon the multiplier shrinks to 1.4×

*Alpha-moves W2a, pre-registered 2026-10-21. Every arm tuned on seed 1336 only
(`results/sweep_20261022T*.json`, 8 settings for baseline and `dirs_only`, 3 for
Muon LR, 3 for σ_qk); results on seeds 1337–1339, PROBE rig, eval every 10.*

**Tuning.** The committed baseline recipe (LR 1e-3, σ 0.02, warmup 100) was within
0.01 nats of the best of its 8-point grid on both corpora; the **tuned baseline** is
LR 2e-3 with warmup 300 (1.4802 vs 1.4902 on modern web, 1.3183 vs 1.3272 on enwik8,
seed 1336). `dirs_only` prefers LR 1e-3 (modern web) / 5e-4 (enwik8) at σ 0.02;
σ 0.01 hurts every arm. Muon: LR 0.01 for both arms on both corpora. `small_qk`:
σ_qk = 1e-5 beat the tuned baseline by 0.008 nats on seed 1336 (3-seed run pending).

**Modern web, AdamW, every arm against the tuned baseline (medians of 3):**

| arm | best loss | steps-to-target vs tuned baseline | artifact |
|---|---|---|---|
| tuned baseline (LR 2e-3, warmup 300) | 1.4848 | 1.0× | in-run |
| **dirs_only** (LR 1e-3, warmup 100) | **1.3809** | **3.3× (2.5–3.5), 3/3** | [`dirs_only_20261022T153244Z`](results/dirs_only_20261022T153244Z.json) |
| dirs_pure (EigenTransfer onto the fresh spectrum, no imprint) | 1.3865 | 2.45× (2.2–2.7), 3/3 | [`dirs_pure_20261022T153518Z`](results/dirs_pure_20261022T153518Z.json) |
| recipe (dirs + imprint + mod wheel) | 1.4907 | never (0/3) | [`recipe_20261022T170852Z`](results/recipe_20261022T170852Z.json) |
| spectral_only | 1.7623 | never | [`spectral_only_20261022T162500Z`](results/spectral_only_20261022T162500Z.json) |
| mimetic (Trockman & Kolter, char-level values) | 1.5074 | never | [`mimetic_20261022T170532Z`](results/mimetic_20261022T170532Z.json) |
| statmatch (70 teacher statistics, no directions) | 1.5994 | never | [`statmatch_20261022T162209Z`](results/statmatch_20261022T162209Z.json) |

**enwik8, AdamW:** tuned baseline 1.3294; `dirs_only` 1.2296, **3.1× (2.9–3.1), 3/3**
([`dirs_only_20261022T153444Z`](results/dirs_only_20261022T153444Z.json));
statmatch 1.5118, never ([`statmatch_20261022T162239Z`](results/statmatch_20261022T162239Z.json)).

**Muon (LR 0.01, both arms):**

| corpus | Muon baseline best | dirs_only best | steps-to-target | artifact |
|---|---|---|---|---|
| modern web | 1.4383 | 1.4060 | **1.4× (1.3–1.5), 3/3** | [`dirs_only_20261022T153457Z`](results/dirs_only_20261022T153457Z.json) |
| modern web, dirs_pure | 1.4383 | 1.4184 | 1.2× (1.1–1.3), 3/3 | [`dirs_pure_20261022T162246Z`](results/dirs_pure_20261022T162246Z.json) |
| modern web, spectral_only | 1.4383 | 1.6262 | never | [`spectral_only_20261022T170400Z`](results/spectral_only_20261022T170400Z.json) |
| enwik8 | 1.3235 | 1.2779 | **1.5× (1.4–1.7), 3/3** | [`dirs_only_20261022T153515Z`](results/dirs_only_20261022T153515Z.json) |
| enwik8, Muon-trained teacher | 1.3235 | 1.2810 | 1.5× (1.4–1.7), 3/3 | [`dirs_only_20261022T162419Z`](results/dirs_only_20261022T162419Z.json) |

**Verdicts against the pre-registered rules.**

- **The strongest control is the tuned baseline itself.** Every analytic or
  statistics-matched init ran *below* it: mimetic +0.02 nats, statmatch +0.11 nats
  (modern web) and +0.18 (enwik8), spectral-only +0.28. The 108-statistic worry
  (that statistics alone explain init gains) does not reproduce on decoder LMs at
  this scale with the 70 statistics we matched. So the ratio against the strongest
  control is the ratio against the tuned baseline: **3.3× / 3.1×, above the 1.2×
  line on both corpora.** (`small_qk` may yet edge the baseline; its 3-seed run is
  pending and cannot change the verdict's sign.)
- **Under Muon the core bet is "AdamW-strong, Muon-weak".** Muon's baseline is
  0.05 nats better than AdamW's, and `dirs_only` under Muon gains only 0.03 nats and
  1.4× / 1.5× in steps — above the pre-registered 1.2× line, so the bet is not
  labelled AdamW-only, but the multiplier is a third of its AdamW value. The
  teacher's optimizer does not matter (Muon-trained teacher: identical ratio). The
  imprint matters more under AdamW (3.3× vs 2.45× without it) than under Muon
  (1.4× vs 1.2×).
- **The mod wheel is confirmed as a brake on adequate data:** `recipe` never reaches
  the tuned baseline's best (as in §11, now against a tuned baseline).
- **Token-equivalent multiplier: refused.** The pre-registered power-law fit
  tolerance (RMSE ≤ 0.01 nats) is not met on any seed (fit RMSE 0.022–0.029 under
  AdamW, 0.037–0.049 under Muon; eval noise at 50 eval batches dominates), so
  `prism_metrics.token_equivalent_multiplier` returns no number, by design. The
  end-of-training gap carries the same information here: `dirs_only` ends 0.09–0.10
  nats below the tuned baseline's *best* under AdamW and 0.03–0.05 under Muon.
- Lineage traces (W6.2) from these runs are in §22.

## 20. Cross-size SVD projection (Runs AF–AJ): heads and head-dims transfer, width does not

*Alpha-moves W3 ([`docs/CROSS-SIZE.md`](docs/CROSS-SIZE.md), pre-registered
2026-10-21: success = ≥ 1.5× steps-to-target vs scratch on case 1 AND > 1× on case 3,
3 of 3 seeds). Operator: residual-stream PCA projection of the teacher's matrices
into the student's coordinates, per-head QK/OV circuit projection, uniform layer
selection, then the ordinary EigenTransfer (`src/prism_crosssize.py`). Modern web,
PROBE rig, 3 seeds, overlap traces every 50 steps.*

| case | teacher → student | steps-to-target vs scratch (median, range) | best loss vs scratch | init loss vs scratch | step-100 loss vs scratch | artifact |
|---|---|---|---|---|---|---|
| 1 width | 12L/768/12h (trained here, 2,000 steps) → 6L/384/6h | **1.0× (reached in 1/3 seeds)** | 1.4924 vs 1.4892 | 5.74 vs 5.65 | **2.25 vs 2.55** | [`dirs_only_20261022T035754Z`](results/dirs_only_20261022T035754Z.json) |
| 2 heads | 6L/384/**12h** → 6L/384/**6h** | **3.7× (2.9–3.9), 3/3** | **1.3827 vs 1.4892** | 4.64 vs 5.65 | 1.78 vs 2.55 | [`dirs_only_20261022T040349Z`](results/dirs_only_20261022T040349Z.json) |
| 3 head-dim | 6L/384/6h → 6L/**192**/6h (scratch baseline at 192) | **1.28× (1.24–1.34), 3/3** | 1.5855 vs 1.6384 | 5.36 vs 5.61 | 2.52 vs 2.60 | [`dirs_only_20261022T040448Z`](results/dirs_only_20261022T040448Z.json) |
| 4 GPT-2 | GPT-2 124M (public weights) → 6L/384/6h bytes, internal matrices only | **1.0× (1/3)** | 1.4900 vs 1.4892 | 5.67 vs 5.65 | 2.57 vs 2.55 | [`dirs_only_20261022T043832Z`](results/dirs_only_20261022T043832Z.json) |
| 4 FRONT | GPT-2 → same student, 3D-DCT truncation (our re-implementation) | never reaches scratch-best | **2.078 vs 1.4892** | 5.58 vs 5.65 | 2.64 vs 2.55 | [`init_state_20261022T041102Z`](results/init_state_20261022T041102Z.json) |
| 1 Weight Selection | 12L/768/12h → 6L/384/6h, uniform element selection (our re-implementation of arXiv 2311.18823) | **4.0× (3.2–4.7), 3/3** | **1.4026 vs 1.4892** | **3.80 vs 5.65** | 1.73 vs 2.55 | [`init_state_20261022T170647Z`](results/init_state_20261022T170647Z.json) |

**Weight Selection wins case 1 outright.** Copying evenly-spaced rows and columns
of the 768-wide teacher's weights into the 384-wide student (content, not
geometry) reaches scratch-best 4.0× faster in every seed and ends 0.09 nats below
it — faster than same-size `dirs_only` (3.3×) though not as low (1.4026 vs 1.3794).
Per the pre-registration this narrows the cross-size claim to the head cells: for
width, index-space weight transfer beats our basis-level projection at this scale,
and this is also, as far as we know, the first language-model data point for Weight
Selection (its paper is vision-only, reported). The GPT-2 case cannot copy weights
(the vocabularies differ), which is where a geometry-only operator would have to
earn its place; it did not here.

**Verdict: the pre-registered success criterion FAILS on case 1.** Projecting a
twice-as-wide teacher's directions through its residual-stream PCA onto the
student buys a real early-window head start (0.31 nats at step 100, 3/3 seeds)
that is gone by the horizon: the projected student never beats scratch-best in
2 of 3 seeds and ends 0.003 nats behind. GPT-2's directions projected onto the
byte-level student do nothing at any step. Two projections *do* transfer:
halving the head count at fixed width keeps the full same-size effect (3.7×
against the same-size 3.3× of §11, best loss 0.11 nats below scratch), and
halving the head dimension gives 1.28× in every seed — the cell CrossGMN reports
0.74× on (reported by arXiv 2610.01649; different scale and task). FRONT's
truncated weights from GPT-2 actively hurt here (0.59 nats behind scratch at the
end), consistent with it being an index-space operator meeting a mismatched
vocabulary and depth.

The overlap traces say the operator did impose the directions: at step 0 the
students' top-16 subspaces overlap the projected reference at 0.58–0.93 (chance
0.03). In case 2 the FFN-up overlap *rises* during training (0.65 → 0.81); in cases
1 and 4 it decays toward chance (0.58 → 0.16, 0.58 → 0.07). Directions that a
wider model uses in its own coordinates are adopted and then abandoned; the
problem is the coordinate map, not the transplant. Whether a μP-native teacher,
a per-layer PCA or Procrustes alignment changes that is the next cell (the
Procrustes and Weight-Selection baselines for cases 1–3 are running as this is
written). The draft preprint in `docs/preprint/` is NOT to be posted; it will
be revised to lead with the head/head-dim result and the width null.

## 21. Prior fusion against the COUNT control (W5) — run 1 void, run 2 running

[`prior_20261022T035740Z.json`](results/prior_20261022T035740Z.json) is kept as a
record and is **void**: its own kill condition fired (`prior` never reached
baseline-best; its step-0 loss sat above uniform). The cause was an off-by-one in
the sparse prior's in-window gather, which fed the logits a confident prediction
of the *current* byte (details in [`docs/NEXT-EXPERIMENTS.md`](docs/NEXT-EXPERIMENTS.md),
W5 block). Fixed and selftested; the identical run is repeating under tag `w5b`.
No claim is drawn from the void run beyond the two readouts that do not depend on
the fusion path: the standalone sparse prior reproduces Act 1 (1.927 bpb on the
student's slot vs 1.906 on the full training bytes), and `dirs_only` reproduces §14
on this rig (4.2× to baseline-best, 1.90 bpb).

## What this does NOT establish

- **Early-window scope.** Runs D and E measure the head start at step 100
  (init-dominated). They prove the transfer is structural and domain-portable;
  they do not show the full no-overfitting arc to convergence on far data (Runs
  A–C are long-horizon but same-data).
- **Sherlock is still English prose** (token-JS 0.027, same character set). A
  genuinely different modality — code, another language — is untested.
- **Spectral vs. generic regularization** is still not isolated by a direct
  control (tuned dropout / weight decay). Run F's teacher-dependence is
  suggestive, not a substitute.
- **Two shared learning rates, not a per-arm-best sweep.**
- **Scale.** 10.65M params, ~1M-token corpora.
- **Finetune-retention (§7) shows old-domain *retention*, not new-domain overfit
  prevention** (plain finetuning didn't itself overfit Sherlock in 1,000 steps), and
  "far" is again Sherlock (same alphabet). The anchor also trades a little late
  adaptation back for retention (a decaying pull would remove it).

## The next experiments, in order

1. **Truly far modality** — the far-corpus arm pointed at source code or another
   language (token-JS ≫ 0.03); find where structural transfer degrades.
2. **Long-horizon far-domain** — the 0%-overlap Sherlock student to convergence:
   does no-overfitting transfer across domains too?
3. **Endurance** — the recipe to 20k–50k steps.
4. **Ablations** — `spectral_only` / `dirs_only` / reg-matched baseline.
5. **Cross-size transfer** — spectrum interpolates; directions need a projection.

## Reproduce it

```bash
cd src
pip install torch numpy transformers tiktoken datasets
python prism_eval.py --student_steps=1500 --eval_every=10 --eval_iters=50   # Run C
python prism_eval.py --method_lr=1e-3 --method_warmup=100                   # Run B
python prism_eval.py --teacher_steps=500 --student_steps=100 --eval_every=10 \
  --eval_iters=40 --batch_size=32 --method_lr=1e-3 --method_warmup=20 \
  --baseline_warmup=20 \
  --overlap=1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.0               # Run D
python prism_eval.py --teacher_sweep=100,250,500,1000,2000 \
  --student_steps=300 --eval_every=20 --eval_iters=40 --batch_size=32       # Run F
python prism_eval.py --teacher_steps=500 --student_steps=100 --eval_every=10 \
  --eval_iters=40 --batch_size=32 --method_lr=1e-3 --method_warmup=20 \
  --baseline_warmup=20 --overlap=1.0,0.75,0.5,0.25,0.0 \
  --far_corpus=data/far.txt --far_val                                       # Run E
python prism_eval.py --report                                               # reprint last artifact
```

The finetune-retention frontier (Runs J, K) runs headless on Modal:

```bash
modal run --detach prism_modal_finetune.py --extra \
 "--tag=r2b --base_steps=2000 --ft_steps=1000 --eval_every=25 --seeds=1337,1338,1339 \
  --arms=base,plain,raw_lo,raw_mid,raw_hi,lowlr_a,lowlr_b,lowlr_c,spectral,shuffled,scratch_ceiling \
  --learning_rate=3e-4 --min_lr=3e-5 --batch_size=32 --block_size=256 --far_corpus=data/far.txt"
```

The eval is stepwise and resumable, records provenance and whether the schedule
was matched, and refuses to score a partial or crashed run. **A number in these
docs must have a matching `results/*.json`.**
