# Next experiments — a handoff for the next agent

You are picking up a mature PRISM codebase with three committed, artifact-backed
results. This doc tells you what's proven, what machinery exists, and the ranked
next experiments — each with a hypothesis, a concrete run command, and what a win
looks like. Read [README](../README.md) "Start here" first for the ground rules;
they are non-negotiable (evidence = committed `results/*.json`, matched schedules for
attribution, wide/shallow probes before deep runs, 3 seeds for a result, no mocks).

## What's already proven

1. **From-scratch spectral transfer** ([RESULTS](../RESULTS.md)): a teacher's spectrum
   → a fresh model trains ~12× faster; the spectrum is data-independent *structure*.
2. **Finetune without forgetting** ([FINETUNE-RETENTION](FINETUNE-RETENTION.md)): keep
   the mod wheel on, self-anchored → up to ~10× less forgetting; it's a raw
   *directional* anchor (not spectral, not just a smaller LR).
3. **The arc** ([UNIFIED-ARC](UNIFIED-ARC.md)): a PRISM-pretrained base is a much
   better finetune-anchor than a plain base *at matched quality* (≈0 forgetting + ~8%
   better adaptation), and it's the **spectral geometry** (a schedule-matched plain base
   doesn't do it).

The through-line: **the spectral geometry is special.** Every experiment below tests a
consequence of that.

## The machinery (all on `master`)

- `src/prism_eval.py` — from-scratch benchmark (`prism-eval/1`). Knobs: `--overlap`,
  `--teacher_sweep`, `--far_corpus`/`--far_val`, `--method_lr`/`--method_warmup`.
- `src/prism_finetune_eval.py` — finetune-retention benchmark (`prism-finetune/1`).
- `src/prism_arc_eval.py` — the base-interaction / arc benchmark (`prism-arc/1`).
- `src/prism_accelerate.py` / `src/prism_finetune.py` — the "apply it" entry points.
- `src/train.py` knobs you'll reuse: `prism_init`, `prism_mod`/`prism_mod_decay`,
  `prism_anchor_mode` (raw|spectral|shuffled) + `prism_anchor_refresh`, `prism_unfold`,
  `val2_dir` (dual-val), `stop_val_target` (matched-quality stop), forced-eval-at-resume.
- Modal runners (own isolated volume each): `prism_modal.py` / `prism_modal_finetune.py`
  / `prism_modal_arc.py`. Pattern: fork a branch → make `prism_modal_<name>.py` with its
  own `Volume.from_name("prism-eval-<name>")` → `modal run --detach`. Fetch:
  `modal volume get prism-eval-<name> nanogpt-prism/results/<file> ./results/` and COMMIT it.
- Ritual before any GPU: `cd src && python prism_selftest.py` (offline, ~1 min) then a
  tiny `--device=cpu` smoke of your driver → a conforming artifact. Then Modal.

## Status update (2026-10-21): the alpha-moves session — pre-registrations for W1–W5

The 30-day alpha scan ([`reports/PRISM alpha in recent research.md`](../reports/PRISM%20alpha%20in%20recent%20research.md))
re-ranked the next moves; [`ALPHA-MOVES-BRIEF.md`](ALPHA-MOVES-BRIEF.md) turns them
into workstreams and [`ALPHA-MOVES-STATUS.md`](ALPHA-MOVES-STATUS.md) tracks them.
Nothing below weakens an earlier pre-registration; the core-bet decision rule,
the three-outcome gates and the two-act framing spec bind unchanged. New
pre-registrations, each committed before its launch:

### W1 — the injection-dial head-to-head (pre-registered 2026-10-21, before any launch)

**Question.** Does one scalar anchor trace the retention ↔ injection frontier
better than the methods the field now uses, and do hard-projection methods (SST)
block injection the way PRISM's strong band does?

**Rigs, unchanged from the committed probes:** facts (`prism_modal_facts.py`
FLAGS: modern-web base, 2,000 base steps, 1,000 finetune steps, LR 3e-4, eval
every 25) and catalog (`prism_modal_catalog.py` FLAGS). New `--tag`s; every arm
forks the same base per seed and differs by one change. Seeds 1337–1339 for
results; **seed 1336 only for tuning**.

**Arms.** Existing: `plain`, `lowlr_b`, `raw_weak` (0.0025), `raw_mid` (0.01),
`raw_exffn`, `raw_exattn`. Open cells: `raw_0035`, `raw_005`, `lowlr_weak`.
PRISM variants: `raw_weak_exemb`, `raw_weak_rows_absent`, `raw_weak_adamreset`,
`raw_weak_adamlerp`, `plain_adamreset`, `raw_aniso`. Controls/competitors
(re-implementations, `src/prism_baselines.py`, deviations listed there):
`out_eps` (ε=1e-4 on the tied embedding, fixed per the paper), `sst`,
`sst_paper` (r=0.99, R100 — the paper's default), `rpsft`, `ewc`, `lora_r4`,
`lora_r16`, `selfdistill`.

**Tuning parity (criterion written before looking).** Each tunable method gets
exactly three settings on seed 1336 (`*_t1..t3` arms; LoRA tunes its LR; the
anisotropic anchor tunes (k, s_top, s_rest)). **Chosen setting = the lowest
forgetting among settings with seen-recall ≥ 0.90 (the parity band plain and
raw_weak occupy); if none reaches 0.90, the highest seen-recall; ties → lower
forgetting.** The same rule on both rigs independently. The chosen settings are
written into `ARM_SPECS` before the 3-seed launch and recorded in the artifact.

**Metrics.** Forgetting (old-domain val climb), recall seen / unseen / v2
paraphrases (`data/facts/recall_prompts_v2.tsv`, 3 never-trained templates),
new-domain val loss, the adaptation-censored guard, per-step wall-clock vs
`plain` (`[timing]` line), and (facts rig) the `[overlap]` subspace traces.
Primary display: seen-recall vs forgetting, every arm, both rigs.

**Pre-registered readouts.**
1. SST (paper default r=0.99): *blocks* if median seen recall is within 10
   points of `raw_mid`'s; *injects at parity* if within 10 points of `plain`'s.
   Unseen-phrasing recall is secondary (its seed variance is high).
2. An arm "beats raw_weak" only if it has BOTH lower forgetting AND higher
   seen recall in **3 of 3 seeds** (per-seed, same base).
3. `raw_weak_exemb`, `raw_weak_rows_absent`, `raw_weak_adamreset`,
   `raw_weak_adamlerp` change forgetting by ≥ 20% (median, vs raw_weak) →
   part of the retention is not directional; < 20% → it is.

**What changes which claim.** If any competitor beats raw_weak under rule 2 on
*both* rigs, README Finding 7 drops "the dial traces the frontier" and Honest
bounds names the winner. If SST injects at parity under rule 1, the brief's
differentiating hypothesis (projection blocks injection) is refuted and said so.

**Cost.** Tuning: ~2 × 25 arm-runs on seed 1336 (~1.9 L4-h each rig ≈ $3).
Results: ~2 × 22 arms × 3 seeds (~5.6 L4-h each rig ≈ $9). Tier A.

### W2a — hardened controls for the core bet, probe scale (pre-registered 2026-10-21, before any launch)

**Question.** Does geometry-at-init (`dirs_only`) still beat a *tuned* baseline
and the analytic / statistics-matched inits the field now expects, under AdamW
and under Muon, on modern web and enwik8?

**Rig.** The PROBE rig unchanged (`prism_modal_modernweb.py` / `prism_modal_enwik8.py`
FLAGS: 2,000-step teacher, 1,500-step students, eval every 10, eval_iters 50,
batch 64×256, seeds 1337–1339). Tuning on **seed 1336 only** via
`src/prism_sweep.py` (artifacts `results/sweep_*.json`).

**Tuning parity (criterion written before looking).** Baseline grid: LR ∈
{5e-4, 1e-3, 2e-3} × init σ ∈ {0.01, 0.02} at warmup 100, plus warmup 300 at
LR ∈ {1e-3, 2e-3} (σ 0.02) — 8 settings; `dirs_only` gets the identical 8. Muon:
`muon_lr` ∈ {0.01, 0.02, 0.04} for the baseline and for `dirs_only` (AdamW
groups at the tuned AdamW LR). **Chosen setting = lowest best val loss within
the 1,500-step horizon on seed 1336**, per corpus, per arm. `small_qk` sweeps
σ_qk ∈ {1e-5, 1e-3, 1e-2} on 1336 by the same rule; `mimetic` and `statmatch`
have one setting each (paper values; the stated statistic list).

**Arms (3 seeds, both corpora).** AdamW at the tuned schedule: tuned baseline,
`small_qk`, `mimetic`, `statmatch`, `spectral_only`, `recipe`, `dirs_only`,
`dirs_pure` (EigenTransfer onto the fresh spectrum, no imprint). Muon: baseline,
`dirs_only`, `dirs_pure`, `spectral_only`; plus `dirs_only` from a Muon-trained
teacher. Every run records `optim` and `teacher_optim` and banks `[overlap]`
traces every 50 steps.

**Metrics.** `prism_score` of each arm vs its in-run baseline; the steps-to-target
ratio of `dirs_only` against the **strongest control** (= the control arm with the
lowest median best val among tuned baseline / small_qk / mimetic / statmatch, per
corpus, computed cross-artifact with `prism_metrics.steps_to_target_ratio` at the
control's best); the token-equivalent multiplier (fit on the strongest control's
curve, RMSE tolerance 0.01 nats, refused otherwise); the end-of-training gap; the
overlap traces (does Muon erase the imprint faster than the transplanted
directions?).

**Pre-registered rules (unchanged from the brief).** If `dirs_only`'s
steps-to-target ratio against the strongest control is **< 1.2× on either
corpus**, that goes into README "Honest bounds" **before any scale spend**. If it
is < 1.2× under Muon, the core bet is labelled "AdamW-only until shown otherwise".

**Cost.** Sweeps ≈ 4.6 L4-h (≈ $4); the 3-seed arms ≈ 2 corpora × 13 runs × ~0.7 L4-h
≈ 18 L4-h (≈ $15). Tier A.

### W3 — cross-size SVD projection (pre-registered 2026-10-21, before any launch)

The full method, cases, baselines and decision rule live in
[`CROSS-SIZE.md`](CROSS-SIZE.md) §"Pre-registration" (committed before launch):
four probe-scale cases on modern web, 3 seeds, PROBE rig — (1) width 768→384 with
a 12L/768/12h teacher, (2) heads 12→6 at width 384, (3) head-dim reduction
6L/384/6h → 6L/192/6h, (4) GPT-2 124M → the 10.65M byte-level student (internal
matrices only); baselines scratch (in-run), Weight Selection, FRONT, mimetic, and
same-size `dirs_only` as the upper reference. **Success = ≥ 1.5× steps-to-target
vs scratch on case 1 AND > 1× on case 3, in 3 of 3 seeds.** Metrics: `prism_score`,
end-of-training gap, overlap traces, teacher compute stated. Draft preprint
(`docs/preprint/cross-size-projection.md`) is NOT to be posted; the user decides.

### W5 — the COUNT control for prior fusion, the Act 2 composition gate (pre-registered 2026-10-21, before any launch)

**Hypothesis.** Fusing a sparse context-6 byte n-gram prior into the logits
*during training* reaches the baseline's best enwik8 loss in fewer steps than
fusing the same prior only at inference, and composes multiplicatively with
geometry-at-init (`dirs_only`).

**Arms** (enwik8, seeds 1337–1339, 2,000-step native teacher, 1,500-step students,
eval every 10, eval_iters 50, batch 32 × block 256 — the §14 rig; matched
schedules LR 1e-3 / warmup 100; fusion strength λ = 1.0; prior = orders 0–6 with
Act 1's μ, counted on the student's training slot, block-edge backoff):
`baseline`, `dirs_only`, `prior`, `dirs_prior`, `baseline_inffuse`, `dirs_inffuse`
(trained without the prior, the same prior fused at every validation point),
`count` (COUNT-style parameter-free input embedding, our re-implementation of
arXiv 2610.20456 §2; deviations in `docs/alpha/W5-NOTES.md`), `count_dirs`.
Validation also at every epoch boundary (`eval_at`).

**Primary metric.** Steps-to-baseline-best (compute_score) against the in-run
plain baseline; secondary: end-of-training loss, token-equivalent multiplier,
init loss.

**Decision rule.** Logit fusion earns its keep **only if `prior` beats
`baseline_inffuse` on steps-to-target in 3 of 3 seeds AND `dirs_prior` beats
`dirs_inffuse` in 3 of 3.** Otherwise COUNT's null (train-time ≈ inference-only)
is replicated at this scale and README Finding 8 / the 30× caveat say so.
**Composition gate:** the "composition multiplies" thesis survives only if
`dirs_prior`'s steps-to-target ratio exceeds `dirs_only`'s by ≥ 1.5× (median);
below 1.2× it saturates and the Act 2 block above records that before any Pythia
spend.

**Kill / void.** If `prior` never reaches baseline-best in any seed, or the prior's
standalone val in the artifact's `prior_info` is not within 0.02 bpb of Act 1's
1.9061, the run is void and investigated, not interpreted. The two-act framing
spec binds every sentence written about these arms.

**Cost.** ≈ 3.8 L4-h ≈ $3. Tier A.

**Run 1 VOID (2026-10-22, `results/prior_20261022T035740Z.json`, kept as a record).**
The kill condition fired: `prior` never reached baseline-best in any seed, and its
step-0 loss (6.41 nats) sat *above uniform* (5.55). Investigation found the sparse
prior's in-window gather built each position's context from the symbols *before*
it, so the fused logits carried a confident prediction of the **current** byte
instead of the next one (the dense `_prior_logp` includes the current byte; the
alignment selftest now pins the convention: 1.28 bits on the next symbol vs 5.5 on
the current, on a synthetic corpus). The standalone sequential score was right all
along (1.927 bpb, within the 0.02 bpb of Act 1 the rule allows), which is why the
build check passed. Fixed in `prism_prior.py`; the run is repeated unchanged
under tag `w5b`. Nothing in the pre-registration above changes.

### W4 — the RL anchor where RL drifts: single-seed pilot (pre-registered 2026-10-22, before any launch)

**Why.** TRL GRPO defaults β = 0.0 and loads no reference model; prime-rl has no
reference KL; verl defaults `use_kl_loss = False` (all verified 2026-10-21). The
pitch is therefore *drift control at zero forward-pass cost in regimes that drift*,
not cutting a cost most recipes already removed.

**Rig.** TRL 1.16.0 `GRPOTrainer`, Qwen2.5-0.5B-Instruct, GSM8K train prompts,
exact-match reward on `#### <answer>`, 300 optimizer steps, 32 completions/step
(8 per prompt, 256 tokens), LR 1e-6, DAPO loss, one A100-40GB per arm
(`prism_modal_rl.py`, driver `src/prism_rl_pilot.py`). Seed 1337 only: every run
is a **probe**; 3-seed confirmations are Tier B.

**Arms.** `beta0`; `kl:β` for β ∈ {0.01, 0.04, 0.1} (swept on the on-policy regime
only — budget); the fixed-W₀ anchor `anchor:s` at s ∈ {0.0025, 0.01, 0.02}; the EMA
anchor `anchor_ema:0.0025:0.01`; WARP-style `warp:0.0025:0.01:0.7` (EMA anchor +
LITI η = 0.7); `anchor_adam:0.0025` (anchor + Adam-moment lerp).

**Drift regimes** (each crossed with β = 0, the best KL and the best anchor):
on-policy (`num_iterations=1`); recycled rollouts `recycle:4` (each batch reused 4
times); stale generation `stale:4` (one generation batch serves 4 optimizer
steps); zero-RL fine-tunability: after RL, LoRA r=8 SFT on IMDB for 200 steps,
then re-score GSM8K (arXiv 2609.33898's protocol).

**Metrics.** GSM8K accuracy (300 held-out test problems, greedy) before/after;
retention on a pinned lm-eval 0.4.13 suite (hellaswag, arc_easy, winogrande;
300 examples each; `acc_norm`); KL(π_θ ‖ π_init) on 32 held-out prompts with
completions sampled from the init model; weight drift, principal-mask overlap
(arXiv 2511.08567's metric, α = 5%, rank-256 reconstruction) and top-k subspace
energy of the update; mean optimizer-step wall-clock; the anchor callback's own
cost; the reference-forward share = (KL-arm step − β0 step)/KL-arm step on the
same regime.

**Pre-registered rule (unchanged from the brief).** The anchor earns its keep in
a regime only if, in 3 of 3 seeds (Tier B), it (i) matches the best KL's retention
within noise at no reward cost AND (ii) beats β = 0 on retention. The single-seed
pilot can only *indicate*; it is written up as a probe (RESULTS §23) and decides
which regime the Tier B seeds go to. If β = 0 does not drift measurably in any
regime (retention within noise of the init, KL(init) small), the honest verdict is
"low need", not "anchor works".

**Amendment (2026-10-22, after the first probe landed, before any further launch).**
At TRL's default LR (1e-6) the β=0 / recycle:4 probe did not move the policy in 300
steps: GSM8K 23.3% → 23.3%, retention flat, KL(π‖π_init) = 0.006, relative weight
drift 4e-5 ([`rl_beta0_recycle4_s1337_w4_20261022T161944Z.json`](../results/rl_beta0_recycle4_s1337_w4_20261022T161944Z.json)).
A regime with no drift cannot separate a drift control from nothing, so a second
wave at **LR 1e-5** (10× the default; a drift regime by construction) is ADDED:
{β0, KL 0.04, anchor 0.0025} × {recycle:4, stale:4}. The default-LR arms run to
completion and are reported as "the stacks' default". The rule above applies
unchanged to every arm; nothing is removed or weakened. Measured cost: 7.3 s/step
on the A100-40, ≈ $1.3 per 300-step arm, so the whole pilot stays near $20.

**Cost.** ≈ 12 arms × ~1.5 A100-h × $2.10 ≈ $35–40 for the full pilot; launched in
priority order (on-policy β0 / KL 0.04 / anchor 0.0025, then the recycle:4 cross)
so the budget can stop it at any point. Under the W4 cap of $100.

## Status update (2026-07-27): the modern-web probe ran — read it first

[RESULTS §11](../RESULTS.md): byte-level FineWeb-Edu bench (`data/modernweb/`,
vocab 256 — **the far-modality vocab gotcha in #2 below is solved**), four
decomposition arms. Verdicts that re-rank this list: **dirs_only wins outright
on modern text** (3.3× to baseline-best, converges 0.11 nats below it, 3/3
seeds) — the Mod Wheel is an *overfitting brake* (endpoint-valuable only on
small data); **spectral_only is below baseline everywhere** — the spectrum is
not the from-scratch lever; and **GPT-2's fingerprint ≈ a native teacher's
spectrum** at parity across size/tokenizer/corpus (`fingerprints/gpt2-124M/`,
eval knobs `--corpus` / `--fingerprint`, runner `prism_modal_modernweb.py`).
New top follow-ups: (a) **dirs_only on Shakespeare** — the missing 2×2 cell
isolating the wheel-as-brake claim; (b) **wheel decay retune / adaptive wheel**
on adequate data (0.01/0.9999 still pulls ~86% at step 1,500 — does a fast
decay recover dirs_only's endpoint?); (c) **cross-size directional projection**
— with the spectrum refuted as the lever, the universal init that could work is
directional (GPT-2's U/Vᵀ projected down); (d) the truly-far modality (#2) is
now unblocked — point `--corpus` at a code corpus.

**Second update (same day): the fact-injection probes ran — both rounds**
([RESULTS §12](../RESULTS.md), `data/facts/` + `fact_recall.py` +
`prism_modal_facts.py`; eval knobs `--old_corpus` / `--ft_val_corpus` /
`--recall_prompts`, train.py knob `--prism_anchor_exclude`). Round 1: the §7
anchor **blocks** novel-fact storage at retention-grade strength (2–3% recall
vs 98–100%) while posting the best val loss — loss is not an injection metric.
Round 2: **the weak band (s=0.0025) is the recipe** — 96% injection at 3× less
forgetting than plain, 2× less than low-LR; selective anchors show facts store
in either attention or FFNs (not FFN-exclusive at this scale), and freeing
FFNs costs more retention than freeing attention. Round 3 ([RESULTS
§13](../RESULTS.md), `data/catalog/`): on a scored structured-decision task
the weak anchor **Pareto-dominates low-LR** (97% vs 88% on novel cases at
half the forgetting) — the specialist's retention dial, demonstrated in the
intelligence-ownership shape. Open cells: low-LR × weak anchor combined; the
dial's shape in (0.0025, 0.005); anchor+replay; and the production-scale test
— **the weight-space anchor as a zero-forward-pass alternative to
KL-to-reference in GRPO/RL post-training** (three arms on a ~9B with prime-rl:
KL-to-ref, no-KL + weak anchor, neither; score task reward, general-benchmark
retention, and training throughput — the anchor arm needs no reference-model
forwards; ~$500 scale, adjacent prior art: Elastic Reset).

## THE CORE BET (2026-07-28) — pre-registered, above everything below

**Claim under test:** on a trusted open model family (Pythia or OLMo), a
PRISM-initialized run reaches the same validation loss — and the same
downstream scores — using meaningfully fewer tokens than the published
baseline. **The primary number is tokens-to-target-loss.** Downstream
metrics are confirmation, not the main event.

**Decision rule, both directions:** if the tokens-to-loss advantage survives
to 410M–1B on a public family with clean comparisons, this is real. If the
advantage fades toward 1× as scale increases, **the core bet has failed,
regardless of how good the small-scale results looked** — no reframing, no
retreat to the side results.

**Explicitly demoted to supporting evidence:** compressor comparisons on
enwik8, small-scale retention wins on synthetic shifts, framing and charts,
and absolute capability against larger models. Useful context; not the thing
that changes training economics.

**Why the comparison can be unusually clean:** Pythia published architecture,
exact data order, schedules, checkpoints, and loss curves. The baseline needs
no trust and no re-run; our arm is the same architecture, same token stream,
same schedule, with exactly one variable changed — the initialization. The
next dollar of GPU spend goes to this track's G2 gate (below), not to further
refinement of the small bench.

**The path from ~3× to "way left / way down" is composition, and it gates
cheap.** Single-lever evidence: geometry-at-init ×2.9–3.3 (three corpora).
The one composition ever measured — prior fusion on top of geometry —
*multiplied* (15× → 30×, Shakespeare, single seed). If that reproduces,
the composed system sits at 6–10× tokens-to-target; a tuned-schedule system
arm adds ~×1.5 more. Therefore **Act 2 on enwik8 (~$3) is re-purposed as the
composition gate for the core bet** — it measures whether lever two
multiplies or saturates, before any Pythia spend. The Pythia chart then
carries two arms against the published curve: the one-variable init arm (the
credible shift) and the best composed system arm (the large shift), labeled
as such. "Way down" needs no extra run: the same trajectories give the
matched-tokens endpoint gap. Cross-size projection (Pythia's own 1B/2.8B
checkpoints as free teachers) is the one unmeasured factor that could raise
the base lever; for Pile-scale priors, infini-gram-style public suffix-array
counts may make the fusion table free.

## The three outcomes (2026-07-28) — what "wildly successful" looks like, and the ladder to each

Ranked by field status; every rung gated, with kill-criteria, so no tier is
funded on hope.

**Outcome 1 — the scaling-curve shift (highest status).** PRISM-init curves of
a known public family (Pythia first: full curves + checkpoints public, so the
gray field needs no trust) shifted left/down on a FLOPs-to-loss plot, holding
from ~100M to 1B+. Ladder: G1 port the ops to the GPT-NeoX family + selftest
($0, real work — RoPE, parallel attn/MLP, new name map); G2 Pythia-70M
single-seed probe, same-size fingerprint from Pythia's own checkpoints (~$100)
— **kill if the shift is <1.2×**; G3 160M, 3 seeds, the chart against
EleutherAI's published curves (~$500); G4 410M (~$1.5k) — the scale test: if
the 410M ratio holds ≥80% of the 70M ratio, the story survives scale and 1B+
(~$10–20k) becomes a fundable claim rather than a wish. The honest unknown all
the way up: our zero-epoch enwik8 result is the right regime evidence, but
10M→1B is 100×.

**Outcome 2 — retention on evals people track (most practical, most probable,
fund first).** The anchor as a ~50-line HF Trainer/optimizer callback (post-
step lerp; W₀ sharded or CPU-offloaded), run on a strong public 1–3B base
through realistic domain/instruction adaptation, scored on lm-eval-harness
suites — the "target gain vs. capability retained" scatter, with the public
leaderboard's own base-vs-finetune deltas as the free gray field. Ladder: G1
callback + 1B smoke (~$20, include a mini dial sweep — **the weak band's
location may shift with Adam scale**); G2 the 1B study, anchored vs full vs
LoRA (~$300–500); G3 7B (~$1–3k). Three internal replications of the dial say
this is the highest probability-per-dollar claim we own; it is also the
artifact fine-tune shippers can use the same week.

**Outcome 3 — the post-training regularizer (most surprising if true).** The
same callback from Outcome 2 dropped into GRPO/DPO in place of (or alongside)
KL-to-reference: measure reward, retention, stability, and throughput — the
anchor needs zero reference-model forwards, which is a measurable compute win
independent of quality. Almost no direct evidence on preference/RL objectives
yet; position against Elastic Reset as the continuous, dialed variant. Run it
only AFTER Outcome 2's port exists (it reuses ~90% of the engineering):
pilot at 1.5–3B (~$500), 9B (~$1.5–3k). Kill honestly if KL's stability role
turns out not to be a weight-space phenomenon.

Sequencing is forced by dependencies: **2 → 3 share one port; 1 is its own
track whose $100 gate (G2) can run in parallel once the NeoX port exists.**
Budget tiers: ~$500 buys Outcome-2-at-1B plus Outcome-1's kill-or-continue
probe; ~$2k adds the 160M chart and the RL pilot; the citable scaling claim
is a $10–20k decision made only after G4 holds. Release hygiene is part of
the claim at every tier: seeds, configs, public training curves — and note
the license's 1e22 threshold does not impede academic reproduction at these
scales.

## The experiments, ranked

### 1. Teacher-free PRISM init — MEASURED (Runs O/P, 2026-07-25)

**Status: run.** `--cross_teacher` landed in `prism_eval.py` and the probe is
committed ([RESULTS §10](../RESULTS.md), artifacts `recipe_20260725T164604Z` /
`…T164640Z`, runner `prism_modal_teacherfree.py`). Verdict: the strong form
(cross ≈ control) is refuted at Sherlock-distance — a Sherlock-only teacher's
fingerprint keeps **4.6× of the control's 9.9×** (~46% of the speedup, ~81% of
the quality gain, no overfitting). Teacher-free init *works*; it isn't *free*.
Open follow-ups, in value order: (a) a teacher-corpus-**size** control (the cross
teacher had 499K vs 803K tokens — is the missing half just data volume?); (b) a
second far corpus + direction to map score-vs-distance; (c) Σ\*-style *averaged*
fingerprints from several corpora — does averaging recover the matched-teacher
half? Original hypothesis below, kept for context.

**Hypothesis (from [project_prism_sigma_star] / the Hutter side-quest):** the spectrum
Σ\* is a *modality* constant, not corpus-specific. If so, you can PRISM-init a fresh
model from a fingerprint extracted from an **unrelated** corpus — no matched teacher —
and the ~12× head start survives. That turns PRISM from "needs a trained teacher" into
a **drop-in universal init** (~128 bytes).

**Run:** extract a fingerprint from a model trained on corpus A, then accelerate a model
on corpus B with it, vs a from-scratch baseline on B:
```bash
cd src
# 1) train an A-teacher and extract its fingerprint (or reuse .prism_cache/)
python prism_extract.py --ckpt out-A/ckpt.pt --out .prism_cache/A
# 2) accelerate on B with A's fingerprint; compare to a plain B run
python prism_accelerate.py --teacher_ckpt out-A/ckpt.pt --out_dir out-B-prism \
    -- --dataset=B --max_iters=1500
```
Better: add a `--cross_teacher` mode to `prism_eval.py` (teacher trained on A, student on
B, scored on B) so it emits a conforming artifact with the prism_score. **Win:** the
prism_score / Δloss on B with A's fingerprint ≈ with B's own teacher. The overlap-0 +
far-corpus results already hint it holds — this makes it the explicit claim. If it holds,
δ-as-OOD-detector, free-half-of-cross-size-transfer, and the continual anchor all follow.

### 2. Truly-far modality (where the story could break)

**Hypothesis:** the spectral transfer + the arc synergy hold within a modality but
degrade at a real modality boundary (English → **code** or another language). This is the
one place the plain recipe might finally need Leo's geometric levers (grassmann/topk/CKA,
already merged as opt-in flags).

**Gotcha to solve first:** the current vocab is Shakespeare's 65 chars; `far.txt`
(Sherlock) shares it. Real code / other languages use characters outside that vocab, so
`_encode_corpus` drops them and destroys the structure. You must either (a) pick a
far corpus over a compatible alphabet, or (b) rebuild the dataset + teacher + models on a
**new, larger vocab** that covers the far corpus (a bigger change — retrain everything).
Flagged as the first design decision. Then rerun `prism_eval.py --far_corpus` and
`prism_arc_eval.py` at that distance and find where Δloss / the arc synergy fall off.

### 3. The continuous single-run (vs sequential)

**Hypothesis (from the unified-arc plan):** one trajectory that pretrains then adapts with
a *morphing* target (teacher-spectrum early via init → self late via a frozen
`prism_unfold` target) matches the sequential pipeline at equal compute — likely *parity*
(the value is operational elegance), not new synergy. Worth one clean run to confirm
parity and rule out a hidden win. Build it as: `prism_init` + `prism_unfold=N` during
phase A, then freeze the target (`prism_unfold=0`) and switch dataset to B, one run.

### 4. Continual A → B → C (multi-domain accumulation)

**Hypothesis:** re-anchor to self after each domain → learn A, then B, then C, retaining
all prior domains. This is Result-2's anchor applied repeatedly = a continual-learning
method sequential single-finetune can't match. Needs a 3rd char corpus (see #2's vocab
gotcha). Metric: retention across ALL prior domains + speed of each new acquisition.

### 5. The "unfold curve" (the Σ\* / Hutter-adjacent test) — FIRST CUT ALREADY RUN

**Hypothesis:** most of a modality's compressibility is a tiny shared prior. Build
size-bounded priors of increasing size, fit on corpus A, and measure held-out
cross-entropy (bits/byte) on a *disjoint* corpus B (same modality); find the knee.

**A CPU first cut has been run** (Shakespeare prior → held-out Sherlock; n-gram priors
by order, size = bz2 bytes):

| prior | size | held-out bits/byte |
|---|---|---|
| order-0 | 367 B | 5.23 |
| order-1 | 3.9 KB | 4.36 |
| order-2 | 27 KB | 3.77 |
| **order-3** | **112 KB** | **3.50 (best)** |
| order-4 | 339 KB | 3.58 |
| order-5 | 893 KB | 3.95 |

**The finding — a knee that then *reverses*.** A tiny shared prior genuinely unfolds
never-seen same-modality text (5.23 → 3.50), but past ~112 KB **bigger priors get
*worse* on held-out text** — high-order Shakespeare n-grams are corpus-specific and
don't transfer. So there is an *optimal shared-prior size, and it's small*: ~⅔ of
English text's compressibility is a shared modality prior you can unfold from a tiny
dictionary; ~⅓ is irreducibly corpus-specific. That's the universal-structure /
specific-content split, shown classically (no PRISM). Harness: `unfold_curve.py`
(stdlib-only, ~130 lines; was written to the session scratchpad — rebuild or recover it).

**Where PRISM earns its slot (the next arm):** the classical prior tops out at 3.5
because n-grams are a *weak* universal model. Swap in a **stronger tiny shared prior** —
a small frozen neural LM, or Σ\* seeding an online nanoGPT — and see if it pushes the
knee *down and left* (more compressibility captured as "shared"). **The decisive test:
does 128 B of Σ\* spectral geometry beat 128 B of n-gram at the tiny end?** If yes, PRISM
earns a slot in compression *and* it's a direct "Σ\* is a modality constant" test. Then
the enwik8-scale version. Not a Hutter entry. Trap to avoid: the lossy keypad-digit
"literal T9" ≤ direct coding (data-processing inequality) — demo only. Full spec in
`project_prism_sigma_star`.

**Framing spec (2026-07-28, binding for any publication of this experiment):**
the "x = 0" result MUST be drawn and written as two acts, or not at all.
**Act 1 — the dot:** labeled "sparse byte n-gram prior alone (CPU counting
pass)". It is prior-dominated by our own measurement (Shakespeare fused-init
2.68 vs prior-alone 2.70 — geometry ≈ rounding error at init) and is NOT a
PRISM claim; its honest comparators are classical compressors, with the table
size stated. **Act 2 — the trajectory:** "same prior + PRISM geometry,
trained" — the PRISM claim lives here and only here: the fused curve leaves
the prior's floor and keeps improving where a pure prior cannot (the
below-the-floor break from the 30× study). "Reached-at-init" / speedup
language attaches to the fused *system trajectory* with the prior explicitly
credited, never to the lone dot. A single point drawn where trained models
end, without these labels, reads as "a trained model that skipped training"
— the exact wrong reading. Decision gate: the free CPU counting pass runs
first; Act 1 landing ~1.80–1.95 bpb on enwik8 val (block-edge backoff fix
required — it is worth ~0.09 bpb at context-5) is what justifies paying for
Act 2 at all.

**Language rule (same binding force):** the phrases "reached at init",
"unbounded speedup", and "already sits at the baseline's final loss" are
banned from any public artifact of this experiment — they get screenshotted
without their parentheticals. The canonical public sentence is: *"A sparse
byte n-gram prior already reaches X bpb on the held-out tail (Act 1). Adding
the PRISM geometry and training produces a trajectory that continues below
that prior floor (Act 2)."* Anything shorter leaks the banned reading.

**Act 1 RAN (2026-07-28, CPU, $0):** val 1.9061 bpb, orders 0–6
interpolated, tuned on dev only
(`results/ngram_floor_20260728T202028Z.json`, tool `src/ngram_floor.py`)
— below xz (1.99) and below the 1,500-step baseline's final val (1.9315).
**Gate PASSED; Act 2 (the fused trajectory) is authorized when funded** —
it needs the training-time sparse gather + block-edge backoff in
`train.py`'s `_prior_logp`, then `prism_prior_eval` arms on enwik8.

### 6. The reach-at-init moonshot (from the 30× hybrid to 1000×)

[Prior-Fused PRISM](PRIOR-FUSED-PRISM.md) already showed the hybrid at **30×**. The
literal **1000×** — the fused model at baseline quality *before any training* — needs the
shared prior to sit *clearly below* the baseline's loss so the fused init already beats it.
Context-3 tops out *at* baseline (2.57 ≈ 2.565 bits/char), a knife-edge, and the `logit_gate`
that would enable reach-at-init backfired (throttled early learning, 30× → 15×). **The one
missing piece: a context-4+ n-gram prior**, which is clearly below baseline but needs a
**sparse build** — the dense V⁴ table (65⁴×65) is too big and `build_ngram_prior.py` times
out on it. Build a sparse (hashed-context) n-gram + a sparse gather in `train.py`'s
`_prior_logp`, confirm the prior dips below baseline on val, then re-run
`prism_prior_eval.py` with the gate *off* (it hurt) and read the reach-at-init speedup on
the `prior` / `prism_prior` arms. Also fix the block-edge inflation (first C-1 positions
per window lack in-window context → currently uniform; back off to a lower order instead).

## Ground rules (again, because they matter)

A number in these docs has a matching committed `results/*.json`. Attribution needs
matched schedules (only then does "only X differs"). Probe wide/shallow first (warmup ≪
steps). 3 seeds for a result; a single seed is a probe. Partial/crashed runs must raise,
never be scored. When you launch a Modal run, watch it, fetch the artifact, and **commit
it** — that's the evidence. And own your errors plainly; this project killed its own
"spectral finetune" hypothesis with a placebo, and overturned an author's prior on the
arc. Truth over hype.
