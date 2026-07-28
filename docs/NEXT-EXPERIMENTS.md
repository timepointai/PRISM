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
