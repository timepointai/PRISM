# PRISM

**Take the geometry of a trained neural network — which subspaces it uses, how
its energy is distributed — and hand it to a fresh model at initialization,
then leave it alone. On the canonical byte-modeling benchmark, that model
reaches from-scratch training's final quality at 42% of the GPU time, and at
equal time converges far below anything from-scratch training ever reaches.
No weights copied, no data shared. Only geometry.**

PRISM extracts that geometry, transplants it, and measures — one committed
artifact per claim — which parts of it carry the effect. This repo has killed
several of its own hypotheses on the way; everything below is what survived.

*(Formerly `nanogpt-prism-shakespeare`, now archived — full history
[there](https://github.com/timepointai/nanogpt-prism-shakespeare); development
continues here.)*

<img src="assets/prism-flashlight.svg" alt="A spectrographic flashlight for models: a trained checkpoint's raw weights enter a prism as white light and split into spectral bands — attention, FFN up, FFN down, embedding — each carrying a spectrum and directions. A reversed prism recombines the bands into a fresh model that trains several times faster. Geometry crosses; content never does." width="100%">

<a href="https://timepointai.github.io/PRISM/docs/how-prism-works.html"><img src="assets/prism-explainer-button.svg" alt="HOW PRISM WORKS — THE VISUAL EXPLAINER: the flashlight metaphor, node-level math, graph-level transfer, and the measurements" width="100%"></a>

---

The bench is deliberately small — nanoGPT-class models (10.65M params), three
seeds for every claim, one L4 — and the corpora run from Shakespeare through a
committed slice of 2024 web text to enwik8. Every number below has an artifact
in [`results/`](results/). The bench is just the pointing hand: **the claims
are about the finger, not what it happens to be pointing at.**

## What the evidence says

### 1. Geometry-at-init beats training from scratch — outright, on real benchmarks

Inject a trained model's directions and spectrum into a fresh model at
initialization, then train it with *no further intervention* (`dirs_only`).
On two corpora where the baseline is at full strength:

- **2024 web text** (byte-level FineWeb-Edu, corpus committed with provenance):
  reaches the baseline's best quality **3.3×** faster (2.9–3.9, resolved, 3/3
  seeds) and **converges 0.11 nats below anything the baseline ever reaches** —
  against a baseline that never overfits ([`RESULTS.md`](RESULTS.md) §11).
- **enwik8** (the Hutter-Prize corpus, standard 90M/5M split): **2.90×**
  (2.72–3.10) to baseline-best, converging **0.152 bits/byte below it**
  (1.780 vs 1.932 bpb). The 90M pool means the student never repeats a byte —
  zero epochs, the least hospitable setting for any memorization explanation.
  It passes the baseline's *final* quality around step 500 and leads by 1.40
  bpb at step 100 ([`RESULTS.md`](RESULTS.md) §14).

### 2. That converts into an energy frontier where every point dominates

Because the transplant does its work at initialization, you can simply train
for less time. Complete short schedules (full warmup and LR decay, not
truncated runs), wall-clock recorded inside each artifact, enwik8:

| arm | bits/byte | vs. baseline's best | GPU time |
|---|---|---|---|
| baseline, 1,500 steps | 1.9315 | — | 414s (100%) |
| **PRISM-init, 600 steps** | 1.9031 | **−0.028** | **172s (42%)** |
| **PRISM-init, 800 steps** | 1.8595 | **−0.072** | **230s (56%)** |
| PRISM-init, 1,500 steps | 1.7796 | −0.152 | 403s (97%) |

Every row beats the baseline's best on every seed; the 600-step *baseline*
control (1.75 nats) shows the short schedule alone does nothing. The teacher
that supplies the geometry is a **one-time mined cost that amortizes**: a
fingerprint mined on the web-text corpus accelerates enwik8 students at
**1.97×**, still converging 0.122 bpb below their baseline — one mining run
serves many corpora ([`RESULTS.md`](RESULTS.md) §16).

### 3. It's the finger, not what it points at

Four escalating experiments built to kill the "the teacher leaked its
content" explanation. All four failed to kill the effect — the transfer is
**structural, not content**:

- **Same corpus, disjoint data:** teacher and student share 100% or 0% of
  their training text — the head start is identical (Δloss 0.57–0.59 either
  way; [`RESULTS.md`](RESULTS.md) §3).
- **Different corpus:** move the student to Sherlock Holmes and score it on
  held-out Sherlock — the advantage *grows* slightly (0.591 → 0.627; §4).
- **A teacher that never saw the corpus:** its fingerprint still delivers
  **4.6×** (vs. 9.9× same-corpus, identical rig) at ~80% of the quality gain
  (§10).
- **Reuse across benchmarks:** the web-text fingerprint on enwik8, above —
  1.97×, still better-than-baseline convergence (§16).

### 4. The directions carry it — and they must be mined, not drawn

The ablation triad answers *which part* of the geometry does the work, and
*where it can come from*:

- **Trained directions win** (finding 1). Remove them and keep everything
  else, and the model sits **below baseline at every step** (§11).
- **Random directions lose** (§11). **Never-trained *structured* directions
  also lose**: analytic DCT bases — smooth, orthonormal, mathematically
  ideal — give a slightly better step-0 loss and then fall behind for the
  rest of training, 3/3 seeds (§15). Well-formedness is not the load-bearing
  property; **training is**. Geometry is *mined* from trained models, not
  *drawn* from mathematics — which makes trained checkpoints a reusable
  commons, and mining + reuse (finding 2) the economically interesting
  operation.
- The effect tracks the mine's quality exactly: it grows with teacher
  training and **saturates right where the teacher converges** (~2,000 steps
  here; longer teachers add nothing), and a barely-trained teacher's geometry
  is *worse than random init* (§5).

### 5. The spectrum is the universal part — honestly priced

The spectrum half of the geometry compresses to **40 numbers** (5 weight
groups × 8 DCT coefficients, ~160 bytes) and crosses model size, tokenizer,
and corpus *simultaneously*: a fingerprint read off **OpenAI's public GPT-2
weights** ([`fingerprints/gpt2-124M/`](fingerprints/gpt2-124M/)) performs at
parity with a purpose-trained teacher's spectrum (1.747 vs 1.762, GPT-2's
marginally better). The honest other half: in spectrum-*only* mode both sit
below baseline — universality is proven, sufficiency is refuted (§11). The
universal init worth building is directional (see What's next).

### 6. Pointed backward, the same machinery retains: ~10× less forgetting

Anchor a trained model to its **own** pre-finetune weights with one line per
optimizer step — `W ← (1−s)·W + s·W₀` — and finetuning on a new domain
forgets **up to ~10× less** while still genuinely learning (the anchored
model beats a from-scratch ceiling on the new domain). It Pareto-dominates
the low-learning-rate alternative (~2× the retention at equal adaptation),
and the attribution matches finding 4: anchoring only the spectrum does
nothing (1.07×), a wrong-spectrum placebo actively harms (0.39×) — it is the
directions being pinned that retains
([`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md)). And the two
uses compound: a PRISM-*pretrained* base finetunes with **≈ zero forgetting
and ~8% better adaptation** at matched quality
([`docs/UNIFIED-ARC.md`](docs/UNIFIED-ARC.md)).

### 7. How hard you pin decides what you can learn — a dial with measured bands

The per-step pull is a **continuous dial between plasticity and retention**,
and three studies price its bands:

- **From scratch on scarce data, full strength is the win:** on 1M-token
  Shakespeare every baseline overfits (1.78 → ~2.31); the pull toward the
  spectral target holds ~1.66–1.67 with no overfitting, reaching baseline
  quality **11.8×** faster as tuned, **7×** schedule-matched
  ([`RESULTS.md`](RESULTS.md) §1–2). On adequate data the same pull costs the
  entire endpoint lead — drop it and init-only wins (§11).
- **Injecting new facts, retention-grade strength blocks learning outright:**
  2–3% closed-book recall vs 98–100% unanchored — while posting the *best*
  validation loss of any arm. **Loss is not an injection metric**; only
  exact-match caught it. The weak band (s≈0.0025) injects at parity with
  plain training while forgetting **3× less** (§12).
- **On a scored task** (a catalog-review workflow with rule-generated ground
  truth, evaluated on never-seen cases), the weak band **Pareto-dominates
  low-LR on both axes**: 97% vs 88% on novel cases at half the forgetting
  (§13). Side finding: facts store in *either* attention or FFNs when the
  other is pinned — storage is not FFN-exclusive at this scale (§12).

### 8. Geometry compounds with statistics

Fuse a fixed shared n-gram prior into the logits (product of experts, so the
model learns only the residual) and stack PRISM on top: PRISM alone 15×, the
n-gram alone 3.8×, **the hybrid 30× — at the best loss of all four arms**,
below even the n-gram's own floor. Single-seed probe; the documented path to
reaching baseline quality *at initialization* runs through a context-4+
sparse prior ([`docs/PRIOR-FUSED-PRISM.md`](docs/PRIOR-FUSED-PRISM.md)).

### The ledger

| study | finding | evidence |
|---|---|---|
| **Modern-web transfer** | geometry-at-init beats a non-overfitting baseline: **3.3×**, **−0.11 nats** beyond its best | [`RESULTS.md`](RESULTS.md) §11 |
| **enwik8** | same effect on the canonical bench: **2.90×**, **−0.152 bpb**, zero-epoch regime | [`RESULTS.md`](RESULTS.md) §14 |
| **The energy frontier** | baseline's full quality at **42%** of the GPU time; −0.072 bpb at **56%**; teacher fingerprint amortizes across corpora at **1.97×** | [`RESULTS.md`](RESULTS.md) §16 |
| **Structure, not content** | head start identical at 100% and 0% data overlap; grows on a different corpus | [`RESULTS.md`](RESULTS.md) §3–4 |
| **Teacher-free init** | a teacher that never saw the corpus keeps **4.6×** of 9.9× and ~80% of the quality gain | [`RESULTS.md`](RESULTS.md) §10 |
| **Attribution: directions** | spectrum-only below baseline; trained directions required — random *and* never-trained analytic bases both refuted | [`RESULTS.md`](RESULTS.md) §11, §15 |
| **Spectrum universality** | GPT-2's 40 public-weight numbers ≈ a native teacher's spectrum across size, tokenizer, corpus | [`RESULTS.md`](RESULTS.md) §11 |
| **From-scratch, scarce data** | **11.8×** tuned / **7×** matched with no overfitting, where every baseline overfits | [`RESULTS.md`](RESULTS.md) §1–2 |
| **Finetune retention** | directional self-anchor → up to **~10× less forgetting**, Pareto-dominating low-LR | [`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md) |
| **The arc** | PRISM-pretrained base: **≈0 forgetting, ~8% better adaptation** at matched quality | [`docs/UNIFIED-ARC.md`](docs/UNIFIED-ARC.md) |
| **Fact injection** | the anchor is an injection↔retention dial: 0.01 blocks storage (2–3% recall), the weak band injects at parity with **3× less** forgetting; LM loss can't see any of it | [`RESULTS.md`](RESULTS.md) §12 |
| **Task specialization** | on a scored workflow, the weak band Pareto-dominates low-LR: 97% on novel cases at half the forgetting | [`RESULTS.md`](RESULTS.md) §13 |
| **Prior-Fused PRISM** | geometry × statistics: **30×** hybrid at the best loss of all arms (single-seed probe) | [`docs/PRIOR-FUSED-PRISM.md`](docs/PRIOR-FUSED-PRISM.md) |

Every number has a matching committed `results/*.json` (loss curves,
provenance, git commit, argv, wall-clock, censoring flags). A claim without an
artifact is not a result in this repo.

<img src="assets/prism-transfer.svg" alt="Left: validation loss at step 100 versus teacher/student data overlap, with the student scored on its own data mixture — the from-scratch baseline sits at ~2.47 while the PRISM recipe sits at ~1.88, and the gap grows from 0.591 at full overlap to 0.627 when the student trains and is scored entirely on Sherlock Holmes. Right: the advantage versus teacher training steps — negative at a 100-step teacher, rising monotonically and saturating at a +0.46 plateau around 2,000 steps, where the teacher itself converges; 4k and 8k teachers are flat." width="100%">

## How it works

Every weight matrix of a trained model factors as `W = U Σ Vᵀ`: the
**directions** (U, V — which subspaces the model decided were worth using) and
the **spectrum** (Σ — how much energy it put in each). Together they describe
the model's *organization* without describing anything it *knows*. PRISM is
three operations on that geometry, with roles the ablations have priced:

1. **EigenTransfer** — blend the fresh model's singular vectors 75% toward the
   teacher's, then re-orthogonalize. *The active ingredient — and the teacher
   must be a trained model: analytic and random substitutes both fail.*
2. **Spectral Imprint** — compress the teacher's singular-value distribution
   to 8 DCT coefficients per weight group (**40 numbers, ~160 bytes**) and
   reshape the fresh model's spectrum to match at init. *Universal across
   models and corpora; supporting role.*
3. **Mod Wheel** — after every optimizer step, pull the weights gently back
   toward a stored target: `W ← (1−s)·W + s·W_target`. *An overfitting brake
   (forward, scarce data) and the retention anchor (backward), with a
   measured strength dial.*

No parameters are copied. The student learns its own content from scratch — it
just doesn't spend its first several hundred steps rediscovering *which
subspaces a trained transformer uses*.

<img src="assets/prism-method.svg" alt="How PRISM works: SVD a trained teacher into directions (U, V) and a spectrum, compress the spectrum to 8 DCT coefficients, inject both into a fresh student, then optionally hold the student toward the spectral target with the mod wheel." width="100%">

<img src="assets/prism-imprint.svg" alt="Deriving the spectral imprint: SVD each trained weight matrix, normalize and average the singular values per group, then least-squares fit 8 cosine (DCT) coefficients. The plot overlays a real group-averaged spectrum against its reconstruction from just 8 coefficients, mean absolute error about 0.03." width="100%">

The same parts run in two directions:

- **Forward (transfer):** inject the geometry at init. Directions do the
  lifting; the spectrum is the cheap universal part; add the wheel only when
  the corpus is small enough to overfit. The fingerprint can be mined once
  and reused across corpora at ~⅔ strength.
- **Backward (retain):** the wheel again, anchored to the model's **own**
  pre-finetune weights — pinning the directions is what prevents forgetting,
  and the strength `s` is the plasticity dial (weak band ≈0.0025 for
  injecting new material, ~0.01–0.02 for maximum retention):

```
W  ←  (1 − s)·W  +  s·W₀        # after each step; W₀ = pre-finetune weights
```

<img src="assets/prism-result.svg" alt="Validation loss over training, nanoGPT Shakespeare, three seeds each. The baseline (LR 1e-3) falls to ~1.78 near step 1,400 then overfits to ~2.30 by step 5,000. The PRISM recipe at LR 5e-4 holds ~1.66; the recipe run at the baseline's own LR of 1e-3, where only the spectral flags differ, holds ~1.67 and also never overfits." width="100%">

## Use it

Two entry points — one per direction. **Transfer** into a fresh model:

```bash
cd src
python prism_accelerate.py \
    --teacher_ckpt path/to/trained/ckpt.pt \
    --out_dir out-accelerated \
    -- --max_iters=2000 --dataset=your_dataset
```

What the measurements say you should know:

- **Match the Mod Wheel to your data regime.** Corpus small enough to
  overfit → keep the recipe as-is (the wheel is the no-overfitting brake).
  Adequate data → append `-- --prism_mod=0.0`: the init-only configuration is
  what beat the modern-web and enwik8 baselines outright.
- **The teacher must be trained** — to convergence and no further; a weak
  teacher is worse than random init. It need *not* be trained on your corpus
  (~⅔ strength survives reuse), but it must exist: analytic direction bases
  don't work.
- Teacher and student must share the same architecture — directional transfer
  is dimension-specific. Cross-size projection is the open frontier.

**Finetune** a trained model without forgetting:

```bash
cd src
python prism_finetune.py \
    --base_ckpt path/to/trained/ckpt.pt \
    --new_data your_new_dataset --retain_val your_old_dataset \
    --out_dir out-finetuned --mod 0.01 --ft_steps 1000
# add --plain to run the same finetune with the anchor off, for comparison
```

`--mod` is the retention/plasticity dial: ~0.01–0.02 to protect an old domain
while adapting style, **≈0.0025 when the finetune must inject genuinely new
material** (facts, bindings, task rules) — at 0.01 the anchor blocks new
content entirely while the loss curve looks great, so score injection with
exact-match probes, never loss alone.

## Start here (humans and agents)

```bash
git clone https://github.com/timepointai/PRISM.git
cd PRISM/src
pip install torch numpy transformers tiktoken datasets
python prism_selftest.py          # 30 offline invariant tests, CPU, ~1 min — start here
python prism_eval.py --teacher_steps=10 --student_steps=10 --eval_every=5 \
  --eval_iters=2 --seeds=1337 --batch_size=4     # tiny end-to-end smoke, CPU, ~1 min
```

Ground rules the tooling enforces, worth knowing before you run anything real:

1. **Evidence = `results/*.json`.** Every run writes a full artifact. A claim
   without a committed artifact is not a result.
2. **Reading a score:** `prism_score` is a ratio — always read it against
   `baseline_best` (a weak baseline inflates it). `left_censored: true` is a
   floor, not a measurement. `schedule_matched: true` means only the PRISM
   flags differed — the attribution-grade comparison.
3. **Runs are resumable:** stages bank to `.prism_runs/<run-key>/`; the same
   command resumes, any changed knob gets a fresh key. Partial or crashed runs
   raise — they are never scored.
4. **Probes must keep `warmup ≪ student_steps`**, or the warmup flattens both
   arms and voids the comparison.
5. **One variable at a time.** Every knob (`--corpus`, `--fingerprint`,
   `--cross_teacher`, `--prism_anchor_exclude`, …) folds into the run key;
   test each against the plain configuration on an otherwise-identical rig.

A machine-readable summary of the method and measurements is embedded at the
bottom of the [visual explainer](https://timepointai.github.io/PRISM/docs/how-prism-works.html) —
a standalone, self-contained page served from GitHub Pages; source:
[`docs/how-prism-works.html`](docs/how-prism-works.html).

## Honest bounds on the claims

- **Scale.** 10.65M params, corpora of 1M–90M tokens. Nothing here has met a
  production model. The GPT-2 extraction crosses into a 124M model for the
  spectrum only. Absolute bpb figures are probe-horizon numbers, not
  literature-competitive.
- **Energy is proxied by wall-clock on one GPU class** (no power metering),
  and the frontier's low end is unmapped below 600 steps. The amortization
  constant — how many corpora one fingerprint usefully serves, as a function
  of distance — has two datapoints (Sherlock↔Shakespeare, modernweb→enwik8).
- **The init-comparison space is partially explored:** trained directions
  win; random and analytic-DCT directions fail. Other structured inits
  (μP-style scalings, orthogonal ensembles at matched spectra) haven't run.
- **The Mod Wheel's decay schedule was tuned on Shakespeare** and never
  retuned for the adequate-data benches; the mirror cell (init-only on
  Shakespeare) hasn't run.
- **The 30× is a single-seed probe** measuring the compounding of two free
  priors; the defensible claim is the hybrid beats PRISM alone at the best
  loss.
- **"Different corpus" tops out at English text** (Sherlock, 2024 web,
  Wikipedia XML). A genuinely different modality — code, another language —
  is untested, though the byte-level benches make it runnable.
- **The fact/task dial is measured at probe scale** in a brutal ~200-epoch
  memorization regime, with high seed variance on unseen-phrasing recall and
  the dial's shape between 0.0025 and 0.01 unmapped.
- **Finetune-retention shows old-domain retention**, not new-domain overfit
  prevention, and trades a little late adaptation for it.

## Reproduce it

**The headline results (Modal, headless, isolated volume per experiment):**

```bash
pip install modal && modal setup                            # one-time auth
modal run --detach prism_modal_modernweb.py --arm dirs      # §11 winner: init-only, modern web
modal run --detach prism_modal_enwik8.py                    # §14: dirs_only on enwik8
modal run --detach prism_modal_energy.py \
  --extra "--method=dirs_only --teacher_steps=2000 --student_steps=600"   # §16 frontier point
modal run --detach prism_modal_cold.py                      # §15: the never-trained control
```

**The dial studies:**

```bash
modal run --detach prism_modal_facts.py      # §12: fact injection, closed-book recall
modal run --detach prism_modal_catalog.py    # §13: scored-task specialization
modal run --detach prism_modal_teacherfree.py --arm cross    # §10: teacher-free init
modal run --detach prism_modal_teacherfree.py --arm control
```

**Local probes** (any GPU, or CPU for smokes) and the finetune-retention /
arc / prior-fusion benchmarks: commands in [`RESULTS.md`](RESULTS.md) and the
docs. Everything is stepwise-resumable and refuses to score partial runs.

## What's next

The ranked, run-command-ready list lives in
[`docs/NEXT-EXPERIMENTS.md`](docs/NEXT-EXPERIMENTS.md). Highlights:

1. **Cross-size directional projection** — with the spectrum priced as
   universal-but-weak and analytic directions refuted, the universal init
   worth building projects a *large trained model's* subspaces down to a
   small one. The most differentiated payoff on the board.
2. **The dial's production test** — the weight-space anchor as a
   zero-forward-pass alternative to KL-to-reference in RL post-training
   (specced at ~$500 on a 9B with prime-rl), plus the low-LR × weak-anchor
   combination cell.
3. **The reach-at-init moonshot** — a context-4+ sparse n-gram prior below
   baseline at init; the path from the 30× hybrid toward passing the eval
   before training starts.
4. **A truly far modality** — the byte-level benches cover code and other
   languages; find where structural transfer degrades.
5. **Frontier bookkeeping** — the sub-600-step low end, the wheel 2×2 mirror
   cell, a second amortization distance, and per-run power metering.

## Repo map

```
README.md                     ← you are here
docs/how-prism-works.html     ← the visual explainer (standalone, self-contained)
docs/FINETUNE-RETENTION.md    ← finetune without forgetting — study + frontier
docs/UNIFIED-ARC.md           ← PRISM pretraining + finetuning compound — the arc
docs/PRIOR-FUSED-PRISM.md     ← T9 × PRISM — the n-gram prior compounds (30×)
docs/NEXT-EXPERIMENTS.md      ← the ranked next experiments + run commands
WHITEPAPER.md                 ← method + experiments in full
RESULTS.md                    ← every committed run and what it does / doesn't show
results/                      ← eval artifacts. the evidence.
archive/                      ← v0.0 / v0.1 writeups + retired notebooks
fingerprints/gpt2-124M/       ← 40 numbers read off OpenAI's public GPT-2 weights
fingerprints/cold-dct/        ← the never-trained fingerprint (refuted control, §15)
data/far.txt                  ← Sherlock Holmes — the far corpus
data/modernweb/               ← FineWeb-Edu slice (2024 web text, byte-level)
data/enwik8/                  ← the Hutter corpus, standard split (downloaded on prepare)
data/facts/                   ← 120 invented facts + paraphrase templates (§12)
data/catalog/                 ← rule-generated catalog-review task (§13)
prism_modal.py                ← headless Modal runner — the original Shakespeare runs
prism_modal_modernweb.py      ← modern-web decomposition (4 arms)
prism_modal_enwik8.py         ← enwik8 bench
prism_modal_energy.py         ← the energy-frontier passes (enwik8 volume)
prism_modal_cold.py           ← the never-trained-directions control
prism_modal_teacherfree.py    ← teacher-free init probe
prism_modal_facts.py          ← fact-injection dial
prism_modal_catalog.py        ← scored-task specialization
prism_modal_finetune.py       ← finetune-retention benchmark
prism_modal_arc.py            ← the arc benchmark
prism_modal_prior.py          ← Prior-Fused (T9 × PRISM) benchmark
prism_modal_leo.py            ← geometric-lever experiments
unfold_curve.py               ← exploratory: the Σ*/"unfold curve" first cut
src/prism_eval.py             ← the from-scratch benchmark (all transfer knobs)
src/prism_finetune_eval.py    ← finetune-retention + injection benchmark (dial arms)
src/prism_arc_eval.py         ← the arc benchmark (matched-quality bases)
src/prism_prior_eval.py       ← the Prior-Fused benchmark (4 arms)
src/fact_recall.py            ← closed-book exact-match recall scorer
src/gen_cold_directions.py    ← synthesizes the never-trained DCT direction frames
src/build_ngram_prior.py      ← builds the shared n-gram prior ("the T9 dictionary")
src/prism_accelerate.py       ← apply PRISM to any checkpoint (transfer entry point)
src/prism_finetune.py         ← finetune any checkpoint without forgetting
src/prism_init.py             ← EigenTransfer + Spectral Imprint + Mod Wheel
src/prism_extract.py          ← extract a fingerprint from any checkpoint (or --hf gpt2)
src/prism_selftest.py         ← 30 offline invariant tests — run before any GPU spend
```

## License

MIT — see [LICENSE](LICENSE).

A standalone clone of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej
Karpathy (MIT, © 2022), not a fork. `model.py`, `configurator.py`, `bench.py`,
`sample.py`, and the `data/` preparers are his; `train.py` is his with PRISM
hooks added (all off by default — a scratch run is byte-identical to nanoGPT).
The PRISM code (`prism_*.py`, `fact_recall.py`, `gen_cold_directions.py`,
`build_ngram_prior.py`, `config/prism_*.py`) is Timepoint Labs', under the
same terms.

---

*A [Timepoint Labs](https://timepointai.com) project by [Sean McDonald](https://x.com/seanmcdonaldxyz).*
