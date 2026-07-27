# PRISM

**Run an SVD over a trained neural network's weights and compress the shape of
what training built into a ~128-byte fingerprint. That fingerprint — not the
weights, not the data — is where the trainability lives.** PRISM extracts it,
transplants it, and measures what it carries. It turns out to carry a lot.

*(Formerly `nanogpt-prism-shakespeare`, now archived — full history
[there](https://github.com/timepointai/nanogpt-prism-shakespeare); development
continues here.)*

<img src="assets/prism-flashlight.svg" alt="A spectrographic flashlight for models: a trained checkpoint's raw weights enter a prism as white light and split into four spectral bands — attention, FFN up, FFN down, embedding — each carrying a spectrum and directions. A reversed prism recombines the bands into a fresh model that trains about 12 times faster. Geometry crosses; content never does." width="100%">

<a href="https://timepointai.github.io/PRISM/docs/how-prism-works.html"><img src="assets/prism-explainer-button.svg" alt="HOW PRISM WORKS — THE VISUAL EXPLAINER: the flashlight metaphor, node-level math, graph-level transfer, and the measurements" width="100%"></a>

---

Everything below is measured on a deliberately small bench — nanoGPT char-level,
10.65M params, three seeds, one L4 — and every number has a committed artifact in
[`results/`](results/). The bench is just the pointing hand: **the claims are
about the finger(print), not what it happens to be pointing at.**

## What's interesting

### 1. Two tiny priors compound to 30× — and the best loss of anything measured

Stack PRISM's *geometric* prior on a fixed *statistical* one — a shared n-gram
prior fused into the logits as a product of experts, so the model only learns the
residual — and the two multiply: PRISM alone 15×, the n-gram alone 3.8×, **the
hybrid 30× to baseline quality, at the best final loss of all four arms** — below
the baseline, below PRISM alone, below the n-gram's own floor. Geometry is what
breaks the residual through the statistical ceiling. (Single-seed probe; the
documented path from here to a literal reach-baseline-*at-init* 1000× needs a
context-4+ sparse prior — see [What's next](#whats-next).) Full study:
[`docs/PRIOR-FUSED-PRISM.md`](docs/PRIOR-FUSED-PRISM.md).

### 2. ~128 bytes of geometry trains a fresh model ~12× faster

Hand a fresh model nothing but a trained one's spectral fingerprint — no weights
copied, no data shared — and it reaches the from-scratch baseline's best quality
in ~100 steps instead of ~1,200 (**11.8× median**, resolved, not a bound; **7×**
with the schedule held identical so *only* the spectral flags differ). And it
**stops overfitting**: through 5,000 steps the baseline collapses from 1.78 to
~2.31 on every seed; the fingerprinted model holds. The advantage tracks the
teacher's geometry exactly — it grows with teacher training and saturates right
where the teacher itself converges; a barely-trained teacher's geometry is noise
imprinted with authority, *worse* than random init.
Evidence: [`RESULTS.md`](RESULTS.md) §1–2, §5.

### 3. It's the finger, not what it points at

Three experiments built to kill the "the teacher leaked its content" explanation,
in escalating order:

- **Same corpus, disjoint data.** Teacher and student get overlapping or disjoint
  halves of 100 random blocks (difficulty-controlled). The head start is
  **identical at 100% and 0% overlap** ([`RESULTS.md`](RESULTS.md) §3).
- **Different corpus.** Move the student to Sherlock Holmes and score it on
  held-out *Sherlock*: the advantage doesn't shrink — it *grows* slightly
  (Δloss 0.591 → 0.627; [`RESULTS.md`](RESULTS.md) §4).
- **Teacher never sees the corpus at all.** Train the teacher *exclusively* on
  Sherlock, transplant its ~128-byte fingerprint into a Shakespeare student: it
  still trains **4.6×** faster at near-identical final quality — about half the
  speedup and ~80% of the quality gain of a same-corpus teacher on the identical
  rig ([`RESULTS.md`](RESULTS.md) §10).

So the fingerprint is not "a Shakespeare model, compressed." It is substantially
a property of the *modality* — transferable, data-independent structure — and a
matched teacher buys the remaining half of the speed, not the phenomenon.

### 4. Pointed backward, it retains instead of transfers: ~10× less forgetting

The same machinery, anchored to a trained model's *own* weights during a
finetune, cuts catastrophic forgetting by **up to ~10×** while the model still
genuinely learns the new domain — Pareto-dominating the lower-learning-rate
frontier (~2× more retention at equal adaptation). The attribution landed on a
negative to the obvious guess: retention is carried by the raw *directions*, not
the spectrum (an only-spectrum anchor does nothing; a wrong-spectrum placebo
harms). And the two uses **compound**: a base that was PRISM-pretrained finetunes
with **≈ zero forgetting** and adapts ~8% better at matched quality — attributably
the spectral geometry, not the schedule. Full studies:
[`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md),
[`docs/UNIFIED-ARC.md`](docs/UNIFIED-ARC.md).

### The ledger

| study | finding | evidence |
|---|---|---|
| **Prior-Fused PRISM** | geometry × statistics compound: **30×** hybrid at the best loss of all arms | [`docs/PRIOR-FUSED-PRISM.md`](docs/PRIOR-FUSED-PRISM.md) |
| **From-scratch transfer** | **11.8×** faster to baseline quality (tuned; **7×** schedule-matched), no overfitting | [`RESULTS.md`](RESULTS.md) §1–2 |
| **Structure, not content** | head start identical at 100% and 0% data overlap; grows on a different corpus | [`RESULTS.md`](RESULTS.md) §3–4 |
| **Teacher-free init** | a fingerprint from a teacher that **never saw the target corpus** keeps **4.6×** of 9.9× and ~80% of the quality gain | [`RESULTS.md`](RESULTS.md) §10 |
| **Finetune retention** | self-anchored Mod Wheel → up to **~10× less forgetting**, beating low-LR on the Pareto frontier | [`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md) |
| **The arc** | a PRISM-pretrained base finetunes with **≈0 forgetting, ~8% better adaptation** at matched quality | [`docs/UNIFIED-ARC.md`](docs/UNIFIED-ARC.md) |

Every number has a matching committed `results/*.json` (loss curves, provenance,
git commit, argv, censoring flags). A claim without an artifact is not a result in
this repo.

<img src="assets/prism-transfer.svg" alt="Left: validation loss at step 100 versus teacher/student data overlap, with the student scored on its own data mixture — the from-scratch baseline sits at ~2.47 while the PRISM recipe sits at ~1.88, and the gap grows from 0.591 at full overlap to 0.627 when the student trains and is scored entirely on Sherlock Holmes. Right: the advantage versus teacher training steps — negative at a 100-step teacher, rising monotonically and saturating at a +0.46 plateau around 2,000 steps, where the teacher itself converges; 4k and 8k teachers are flat." width="100%">

## How it works

Every weight matrix of a trained model factors as `W = U Σ Vᵀ`: the
**directions** (U, V — which subspaces the model decided were worth using) and
the **spectrum** (Σ — how much energy it put in each). Together they describe the
model's *organization* without describing anything it *knows*. PRISM is three
small operations on that geometry:

1. **Spectral Imprint** — compress the teacher's singular-value distribution to
   8 DCT coefficients per weight group (**~128 bytes total** — the fingerprint)
   and reshape a fresh model's spectrum to match at init.
2. **EigenTransfer** — blend the fresh model's singular vectors 75% toward the
   teacher's, then re-orthogonalize.
3. **Mod Wheel** — after every optimizer step, pull the weights gently back
   toward a stored target: `W ← (1−s)·W + s·W_target`.

No parameters are copied. Only geometry. The student learns its own content from
scratch — it just doesn't spend its first thousand steps rediscovering *how a
transformer should be shaped*, and it doesn't drift out of that shape later
(which is what overfitting looks like, geometrically).

<img src="assets/prism-method.svg" alt="How PRISM works: SVD a trained teacher into directions (U, V) and a spectrum, compress the spectrum to 8 DCT coefficients, inject both into a fresh student, then hold the student toward the spectral target with the mod wheel." width="100%">

<img src="assets/prism-imprint.svg" alt="Deriving the spectral imprint: SVD each trained weight matrix, normalize and average the singular values per group, then least-squares fit 8 cosine (DCT) coefficients. The plot overlays a real group-averaged spectrum against its reconstruction from just 8 coefficients, mean absolute error about 0.03." width="100%">

The same three parts run in two directions, and the measurements split the
geometry cleanly between them:

- **Forward (transfer):** the **spectrum carries structure** — data-independent,
  portable across corpora, the thing findings 1–3 above are about.
- **Backward (retain):** the **directions carry content** — domain-specific, and
  *pinning* them is what prevents forgetting. The whole backward method is one
  line, applied after each optimizer step of a finetune:

```
W  ←  (1 − s)·W  +  s·W₀        # W₀ = the pre-finetune weights; s ≈ 0.01–0.02
```

Gradient descent still dominates each step, so the model keeps learning the new
domain — but it never drifts far from what it already knew. `s` is the
retention/plasticity dial.

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

It extracts the spectral fingerprint from the checkpoint, then trains a fresh
model initialized and regularized by it. Two things the measurements say you
should know:

- **The teacher must be trained.** A weak teacher is *worse than random init*;
  the advantage saturates once the teacher converges — train it to convergence,
  no further.
- Teacher and student must share the same architecture — the directional transfer
  is dimension-specific. Cross-size transfer is future work.

**Finetune** a trained model without forgetting:

```bash
cd src
python prism_finetune.py \
    --base_ckpt path/to/trained/ckpt.pt \
    --new_data your_new_dataset --retain_val your_old_dataset \
    --out_dir out-finetuned --mod 0.01 --ft_steps 1000
# add --plain to run the same finetune with the anchor off, for comparison
```

`--mod` is the retention/plasticity dial (higher = more retention, slightly less
adaptation); `--retain_val` scores the old domain alongside the new so you can see
the forgetting the anchor prevents.

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
   floor, not a measurement. `schedule_matched: true` means only the spectral
   flags differed — the attribution-grade comparison.
3. **Runs are resumable:** stages bank to `.prism_runs/<run-key>/`; the same
   command resumes, any changed knob gets a fresh key. You cannot accidentally
   resume one experiment onto another. Partial or crashed runs raise — they are
   never scored.
4. **Probes must keep `warmup ≪ student_steps`**, or the warmup flattens both
   arms and voids the comparison.
5. **One variable at a time.** Lever knobs (`--align_mode`, `--align_topk`,
   `--cka`, `--cross_teacher`, …) fold into the run key; test each against the
   plain recipe on an otherwise-identical rig, never stacked.

A machine-readable summary of the method and all measurements is embedded at the
bottom of the [visual explainer](https://timepointai.github.io/PRISM/docs/how-prism-works.html) —
a standalone, self-contained page (no scripts, no external requests) served from
GitHub Pages; source: [`docs/how-prism-works.html`](docs/how-prism-works.html).

## Honest bounds on the claims

Every number above has a committed artifact, and every artifact has a scope:

- **The 30× is a single-seed probe** and measures the compounding of two free
  priors (the fused arms start near baseline) — the defensible claim is that the
  hybrid beats PRISM alone *and* reaches the best loss, not that PRISM alone is
  30×. The three-seed spine of the story is the transfer/retention work.
- **The transfer probes are early-window measurements** (step-100,
  init-dominated, 3 seeds, matched LR). They prove the head start is structural
  and domain-portable; the long-horizon no-overfitting runs (1,500 / 5,000
  steps) are same-data.
- **"Different corpus" means Sherlock Holmes** — a different author, but still
  English prose over the same character vocabulary (token-JS ≈ 0.03). A genuinely
  different modality (code, another language) is untested and is where the plain
  recipe may finally need the geometric levers.
- **Teacher-free init is measured at one distance, in one direction**
  (Sherlock → Shakespeare), and the cross teacher trained on ~40% less data than
  the control's — a teacher-corpus-size control is the immediate follow-up before
  reading the "missing half" as a distance effect.
- **Spectral vs. generic regularization is not fully isolated** — the
  teacher-strength dependence argues the spectral target matters, but a
  tuned-dropout/weight-decay control has not been run.
- **Scale:** 10.65M params, ~1M-token corpora. Nothing here has met a production
  model.
- **In finetuning, the retention shown is old-domain retention**, not new-domain
  overfit prevention; the anchor trades a little late adaptation for it.

## Reproduce it

**Modal (headless — how the committed runs were produced):**

```bash
pip install modal && modal setup                        # one-time auth
modal run --detach prism_modal.py \
  --extra "--student_steps=1500 --eval_every=10 --eval_iters=50"   # the 11.8× run
modal run --detach prism_modal.py \
  --extra "--method_lr=1e-3 --method_warmup=100"        # the matched-LR control (7×)
```

**The transfer probes (local or Modal):**

```bash
cd src
python prism_eval.py --teacher_steps=500 --student_steps=100 --eval_every=10 \
  --eval_iters=40 --batch_size=32 --method_lr=1e-3 --method_warmup=20 \
  --baseline_warmup=20 --overlap=1.0,0.75,0.5,0.25,0.0        # overlap sweep
python prism_eval.py --teacher_sweep=100,250,500,1000,2000 \
  --student_steps=300 --eval_every=20 --eval_iters=40 --batch_size=32   # teacher lever
python prism_eval.py --report                                  # reprint last artifact
```

The cross-domain arm reproduces with
`--overlap=1.0,0.75,0.5,0.25,0.0 --far_corpus=data/far.txt --far_val` (plus the
probe schedule above).

**The teacher-free probe** (two arms, isolated volumes; only `--cross_teacher`
differs):

```bash
modal run --detach prism_modal_teacherfree.py --arm cross    # Sherlock-only teacher
modal run --detach prism_modal_teacherfree.py --arm control  # same-corpus teacher
```

**The finetune-retention frontier and the other benchmarks** run headless on their
own isolated Modal volumes — `prism_modal_finetune.py`, `prism_modal_arc.py`,
`prism_modal_prior.py` (commands in [`RESULTS.md`](RESULTS.md) and the docs).

## What's next

The ranked, run-command-ready list lives in
[`docs/NEXT-EXPERIMENTS.md`](docs/NEXT-EXPERIMENTS.md). Highlights:

1. **The reach-at-init moonshot** — the Prior-Fused hybrid reaches baseline
   quality *at init* if the shared prior sits clearly below baseline; that needs a
   **context-4+ sparse n-gram** (dense V⁴ times out). The documented path from the
   30× hybrid toward a literal 1000×.
2. **A truly far modality** — point the far-corpus arm at source code or another
   language (needs a vocab decision) and find where structural transfer degrades;
   that's where the opt-in geometric levers (Grassmann pairing, top-k, CKA —
   contributed in PR #1) get their real test.
3. **Close the teacher-free gap** — a teacher-corpus-size control (was the
   missing half of the speedup just 40% less teacher data?), a second far corpus
   to map score-vs-distance, and averaged multi-corpus (Σ\*) fingerprints.
4. **Long-horizon far-domain, continual A→B→C, cross-size transfer, the "unfold
   curve"** — all specified with run commands in the doc.

## Repo map

```
README.md                     ← you are here
docs/how-prism-works.html     ← the visual explainer (standalone, self-contained)
docs/FINETUNE-RETENTION.md    ← finetune without forgetting — study + frontier
docs/UNIFIED-ARC.md           ← PRISM pretraining + finetuning compound — the arc
docs/PRIOR-FUSED-PRISM.md     ← T9 × PRISM — the n-gram prior compounds (30×)
docs/NEXT-EXPERIMENTS.md      ← the ranked next experiments + run commands
WHITEPAPER.md                 ← method + experiments in full
RESULTS.md                    ← the committed runs and what they do / don't show
results/                      ← eval artifacts. the evidence.
archive/                      ← v0.0 / v0.1 writeups + retired notebooks
prism_modal.py                ← headless Modal runner (the committed from-scratch runs)
prism_modal_finetune.py       ← isolated runner — finetune-retention benchmark
prism_modal_arc.py            ← isolated runner — the arc benchmark
prism_modal_prior.py          ← isolated runner — Prior-Fused (T9 × PRISM) benchmark
prism_modal_teacherfree.py    ← isolated runner — the teacher-free init probe
prism_modal_leo.py            ← isolated runner — geometric-lever experiments
unfold_curve.py               ← exploratory: the Σ*/"unfold curve" first cut
src/prism_eval.py             ← the from-scratch benchmark (all transfer knobs)
src/prism_finetune_eval.py    ← the finetune-retention benchmark
src/prism_arc_eval.py         ← the arc benchmark (matched-quality bases)
src/prism_prior_eval.py       ← the Prior-Fused benchmark (4 arms)
src/build_ngram_prior.py      ← builds the shared n-gram prior ("the T9 dictionary")
src/prism_accelerate.py       ← apply PRISM to any checkpoint (transfer entry point)
src/prism_finetune.py         ← finetune any checkpoint without forgetting
src/prism_init.py             ← Spectral Imprint + EigenTransfer + Mod Wheel
src/prism_extract.py          ← extract a fingerprint from any checkpoint
src/prism_selftest.py         ← 30 offline invariant tests — run before any GPU spend
data/far.txt                  ← Sherlock Holmes (Project Gutenberg #1661) — the far corpus
```

## License

MIT — see [LICENSE](LICENSE).

A standalone clone of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej
Karpathy (MIT, © 2022), not a fork. `model.py`, `configurator.py`, `bench.py`,
`sample.py`, and the `data/` preparers are his; `train.py` is his with PRISM hooks
added (all off by default — a scratch run is byte-identical to nanoGPT). The PRISM
code (`prism_*.py`, `build_ngram_prior.py`, `config/prism_*.py`) is Timepoint
Labs', under the same terms.

---

*A [Timepoint Labs](https://timepointai.com) project by [Sean McDonald](https://x.com/seanmcdonaldxyz).*
