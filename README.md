# PRISM

**Transfer the *geometry* of a trained neural network — not its weights, not its
data — and training gets dramatically faster, forgets less, and compounds with
other priors.** Validated end-to-end on nanoGPT char-level (10.65M params), three
seeds, every number backed by a committed artifact in [`results/`](results/).

*(Formerly `nanogpt-prism-shakespeare`, now archived — full history
[there](https://github.com/timepointai/nanogpt-prism-shakespeare); development
continues here.)*

<img src="assets/prism-flashlight.svg" alt="A spectrographic flashlight for models: a trained checkpoint's raw weights enter a prism as white light and split into four spectral bands — attention, FFN up, FFN down, embedding — each carrying a spectrum and directions. A reversed prism recombines the bands into a fresh model that trains about 12 times faster. Geometry crosses; content never does." width="100%">

<a href="https://timepointai.github.io/PRISM/docs/how-prism-works.html"><img src="assets/prism-explainer-button.svg" alt="HOW PRISM WORKS — THE VISUAL EXPLAINER: the flashlight metaphor, node-level math, graph-level transfer, and the measurements" width="100%"></a>

---

Every trained model's weight matrices have a geometry: `W = U Σ Vᵀ` — the
**directions** (U, V: which subspaces the model uses) and the **spectrum** (Σ: how
much energy it puts in each). PRISM is three small operations on that geometry:

1. **Spectral Imprint** — compress the teacher's singular-value distribution to
   8 DCT coefficients per weight group (**~128 bytes total**) and reshape a fresh
   model's spectrum to match at init.
2. **EigenTransfer** — blend the fresh model's singular vectors 75% toward the
   teacher's, then re-orthogonalize.
3. **Mod Wheel** — after every optimizer step, pull the weights gently back toward
   a stored target: `W ← (1−s)·W + s·W_target`. Pointed at a spectral target it
   regularizes a from-scratch run; pointed at a model's *own* weights it prevents
   catastrophic forgetting.

No parameters are copied. Only geometry. The through-line of every result below:
**the spectral geometry is special** — it is transferable, data-independent
structure, and it compounds.

## The results

| study | finding | evidence |
|---|---|---|
| **From-scratch transfer** | a teacher's spectral geometry → a fresh model reaches baseline quality **11.8×** faster (tuned; **7×** schedule-matched), and stops overfitting | [`RESULTS.md`](RESULTS.md) §1–2 |
| **Structure, not content** | the head start is **identical at 100% and 0%** teacher/student data overlap, and *grows* slightly when the student trains and is scored on a different corpus | [`RESULTS.md`](RESULTS.md) §3–4 |
| **Finetune retention** | Mod Wheel self-anchored during a finetune → up to **~10× less catastrophic forgetting**, Pareto-dominating low-LR; attribution: a *raw directional* anchor, not spectral | [`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md) |
| **The arc** | a PRISM-*pretrained* base is a far better finetune-anchor at matched quality: **≈0 forgetting, ~8% better adaptation** — attributably the spectral geometry | [`docs/UNIFIED-ARC.md`](docs/UNIFIED-ARC.md) |
| **Prior-Fused PRISM** | fuse a fixed T9-style n-gram prior into the logits and stack PRISM: **30×** hybrid — double PRISM alone — at the best loss of all arms | [`docs/PRIOR-FUSED-PRISM.md`](docs/PRIOR-FUSED-PRISM.md) |

| **Teacher-free init** | a fingerprint from a teacher that **never saw the target corpus** still trains it **4.6×** faster at near-identical final quality (vs. 9.9× same-corpus, identical rig) — portable, not yet free | [`RESULTS.md`](RESULTS.md) §10 |

Every number has a matching committed `results/*.json` (loss curves, provenance,
git commit, argv, censoring flags). A claim without an artifact is not a result in
this repo.

<img src="assets/prism-transfer.svg" alt="Left: validation loss at step 100 versus teacher/student data overlap, with the student scored on its own data mixture — the from-scratch baseline sits at ~2.47 while the PRISM recipe sits at ~1.88, and the gap grows from 0.591 at full overlap to 0.627 when the student trains and is scored entirely on Sherlock Holmes. Right: the advantage versus teacher training steps — negative at a 100-step teacher, rising monotonically and saturating at a +0.46 plateau around 2,000 steps, where the teacher itself converges; 4k and 8k teachers are flat." width="100%">

## Why this is interesting

The obvious explanation for any teacher→student speedup is that the teacher leaked
its *content*. The experiments were built to kill that explanation, and they did:

1. **Same corpus, disjoint data.** Cut the corpus into 100 random blocks, give
   teacher and student overlapping or disjoint halves (both spanning the whole
   corpus, so difficulty is controlled). The early advantage is **identical at
   100% and 0% overlap** ([`…T050203Z`](results/recipe_20260721T050203Z.json)).
2. **Different corpus entirely.** Swap the student's data for Sherlock Holmes and
   score it on held-out *Sherlock*. A Shakespeare teacher's geometry accelerates
   learning of Sherlock at least as much as Shakespeare — Δloss 0.627 vs. 0.591
   ([`…T161208Z`](results/recipe_20260721T161208Z.json)).

3. **Teacher never sees the corpus at all.** Train the teacher *exclusively* on
   Sherlock and use its ~128-byte fingerprint to init a Shakespeare student: it
   still reaches baseline-best quality **4.6×** faster (median of 3, resolved),
   at a final loss the baseline never touches — about half the speedup and ~80%
   of the quality gain of a same-corpus teacher on the identical rig
   ([`…T164604Z`](results/recipe_20260725T164604Z.json) vs.
   [`…T164640Z`](results/recipe_20260725T164640Z.json)). The geometry is
   substantially a *modality* property; a matched teacher still buys the other
   half.

And the effect tracks exactly what it should if the geometry is what matters: the
advantage grows with teacher training and **saturates right where the teacher
itself converges** (≈2,000 steps; 4k/8k teachers add nothing —
[`…T143246Z`](results/recipe_20260721T143246Z.json),
[`…T172238Z`](results/recipe_20260721T172238Z.json)). A barely-trained teacher's
geometry is noise imprinted with authority — actively *worse* than random init.

## The other direction: finetuning without forgetting

Everything above hands a trained model's geometry to a *fresh* one. The same Mod
Wheel, pointed the other way — anchored to a trained model's **own pre-finetune
weights** — lets it adapt to a new domain without wrecking the old one:

```
# after each optimizer step, for every weight matrix:
W  ←  (1 − s)·W  +  s·W₀        # s ≈ 0.01–0.02, constant
```

| finetune arm (Shakespeare → Sherlock, 3 seeds) | forgets Shakespeare | learns Sherlock | vs. plain |
|---|---|---|---|
| plain finetune | +0.43 | 1.25 | 1× |
| **Mod Wheel anchor (s = 0.02)** | **+0.04** | 1.37 | **~10× less forgetting** |

The anchored model still beats a from-scratch Sherlock model — it genuinely learns
the new domain. The attribution landed on a **negative** to the obvious guess:
anchoring only the *spectrum* does nothing for retention (1.07×), a wrong-spectrum
placebo actively harms (0.39×), and the anchor Pareto-dominates the low-LR
frontier (~2× more retention at equal adaptation). The protection is a raw
**directional / whole-weight** anchor. That splits PRISM's geometry cleanly:

- **Spectrum → structure.** Data-*independent*; *transfer* it into fresh models.
- **Directions → content.** Domain-*specific*; *pin* them to retain what's known.

Full study, frontier, and bounds: [`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md).

### The two directions compound

A base that was itself PRISM-*pretrained* makes a much better thing to
anchor-*finetune*: at **matched** Shakespeare quality, a PRISM base finetunes on
Sherlock with **≈ zero forgetting** (vs. +0.05 nats plain) and adapts **~8%
better** — and a schedule-matched plain control confirms it's the spectral
geometry, not the learning rate. Full study: [`docs/UNIFIED-ARC.md`](docs/UNIFIED-ARC.md).

### …and they compound with a statistical prior

Fuse a tiny fixed **shared n-gram prior** into the logits as a product of experts
(`final = model_logits + λ·log p_ngram`), so the model learns only the residual,
and stack PRISM on top. Measured as steps-to-baseline-quality: PRISM alone 15×,
the n-gram alone 3.8×, **the hybrid 30× — at the best loss of all four arms**,
below even the n-gram's own floor. Statistics from T9, geometry from PRISM, and
they stack. Full study: [`docs/PRIOR-FUSED-PRISM.md`](docs/PRIOR-FUSED-PRISM.md).

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

<img src="assets/prism-method.svg" alt="How PRISM works: SVD a trained teacher into directions (U, V) and a spectrum, compress the spectrum to 8 DCT coefficients, inject both into a fresh student, then hold the student toward the spectral target with the mod wheel." width="100%">

<img src="assets/prism-result.svg" alt="Validation loss over training, nanoGPT Shakespeare, three seeds each. The baseline (LR 1e-3) falls to ~1.78 near step 1,400 then overfits to ~2.30 by step 5,000. The PRISM recipe at LR 5e-4 holds ~1.66; the recipe run at the baseline's own LR of 1e-3, where only the spectral flags differ, holds ~1.67 and also never overfits." width="100%">

## Honest bounds on the claims

Every number above has a committed artifact, and every artifact has a scope:

- **The transfer probes are early-window measurements** (step-100, init-dominated,
  3 seeds, matched LR). They prove the head start is structural and
  domain-portable; the long-horizon no-overfitting runs (1,500 / 5,000 steps) are
  same-data.
- **"Different corpus" means Sherlock Holmes** — a different author, but still
  English prose over the same character vocabulary (token-JS ≈ 0.03). A genuinely
  different modality (code, another language) is untested and is where the plain
  recipe may finally need the geometric levers.
- **Spectral vs. generic regularization is not fully isolated** — the
  teacher-strength dependence argues the spectral target matters, but a
  tuned-dropout/weight-decay control has not been run.
- **Teacher-free init is measured at one distance, in one direction**
  (Sherlock → Shakespeare, token-JS 0.028), and the cross teacher trained on ~40%
  less data than the control's — a teacher-corpus-size control is the immediate
  follow-up before reading the "missing half" as a distance effect.
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

1. **A truly far modality** — point the far-corpus arm at source code or another
   language (needs a vocab decision) and find where structural transfer degrades;
   that's where the opt-in geometric levers (Grassmann pairing, top-k, CKA —
   contributed in PR #1) get their real test.
2. **The reach-at-init moonshot** — the Prior-Fused hybrid reaches baseline
   quality *at init* if the shared prior sits clearly below baseline; that needs a
   **context-4+ sparse n-gram** (dense V⁴ times out). The documented path from the
   30× hybrid toward a literal 1000×.
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
