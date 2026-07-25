# PRISM — handoff to the next agent

> **Status update (2026-07-25):** the migration + rebrand below is **done**
> (selftest 30/30; Pages re-enabled; runners repointed at PRISM/main). The
> teacher-free init experiment (§6's biggest lead) has been **run** — see
> [RESULTS §10](RESULTS.md) and `docs/NEXT-EXPERIMENTS.md` #1 for the verdict
> (~half the speedup survives, quality fully survives) and the ranked follow-ups.

You are picking up **PRISM** in its new home, `timepointai/PRISM`. This document is your
full orientation — read it first. It explains what exists, where it is, what to do, and
the rules that are non-negotiable.

## 1. The state of this repo right now

This is a **fresh scaffold**: this `HANDOFF.md`, a placeholder `README.md`, and the MIT
`LICENSE`. **There is no code here yet.**

Everything — the complete codebase, all committed `results/*.json` evidence, every doc,
and the full git history — lives in the **now-archived** predecessor repo:

- GitHub (archived, read-only, still cloneable): **`timepointai/nanogpt-prism-shakespeare`**
  <https://github.com/timepointai/nanogpt-prism-shakespeare>
- Local clone (synced to `a4d88a4`): **`~/dev/GitHub/timepointai/nanogpt-prism-shakespeare`**

The rename from `nanogpt-prism-shakespeare` → `PRISM` is a rebrand: the project outgrew a
name that read like a nanoGPT fork. It is still open source (MIT).

## 2. Your first job — migrate + rebrand (housekeeping, then you're at the frontier)

1. **Bring the code over.** Two clean options — pick per Sean's preference:
   - *Fresh start (simplest):* copy the working tree from the old local clone into this
     repo, commit as the first real commit, push. History lives in the archived old repo.
   - *History-preserving:* add the old repo as a remote and push its history here (this
     will replace the current placeholder root — re-add the placeholder/handoff on top,
     or fold this HANDOFF into the migrated tree). More work; only if Sean wants the
     commit history carried forward.
2. **Rebrand in-tree:** `Prism` / `nanogpt-prism-shakespeare` → `PRISM` where it's the
   project name (not where it refers to the upstream nanoGPT base by Karpathy — keep that
   attribution intact in `LICENSE` and `model.py`).
3. **Update the Modal runners** (`prism_modal*.py`): they hard-code
   `REPO_URL = ".../nanogpt-prism-shakespeare.git"` and `BRANCH = "..."`. Point them at
   PRISM. Each uses its own isolated Modal volume (`prism-eval`, `prism-eval-finetune`,
   `prism-eval-arc`, `prism-eval-prior`) — keep that isolation.
4. **Write the real README** — the old `README.md` (in the archived repo) is the template;
   it already covers all four results. Adapt it, drop the "being set up" banner.
5. **Re-enable GitHub Pages** for the visual explainer (`docs/how-prism-works.html`) on
   the PRISM repo (the old Pages URL was `timepointai.github.io/nanogpt-prism-shakespeare/...`).
6. Sanity-gate the migration: `cd src && python prism_selftest.py` (30 offline tests, ~1
   min) must pass before you trust anything.

## 3. What PRISM is (the science)

Every trained model's weights have a **geometry**: per weight matrix `W = U Σ Vᵀ` →
**directions** (U, V singular vectors) + **spectrum** (Σ singular values). PRISM is three
ops: **Spectral Imprint** (reshape a fresh model's spectrum to a teacher's, ~128 bytes of
DCT coeffs), **EigenTransfer** (blend singular vectors 75% toward the teacher's at init),
and the **Mod Wheel** (a per-step pull back toward a stored spectral target,
`W ← (1−s)·W + s·W_target`). All validated on nanoGPT char-level, 10.65M params, 3 seeds,
Modal L4, with an absolute evidence bar.

**The through-line of every result: the spectral geometry is special.**

## 4. The four published results (all in the archived repo, with docs)

| result | finding | doc (in the old repo) |
|---|---|---|
| **from-scratch transfer** | spectrum = transferable, data-independent *structure* → **~12×** faster, no overfitting; survives 0-overlap and cross-corpus | `README.md` + `RESULTS.md` + `WHITEPAPER.md` |
| **finetune-retention** | keep the Mod Wheel on, self-anchored → **~10×** less catastrophic forgetting; attribution shows it's a *raw directional* anchor (L2/EWC-lite), not spectral, and beats low-LR | `docs/FINETUNE-RETENTION.md` |
| **the arc** | a PRISM-*pretrained* base is a much better finetune-anchor at *matched* quality (≈0 forgetting, ~8% better adapt) — attributably the spectral geometry; pretraining + finetuning **compound** | `docs/UNIFIED-ARC.md` |
| **prior-fusion (T9 × PRISM)** | fuse a fixed shared n-gram prior into the logits (product of experts) → **30×** hybrid, double PRISM, best loss; a context-3 n-gram ≈ the neural baseline. Geometry compounds with statistics | `docs/PRIOR-FUSED-PRISM.md` |

The complete story is also in the standalone visual explainer `docs/how-prism-works.html`
(with a machine-readable agent-summary block at the bottom you should read).

## 5. The machinery (all in the old repo's `src/`)

- Benchmarks (each emits committed JSON artifacts, is stepwise-resumable, raises on
  partial runs): `prism_eval.py` (from-scratch), `prism_finetune_eval.py`,
  `prism_arc_eval.py`, `prism_prior_eval.py`.
- "Apply it" entry points: `prism_accelerate.py` (transfer), `prism_finetune.py` (anchor).
- Core: `prism_init.py` (Imprint + EigenTransfer + Mod Wheel + `spectral_target`),
  `prism_extract.py`, `build_ngram_prior.py`. `train.py` is nanoGPT's with all the PRISM
  hooks (all off by default → scratch runs byte-identical).
- `prism_selftest.py` — 30 offline invariant tests. Run before any GPU spend.
- **Modal ops:** fork a branch → make `prism_modal_<name>.py` with its own
  `Volume.from_name("prism-eval-<name>")` → `modal run --detach`. Watch `modal app logs
  <ap-id>`, then `modal volume get prism-eval-<name> nanogpt-prism/results/<file>
  ./results/` and **commit the artifact** — that's the evidence. L4 is the sweet spot;
  probes ~10–20 min. Ritual: selftest → tiny `--device=cpu` smoke → Modal.

## 6. Next experiments (the frontier)

The archived repo's **`docs/NEXT-EXPERIMENTS.md`** is the ranked, run-command-ready list.
Highlights:

- **Teacher-free init (the biggest lead):** if the spectrum is a *modality* constant,
  PRISM-init from an *unrelated* corpus's fingerprint — **no teacher** — and keep the ~12×.
  Cheapest high-value test. See also the `project_prism_sigma_star` memory / the Σ* idea.
- **The reach-at-init moonshot:** the prior-fusion hybrid hits baseline quality *at init*
  if the shared prior is *below* baseline — needs a **context-4+ sparse n-gram** (dense V⁴
  times out) + a block-edge fix. The documented path from the 30× hybrid to a literal 1000×.
- Truly-far modality (code — needs a new char corpus/vocab), continuous single-run vs
  sequential, continual A→B→C, the "unfold curve" (a first cut exists: `unfold_curve.py`).

## 7. Ground rules (non-negotiable — this project killed its own hypotheses honestly)

- **Evidence = a committed `results/*.json`.** A number in a doc without a matching
  artifact is not a result.
- **Attribution needs matched schedules** (only then does "only X differs"). Use
  **ratio/delta metrics vs an explicit control**, never bare absolutes.
- **Wide/shallow probes first**, deep runs after. **3 seeds** for a result; a single seed
  is a probe. Partial/crashed runs must **raise**, never be scored.
- **No mocks** — everything hits the real training loop.
- **Own your errors plainly.** This repo refuted its own "spectral finetune" hypothesis
  with a placebo, overturned an author's prior on the arc, and reported the prior-fusion
  gate backfiring. Truth over hype, always.

## 8. Working with Sean (the owner)

Lead with the **maximal advantage**, caveats as **bounded caveats** ("don't bury the
lede"). He prefers **wide/shallow probes** over deep runs and is decisive. **Paid GPU runs
are pre-authorized**, two-at-once is fine. The evidence bar is absolute — he initiated it.
He runs long autonomous loops and expects honest reporting, including clean negatives.

## 9. Housekeeping notes

- **Commits:** `type: description` (`feat`/`fix`/`docs`/`chore`/`results`/`refactor`).
  **NO** "Co-Authored-By", **NO** AI/tool attribution, **NO** emoji in commit messages.
  Applies across all timepointai repos.
- **Sean-only open item:** rotate the Modal token (long-standing carryover). Modal auth is
  `realityinspector`; volumes `prism-eval` / `prism-eval-finetune` / `prism-eval-arc` /
  `prism-eval-prior` hold each benchmark's resume state.
- Open source (MIT). The nanoGPT base is Andrej Karpathy's (MIT, © 2022) — keep that
  attribution when you migrate `model.py`/`configurator.py`/`sample.py`/`bench.py`/`data/`.

Welcome to PRISM. The frontier is teacher-free init and the reach-at-init 1000× — but
migrate + rebrand first, and run the selftest before you trust anything.
