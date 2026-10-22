# W2b — the Pythia ladder: G1 port, harness and gated runner (prepared, not launched)

*Alpha-moves workstream W2b (brief §5). Built 2026-10-21 on branch `exp/alpha-w2b`.
Nothing here has run on a GPU; every number below is an estimate and is labelled
as one. The pre-registration in `docs/NEXT-EXPERIMENTS.md` (core bet, G2–G4 gates)
is restated at the end and is not weakened.*

## What was built

| file | what |
|---|---|
| `src/prism_neox.py` | PRISM on GPT-NeoX/Pythia: group classification, per-head QKV unpack/repack (exact), fingerprint extraction (per-matrix U/Vt in the canonical [Q;K;V] layout + 8-coefficient group spectra), `apply_prism_neox` (EigenTransfer, optional Spectral Imprint) onto Pythia's **own published `step0`**. `align=0, imprint=False` is an exact no-op, so the baseline arm is the published init bit for bit. |
| `src/pile_mmap.py` | Pythia's exact token stream from the published Megatron `MMapIndexedDataset` (`EleutherAI/pile-standard-pythia-preshuffled`): `.idx` reader, sharded `.bin` access, iteration→sequence mapping, HTTP range fetching of only the bytes a gate needs, a synthetic-dataset writer in the same format for tests. |
| `src/pythia_train.py` | The ladder harness: one arm per run (`baseline` / `prism` / `control`), the published schedule, resumable, train.py-style eval lines, `prism-pythia/1` artifact, `--published_curve` (scores the published checkpoints on the same held-out slice), `--lm_eval` (lm-eval 0.4.13, Pythia's 0-shot suite), `--score` (steps-to-target, token-equivalent multiplier, donor-adjusted multiplier). |
| `prism_modal_pythia.py` | Gated runner: `--gate dry|g2|g3|g4`, cost table before any spawn, CPU-only data preparation first, one detached GPU call per (seed, arm), one shared Volume. |
| `src/prism_selftest.py` | +18 checks (85 total pass): QKV round-trip and agreement with the HF forward, classification coverage, orthonormal fingerprints, exact no-op, end-to-end on a 2-layer synthetic NeoX, the MMIDIDX reader (sharded, straddling, windowed), and three checks on the cached real pythia-70m checkpoints. |

## Facts verified against the live sources (2026-10-21)

- `transformers` 5.9.0 `modeling_gpt_neox.py`: `qkv.view(*, -1, 3*head_size)` then
  `chunk(3, dim=-1)` — the fused weight is packed **per head** (q,k,v contiguous
  inside each head's 3·head_size rows), not in three d-row blocks. `qkv_unpack`
  converts to nanoGPT's layout; the selftest checks the unpacked Q block against
  the HF forward's per-head query.
- pythia-70m `config.json`: hidden 512, 6 layers, 8 heads, intermediate 2048,
  rotary_pct 0.25, use_parallel_residual true, tie_word_embeddings **false**,
  vocab 50304. Revisions `step0` … `step143000` exist; `main` == step143000.
- Extra-seed repos exist with `step0`: `EleutherAI/pythia-{70m,160m,410m}-seed{1,2,…}`
  (155 branches each). "3 seeds" in G3 therefore means three **published**
  initializations (seed1–3), not three re-draws of ours; the teacher is the
  seed-0 `main` checkpoint for all of them (one fingerprint per size).
- `document.idx` header (HTTP range read): magic `MMIDIDX`, version 1, dtype code
  8 = uint16, N = 146,432,000 = 143,000 × 1,024 sequences, D = 1; every sequence is
  2,049 tokens and pointer[i] = 4,098·i exactly. Shards 0–19 are 30,000,000,000
  bytes each (shard 20: 78,336,000 bytes); the official unshard script
  concatenates them, so global offset → shard `o // 30e9`.
- 70M yml: lr 1e-3, min_lr 1e-4, cosine over 143,000 iters, 1% warmup, betas
  0.9/0.95, eps 1e-8, wd 0.1, clip 1.0, 1,024 × 2,048 tokens per iteration, fp16.
  160M: 6e-4/6e-5. 410M: 3e-4/3e-5.

## Deviations (all recorded in every artifact's `config.deviations`)

1. **bf16 autocast** instead of fp16 with dynamic loss scaling.
2. **Gradient accumulation** over micro-batches on one GPU (default 32 × 32 = 1,024
   sequences, in Pythia's order within the iteration) instead of 32-way data
   parallelism. Same batch, same order, different reduction order of the sum.
3. **Held-out slice**: sequences `[140000·1024, 140000·1024 + eval_seqs)` of the
   training stream (iteration 140,000 of the published run, shard 19, an 8 MB
   window). No gate trains within 130,000 iterations of it; every published
   checkpoint ≤ step139000 had not seen it either, which is what makes the
   published-curve reproduction valid. The Pile validation set is not used.
   The teacher (step143000) *has* trained on it; it is never scored as a comparator.
4. **Schedule not rescaled**: a gate stops at `--iters` but runs the published
   143,000-iteration cosine, so step k of our baseline is comparable to published
   `step<k>`.
5. **FlashAttention** via torch SDPA (whatever `transformers` picks), not the
   Megatron kernels.

## Reproduction check (stated before any launch)

The baseline arm's held-out loss must be **within 0.05 nats of the published
checkpoint's held-out loss at every matched step up to the horizon**
(`--published_curve` scores step0, 1, 2, 4, …, 512, 1000, 2000, … ≤ iters on the
same slice). `pythia_train.py --score` writes `ratio_reportable: false` and
withholds nothing but says so until the check passes. If it fails, the first
suspects are deviations 1–2 (bf16, accumulation) and the data mapping; the
published checkpoints then stay the gray field and our baseline is debugged
before any PRISM number is quoted. A 2023 recipe re-run in different code is not
a fair comparator until it tracks the published curve.

## Accounting per gate

- steps-to-target against the in-run baseline (`compute_score` semantics);
- the token-equivalent multiplier at end of training from a monotone power-law
  fit to the baseline curve (`prism_metrics.token_equivalent_multiplier`,
  RMSE tolerance 0.02 nats, refuses otherwise);
- donor compute: the teacher is the published final checkpoint, 6·N·D with
  D = 299,892,736,000 tokens, quoted amortised over one student
  (`donor_adjusted_multiplier`) — the fingerprint is reusable, the published
  checkpoint already exists, so this is the *worst-case* accounting;
- optimizer recorded (AdamW, published hyper-parameters);
- `--lm_eval`: Pythia's 0-shot suite via lm-eval-harness 0.4.13 at matched
  tokens (lambada_openai, piqa, winogrande, arc_easy, arc_challenge, sciq, logiqa).

## Cost table (estimates; H100 $3.95/h, A100-80 $2.50/h, L4 $0.80/h on Modal, 2026-10-21)

Model: FLOPs/token = 6·N_eff + 12·L·d·T (N_eff = non-embedding params + `embed_out`,
T = 2048); tokens/iter = 2,097,152; H100 bf16 dense 989 TFLOP/s at an **assumed
30% MFU** (small models rarely do better); A100 is ~2× worse per dollar, so H100.

| gate | model | plan | FLOPs/iter | s/iter | $/iter | iters total | GPU $ (est.) | with overheads (est.) | tier |
|---|---|---|---|---|---|---|---|---|---|
| dry | 70M | 1 × {baseline, prism} × 20 iters, L4 | 7.2e14 | ~20 | — | 40 | < $1 | ~$1 (84 MB + 2 MB data) | A |
| G2 | 70M | 1 seed × {baseline, prism} × 8,000 | 7.2e14 | 2.4 | $0.0027 | 16,000 | ~$43 | ~$60–75 (33.6 GB fetch, ~19 published evals, lm-eval) | B, ~$100 |
| G3 | 160M | 3 published seeds × {baseline, prism} × 8,000 | 2.0e15 | 6.8 | $0.0075 | 48,000 | ~$360 | ~$420–500 | B, ~$500 |
| G4 | 410M | 2 published seeds × {baseline, prism, control} × 8,000 | 5.7e15 | 19.3 | $0.0211 | 48,000 | ~$1,010 | ~$1.2k | C, ~$1.5k |

Why 8,000 iterations: 16.8B tokens (5.6% of the published run), past the 1,430-step
warmup and into the early cosine where Pythia's published loss is still falling
fast — the regime where an init effect is largest and where a steps-to-target
ratio can be measured against a baseline that has not plateaued. The same horizon
at every size keeps the ladder comparable for G4's "holds ≥ 80% of the 70M ratio"
rule. If the measured MFU is far from 30%, the runner's table re-prices before the
spawn (edit `MFU` / `GATES[...]['iters']`); nothing launches without the printout.

Memory: 410M in bf16 autocast with fp32 master weights and AdamW states ≈ 6.5 GB +
activations at micro-batch 32 × 2,048 — fits an 80 GB H100 (drop `--micro_bs` to
16 if not).

## Launch commands (G2+ only with Tier B/C authorization)

```bash
# push the branch first — the runner clones exp/alpha-w2b at run time
modal run prism_modal_pythia.py --gate dry          # Tier A: real 70M, 20 iters, L4, artifact check
modal run --detach prism_modal_pythia.py --gate g2  # Tier B
modal run --detach prism_modal_pythia.py --gate g3  # Tier B, after G2 passes and W2a's controls have landed
modal run --detach prism_modal_pythia.py --gate g4  # Tier C — needs /work/pile/control_init_410m.pt from W2a's strongest control
# fetch + commit
modal volume get prism-alpha-pythia nanogpt-prism/results/pythia_<stamp>.json ./results/
cd src && python3 pythia_train.py --score ../results/pythia_<baseline>.json ../results/pythia_<prism>.json --tag g2
```

## Pre-registration restated (from docs/NEXT-EXPERIMENTS.md; unchanged)

**Claim under test:** on a trusted open family (Pythia), a PRISM-initialized run
reaches the same validation loss — and the same downstream scores — using
meaningfully fewer tokens than the published baseline. The primary number is
tokens-to-target-loss; downstream metrics are confirmation.

**Decision rule, both directions:** if the tokens-to-loss advantage survives to
410M–1B with clean comparisons, this is real. If the advantage fades toward 1× as
scale increases, the core bet has failed, regardless of how good the small-scale
results looked — no reframing, no retreat to the side results.

**Gates:** G1 port + selftest ($0) — done here. **G2** Pythia-70M single seed,
same-size fingerprint from Pythia's own checkpoints (~$100) — **kill if the shift
is < 1.2×**. **G3** 160M, 3 seeds, the chart against the published curves (~$500).
**G4** 410M (~$1.5k): if the 410M ratio holds ≥ 80% of the 70M ratio the story
survives scale and 1B+ becomes a fundable claim. The Act 2 composition gate on
enwik8 (W5) runs before any Pythia spend. Arms per scale: baseline (step0), PRISM
(step0 + EigenTransfer), and from G3 on the strongest control from W2a; AdamW
matching Pythia's recipe; a Muon arm is a separate, pre-registered extension.
No PRISM ratio is reported until the baseline reproduces the published curve
within the tolerance above.

## Open items for the parent

- `--control_init` expects a state dict built by W2a's strongest control for the
  Pythia architecture (statmatch / small_qk / mimetic ported to NeoX names); not
  built here — the hook is in place.
- The published-curve evaluation downloads ~19 checkpoints (70M: ~170 MB each);
  for 410M that is ~19 × 1.6 GB on the Volume. Acceptable; noted.
- CPU dry runs use the published schedule's warmup, so 20 synthetic iterations
  barely move the loss — the smoke checks plumbing and the artifact, not learning.
