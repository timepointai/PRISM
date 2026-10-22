# W5 — the COUNT control for prior fusion (Act 2 composition gate)

*Branch `exp/alpha-w5`, built 2026-10-21. Prepared, not launched. Every number below
that is not a build statistic comes from a committed artifact or is labeled "reported".*

## What was built

| piece | file | what it does |
|---|---|---|
| sparse prior | `src/prism_prior.py` (`SparsePrior`, `build_tables`) | Orders 0..C maximum-likelihood next-symbol tables, exact contexts (sorted int64 keys + fp16 `(n_ctx, V)` rows), Jelinek–Mercer interpolation with per-order μ exactly as `ngram_floor.py`. `.logp(X)` gathers `(B,T,V)` log-probs on the training device from in-window context with **block-edge backoff**: at window position *t* orders *k > t* are misses, so the mixture backs off to the orders that fit (position 0 gets the unigram mixture, not uniform). `.score_sequential()` is the Act-1 protocol. |
| builder | `src/build_sparse_prior.py` | Counts `data/<slot>/train.bin` minus a 1M-symbol dev tail; μ tuned greedily on dev (ngram_floor grid) or taken from Act 1 (`--mus act1`); reports dev and sequential val bits/symbol. |
| COUNT embedding | `src/prism_prior.py` (`install_count_embedding`, `count_embedding`) | Our re-implementation of arXiv 2610.20456 §2: `f_t = (1/J) Σ_j Σ_{k≤K} S(w_k|h_{t,j}) B[w_k]` with B = the tied unembedding, stupid-backoff scores (0.4), top-K continuations, orders 1–6, `h⁰ = f + e`, amplification `√d·LayerNorm`. Installed by replacing `model.forward`; model.py and the checkpoint schema untouched, zero new parameters. |
| train.py | `eval_at` knob (extra eval steps, additive); `_loss` skips fusion when `prior_strength == 0` (so count arms are not logit-fused) | both default-off; golden check identical. |
| harness | `src/prism_prior_eval.py` | `--corpus`, `--sparse_context`, `--sparse_cmin`, `--sparse_mus`, `--count_cfg`, `--epoch_evals`; arms `dirs_only`, `dirs_prior`, `baseline_inffuse`, `dirs_inffuse`, `count`, `count_dirs` beside the original four; token-equivalent multiplier (`prism_metrics.py`) per arm; epoch-boundary evals; schema `prism-prior/1`, additive keys only (`config.prior_info`, `config.eval_at`, `runs[].arms[].token_equivalent`, `…arm_spec`, `…step_ms`). |
| runner | `prism_modal_alpha_w5.py` | branch `exp/alpha-w5`, volume `prism-alpha-w5`, prints the cost estimate before spawning. |
| selftest | `src/prism_selftest.py` (+15 checks, 82 total) | sparse prior == `ngram_floor.mix()` to 1e-5 (fp32 rows) / 1e-3 (fp16 rows); rows normalise; block-edge backoff is not uniform and equals the standalone shorter-context prediction; stupid-backoff ratio and 0.4 rules; COUNT shapes, zero parameters, zero-table reduction to the plain forward, √d·LN amplification, gradient flow. |

## The enwik8 context-6 build (this laptop, CPU)

Counted on the first 89M bytes (the Act-1 protocol: last 1M of train held out), μ from Act 1:

| order | contexts | distinct (k+1)-grams | fp16 rows + int64 keys |
|---|---|---|---|
| 0 | 1 | 205 | 0.0 MB |
| 1 | 205 | 18,065 | 0.1 MB |
| 2 | 18,065 | 236,005 | 9.4 MB |
| 3 | 236,005 | 1,131,172 | 122.7 MB |
| 4 | 1,131,172 | 3,292,711 | 588.2 MB |
| 5 | 3,292,711 | 6,720,191 | 1,712.2 MB |
| 6 | 6,720,191 | 11,431,002 | 3,494.5 MB |

Total **5.93 GB** on the device at `c_min = 1` (exact), built in 156 s. That fits a 24 GB
L4 beside the 10.65M model with >15 GB to spare, so **no pruning is needed**
(`--sparse_cmin` remains available). Sequential val score **1.9062 bits/byte** on bytes
90–95M — Act 1 (`results/ngram_floor_20260728T202028Z.json`) measured 1.9061 with
fp64 counts; the 1e-4 gap is the fp16 row storage. The dev-tail score is 1.8922.

In the harness the prior is counted on the **student's training slot**
(`data/shakespeare_eval/train.bin` after `setup()`, 72M bytes for enwik8 = the 80%
pool split the student trains on) so the prior sees exactly the model's training data;
the Act-1 μ are reused for enwik8 and the resulting standalone val number is recorded in
the artifact's `config.prior_info`.

## Deviations from the papers (stated)

- **COUNT amplification.** The paper's §3.3 formula is `√d·LayerNorm(h⁰)` with a learned
  gain and bias; we use a **parameter-free** LayerNorm so the arm adds zero parameters and
  the checkpoint schema is unchanged (the first block's `ln_1` can absorb a gain). `amp:none`
  is available. The paper's earlier description reads "d · LayerNorm" in one place; we
  follow the equation.
- **Positional embeddings.** COUNT's models use RoPE; nanoGPT adds learned absolute
  positions, so `x = drop(amp(e + f) + pos)`.
- **Gradient through B.** `f_t` is a weighted sum of unembedding rows; the paper does not
  detach, so gradients flow into the tied `wte` through `f_t` as well as through the head.
- **Table scope.** COUNT counts over its full pretraining corpus (Dolma 3, 797 GB table);
  ours is the training slot of a 72M-byte corpus at byte level, orders ≤ 6, exact (no
  `c_min` pruning, so "longest stored suffix" = longest seen suffix).
- **Scores.** Stupid backoff with the paper's 0.4 factor; top-K = 8 as in their training
  runs; orders combined by the plain average (J = 6) as in Eq. 1.
- **Logit fusion** is product-of-experts in log space (`logits + λ·log p`), the original
  Prior-Fused protocol, not COUNT's additive/multiplicative late-fusion variants.
- **Inference-only fusion** (`*_inffuse` arms) trains without the prior and fuses the same
  prior with the same λ at every validation point, so the two curves are scored on the
  same metric at every step.

## Pre-registration (to be committed to docs/NEXT-EXPERIMENTS.md before launch)

**Hypothesis.** Fusing a sparse context-6 n-gram prior into the logits *during training*
reaches the baseline's best enwik8 loss in fewer steps than fusing the same prior only at
inference, and composes multiplicatively with geometry-at-init (`dirs_only`), per the
Act 2 composition gate in NEXT-EXPERIMENTS.

**Arms** (enwik8, 3 seeds 1337–1339, 2,000-step native teacher, 1,500-step students,
eval every 10 with eval_iters 50, batch 32 × block 256, the §14 rig): `baseline`,
`dirs_only`, `prior`, `dirs_prior`, `baseline_inffuse`, `dirs_inffuse`, `count`,
`count_dirs`. Matched schedules (LR 1e-3, warmup 100) for all arms. λ = 1.0.

**Primary metric.** Steps-to-baseline-best (compute_score) against the in-run plain
baseline. Secondary: end-of-training loss, the token-equivalent multiplier, init loss.

**Decision rule.** Logit fusion earns its keep **only if `prior` beats `baseline_inffuse`
on steps-to-target in 3 of 3 seeds AND `dirs_prior` beats `dirs_inffuse` in 3 of 3**.
Otherwise COUNT's null (train-time ≈ inference-only, reported at 190M) is replicated at
this scale and README Finding 8 / the 30× caveat say so. The composition gate: the core
bet's "composition multiplies" thesis survives only if `dirs_prior`'s steps-to-target
ratio exceeds `dirs_only`'s by ≥ 1.5× (median); below 1.2× it saturates and
NEXT-EXPERIMENTS' Act 2 block records that before any Pythia spend.

**Kill condition.** If `prior` fails to reach baseline-best in any seed, or the prior's
standalone val number in `prior_info` is not within 0.02 bpb of Act 1's 1.9061, the run is
void and investigated, not interpreted.

**Framing.** The binding two-act framing spec applies: the prior's standalone number is
Act 1 and never a PRISM claim; "reached at init" is a flag, never a headline.

## Cost estimate

Per the runner (`SEC_PER_STEP = 0.27` from the committed enwik8 dirs_only run; prior/count
gather overhead measured on CPU as a ratio, see below): 3 seeds × (2,000 teacher + 8 arms ×
1,500 steps × overhead) ≈ **3.8 L4-h ≈ $3.02** at $0.80/h. Tier A; under the
$40 per-run cap.

CPU overhead measurement (tiny 2L/64d model, batch 4 × block 32, 20 steps): mean step time 5,753 ms plain vs 5,431 ms logit-fused vs 7,205 ms
eval-only-fused vs 5,141 ms COUNT — i.e. within run-to-run noise on a CPU-bound 10.65M model
(the gather is 7 `searchsorted` calls plus 8,192×256 row gathers per step; the COUNT arm adds
six (B,T,256)@(256,384) matmuls and a top-K). The runner therefore keeps conservative upper-bound
ratios (1.25× logit fusion, 1.05× eval-only, 1.4× COUNT); the artifact's per-arm `step_ms` gives
the measured GPU figure after the run.

## Launch (after the pre-registration commit)

```bash
git push origin exp/alpha-w5
modal run --detach prism_modal_alpha_w5.py --extra \
 "--corpus=enwik8 --sparse_context=6 --teacher_steps=2000 --student_steps=1500 \
  --eval_every=10 --eval_iters=50 --batch_size=32 --block_size=256 --seeds=1337,1338,1339 \
  --arms=baseline,dirs_only,prior,dirs_prior,baseline_inffuse,dirs_inffuse,count,count_dirs --tag=w5"
# fetch + commit:
modal volume get prism-alpha-w5 nanogpt-prism/results/prior_<stamp>.json ./results/
```

## CPU smoke

`prism_prior_eval.py --corpus=enwik8 --sparse_context=4 --teacher_steps=10 --student_steps=10
--eval_every=5 --eval_iters=2 --batch_size=4 --block_size=32 --seeds=1337 --device=cpu --tag=w5smoke`
over all eight arms: exit 0, conforming `prism-prior/1` artifact (then deleted — smoke artifacts
are not evidence). The context-4 prior counted on the 72M-byte student slot: 1,006,689 order-4
contexts, 645 MB, standalone val 2.3235 bits/byte (context 4; the context-6 launch prior is the
1.906 one above). `token_equivalent` correctly refused on the 3-point smoke curves ("need at
least 4 eval points"). Golden check after the harness swap: the legacy Shakespeare prior smoke is
still identical to `main`.
