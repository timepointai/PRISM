# W3 — cross-size SVD projection: build notes

*Branch `exp/alpha-w3` (from `alpha-moves` @ 630f181). Written 2026-10-21.
Nothing here has run on a GPU; every launch below is prepared, cost-estimated,
and waits for the session owner.*

## What was built

| file | what |
|---|---|
| `src/prism_crosssize.py` | the projection operator (residual-stream PCA; Procrustes variant), the fingerprint writer, the Weight-Selection and FRONT init builders, a CLI |
| `src/prism_eval.py` (additive) | `--teacher_n_layer/n_embd/n_head`, `--student_n_layer/n_embd/n_head` (both arms), `--crosssize=pca\|procrustes`, `--crosssize_hf=gpt2`, `--xsize_hidden`, `--xsize_layers`, `--init_builder=weightsel\|front`, `--front_r`; all fold into `method_knobs` (run key) and `config` |
| `src/prism_selftest.py` (+18 checks) | same-size projection reproduces teacher directions and every head's QK/OV circuit; student shapes; head selection; layer selection; orthonormal fingerprint; apply_prism consumes it; Procrustes recovers a known rotation; identity PCA at equal width; head-dim reduction; Weight Selection exact at equal size and stride-2 at half size; FRONT identity at equal size, shapes, r<1 masking; init_state loader round trip |
| `prism_modal_alpha_w3.py` | runner: 4 cases × 6 methods, six Volume slots, cost estimate per launch |
| `docs/CROSS-SIZE.md` | method, cases, baselines, metrics, pre-registration |
| `docs/preprint/cross-size-projection.md` | DRAFT preprint skeleton — results TODO; **not to be posted** |

No edits to `train.py` or `prism_init.py` (the W0 hooks `init_state`,
`init_mode`, `mup_base_width` were enough).

## Operator decisions (stated, as the brief asks)

- **Projector:** residual-stream PCA (teacher activations on 8×8×256 calibration
  tokens, fixed seed 20261021) — top-d_S axes pooled over every block input and
  the final-LN input. **Identity when d_T = d_S** (a full-rank rotation would
  change the function through the elementwise LayerNorm and buy nothing).
- **Hidden width:** PCA of post-GELU hidden activations per layer (default);
  SVD truncation of the projected `c_fc` as the alternative (`--xsize_hidden=svd`).
- **Heads:** per-head QK and OV circuit products projected, then rank-hd_S SVD
  re-factorisation; QK scaled by √(hd_S/hd_T). **Head-count reduction keeps the
  heads with the largest projected circuit energy** ‖PᵀM_hP‖_F·‖PᵀO_hP‖_F.
- **Depth:** uniform selection ⌊i·L_T/L_S⌋ (12→6 = even layers); averaging is a
  flag.
- **Spectra:** the TEACHER's group-averaged profiles (size-agnostic imprint);
  `--spectra_from=projected` available.
- **Embeddings:** projected (`wte P`, `wpe P`) when the vocabulary matches; left
  fresh for GPT-2 → bytes.
- **Procrustes:** teacher PCA coordinates rotated onto a fresh student's own
  residual PCA coordinates on the same tokens (unit-scaled; R = UVᵀ); same
  tokenisation only, so case 4 runs PCA.
- **Weight Selection (our re-implementation):** first-N layers, stride-⌊t/s⌋
  evenly spaced indices from 0 (the paper's worked example), the same index set
  for every tensor sharing a dimension, LN gains selected, embeddings copied
  when the vocab matches. No rescaling (none in the paper); no distillation.
- **FRONT (our re-implementation of the operator):** each weight type stacked
  across ALL teacher layers (L×out×in), `scipy.fft.dctn(norm='ortho')`,
  learngene mask at ratio r (default 1.0 — the truncation to the target already
  removes the high frequencies; the paper's plateau is r∈[0.5,0.83] and
  `--front_r` exposes it), truncate the high-frequency side to L_S×out_S×in_S
  (zero-pad to grow), `idctn(norm='ortho')`. No rescaling (none in the paper).
  Embeddings/LayerNorms fresh (the paper randomly initialises unselected
  modules). **FRONT+'s spectral-regularised refinement is not implemented.**
- **Not function-preserving:** the projected network is not the teacher's
  function in the student's coordinates (LayerNorm); only the SVD directions are
  transplanted. Said in docs/CROSS-SIZE.md.

## Smokes (CPU, tiny: teacher 10 steps, student 10, eval 5, seed 1337, batch 4) — all passed

Every smoke wrote a complete, conforming `prism-eval/1` artifact with
`config.crosssize` / `config.init_builder` / `teacher_dims` / `student_dims`
populated and the knobs folded into the run key; the smoke artifacts were
deleted afterwards (not evidence). The default run key is unchanged
(`dirs_only_t10_s10_e5_i2_seeds1337`), so historical runs still resume.

| smoke | result |
|---|---|
| dirs_only, `--crosssize=pca`, 4L/128/4h → 2L/64/2h, `--track_overlap=5` | pass; 10 direction sets; residual PCA explains 0.934 of variance; overlap traces banked |
| dirs_only, `--crosssize=procrustes`, same shapes | pass; Procrustes relative residual 1.135 (random student — expected near 1) |
| init_state + `--init_builder=front` / `weightsel` | pass (student 2L/64/2h, both arms) |
| mimetic at 2L/64/2h | pass |
| case-3 shape: 6L/384/6h teacher → 6L/192/6h student, `--crosssize=pca` | pass; 26 direction sets; PCA explains 0.990 |
| GPT-2 → 2L/64/2h on modernweb (`--crosssize_hf=gpt2`) | pass; 8 direction sets (internal matrices; embeddings fresh); PCA explains 0.909 |
| FRONT from GPT-2 (`--crosssize_hf=gpt2 --init_builder=front`) | pass |
| golden check (`golden.sh w3` vs main's `.evals`) | **identical** on all four smokes |
| `prism_selftest.py` | 85/85 (18 new) |

Bugs found and fixed on the way: the method arm was not receiving the student
architecture flags (only the baseline was — the first two cross-size smokes
silently ran a 6L/384 student that skipped the mismatched directions); a
missing module-level `pickle` import on the HF path; the hidden/layer knobs
were being recorded as `None` on every run (would have changed historical run
keys).

**Teacher step-time ratio, CPU check:** batch 8, block 256, iters 1–2:
10.65M model 1.32–1.37 s/step; 12L/768 (85.0M) 4.18–4.87 s/step ⇒ **≈3.4× on
this CPU**, against **7.6× by FLOP count**. The L4 sits in between (the 10.65M
model under-utilises it); the runner uses 7.6× as the conservative estimate.

## Cost table (L4 at $0.80/h; 3 seeds; teachers cached per slot after the first method)

Student step time measured on the committed Run R/X: 1,500 steps ≈ 413 s
(0.275 s/step incl. dense eval). Teacher-to-student step ratio for a 12L/768
model at T=256 by FLOP count, (6N + 12·L·d·T) per token: 538M vs 71M ⇒ **7.6×**.
(CPU check of the ratio: see the parent's status doc if measured; the L4 may
be less than 7.6× because the 10.65M model under-utilises it.)

| launch (`--case --method`) | teacher | students (2 arms) | total | est. USD |
|---|---|---|---|---|
| case 1, xsize (first on its slot) | 3 × 2,000 × 0.275 × 7.6 s ≈ 3.5 h | 3 × 2 × 1,500 × 0.275 s ≈ 0.7 h + evals 0.45 h | ~4.6 h | **$3.7** |
| case 1, each further method (teacher cached) | 0 | ~1.15 h | ~1.15 h | $0.9 |
| case 2, xsize (first) | 3 × 2,000 × 0.275 s ≈ 0.5 h | ~1.15 h | ~1.6 h | $1.3 |
| case 2, each further method | 0 | ~1.15 h | | $0.9 |
| case 3, xsize (first; 192-wide students ≈ half the step) | 0.5 h | ~0.8 h | ~1.3 h | $1.0 |
| case 3, each further method | 0 | ~0.8 h | | $0.65 |
| case 4, xsize (GPT-2 teacher = public weights, no training) | 0 | ~1.15 h | ~1.15 h | $0.9 |
| case 4, front / weightsel | 0 | ~1.15 h | | $0.9 |
| **all four cases × {xsize, xsize_procrustes (1–3), weightsel, front, mimetic (1–3)}** | | | **~17 h** | **≈ $14** |

Per-launch estimates stay far under the $40 per-run cap; the whole study is
Tier A. The runner prints the estimate before spawning (`--dry` prints only).

## Launch commands (after review; push `exp/alpha-w3` first)

```bash
# case 1 — the headline cell; run the methods on ONE slot so the 12L/768 teacher is trained once
modal run --detach prism_modal_alpha_w3.py --case 1 --method xsize --slot a
modal run --detach prism_modal_alpha_w3.py --case 1 --method xsize_procrustes --slot a   # after xsize lands
modal run --detach prism_modal_alpha_w3.py --case 1 --method weightsel --slot a
modal run --detach prism_modal_alpha_w3.py --case 1 --method front --slot a
modal run --detach prism_modal_alpha_w3.py --case 1 --method mimetic --slot a
# cases 2 and 3 in parallel on their own slots
modal run --detach prism_modal_alpha_w3.py --case 2 --method xsize --slot b
modal run --detach prism_modal_alpha_w3.py --case 3 --method xsize --slot c
# case 4 — no teacher stage
modal run --detach prism_modal_alpha_w3.py --case 4 --method xsize --slot d
modal run --detach prism_modal_alpha_w3.py --case 4 --method front --slot d
# optional μP arms (prepared): add --extra "--mup_base_width=384"  (teacher and student share the knob via train.py)
# fetch:  modal volume get prism-alpha-w3-<slot> nanogpt-prism/results/<method>_<stamp>.json ./results/
```

Sequential launches on the same slot share the teacher cache but must not run
concurrently (one run lock / data slot per Volume). `--track_overlap=50` can be
added through `--extra` for the subspace traces (both arms).

## Tier B (prepared only): Pythia-1B / 2.8B → Pythia-160M / 410M

The same operator applies to GPT-NeoX once W2b's port (`src/prism_neox.py`)
exists: residual PCA on Pythia-1B's (d=2048) residual stream over a Pile
calibration slice, P ∈ R^{2048×768} for the 160M student (or 2048×1024 for
410M), per-head circuits with Pythia's per-head QKV packing, parallel-residual
layers selected uniformly (16→12 or 16→24 is a *growth* in depth for 410M —
use `average`/repeat, state it), rotary dims untouched (rotary acts on q/k
after projection), untied embeddings projected separately (`embed_in P`,
`embed_out P`). Student = Pythia's published step0 init + EigenTransfer, trained
on the W2b harness (same token stream and schedule as the published runs);
baseline = step0 as-is. Cost sits with G2/G3 in `docs/NEXT-EXPERIMENTS.md`
(~$100 at 70M-scale smoke, ~$500 at 160M, 3 seeds). Not launched.

## Deviations from the brief

- μP: `mup_base_width` exists (W0) but the first comparison runs the committed
  non-μP PROBE recipe, so it is read against the same-size reference (Run R); μP
  arms are prepared via `--extra`.
- Case 2's teacher (6L/384/12h) and case 1's (12L/768/12h) are *new* teachers
  trained inside the run; only case 3 reuses Run R's teacher shape.
- FRONT+ refinement and Weight Selection + distillation are out of scope (operator-only re-implementations).
