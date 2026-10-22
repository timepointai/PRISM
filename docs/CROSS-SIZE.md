# Cross-size SVD projection

*Alpha-moves W3. Code: [`src/prism_crosssize.py`](../src/prism_crosssize.py);
runner: [`prism_modal_alpha_w3.py`](../prism_modal_alpha_w3.py); build notes:
[`alpha/W3-NOTES.md`](alpha/W3-NOTES.md). Status 2026-10-21: method, baselines,
selftests and CPU smokes done; GPU runs prepared, not launched. Every number
below that is not marked "reported by" will come from a committed `results/*.json`.*

## The question

EigenTransfer rotates a fresh model's singular directions toward a teacher's of
the *same shape*. The directions carry the from-scratch effect (RESULTS §11, §14,
§15), the spectrum is universal but weak, and analytic directions fail. So the
universal init worth building is directional — and the teachers worth mining
are large. That needs a map from a large model's SVD subspaces into a smaller
model's coordinates. Nobody has projected a large LM's SVD subspaces into a
smaller LM and trained it at ≥100M; the learngene group (FRONT, SPA-FRONT) works
in weight-index space one step away ([report](../reports/PRISM%20alpha%20in%20recent%20research.md)).

## The operator

Every matrix of a transformer reads from or writes to the residual stream, or
both through an intermediate width. The student's residual stream is given the
teacher's **principal coordinates**:

1. **Residual PCA.** Run the teacher on a calibration slice of its own training
   data (8 batches × 8 × 256 tokens, fixed seed); pool the residual vectors at
   every block input and the final LayerNorm's input; P ∈ R^{d_T×d_S} = the top
   d_S principal axes. (When d_T = d_S the identity is used: a full-rank
   rotation would change the function for nothing and is what makes the
   same-size case reproduce the teacher's directions exactly — selftested.)
2. **Project.** With y = x Wᵀ: residual-reading matrices become W P, residual-
   writing ones Pᵀ W, in general W̃ = P_outᵀ W P_in.
   - *MLP width* via PCA of the post-GELU hidden activations, P_h per layer
     (alternative: SVD truncation of the projected c_fc): W̃_fc = P_hᵀ W_fc P,
     W̃_proj = Pᵀ W_proj P_h.
   - *Attention heads* via the gauge-invariant per-head circuit products
     ([arXiv 2610.18471](https://arxiv.org/abs/2610.18471) names the GL
     symmetry that makes W_Q and W_K alone non-invariant): M_h = W_q_hᵀ W_k_h
     and O_h = W_v_hᵀ W_o_hᵀ, both d_T×d_T. Project PᵀM_hP, PᵀO_hP, re-factor at
     the student head dim by a rank-hd_S SVD (W̃_q = Σ^½U_kᵀ, W̃_k = Σ^½V_kᵀ,
     W̃_v = Σ^½U_kᵀ, W̃_o[:,h] = V_kΣ^½); the QK factor carries √(hd_S/hd_T) so
     the 1/√hd attention scale is invariant. **Head-count reduction keeps the
     heads with the largest projected circuit energy** (‖PᵀM_hP‖_F·‖PᵀO_hP‖_F).
   - *Embeddings* (same vocabulary only): wte P, wpe[:T_S] P.
3. **Depth.** Uniform layer selection: student layer i takes teacher layer
   ⌊i·L_T/L_S⌋ (12→6 = even layers). Averaging the L_T/L_S-layer groups is a
   flag (`--xsize_layers=average`), not the default.
4. **Fingerprint.** SVD each W̃ for U/Vᵀ (directions.pt in student shapes), and
   the teacher's own group-averaged spectra (the imprint is size-agnostic).
   `prism_eval.py --fingerprint=<dir> --method=dirs_only` then runs the normal
   EigenTransfer + imprint, unchanged.

**Alternative operator — Procrustes** (`--crosssize=procrustes`, precedent
[arXiv 2610.15322](https://arxiv.org/abs/2610.15322)): the teacher's PCA
coordinates are rotated onto a fresh student's own residual PCA coordinates on
the same tokens (R = UVᵀ of c_Tᵀc_S after unit scaling; P ← P_T R P_Sᵀ). It
needs the same tokenisation, so the GPT-2 case uses PCA.

**What the projection is not.** LayerNorm is elementwise, not rotation-
invariant, so the projected network is not function-preserving in the
student's coordinates. That is fine for PRISM: only the *directions* are
transplanted (then re-orthogonalised and imprinted); no weights are copied.

## μP

The brief asks for μP on teacher and student (width generalisation holds under
μP only — [arXiv 2610.00420](https://arxiv.org/abs/2610.00420), reported for
metanetworks). train.py carries `mup_base_width` (μP-lite: hidden init std
√(b/width), logit multiplier b/width, hidden LR multiplier b/width). The
pre-registered cases below run at the committed PROBE recipe first, with
`--extra "--mup_base_width=384"`-style μP arms as a prepared follow-up, because
the committed same-size reference (Run R) is non-μP and the first comparison
must be against it.

## Cases (probe scale, modern web, 3 seeds, the PROBE rig of Run R)

| case | teacher | student | what it isolates |
|---|---|---|---|
| 1 | 12L / 768 / 12 heads (hd 64), trained 2,000 steps on modern web | 6L / 384 / 6 heads (hd 64) | width + depth reduction at fixed head dim |
| 2 | 6L / 384 / 12 heads (hd 32) | 6L / 384 / 6 heads (hd 64) | head-count reduction at fixed width |
| 3 | 6L / 384 / 6 heads (hd 64) — Run R's teacher | 6L / 192 / 6 heads (hd 32) | head-dimension reduction — CrossGMN's failing case (0.74×, [arXiv 2610.01649](https://arxiv.org/abs/2610.01649), reported) |
| 4 | GPT-2 124M (12L / 768 / 12, BPE) — public weights | 6L / 384 / 6 (bytes) | a real large teacher; internal matrices only (vocabularies differ) |

**Baselines, every case:** scratch (the in-run baseline, same seed and schedule);
Weight Selection ([arXiv 2311.18823](https://arxiv.org/abs/2311.18823), our
re-implementation: first-N layers, evenly spaced element selection, consistent
indices, embeddings copied when the vocab matches); FRONT
([arXiv 2603.07523](https://arxiv.org/abs/2603.07523), our re-implementation of
the 3D-DCT truncate/pad operator, orthonormal DCT, r = 1.0, no FRONT+
refinement); mimetic init ([arXiv 2305.09828](https://arxiv.org/abs/2305.09828),
`--method=mimetic`, cases 1–3); and same-size `dirs_only` as the upper reference
(Run R, committed: 3.3× on modern web). All re-implementations list their
deviations in the module docstring and are recorded in `arm_specs`/`method_knobs`.

**Metrics:** `prism_score` (steps-to-target against the in-run scratch baseline),
the end-of-training gap, the token-equivalent multiplier (`prism_metrics.py`),
`[overlap]` traces against the projected teacher directions (`--track_overlap=50`),
and the teacher's compute stated (6·N·D; for GPT-2, "donor = public weights").

## Pre-registration (2026-10-21, before any launch)

- **Success** = cross-size `dirs_only` reaches the scratch baseline's best loss
  **≥ 1.5× faster on case 1 and > 1× on case 3, in 3 of 3 seeds**.
- A case where the projected init is *slower* than scratch in ≥ 2 of 3 seeds is
  reported as a negative in RESULTS §20 and README "Honest bounds", by name.
- Weight Selection and FRONT are reported on the same axes; if either beats the
  projection on case 1 in 3/3 seeds, the preprint's claim narrows to "a basis-
  level operator that matches index-space transfer", not "beats".
- The upper reference (same-size Run R, 3.3×) bounds what a perfect projection
  could recover; the fraction recovered is reported per case.
- Tier A: cases 1–4 at probe scale (~$11 total, table in W3-NOTES). Tier B
  (prepared, not launched): Pythia-1B/2.8B → 160M/410M on the W2b NeoX harness.

## Results (2026-10-22; RESULTS §20; artifacts in `results/`)

| case | teacher → student | steps-to-target vs scratch (median, range, seeds reached) | best loss (method vs scratch) | step-100 loss | artifact |
|---|---|---|---|---|---|
| 1 width | 12L/768/12h → 6L/384/6h | **1.0× (1/3 reached)** | 1.4924 vs 1.4892 | 2.25 vs 2.55 | `dirs_only_20261022T035754Z` |
| 2 heads | 6L/384/12h → 6L/384/6h | **3.7× (2.9–3.9), 3/3** | 1.3827 vs 1.4892 | 1.78 vs 2.55 | `dirs_only_20261022T040349Z` |
| 3 head-dim | 6L/384/6h → 6L/192/6h | **1.28× (1.24–1.34), 3/3** | 1.5855 vs 1.6384 | 2.52 vs 2.60 | `dirs_only_20261022T040448Z` |
| 4 GPT-2 | GPT-2 124M → 6L/384/6h (internal matrices) | **1.0× (1/3)** | 1.4900 vs 1.4892 | 2.57 vs 2.55 | `dirs_only_20261022T043832Z` |
| 4 FRONT | GPT-2 → same student (3D-DCT truncation, r=1) | never | 2.078 vs 1.4892 | 2.64 vs 2.55 | `init_state_20261022T041102Z` |
| ref | same-size dirs_only (Run R) | 3.3× (2.9–3.9) | 1.3794 vs 1.4892 | — | `dirs_only_20260727T154359Z` |

**Verdict against the pre-registration: FAIL on case 1, PASS on case 3.** The
width projection buys a real early head start (−0.31 nats at step 100, every seed)
that the student gives back by the horizon; the GPT-2 projection does nothing;
FRONT's truncated GPT-2 weights are far below scratch. The two cells that transfer
are the head cells: halving the head count at fixed width recovers the full
same-size effect (3.7× vs 3.3×; 112% of the reference), and halving the head
dimension gives 1.28× in every seed — CrossGMN's reported failure cell (0.74×,
reported by arXiv 2610.01649 at ~210k params, different task).

The overlap traces show the operator imposed the directions in every case
(step-0 overlap with the projected reference 0.58–0.93 against chance ≈ 0.03) and
that the student *keeps* them only in the head cases (case 2 FFN-up overlap rises
from 0.65 to 0.81 during training; cases 1 and 4 decay to 0.16 and 0.07). The
transplant works; the residual-coordinate map across widths does not carry
useful geometry at this scale. Weight-Selection baselines for cases 1–3 are
running; Procrustes and mimetic are prepared. Honest bounds: probe scale, one
corpus, one calibration slice, uniform depth selection only, no μP arm run yet.

## Deliverables

- `RESULTS.md` §20 (verdict in the title), `docs/CROSS-SIZE.md` (this file,
  updated with numbers), `docs/preprint/cross-size-projection.md` (DRAFT; the
  user decides whether anything is ever posted).
