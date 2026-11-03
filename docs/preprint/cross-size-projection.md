# Projecting a Transformer's Singular Subspaces into a Smaller Model: Cross-Size Geometry Transfer at Initialization

**DRAFT — DO NOT SUBMIT OR POST.** Timepoint Labs working draft, 2026-10-21. The
results section is a placeholder until the pre-registered runs in
[`docs/CROSS-SIZE.md`](../CROSS-SIZE.md) land; every number in it must point at a
committed `results/*.json`. External numbers are "reported by" their papers.

## Abstract

Trained transformers factor every weight matrix as W = UΣVᵀ. Prior work on
PRISM showed that transplanting a same-size teacher's singular *directions* (U, V)
into a fresh model at initialization — with no weights copied — reaches the
from-scratch baseline's quality 2.9–3.3× faster at 10.65M parameters on modern
web text and enwik8, and that the directions, not the spectrum, carry the effect
(random and analytic directions both fail). The limitation is dimensional: U and
V live in the teacher's residual-stream coordinates, so a larger teacher could
not seed a smaller student. We give an operator that removes that limit. The
student's residual stream is expressed in the teacher's principal coordinates
(PCA of the teacher's residual activations on a calibration slice); matrices that
read from or write to the residual stream are projected with those axes, the MLP
width through PCA of the hidden activations, and attention through the per-head
QK and OV circuit products — the gauge-invariant objects — re-factored at the
student's head dimension by a rank-truncated SVD. The projected matrices are
then used only for their singular directions, which EigenTransfer blends into
the fresh student. We evaluate four pre-registered cells against scratch and
against our re-implementations of Weight Selection, FRONT's frequency-domain
operator and mimetic initialization,
with steps-to-target, token-equivalent multipliers and subspace-overlap traces.
The cells are width + depth halving, width-only halving (which also halves the
head dimension, where learned cross-architecture operators fail), a head
re-partition at identical shapes, and GPT-2 124M into a 10.65M byte-level model.
At 10.65M parameters on byte-level web text (3 seeds), halving width alone
(6L/384 → 6L/192) gives 1.28× to the scratch baseline's best loss in every seed.
That is the cell where the learned CrossGMN operator reports 0.74×. Halving width
and depth together (12L/768 → 6L/384) buys an early head start that is gone by the
horizon (1.0×). Index-space Weight Selection (our re-implementation) beats the projection in both
size-changing cells: 4.0× on width + depth and 8.2× on width only. Projecting
from GPT-2 124M does nothing, and FRONT's truncated GPT-2 weights (our
re-implementation) trail scratch by
0.59 nats. The head re-partition cell keeps the same-size effect (3.7×), but it
changes no parameter shape, so it is not a size change. The pre-registered
criterion (≥ 1.5× on the width + depth cell, > 1× on the width-only cell)
therefore fails on the first and passes on the second.

## 1. Introduction

- Efficient-pretraining methods that reuse a smaller checkpoint (stacking,
  bert2BERT, LiGO, G_stack) copy weights and stop at parity; they need a
  same-family *smaller* model. Methods that go the other way — a large source
  into a small target — are new: FRONT keeps low-frequency DCT coefficients of
  weights stacked across layers and truncates or pads them to the target size
  (31.0% FLOPs saved for GPT-2 Base→Small, reported by arXiv 2603.07523);
  SPA-FRONT adds permutation alignment (up to 58.7% LM FLOPs, reported, abstract
  only); CrossGMN learns a cross-architecture operator at ≤210k parameters and
  fails on attention-dimension (0.74×) and head-count (0.86×) reduction
  (reported by arXiv 2610.01649); Weight Selection takes evenly spaced slices
  of a larger model's weights (vision only, reported by arXiv 2311.18823).
- All of these operate in weight-*index* space, which is why permutation
  alignment is needed. Singular directions are basis-level objects: they are
  defined up to the residual-stream coordinate system, and that is the only
  thing that has to be mapped.
- Contributions: (i) the projection operator; (ii) the first measurement of
  tokens-to-target for a cross-size SVD-direction init on a decoder LM, with
  the head-dimension case where the learned operator fails; (iii) the
  same-rig comparison against index-space transfer (Weight Selection, FRONT)
  and analytic init (mimetic), none of which had been run against an SVD-space
  method. The headline number is the width-only cell: 1.28× (1.24–1.34) against scratch, 3/3, with directions only and no weights copied. Weight Selection (our re-implementation) beats it there (8.2×) and in the width + depth cell (4.0×, where the projection is a null). As pre-registered, the claim is therefore "a basis-level operator that does not match index-space transfer at this scale", not "beats".

## 2. Method

### 2.1 Same-size EigenTransfer (background)
For each matrix, SVD the fresh init W_0 = U_0Σ_0V_0ᵀ and the teacher W_T =
U_TΣ_TV_Tᵀ; blend U ← orth((1−α)U_0 + αU_T), likewise V, with α = 0.75; impose
the teacher's group-averaged spectrum compressed to 8 DCT coefficients (Spectral
Imprint); W ← UΣ̃Vᵀ at the fresh init's Frobenius norm. Trained directions are
required: random directions (RESULTS §11) and never-trained analytic DCT bases
(§15) both fail.

### 2.2 The residual-coordinate map
Let x_ℓ ∈ R^{d_T} be the teacher's residual stream at block ℓ's input. On a
calibration slice we pool x_ℓ over positions and blocks (and the final-LN
input) and take the top-d_S principal axes P ∈ R^{d_T×d_S} (orthonormal
columns). With y = xWᵀ for a Linear weight W (out×in):
- residual-reading: W̃ = W P (out×d_S);
- residual-writing: W̃ = Pᵀ W (d_S×in);
- MLP: P_h ∈ R^{4d_T×4d_S} from PCA of the post-GELU hidden activations,
  W̃_fc = P_hᵀ W_fc P, W̃_proj = Pᵀ W_proj P_h (alternative: P_h = top left singular
  vectors of W_fc P);
- attention, per head h with head dim hd_T: the circuit products
  M_h = W_{q,h}ᵀ W_{k,h} and O_h = W_{v,h}ᵀ W_{o,h}ᵀ (both d_T×d_T) are invariant to
  the GL(hd) reparameterisation of each factor pair (arXiv 2610.18471), so they
  are what we project: M̃_h = PᵀM_hP, Õ_h = PᵀO_hP; a rank-hd_S SVD re-factors
  each (W̃_q = Σ^½U_kᵀ, W̃_k = Σ^½V_kᵀ; W̃_v = Σ^½U_kᵀ, W̃_o[:,h] = V_kΣ^½), with the
  √(hd_S/hd_T) factor keeping the 1/√hd attention scale invariant. If the
  student has fewer heads, the heads with the largest projected circuit energy
  ‖M̃_h‖_F‖Õ_h‖_F are kept.
- depth: student layer i takes teacher layer ⌊i·L_T/L_S⌋ (averaging the group
  is an option);
- embeddings (same vocabulary): wte P, wpe P.
The projected network is not function-preserving (LayerNorm is not rotation-
invariant); only the SVD directions of the W̃ are used, by EigenTransfer, with
the teacher's spectrum imprinted as before.

### 2.3 Procrustes variant
Rotate the teacher's PCA coordinates onto a fresh student's own residual PCA
coordinates on the same tokens (orthogonal Procrustes; precedent: SuperAI aligns
PCA subspaces across heterogeneous architectures, arXiv 2610.15322):
P ← P_T R P_Sᵀ. Same-tokenisation only.

### 2.4 Parameterisation
μP-lite (hidden init std √(b/d), logit multiplier b/d, hidden LR multiplier b/d)
is available for both teacher and student (arXiv 2610.00420 reports width
generalisation of weight-space maps under μP only); the first comparison is at
the committed non-μP recipe so it can be read against the same-size reference.

## 3. Experiments

Rig: nanoGPT, byte-level FineWeb-Edu slice (4.5M tokens), 3 seeds, dense eval,
matched method/baseline schedules (LR 1e-3, warmup 100, 1,500 steps), the
committed probe of RESULTS §11. Cases: (1) 12L/768/12h → 6L/384/6h (width and
depth halve together); (2) 6L/384/12h → 6L/384/6h (identical parameter shapes; only the
head partition changes, so this is same-size transfer); (3) 6L/384/6h → 6L/192/6h
(width only; the head dimension halves with it); (4) GPT-2 124M → 6L/384/6h, internal
matrices only. Baselines: scratch (in-run),
Weight Selection, FRONT (operator only, r = 1.0), mimetic; upper reference:
same-size dirs_only (3.3×, RESULTS §11). Metrics: steps-to-target
(`prism_score`), end-of-training gap, token-equivalent multiplier against a
fitted baseline curve (refused when the fit RMSE exceeds 0.01), subspace-overlap
traces of the student's top-16 directions against the projected teacher
directions, donor compute (6·N·D).

### 3.1 Results (3 seeds each; medians, ranges)
| case | method | prism_score (median, range; seeds reached) | best loss vs scratch | step-100 loss vs scratch | artifact |
|---|---|---|---|---|---|
| 1 | xsize (pca), 12L/768/12h → 6L/384/6h | 1.0× (1/3) | 1.4924 vs 1.4892 | 2.25 vs 2.55 | `results/dirs_only_20261022T035754Z.json` |
| 1 | xsize (procrustes) | prepared, not run | | | |
| 1 | weight selection (our re-implementation) | **4.0× (3.2–4.7; 3/3)** | **1.4026 vs 1.4892** | 1.73 vs 2.55 | `results/init_state_20261022T170647Z.json` |
| 1 | FRONT | prepared | | | |
| 1 | mimetic | prepared | | | |
| 2 | xsize (pca), head partition 12 → 6 at identical shapes (same-size transfer) | 3.7× (2.9–3.9; 3/3) | 1.3827 vs 1.4892 | 1.78 vs 2.55 | `results/dirs_only_20261022T040349Z.json` |
| 2 | weight selection (identical shapes: near-copy of the trained teacher; degenerate upper bound) | 49× (49.0–49.7; 3/3) | 1.3679 vs 1.4892 | 1.45 vs 2.55 | `results/init_state_20261102T214129Z.json` |
| 3 | xsize (pca), width only 384 → 192 (head dim 64 → 32 with it) | **1.28× (1.24–1.34; 3/3)** | 1.5855 vs 1.6384 | 2.52 vs 2.60 | `results/dirs_only_20261022T040448Z.json` |
| 3 | weight selection (our re-implementation) | **8.2× (7.5–8.3; 3/3)** | **1.4323 vs 1.6384** | 1.91 vs 2.60 | `results/init_state_20261102T214752Z.json` |
| 4 | xsize (pca, GPT-2 124M → bytes, internal matrices) | 1.0× (1/3) | 1.4900 vs 1.4892 | 2.57 vs 2.55 | `results/dirs_only_20261022T043832Z.json` |
| 4 | FRONT (GPT-2, r = 1.0) | never reaches scratch-best | 2.078 vs 1.4892 | 2.64 vs 2.55 | `results/init_state_20261022T041102Z.json` |
| ref | same-size dirs_only | 3.3× (2.9–3.9) | 1.3794 vs 1.4892 | — | `results/dirs_only_20260727T154359Z.json` |

Subspace-overlap traces (top-16, mean cos², chance ≈ 0.03): at step 0 the
students carry the projected directions in every case (0.58–0.93); in case 2 the
FFN-up overlap rises during training (0.65 → 0.81), in cases 1 and 4 it decays to
0.16 and 0.07. The transplant is faithful. Whether the width + depth failure comes
from depth, from the combination, or from the wider teacher is not separated by
these cells (the isolating cells 6L/768 → 6L/384 and 12L/384 → 6L/384 were not run).

Pre-registered success: ≥1.5× on case 1 and >1× on case 3, in 3/3 seeds.

## 4. Related work
FRONT / SPA-FRONT (frequency-domain learngenes across sizes; index space;
language baselines were scratch and distillation only); CrossGMN (learned
equivariant cross-architecture operator, toy scale, fails on attention-width
and head reduction); Weight Selection (slice a larger pretrained model; vision);
LiGO / bert2BERT / stacking (growth, copies weights, parity ceiling); mimetic
initialization (closed-form attention products); μP width generalisation of
weight-space maps (arXiv 2610.00420); Procrustes-aligned PCA subspaces across
architectures (arXiv 2610.15322); gauge symmetries of attention factors
(arXiv 2610.18471); PRISM's same-size results and ablations (RESULTS §11–§16).

## 5. Limitations
Probe scale (10.65M students, 85M–124M teachers); one modern-web corpus; the
projected network is not function-preserving and is used only for directions;
head selection by projected energy is a heuristic; FRONT is re-implemented
without its spectral-regularised refinement, Weight Selection without
distillation; no μP arm in the first comparison; the Pythia-1B/2.8B →
160M/410M extension is prepared, not run.

## Reproducibility
`prism_modal_alpha_w3.py --case <1-4> --method <xsize|xsize_procrustes|weightsel|front|mimetic>`;
every artifact records the commit, argv, both arms' curves and the projection
report (`crosssize.json` next to the fingerprint).
