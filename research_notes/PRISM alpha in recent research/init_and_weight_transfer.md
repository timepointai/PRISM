# Initializing networks from trained-model structure: weight-geometry transfer, model growth, cross-size projection (window 2026-09-21 → 2026-10-21)

Method and coverage notes, for the report writer:
- **Sources and search method.** I ran about 70 date-filtered arXiv API queries using `submittedDate` and `lastUpdatedDate` between 2026-09-21 and 2026-10-21, across all categories, then screened them to cs.LG/CL/CV/AI/NE and stat.ML. I also ran about 55 keyword queries against the OpenReview search API for **ICLR 2027 submissions**. The OpenReview API reports these submissions as made public on **2026-10-03** (`odate` field, checked on sample submissions), which puts them inside the window. They are anonymous and unreviewed, so treat their claims as preprint-grade. Several were *created* (`cdate`) in Aug–Sept 2026 but were not public until Oct 3.
- **Coverage limits.**
  - The arXiv API keyword matching is fuzzy and stemmed, so some relevant papers may have been missed.
  - Hugging Face Papers pages render client-side and could not be read.
  - OpenReview HTML and PDFs sit behind a browser challenge, so I saw **only abstracts** for ICLR 2027 submissions, with no full text.
  - Semantic Scholar was not queried.
- **Out-of-window items** appear only as labeled context, with their dates.
- **PRISM baseline used for comparison.** Same-architecture teacher. Per-matrix SVD directions are blended 75% toward the teacher and re-orthogonalized (EigenTransfer). The spectrum is compressed to 8 DCT coefficients per weight group (Spectral Imprint). No weights are copied. Results: 2.9–3.3× faster to baseline-best loss at 10.65M params.

---

## Q1. Has anyone in the window published geometry-only or spectrum/subspace-based initialization from trained models, and how close is it to EigenTransfer + Spectral Imprint?

### Takeaway
No in-window paper does what EigenTransfer does: rotate a fresh model's own singular vectors toward a same-architecture teacher's and re-orthogonalize, with no weight copy. The direction-transfer core of PRISM is **not scooped**.

The neighborhood got crowded, though. Four results bear on it:
- **SAHyper** combines *pretrained singular values* with *generated orthogonal bases*. That is the mirror image of PRISM: Spectral Imprint without EigenTransfer.
- **"What Transfers from Non-Visual Procedural Warm-Up?"** shows that refactoring a teacher's QK/OV *operators* into fresh projection matrices keeps 103–121% of the transfer gain. This is strong external support for "transfer geometry, not weights."
- **Procedural Core** reports that rank truncation and component shuffling destroy its gains, so structure matters beyond statistics.
- **"A Lasting Bias of Initialization"** is the main threat. In ViTs, imposing **108 scalar statistics** on a random init matched the gains of procedural warm-up.

### Cited Findings

#### Master inventory of in-window items (verdicts relative to PRISM)
| Date (public) | ID / URL | Core claim | Scale tested | Verdict vs PRISM |
|---|---|---|---|---|
| 2026-09-29 (arXiv); also ICLR'27 sub | [arXiv 2609.37631 Procedural Core](https://arxiv.org/abs/2609.37631) | A ~1M-param recurrent transformer is trained on procedural data, then its weights are tiled to init transformers of any width and depth. ViT-B ImageNet top-1 is 79.8 vs 77.6 (default) and 79.5 (mimetic). LM: ~4% lower validation perplexity. | ViT-B 85M. GPT-2 124M on 2B tokens of FineWeb-Edu and CodeParrot. | **Closest "reusable trained-structure init."** It validates "structure > statistics" and is composable, since it could serve as a data-free teacher. It partly threatens cross-size novelty, but only in vision. |
| 2026-10-01 | [arXiv 2610.01649 CrossGMN](https://arxiv.org/abs/2610.01649) | An equivariant graph metanetwork maps (trained source, target init) to a refined smaller target init. Median distillation speedup is up to 8.89×. | ≤ ~210k params; tiny ViTs | **Occupies the cross-size cell only at toy scale.** It *fails* when shrinking attention width or heads. That leaves a gap PRISM could fill. |
| 2026-09-30 | [arXiv 2610.00420 Transferable Graph Metanetworks](https://arxiv.org/abs/2610.00420) | Weight-space metanetworks generalize across input widths, robustly up to 42× the training width, under μP. | Small nets | **Composable.** Evidence that μP makes trained weight geometry width-consistent, which matters for cross-size projection. |
| 2026-10-03 (ICLR'27) | [OpenReview A04pxSDaxO "Reparameterize Before You Extract" (SPA-FRONT)](https://openreview.net/forum?id=A04pxSDaxO) | Training-free permutation alignment raises low-frequency DCT energy of pretrained weights before FRONT extraction, for variable-sized init. Claims 30.0% fewer epochs to target in vision and up to 58.7% fewer LM pretraining FLOPs. | LM sizes not visible (abstract only) | **Scoop risk for the cross-size bet.** The learngene group already does cross-size, DCT-compressed transfer of trained weights. It moves (compressed) weight *values*, not SVD geometry. |
| 2026-10-03 (ICLR'27) | [OpenReview 4PLiP3jL6H SAHyper](https://openreview.net/forum?id=4PLiP3jL6H) | A graph hypernetwork whose "Structure-Aware Spectral Inheritance combines pretrained singular values with generated orthogonal bases." It generates a ViT-B at 63.86% top-1 with no training. | ViT-B | **Partial scoop of the Spectral Imprint idea**, in a hypernetwork setting and judged on zero-shot weights rather than training speed. |
| 2026-10-03 (ICLR'27) | [OpenReview II0OaQcnmX "What Transfers from Non-Visual Procedural Warm-Up?"](https://openreview.net/forum?id=II0OaQcnmX) | QK and OV operators from a Dyck-warmed ViT are refactored into fresh projections, with the rest randomly initialized. ViT-B/IN-1K top-1 is 80.54 vs 77.72 (scratch) and 79.50 (full warm-up). | ViT-B, IN-1K | **Validates the "geometry not weights" principle.** Composable: target the QK/OV operator subspaces. |
| 2026-10-03 (ICLR'27) | [OpenReview DzLzZYD84Y "A Lasting Bias of Initialization"](https://openreview.net/forum?id=DzLzZYD84Y) | 108 layerwise scalar statistics taken from warmed-up ViTs and imposed on a random init match or exceed the warm-up gains. | ViTs | **Threat / required control.** It suggests statistics alone may explain some init gains. |
| 2026-10-03 (ICLR'27) | [OpenReview XubgOIuvFd Config2Weights](https://openreview.net/forum?id=XubgOIuvFd) | A generative model trained on a collection of pretrained LLM weights produces an init for a new architecture from its config alone. "Competitive" with weight-expansion methods, including on unseen architectures. | Unspecified in abstract | **Threatens the cross-size cell from the generative side.** Needs a model zoo. Evidence in the abstract is weak. |
| 2026-10-03 (ICLR'27) | [OpenReview 1YZBBp90ee "A Night at the LLM Zoo"](https://openreview.net/forum?id=1YZBBp90ee) | A zoo of 300 GPT-2s (51M, 124M, 355M; N=100 each) across data mixtures. A flow-matching generator over Gram-PCA-compressed weights produces complete pretrained LLMs. Code released. | ≤355M | **Composable.** A test bed for PRISM fingerprints across data mixtures. |
| 2026-10-03 (ICLR'27) | [OpenReview YMkHmZjJ6B "Optimizer Lineage"](https://openreview.net/forum?id=YMkHmZjJ6B) | Muon pretraining raises per-layer stable rank 6–14× vs Adam. The gap is causal and survives an iso-loss control. | 124M, 350M GPT | **Matters for PRISM generality.** The teacher's optimizer stamps its spectrum. |
| 2026-09-25 | [arXiv 2609.35852 Row/Column Scale Fields](https://arxiv.org/abs/2609.35852) | Row and column log-RMS "scale fields" of trained weights are consistent across Pythia sizes. Flattening channel gain raises loss while matrix norms are preserved. | Pythia, 4 sizes | **Composable.** A non-spectral fingerprint that may transfer across sizes. |
| 2026-10-03 (ICLR'27) | [OpenReview lUcBzk2cKw "Weight-Scale Growth Tracks Training Effort"](https://openreview.net/forum?id=lUcBzk2cKw) | Weight-scale growth tracks AdamW optimization work, not learning quality. | Small LMs | **Validates** PRISM's "spectrum is necessary but not sufficient." |
| 2026-09-28 | [arXiv 2609.35745 "Copy the Same, Distill the Difference"](https://arxiv.org/abs/2609.35745) | For softmax→linear ViTs, MLP weights are operator-agnostic and copyable, while attention weights are operator-specific and copying barely helps. | Various ViT sizes | **Informs per-matrix priorities** for cross-architecture transfer. |
| 2026-10-13 | [arXiv 2610.17914 LinMorph](https://arxiv.org/abs/2610.17914) | A learned weight-space morphism maps softmax-attention params to Gated DeltaNet inits for video DiTs. "Efficient linearization depends… critically on how pretrained knowledge is transferred before recovery." | Video DiTs | **Adjacent** (cross-operator init). |
| 2026-10-17 | [arXiv 2610.24664 Knowledge Diversion](https://arxiv.org/abs/2610.24664) | Learngene group: random-init weights are factorized into rank-one "Knowledge Units," which pretraining organizes into learngenes and tailors. | Image generation | **Adjacent.** Shows the learngene group is active and productive. |
| 2026-09-30 / 2026-10-03 | [arXiv 2609.39827](https://arxiv.org/abs/2609.39827); [OpenReview 73o2bc3VP2 "Rethinking Pre-pretraining"](https://openreview.net/forum?id=73o2bc3VP2) | Conflicting results on whether init-like synthetic warm-ups survive scale (see Q5). | 500M–7B; ≤881M | **Q5 evidence.** |
| 2026-10-03 (ICLR'27) | [OpenReview a8SnZ5mFCw "Small LLMs: Pruning vs. Training from Scratch"](https://openreview.net/forum?id=a8SnZ5mFCw) | Pruned Llama-3.1-8B inits beat random at matched tokens, but the edge narrows as tokens grow. | 8B parent | **Q5 evidence.** |
| 2026-10-03 (ICLR'27) | [OpenReview 9YVNlDplEi "How Model Growth, Boundary Operators, and Recursion Improve Scaling Laws"](https://openreview.net/forum?id=9YVNlDplEi) | Untied-Grow gives a 1.55× compute-efficiency gain at 10^20 FLOPs and *improves the scaling exponent*. | 10^18–10^20 FLOPs, extrapolated to 10^21 | **Bar to beat and composable** (Q2 and Q5). |

#### Detail on the closest relatives
- **Procedural Core: method.** Expansion is *tiling, not SVD*. Depth: first and last blocks are copied and the recurrent middle block is replicated. Width: `W̃_ij = W_(i mod d_out),(j mod d_in)`, rescaled to the source Frobenius norm. — [arXiv 2609.37631 (HTML)](https://arxiv.org/html/2609.37631v1)
- **Procedural Core: ablations that echo PRISM.** "Rank truncation hurts Procedural Core much more sharply than the direct warm-up weights." Shuffling components while keeping weight statistics shows "V/proj causes the largest drop; Q/K keeps most of the gain." A parameter-matched non-recurrent core gives +1.7% on CIFAR-100, vs +4.7% for the recurrent core. — [arXiv 2609.37631 (HTML)](https://arxiv.org/html/2609.37631v1)
- **Procedural Core: its LM experiment is not cross-size.** The 12-layer, 768-wide, tied-middle-block auxiliary model was trained 2,000 steps (~131M tokens) of k-Dyck. The 124M target is the same shape. Gain is ~4% validation perplexity after 2B tokens. No tokens-to-target figure is given, and no code is released yet ("upon publication"). — [arXiv 2609.37631 (HTML)](https://arxiv.org/html/2609.37631v1)
- **SAHyper.** "Structure-Aware Spectral Inheritance combines pretrained singular values with generated orthogonal bases and coordinates related bases within attention blocks." — [OpenReview 4PLiP3jL6H](https://openreview.net/forum?id=4PLiP3jL6H)
- **"What Transfers…": operator transfer beats raw copying.** On CIFAR-100 and Tiny-ImageNet, "late-block QK+OV initialization reaches 75.17 ± 0.30% … and 60.41 ± 0.64% … compared with 74.74 ± 0.51% and 60.30 ± 0.70% for full warm-up transfer, retaining 121% and 103% of its gain over scratch." — [OpenReview II0OaQcnmX](https://openreview.net/forum?id=II0OaQcnmX)
- **"What Transfers…": factorization coordinates matter.** "Exact alternative factorizations further show that optimization can depend on factor coordinates even when the transferred operators are fixed." — [OpenReview II0OaQcnmX](https://openreview.net/forum?id=II0OaQcnmX)
- **"What Transfers…": source matters.** With a supervised-ImageNet source, the same QK+OV intervention "recovers only 15% of the full transfer gain." — [OpenReview II0OaQcnmX](https://openreview.net/forum?id=II0OaQcnmX)
- **"A Lasting Bias of Initialization."** "We extract a total of 108 of these scalar statistics … and show that imposing them on a randomly initialized ViT is sufficient to match or exceed the improvements from procedural warm-up." The statistics include effective weight scales, LayerNorm stats, low attention entropy, negative MLP pre-activations, and late-layer residual write ratios. — [OpenReview DzLzZYD84Y](https://openreview.net/forum?id=DzLzZYD84Y)
- **Optimizer Lineage.** Muon "raises the per-layer stable rank of hidden matrices by 6–14× (from ~17 to ~121)" vs Adam. At 350M a "~12–25× stable-rank gap" persists under an iso-loss control. — [OpenReview YMkHmZjJ6B](https://openreview.net/forum?id=YMkHmZjJ6B)
- **Row/column scale fields.** "Edits of a frozen checkpoint separate reciprocal scale balance … from relative channel gain: flattening the gain increases in-distribution loss while preserving matrix norms and the balanced core." — [arXiv 2609.35852](https://arxiv.org/abs/2609.35852)
- **Pre-window corroboration of PRISM's spectrum-only null (2026-07-10, v1 only).** "Complexity-Guided Component-wise Initialization" imitated the component-wise magnitudes and spectral profiles of 11 pretrained GPT-2-style checkpoints. It found the spectral inits change spectral structure but give no performance advantage, and reusing pretrained weights stays competitive. The authors conclude that "effective reuse likely requires preserving richer information." — [arXiv 2607.09204](https://arxiv.org/abs/2607.09204)
- **Pre-window context (2026-02).** "Beyond Student: An Asymmetric Network for Neural Network Inheritance" reportedly initializes a new network's modules from a teacher by truncated SVD. I saw only a search snippet and did not verify it. — [arXiv 2602.09509](https://arxiv.org/pdf/2602.09509)

### Inferences
- **EigenTransfer is still unoccupied.** It keeps a fresh model's own random singular vectors and rotates them partway toward the teacher's. No in-window work does this, so PRISM's "directions, not weights" claim stands as distinct. The two nearest neighbors bracket it from either side:
  - SAHyper keeps the teacher's *spectrum* with new bases.
  - II0OaQcnmX keeps the teacher's full *operator* with new factor coordinates.
  - PRISM can position itself as the controlled middle point: directions with partial blending, spectrum compressed to 40 numbers.
- **Independent corroboration of PRISM's ablations.** "Spectrum-only is insufficient" now has independent support (2607.09204 pre-window, plus Weight-Scale Growth and Optimizer Lineage in window). "Structure > statistics" is supported by Procedural Core's rank-truncation and shuffling ablations. PRISM should cite these as convergent evidence.
- **The main threat to PRISM's mechanism claim is the 108-statistic result.** PRISM's existing controls (spectrum-only; DCT analytic directions) do not match *activation* statistics such as attention entropy, LayerNorm stats, and residual write ratios. A **statistics-matched control** would directly answer this critique: impose the teacher's layerwise activation and weight statistics with random directions.
- **Teacher and student optimizer may matter.** Optimizer Lineage implies a Muon teacher and an Adam teacher carry very different spectra. PRISM should record which optimizer its teacher used. A Muon student may also wash out an imposed spectrum faster.
- **II0OaQcnmX suggests targeting QK/OV geometry.** Its finding that late-block QK/OV operators carry most of the effect, together with Procedural Core's finding that V/proj is most structure-sensitive, suggests a cheap PRISM ablation: apply EigenTransfer to QK/OV only versus MLP only.

### Gaps
- I could not read full texts of ICLR 2027 submissions (OpenReview challenge). Model sizes, baselines, and whether any of them compare against mimetic or SVD-direction inits are unknown beyond the abstracts.
- I found no in-window paper that tests SVD-direction transfer *with re-orthogonalization* from a same-architecture natural-data teacher, so there is no direct replication or refutation of EigenTransfer.
- Procedural Core's code is not released, so its expansion-plus-LM claim cannot yet be independently checked.

---

## Q2. What's new in model growth, width/depth expansion, weight subcloning, and growing from a small checkpoint, and what savings and scales are reported?

### Takeaway
Growth work in the window is active and increasingly scale-credible:
- Untied-Grow claims an **exponent** improvement, 1.55× compute efficiency at 10^20 FLOPs.
- KITE grows to a 67B MoE.
- Growing to See saves 13.9–24.8% of FLOPs on SSL ViTs.

New theory and diagnostics show where growth loses value. River-Hill finds an LR reset discards 45–50% of the source's annealing progress. I found no new HyperCloning, LiGO, or subcloning successor papers in the window. Growth still copies weights, which PRISM does not, so the two are composable rather than competing.

### Cited Findings
- **Untied-Grow scaling ladders.** "More than 30 distinct scaling ladders over compute budgets of approximately 10^18 to 10^20 FLOPs… our best architectural variant, *Untied-Grow*, achieves a 1.55× compute-efficiency gain at 10^20 FLOPs, and the trend extrapolates to 10^21 FLOPs, where our model matches GPT-3 13B on CORE." Better architectures "can improve the *exponents*." — [OpenReview 9YVNlDplEi (public 2026-10-03)](https://openreview.net/forum?id=9YVNlDplEi)
- **River-Hill Budgets.** When an annealed source is grown and its LR schedule reset, "the river-valley view predicts that 45% of progress made during the decay to 10% peak learning rate is lost upon reset, and up to 50% for a decay to zero." The growth operator does not recover this (standard BERT configuration). — [OpenReview VQ3NV9KF1P](https://openreview.net/forum?id=VQ3NV9KF1P)
- **KITE / Step Scale Transformer.** It trains "from a smaller size to a larger size (i.e., saving training costs via upcycling)" and places new parameters where they don't affect attention KV. At comparable cumulative compute, a 67B MoE (2.15B active) achieves lower training loss than 47B and 63B MoE baselines, while cutting estimated inference cost by 6.7% and 31.6%. — [arXiv 2609.27294 (2026-09-23)](https://arxiv.org/abs/2609.27294)
- **Growing to See.** Progressive depth growth for MAE and DINOv3 "improves downstream performance while reducing training FLOPs by 13.9–24.8%, with ImageNet linear-probe accuracy gains of up to 3.36 percentage points." — [arXiv 2610.16817 (2026-10-12)](https://arxiv.org/abs/2610.16817)
- **Transformer Depth Up-Scaling, infinite-depth view.** Depth up-scaling is formalized as grid refinement of a controlled ODE. Scaled stacking gives O(L^-1) terminal-state error and O(L^-1)-consistent directional derivatives. Layer concatenation degrades more with base depth, while in-place stacking is more stable. Tested on Qwen3 and with GPT-2 progressive training. — [OpenReview jqkmTgKxbg](https://openreview.net/forum?id=jqkmTgKxbg)
- **Depth growth and ICL.** For models up to 3B, grown and looped models "acquire ICL earlier" and "depth growth propagates early induction heads across layers." — [OpenReview MOIKCu9eSG](https://openreview.net/forum?id=MOIKCu9eSG)
- **What growth adds.** A Transformer grown from 253M to 857M parameters adds K=36 functionally independent directions, and all of them persist through continual learning. — [OpenReview 0xESqqBEgu](https://openreview.net/forum?id=0xESqqBEgu)
- **Delayed Depth Expansion.** At 155M and 314M, training a shallower dense model first and converting it to looped depth later "substantially reduces cumulative Transformer-block FLOPs at matched validation loss." — [OpenReview L49FACvzY7](https://openreview.net/forum?id=L49FACvzY7)
- **Sketch-Schur-OST.** Function-preserving expansion that picks new directions complementary to the inherited tangent space and conditions their coordinates for the optimizer. Abstract only; no scale stated. — [OpenReview VBI2qYva2x](https://openreview.net/forum?id=VBI2qYva2x)
- **CAGE-NAS.** Grows only when a certified functional-gradient step fails, then applies function-preserving expansion. Results are in a controlled toy setting: 99.8th percentile of admissible architectures by held-out RMSE. — [arXiv 2610.01173 (2026-10-01)](https://arxiv.org/abs/2610.01173)
- **Upcycling from trained checkpoints (weight-copying; title/abstract level).**
  - UpLoop, from Qwen3-4B Base: +6.17% average accuracy over 20 benchmarks at R=3. — [arXiv 2610.25004](https://arxiv.org/abs/2610.25004)
  - MASKerade, dense-to-MoE. — [arXiv 2610.07809](https://arxiv.org/abs/2610.07809)
  - STAPLE, static sparse upcycling. — [arXiv 2610.12599](https://arxiv.org/abs/2610.12599)
  - DivMoE. — [arXiv 2610.11317](https://arxiv.org/abs/2610.11317)
  - GrowRL grows zero-initialized attention and FFN "pathways" for RL and transfers them across backbones. — [arXiv 2610.16489](https://arxiv.org/abs/2610.16489)
- **Pre-window context (May 2026).** "When is Warmstarting Effective for Scaling Language Models?" reports that the warm-start advantage over scratch holds only within a bounded token budget, with ≤20 tokens/parameter more likely to show gains. This is from a search snippet; I did not read the full text. — [arXiv 2605.13405](https://arxiv.org/abs/2605.13405)
- **Pre-window landmarks for reference (dates as published).**
  - G_stack (2024-05) matched a 7B baseline's loss at 300B tokens using 194B tokens (54.6% speedup). — [arXiv 2405.15319](https://arxiv.org/abs/2405.15319)
  - HyperCloning (2024-09) is function-preserving width expansion tested on OPT, Pythia, and OLMo. — [arXiv 2409.12903](https://arxiv.org/abs/2409.12903)

### Inferences
- **Growth sets PRISM's real bar at scale.** Growth's credible numbers (1.55× at 10^20 FLOPs; 54.6% at 7B, pre-window) are what PRISM's pre-registered bet (a) on Pythia/OLMo 410M–1B must be compared against. A tokens-to-target ratio meaningfully above ~1.5× at ≥410M would be notable. At ~1.1× it would sit below growth.
- **River-Hill supports PRISM's no-weight-copy design.** Weight-copying growth suffers an LR-reset offset that discards annealing progress. PRISM starts a fresh model on a fresh schedule, so it should not pay this offset. That is an argument *for* the design, though no paper tests it.
- **Growth and PRISM look composable.** One option: grow a small model, then imprint geometry from a larger teacher on the newly added blocks instead of zero or noise init. Another: use PRISM to initialize the seed model of a growth ladder. I found no paper doing either.

### Gaps
- I found no new HyperCloning, LiGO, bert2BERT, MSG, or weight-subcloning *successor* papers in the window. The arXiv keyword search may have missed some.
- Untied-Grow's operator details and its tokens-to-target at fixed loss are not in the abstract.
- KITE's exact training-compute savings relative to from-scratch are not stated in the abstract. It reports loss at comparable compute.

---

## Q3. Any work on cross-size or cross-architecture transfer of subspaces or structure? Is PRISM's cross-size cell still empty?

### Takeaway
**The cell is not empty, but the specific sub-cell is.** Several groups now transfer trained structure across sizes:
- The learngene group's **FRONT / SPA-FRONT** keeps low-frequency DCT coefficients of trained weights, then truncates or pads them to the target size. FRONT reports 31–47.5% FLOPs savings on BERT/RoBERTa/GPT-2 Base→Small. SPA-FRONT claims up to 58.7% LM FLOPs savings.
- **Procedural Core** tiles a 1M-param core to ViT-B.
- **Config2Weights** generates LLM inits for unseen configurations.
- **CrossGMN** learns cross-architecture maps at toy scale, and fails on attention-width reduction.

Still unoccupied: projecting a large natural-data-trained LM's **SVD subspaces** (basis-level geometry, not raw weight values) into a smaller LM, validated by tokens-to-target at ≥100M. FRONT is the scoop risk to watch. Its group publishes rapidly and already operates on frequency-compressed trained structure across sizes.

### Cited Findings
- **SPA-FRONT (ICLR 2027 submission, public 2026-10-03).** FRONT shows "low-frequency components of pre-trained weights are informative for foundational and transferable knowledge." SPA exploits the fact that "functionally equivalent parameterizations within the same permutation-equivalence class can exhibit different spectral energy distributions." It uses depth-wise bipartite matching and width-wise anchored joint permutation. Claimed results: "reducing vision-model training epochs to reach target accuracy by 30.0% on average and cutting language-model pre-training FLOPs by up to 58.7%." Code: anonymous.4open.science/r/SPA-FRONT. Keywords: "Model Initialization, Variable-Sized Models, Permutation Invariance." — [OpenReview A04pxSDaxO](https://openreview.net/forum?id=A04pxSDaxO)
- **FRONT base method (pre-window; v1 2026-03-08, v3 2026-06-03).** Authors: Shen, Feng, Xie, Lv, Geng. It applies a **3D-DCT** to each weight type stacked across layers (L × d_in × d_out), keeps a low-frequency block, then zero-pads or truncates in frequency space to the target depth and width before the inverse DCT. LM experiments use BERT, RoBERTa, and GPT-2, Base (12-layer) → Small (6-layer), on GLUE. FLOPs reductions are BERT-S 47.5%, RoBERTa-S 42.9%, GPT-S 31.0% (average 40.5%). Language baselines are only scratch and knowledge distillation. Vision baselines include Mimetic-Init, Weight Selection, LiGO, and learngene variants, with up to 15× faster convergence. The largest source is DeiT-B (86.7M). Code: github.com/LUcy0505/FRONT. — [arXiv 2603.07523](https://arxiv.org/abs/2603.07523); [HTML v3](https://arxiv.org/html/2603.07523v3); [GitHub](https://github.com/LUcy0505/FRONT)
- **Contested FRONT text.** A Pith review said the manuscript text it received "did not state, derive, or test the core low-frequency claim." This may reflect a mismatched text version. — [Pith](https://pith.science/paper/2603.07523)
- **CrossGMN.** It reformulates cross-architecture operators to take "a trained source network and an initialization of the target network," invariant to source permutations and equivariant to target permutations. It refines a pruning-based anchor init. Results:
  - Median speedups: 8.89× on MNIST INRs for width+depth reduction; 3.78× cross-dataset.
  - ViTs: depth reduction 1.21×, head+depth 1.18×.
  - Head-count-only reduction 0.86× and attention-dimension reduction 0.74× (no speedup).
  - Largest model is ~210k params. Upscaling and cross-family transfer are left to future work.
  — [arXiv 2610.01649](https://arxiv.org/abs/2610.01649); [HTML](https://arxiv.org/html/2610.01649v1)
- **Transferable Graph Metanetworks.** Metanetworks are made invariant to "the ways in which networks of different widths represent the same function." Performance "remains robust up to 42× the training width" on μP-trained inputs, with proofs via infinite-width theory. They "explain why it can fail under other parameterizations." — [arXiv 2610.00420](https://arxiv.org/abs/2610.00420)
- **Config2Weights.** "A generative initialization framework that learns from a collection of pretrained LLM weights to initialize new architectures conditioned solely on their architectural configuration… including unseen architectures and configurations… competitive performance against existing weight-expansion methods." — [OpenReview XubgOIuvFd](https://openreview.net/forum?id=XubgOIuvFd)
- **Procedural Core (cross-size via tiling).** It expands "a 1M-parameter core to initialize an 85M-parameter ViT-Base," improving ImageNet top-1 by 2.2 pp. Gains grow with depth on CIFAR-100 ViT-Tiny, from +2.0% at 6 layers to +2.8% at 24. — [arXiv 2609.37631](https://arxiv.org/abs/2609.37631); [HTML](https://arxiv.org/html/2609.37631v1)
- **Nested and elastic alternatives (sub-models without projection).**
  - ShellElastic trains nested sub-networks in one supernet with a μP extension (μE) so all sub-networks are maximally updated at once. Its sub-networks "outperform MatFormer." — [arXiv 2610.15124](https://arxiv.org/abs/2610.15124)
  - "Your Transformer Is Secretly Elastic" turns pretrained transformers from 22M to 8B into a budget continuum by jointly ranking neurons and heads, with no retraining. — [OpenReview nEDVIjb07v](https://openreview.net/forum?id=nEDVIjb07v)
- **Cross-architecture functional subspaces (not init).** SuperAI aligns PCA subspaces of source and host models with orthogonal Procrustes to inject capabilities across heterogeneous architectures. — [arXiv 2610.15322](https://arxiv.org/abs/2610.15322)
- **Learned subspaces for compression (a possible projection operator).** LSP learns orthogonal projectors per layer, initialized from whitened SVD truncation, end-to-end against the dense model's KL. Tested on OPT-125M/1.3B, Qwen3-4B, and Llama-2-7B. At −70% compression, Llama-2-7B reaches 10.9 WikiText-2 perplexity vs 13.3 for the best baseline. — [arXiv 2609.40127](https://arxiv.org/abs/2609.40127)
- **Pre-window context.** "Geometric Foundation Model Distillation for Efficient Lunar 3D Reconstruction" (arXiv 2607.01851, July 2026) is reported to build a smaller student's starting weights from the teacher's leading singular directions. I saw only a search snippet and did not verify it. — [arXiv 2607.01851](https://arxiv.org/pdf/2607.01851)

### Inferences
- **FRONT's dependence on weight coordinates is PRISM's opening.**
  - FRONT and SPA-FRONT work in *index space*: a DCT over weight-matrix coordinates. That is why SPA has to permutation-align weights first.
  - PRISM works in *SVD space*, where directions are defined up to basis. That is arguably the more principled object for cross-size transfer.
  - PRISM still has to solve the ambient-coordinate problem. Mapping a d_large × k subspace into d_small needs a residual-stream correspondence. Options are a random or learned projection, a μP-consistent embedding, or LSP/SuperAI-style Procrustes alignment.
- **Scoop risk is real and near-term.** The learngene group (Xin Geng / Fu Feng) published FRONT (Mar 2026), SPA-FRONT (Oct 2026), and Knowledge Diversion (Oct 2026). Their next step, SVD- or eigen-domain learngenes, would land squarely on PRISM's bet (b). PRISM should publish its cross-size projection result, or at least a clear preprint of the method, soon.
- **Comparing PRISM to FRONT.**
  - FRONT's LM evidence is thin: GLUE, BERT/GPT-2 Base→Small, GPT-S 31% FLOPs. Its LM baselines are only scratch and KD.
  - A PRISM cross-size paper that reports tokens-to-target on a modern decoder LM against FRONT, SPA-FRONT, Weight Selection, and mimetic init would be clearly differentiated.
- **CrossGMN shows where the hard part is.** It fails specifically on attention-dimension and head reductions. That is the hardest part of cross-size transfer and the place a geometry-based method could show its value.
- **μP likely helps cross-size projection.** Metanetwork width-generalization works under μP and fails otherwise, so PRISM's cross-size projection is likely better-behaved when teacher and student both use μP.

### Gaps
- I could not access SPA-FRONT's full PDF or anonymous repo. The LM source and target sizes behind the "up to 58.7%" figure are unknown.
- Config2Weights' model sizes, baselines ("existing weight-expansion methods"), and absolute speedups are not in the abstract.
- I found no paper that projects *SVD subspaces* from a larger trained LM into a smaller LM and trains it. Absent better coverage, this looks unoccupied.

---

## Q4. Any new analytic, mimetic, or closed-form inits that make attention or MLPs "look trained," with LM gains at scale?

### Takeaway
No in-window paper reports a closed-form or mimetic init giving LM gains at ≥100M scale. In-window analytic-init work is mostly theory or ViT-scale.

The more consequential in-window results are about **controls**:
- **"How QK Initialization Shapes Gradients"** shows a broad low-loss plateau down to σ_qk = 10^-5 in 124M LMs, so a well-chosen QK init is a cheap baseline improvement.
- **"Rethinking Pre-pretraining"** finds that prior synthetic-warm-up gains were partly an artifact of a weak baseline init.
- **"A Lasting Bias of Initialization"** shows a ~108-number statistical init can reproduce procedural warm-up gains in ViTs.

These set the bar for what PRISM's baseline must include.

### Cited Findings
- **QK init recipe at 124M.** Theory gives (2 ln T)^{1/4} as an upper reference for σ_qk. Experiments with "100-layer attention stacks and 124M-parameter language models" show "AdamW and Muon exhibiting a broad low-loss plateau down to σ_qk = 10^-5." Recipe: "favor small, nonzero QK initialization." — [OpenReview 419qaAMpme](https://openreview.net/forum?id=419qaAMpme)
- **Init artifacts in pre-pretraining.** "Broader PPT gains reported in prior work can be attributed to the weight initialization scheme. In an NCA reproduction, we recover a 5.4% PPT advantage under the original initialization; changing the initialization improves both methods (the no-PPT baseline by 9.0%), leaving them indistinguishable." — [OpenReview 73o2bc3VP2](https://openreview.net/forum?id=73o2bc3VP2)
- **Statistics-matched init.** 108 scalar layerwise statistics imposed on a random ViT init "match or exceed the improvements from procedural warm-up," with persistent effects consistent with implicit regularization. — [OpenReview DzLzZYD84Y](https://openreview.net/forum?id=DzLzZYD84Y)
- **QK geometry family (withdrawn).** A continuous family of joint query–key geometries at init has closed-form score statistics. The apparent geometry effect "is explained by scale," and the authors say it "is not yet a portable initialization recipe." The submission is marked **withdrawn**. — [OpenReview 5kUSidIv48](https://openreview.net/forum?id=5kUSidIv48)
- **Pattern-formation priors.** Choosing task-aligned positional encoding and weight init yields "improved data efficiency and accelerated optimization on controlled sequence tasks and with ConViT on CIFAR-10." — [arXiv 2609.37921 (2026-09-29)](https://arxiv.org/abs/2609.37921)
- **Mimetic init as a comparison point.** Procedural Core compares against mimetic init on ViT-B/ImageNet: Mimetic 79.5 ± 0.7 vs Procedural Core 79.8 ± 0.6 vs default 77.6 ± 0.2. — [arXiv 2609.37631 (HTML)](https://arxiv.org/html/2609.37631v1)
- **Revised in window: QK priors get overwritten.** "Making Transformers Logical" (v1 2026-08-01, **v2 2026-10-14**) reports that QK-initialization priors "can be rapidly overwritten during training and fail to change the learned extrapolation rule" on Boolean tasks. A learnable additive attention-logit bias persists. — [arXiv 2608.00418](https://arxiv.org/abs/2608.00418)
- **Pre-window context.**
  - "Conditioned Initialization for Attention" (2026-09-07, ICLR 2026; two weeks before the window) sets QKV weights to reduce the attention Jacobian's condition number and reports gains over default and mimetic init. — [arXiv 2609.07086](https://arxiv.org/abs/2609.07086)
  - "Mimetic Initialization of MLPs" (2026-02). — [arXiv 2602.07156](https://arxiv.org/abs/2602.07156)

### Inferences
- **PRISM's baseline is now more vulnerable than its method.** A reviewer will ask whether the from-scratch baseline used a tuned init. That means small-σ QK, μP or tuned std, and possibly mimetic or conditioned attention init. "Rethinking Pre-pretraining" shows a 9% baseline improvement from init alone erased a claimed effect. PRISM should rerun its 3× claim against (i) a tuned-init baseline, (ii) a mimetic-init baseline, and (iii) a 108-style statistics-matched init.
- **PRISM's own controls echo this literature.** Its finding that never-trained analytic DCT directions fail is consistent with analytic and mimetic inits giving modest, ViT-centric gains. PRISM's distinctive claim is that *trained* directions add value beyond any closed-form structure. That claim is stronger if it survives the controls above.

### Gaps
- I found no in-window paper reporting mimetic or analytic init gains on decoder LMs at ≥100M with tokens-to-target accounting.
- "Lasting Bias" is ViT-only. Whether a statistics-matched init reproduces gains in LMs, or reproduces PRISM's gains specifically, is untested.

---

## Q5. Is there evidence about whether init-based speedups persist, shrink, or vanish with scale (tokens-to-target at 100M–7B)?

### Takeaway
The evidence is **mixed and conflicting**, and none of it tests geometry-only init directly:
- **Persists:** synthetic pre-pretraining saves "at least 21B PT tokens at the 3B scale" across 500M–7B models and up to 100B tokens. Depth growth improves the scaling exponent (1.55× at 10^20 FLOPs).
- **Shrinks or vanishes:** compute-matched pre-pretraining "never outperforms" natural-language training (881M params, 50B tokens), and additive gains "vanish by 1B natural-language tokens."
- **Narrows:** the advantage of pruned-parent inits from Llama-3.1-8B narrows as the token budget grows.
- **Fades or doesn't appear:** Procedural Core's DINO gain shrinks over training, and there is no gain under the stronger DeiT-III recipe.

The default expectation for PRISM is that its 3× at 10.65M will shrink at 410M–1B. The literature leaves room for a durable residual if the effect behaves like an exponent change rather than a constant offset.

### Cited Findings
- **Survives scale.** "A comprehensive study on PPT spanning five PPT tasks, four PT data mixtures, four parameter scales (500M to 7B), and PT budgets of up to 100B tokens… gains of PPT persist at scale, e.g., saving at least 21B PT tokens at the 3B scale." Gains come from tasks that improve long-range retrieval, not from a grammatical prior. — [arXiv 2609.39827 (2026-09-30)](https://arxiv.org/abs/2609.39827); also [OpenReview P8cMuhbbfz](https://openreview.net/forum?id=P8cMuhbbfz)
- **Contradicts it.** Under compute matching, synthetic PPT "never outperforms pure natural-language training… (881M parameters trained on 50B tokens)." When additive, PPT gains "vanish by 1B natural-language tokens," and "knowledge acquired during PPT fails to survive subsequent natural-language training." — [OpenReview 73o2bc3VP2](https://openreview.net/forum?id=73o2bc3VP2). This contradicts [arXiv 2609.39827](https://arxiv.org/abs/2609.39827). The difference is likely accounting (compute-matched vs additive) and baseline-init strength. Neither paper's full text was compared.
- **Strong controls flip results at small scale.** In small decoders, Dyck warm-up beats scratch in 15/15 pairs by final bits per byte. But "a first-order Markov surrogate strictly beats Dyck in 29/30," and "retention of natural embedding/head parameters" wins all 30 cells. — [OpenReview XQe6IhpfHW](https://openreview.net/forum?id=XQe6IhpfHW)
- **Effects persist beyond perplexity.** Allocating 1% of tokens to a stack task improves MuSiQue multi-hop QA by up to 3.9 F1 "despite comparable language-modeling perplexity." "The early advantage persists through billions of subsequent language tokens." Mixing the data in rather than front-loading it is "far less effective," and exposure after pretraining "completely removes the benefits." — [arXiv 2609.38764 (2026-09-30)](https://arxiv.org/abs/2609.38764)
- **Parent init narrows with tokens (8B scale).** Pruning Llama-3.1-8B at ratios 0.5–0.8: "with the same training token budget, pruned initialization consistently outperforms random initialization… although the advantage narrows as the training token budget grows." Given the full pipeline token budget, "coarser structured pruning can be matched or surpassed" by scratch. — [OpenReview a8SnZ5mFCw](https://openreview.net/forum?id=a8SnZ5mFCw)
- **Growth improves exponents.** Untied-Grow gives 1.55× compute efficiency at 10^20 FLOPs, the trend extrapolates to 10^21, and "better architectures can improve the *exponents*." — [OpenReview 9YVNlDplEi](https://openreview.net/forum?id=9YVNlDplEi)
- **Growth loses annealing progress.** An LR reset at growth discards 45–50% of annealing progress. — [OpenReview VQ3NV9KF1P](https://openreview.net/forum?id=VQ3NV9KF1P)
- **Init gains shrink with training length and recipe strength.** In Procedural Core, the DINO ViT-S kNN gain narrows from +0.6 pp (epoch 100) to +0.3 (epoch 300). Under the DeiT-III recipe, ImageNet does not improve (82.4% vs 82.5%). — [arXiv 2609.37631 (HTML)](https://arxiv.org/html/2609.37631v1)
- **Inductive bias without init at GPT-2 Medium.** Structure–residual biases (not an init) give a 1.44% NLL improvement at 300M tokens that "persist with training runs up to 20B tokens." — [OpenReview kSIZLpN6sn](https://openreview.net/forum?id=kSIZLpN6sn)
- **Pre-window context.**
  - Procedural pretraining (2026-01, ICML 2026): up to 1.3B, front-loading 0.1–0.3% procedural data reaches the same loss with 55/67/86% of the original data on C4, CodeParrot, and DeepMind-Math. From a search summary. — [arXiv 2601.21725](https://arxiv.org/abs/2601.21725v1)
  - Warmstarting (2026-05): gains hold only within a bounded token budget. From a snippet. — [arXiv 2605.13405](https://arxiv.org/abs/2605.13405)

### Inferences
- **Shrinkage is the base rate for anything that only changes the starting point.** A pre-registered bet (a) result of "reaches published Pythia/OLMo loss in X% fewer tokens" would be among the first tokens-to-target measurements of a *geometry-only* init at 410M–1B. No in-window paper made one, so the result is publishable whether it is positive or null.
- **Use the "survives scale" design as a template.** 2609.39827 counts tokens saved at a matched downstream or loss target across 500M–7B. PRISM's Pythia/OLMo bet could adopt the same accounting (tokens saved at the target, plus downstream evaluations), which would make the comparison direct.
- **Expect a reviewer critique on compute accounting.** Following "Rethinking Pre-pretraining," PRISM should report (i) compute-matched results that include teacher training cost (or argue the teacher is a sunk public checkpoint), and (ii) results against a strong baseline init.
- **PRISM's saturation finding fits the growth literature.** The effect saturates once the teacher converges, and a barely-trained teacher is worse than random. This parallels River-Hill (annealing state matters) and Optimizer Lineage (teacher training history stamps the spectrum). The teacher's training regime is a first-class variable.
- **Measure capabilities, not just loss.** "Lasting Effects" shows init-like interventions can change capabilities (multi-hop reasoning) at equal perplexity, so PRISM's scale-up should evaluate downstream tasks, not only loss.

### Gaps
- No in-window paper measures tokens-to-target for an SVD- or spectrum-based init from a trained teacher at 100M–7B. The question remains open and is PRISM's to answer.
- I could not reconcile the 2609.39827 vs 73o2bc3VP2 conflict in detail because the OpenReview full text was inaccessible.
- Scale behavior of FRONT, SPA-FRONT, and Config2Weights is unknown beyond small BERT/GPT-2 sizes.
