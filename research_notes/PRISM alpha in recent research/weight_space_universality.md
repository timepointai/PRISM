# Weight-Space Structure & Universality: Shared Subspaces, Spectra, and Fingerprints (window: 2026-09-21 → 2026-10-21)

Scope and method: I searched arXiv (keyword searches sorted by newest announcement), plus general web search, for work first made public or substantially revised between 2026-09-21 and 2026-10-21, and checked each in-window item on its arXiv abstract page. **Date caveat:** arXiv assigns an ID when a paper is *announced*, not when it is submitted. Two 2609.* papers below list August submission dates. They were evidently held in moderation and first made public in September, so I count them as in-window but flag them (†). The arXiv search listings I used were truncated (only the first ~100–200k characters were read), so the inventory may miss some window papers. Pre-window items appear only as date-labeled context.

**Inventory index (in-window items; verdict codes: S = supports PRISM premise, U = undercuts/complicates, C = composable, G = leaves a gap PRISM fills, P = provenance/IP relevance)**

| arXiv ID | Date (v1 / version in window) | Title (short) | Scale | Verdict |
|---|---|---|---|---|
| [2512.05117v3](https://arxiv.org/abs/2512.05117v3) | v1 4 Dec 2025; **v3 5 Oct 2026** | Universal Weight Subspace Hypothesis | 1,200+ models (500 Mistral-7B LoRAs, 500 ViTs, 50 LLaMA-8B) | S, C, U(scope) |
| [2609.26165](https://arxiv.org/abs/2609.26165) † | submitted 9 Aug 2026; announced Sept 2026 | Spectral Tail Interventions in Decoder-Only LMs | 3 base checkpoints + 1 Phi instruct; 5 reasoning benchmarks | S, C |
| [2609.26147](https://arxiv.org/abs/2609.26147) † | submitted 23 Aug 2026; announced Sept 2026 | Block-Level Weight-Space Structure Persists Under Post-Training | 8 configs, 4 families, ~1B–7B | P |
| [2609.23087](https://arxiv.org/abs/2609.23087) (borderline) | submitted Sat 19 Sep 2026 | Neural Spectral Capacity | 7 families; pruning LLaMA-7B→5.7B | G |
| [2609.27252](https://arxiv.org/abs/2609.27252) | 23 Sep 2026 | What Converges in the Platonic Representation Hypothesis? | VLMs, video-text (representations only) | S(partial), U |
| [2609.38081](https://arxiv.org/abs/2609.38081) | 29 Sep 2026 | Hessian Null Space Continuation | RNNs, ImageNet ViTs, RL agents | U, P (evasion) |
| [2610.00420](https://arxiv.org/abs/2610.00420) | 30 Sep 2026 | Transferable Graph Metanetworks | robust to 42× training width (μP inputs) | C, G |
| [2610.00929](https://arxiv.org/abs/2610.00929) | v1 1 Oct, v2 2 Oct 2026 (NeurIPS 2026) | Platonic Task Arithmetic | 6 families, 8 tasks | S(function-level), U(coordinate-level) |
| [2610.01649](https://arxiv.org/abs/2610.01649) | 1 Oct 2026 | CrossGMN: cross-architecture weight-space transformations | INRs, MLPs, CNNs, ViTs (small) | C, competitor to cross-size bet |
| [2610.11730](https://arxiv.org/abs/2610.11730) | 8 Oct 2026 | Spectral Weight Decay | LLaMA 124M–500M | C |
| [2610.13123](https://arxiv.org/abs/2610.13123) | 9 Oct 2026 (NeurIPS 2026) | Why Heavy-Tailed Weight Spectra Predict Model Quality | "large-scale" CNNs and LLMs | S, C |
| [2610.18471](https://arxiv.org/abs/2610.18471) | 13 Oct 2026 | Graph Metanetworks for Transformer LMs | not stated; two new transformer zoos | P, C |
| [2610.21995](https://arxiv.org/abs/2610.21995) | 15 Oct 2026 | Interfaces and Modules / Universal Weight Encoder | 1,200 transformers × 100 architectures | C, P |
| [2610.23616](https://arxiv.org/abs/2610.23616) | 16 Oct 2026 | The Sign Code | overparameterized vision classifiers | adjacent |
| [2610.17914](https://arxiv.org/abs/2610.17914) | 13 Oct 2026 (listing only) | LinMorph: learned weight morph for init | video DiTs | C |
| [2610.13536](https://arxiv.org/abs/2610.13536) | 9 Oct 2026 (listing only) | Unsupervised discovery of subliminal learning | not stated | P (adjacent) |
| [2610.28434](https://arxiv.org/search/?query=fingerprinting+models&searchtype=abstract&order=-announced_date_first) | 20 Oct 2026 (listing only) | Every Model Leaves a Trace (behavioral fingerprint) | not stated | P (non-weight) |

---

## Q1. Do different models converge to shared or alignable weight subspaces or spectra?

### Takeaway
Window-period evidence agrees with PRISM's split. Spectra behave like a low-dimensional, quasi-universal family statistic. *Learned directions* carry function and are only partly shared: they are shared within an architecture (universal-subspace v3), but at the coordinate level they differ across models and line up only at the level of function or relational structure (Platonic Task Arithmetic, PRH "structure over geometry"). A new counterweight (Hessian Null Space Continuation) shows that many internally distinct weight solutions implement the same function, so "universality" cannot mean unique weights.

### Cited Findings
- **Universal Weight Subspace Hypothesis, v3 (2026-10-05).** Analyzes over 1,200 models: 500 Mistral-7B LoRAs, 500 ViTs and 50 LLaMA-8B. Reports that networks converge to shared spectral subspaces regardless of initialization, task or domain, and that "a few principal directions" capture most variance. v1 was 4 Dec 2025, v2 6 Dec 2025, v3 5 Oct 2026 (56 pages; file grew from 14.3 MB to 17.5 MB). The abstract page does not say what changed in v3. — [arXiv 2512.05117v3](https://arxiv.org/abs/2512.05117v3)
- A secondary summary, not the paper, puts the figure at "about 16 directions per layer" across ~1,100 models. — [Rohan Paul on X](https://x.com/rohanpaul_ai/status/2014647799011541032) (secondary; unverified against the paper)
- The universality is **within a shared architecture**. The abstract speaks of "sparse, joint subspaces within shared architectures." — [arXiv 2512.05117v3](https://arxiv.org/abs/2512.05117v3)
- **Spectral Tail Interventions (announced Sept 2026†; submitted 9 Aug 2026).** Across all 20 model×task cells, editing the *learned* upper spectral tail damaged performance more than the mean of five spectrum-matched Haar (random-direction) controls. 18 of 20 paired contrasts stayed significant after Holm correction. It also derives a finite-width bound linking the inverse participation ratio of squared singular values to pre-softmax logit kurtosis, but only for the QK block. Covers 3 base checkpoints plus an instruction-tuned Phi, on 5 reasoning benchmarks. The page does not name the models or sizes. — [arXiv 2609.26165](https://arxiv.org/abs/2609.26165)
- **Why Heavy-Tailed Weight Spectra Predict Model Quality (2026-10-09; NeurIPS 2026; Mahoney group).** Derives a DNN marginal likelihood computed only from final-layer weights, which predicts model quality across large CNNs and LLMs. The limiting log-marginal likelihood under the heavy-tailed High-Temperature Marchenko–Pastur (HTMP) spectrum is convex in a "heaviness" parameter, and its optimum empirically predicts over-fitting. This gives the first rigorous statistical grounding for WeightWatcher-style heavy-tail metrics. — [arXiv 2610.13123](https://arxiv.org/abs/2610.13123)
- **What Converges in the Platonic Representation Hypothesis? (2026-09-23).** After calibration, *relational structure* (which samples are related) converges at both local and global scales. *Metric geometry* (distances and similarities) converges much more weakly, and stricter distance agreement "flattens the capacity-dependent trend." The study analyzes representations of VLMs and video-text models, not weights. — [arXiv 2609.27252](https://arxiv.org/abs/2609.27252)
- **Platonic Task Arithmetic (v1 2026-10-01, v2 2026-10-02; NeurIPS 2026).** Task-specialized models behave alike, but their parameter updates "share no common coordinate system." The authors model each update as a "shadow" of a model-agnostic platonic task vector and report a model-specific residual with norm *comparable to* the shared component. Cross-model transfer through "Universal Task Descriptors" keeps 74–80% of the gain the target's own descriptors give. Covers 6 model families and 8 tasks. — [arXiv 2610.00929](https://arxiv.org/abs/2610.00929)
- **Hessian Null Space Continuation (2026-09-29).** Many different internal mechanisms coexist within one mode-connected low-loss region while computing the same function. In ImageNet ViTs, HNC finds representations that differ from the original *more than any independently trained model with a different architecture or objective*. Standard gradient training does not reveal this diversity. Also tested on RNNs and RL agents. — [arXiv 2609.38081](https://arxiv.org/abs/2609.38081)
- Context (pre-window, 2026-08-07, COLM 2026): spectral energy (the singular-value distribution) reliably separates *independently trained models and different families*. Subspace alignment handles finer distinctions among close relatives. Tested on 110+ open-weight LLM pairs. — [arXiv 2608.07786](https://arxiv.org/abs/2608.07786)
- Context (2026-06): low-barrier linear mode connectivity has been scaled to billion-parameter pretrained transformers. The method uses residual-space rotations and head and FFN permutations, i.e. alignment needs more than permutations. — [arXiv 2606.23607](https://arxiv.org/abs/2606.23607)
- I found no dedicated replication or rebuttal of the universal-subspace hypothesis in the window. Search turned up only informal commentary arguing that the subspace is per-architecture and that spectral concentration ≠ functional preservation. I could not tie that commentary to a verifiable primary source. — [alphaXiv page for 2512.05117](https://www.alphaxiv.org/abs/2512.05117) (commentary venue; treat as unverified)

### Inferences
- **Supports PRISM.** 2609.26165's spectrum-matched Haar controls are the same experiment as PRISM's "random directions fail," run as an *ablation* on large trained LMs. The specific learned directions carry the function, not the spectrum. This is independent, in-window corroboration.
- **Tension PRISM can resolve.** PRISM finds GPT-2's spectrum works as well as a native teacher's across size, tokenizer and corpus, so spectra are functionally interchangeable for init. Yet 2608.07786 finds spectral energy *discriminates* families. Both can hold if spectra differ measurably but in ways that don't matter for optimization. PRISM's 40-number DCT fingerprint is well placed to test this directly: compare discriminability (between-family distance) with functional interchangeability (init efficacy).
- **HTMP link.** 2610.13123 puts trained spectra in a family indexed by a heaviness parameter. PRISM's 8 DCT coefficients per group may be a redundant parameterization of ~1–2 HTMP/power-law parameters (α). That would explain *why* the spectrum compresses and transfers. This is testable by regressing PRISM's DCT coefficients on fitted α or heaviness.
- **Partial transfer is the norm.** PRISM's ~2/3 cross-corpus transfer resembles Platonic Task Arithmetic's 74–80% cross-model transfer and PRH's "relational structure converges, metric geometry less." A consistent reading: PRISM's mined directions contain a shared component plus a model- or corpus-specific residual of comparable size.
- **Caution (undercut).** HNC means a trained network's specific U/V are one of many function-equivalent choices. PRISM's "directions carry organization" is about directions as an optimization *prior* for a fresh model, not a unique identity of the function.

### Gaps
- What v3 of 2512.05117 changed (new experiments? cross-architecture claims?) is not stated on the abstract page. The full 56-page PDF was not reviewed.
- No window paper tests cross-*seed* or cross-*family* alignment of per-matrix SVD directions in pretrained LLMs (as opposed to LoRA populations or representations).
- WeightWatcher project updates in the window: not checked.

---

## Q2. How does weight geometry form during training (singular-vector stabilization, spectral evolution, rank collapse, emergence timing)?

### Takeaway
No in-window paper directly measures *when singular vectors stabilize* in LLM pretraining. The window adds one timing result: spectral concentration (inverse participation) *precedes* accuracy transitions. Recent pre-window work describes a two-timescale picture: fast, transient compression and an early-stabilizing spectrum, followed by slow, persistent spectral-shape differentiation. That picture explains PRISM's "saturates when teacher converges" finding and gives PRISM a measurable hypothesis to test.

### Cited Findings
- **In-window:** inverse participation (spectral concentration) precedes pooled accuracy transitions under a matched crossing rule. A residualized tail-aware LoRA reaches its targets *earlier* than standard LoRA and PiSSA, but the final-score intervals overlap. — [arXiv 2609.26165](https://arxiv.org/abs/2609.26165)
- **Context (2026-04-03), Spectral Lifecycle of Transformer Training.** Full SVD of every weight matrix every 25 steps (150,000+ snapshots) across 30M–285M custom models, validated on 9 models from GPT-2, Pythia and custom families (30M–1B). Findings:
  - Stable-rank "compression waves" travel early→late layers and then reverse.
  - The power-law α develops a persistent depth gradient.
  - Q/K projections carry the depth-dependent dynamics while V/O compress uniformly.
  - The author's two-timescale theory separates "fast compression (how much structure)" from "slow shape differentiation (what kind)."
  - Scaling law Δα ∝ L^0.26 (R² = 0.99).

  — [arXiv 2604.22778](https://arxiv.org/abs/2604.22778) (PDF text verified)
- **Context (May 2026), Stability of Singular Distribution.** The trace-normalized singular-value spectrum settles early while the parameter matrices keep changing. This stabilization coincides with the onset of a slow-improvement phase in GPT-2 and LLaMA pretraining. A third-party review calls the causal story correlational. — [arXiv 2605.26489](https://arxiv.org/abs/2605.26489) (abstract and search snippets only; not fetched in full)
- **Context (2024), Yunis et al.** Top singular vectors stabilize well before training ends, and the rank-minimization bias tracks grokking. — [Lacuna summary of "Approaching Deep Learning through the Spectral Dynamics of Weights"](https://lacuna.tiptreesystems.com/work/approaching-deep-learning-through-the-spectral-dynamics-v5jc) (secondary summary)
- **Context (2026-09-05, just pre-window).** SVD compression of attention projections suppresses rank collapse at initialization but *accelerates* it in pretrained GPT-2 and Pythia. Much of this is attributed to the subspace SVD selects. — [arXiv 2609.06341](https://arxiv.org/search/?query=%22singular+value%22+weights+transformer&searchtype=abstract&order=-announced_date_first) (listing summary only)
- **Context.** Training dynamics also reshape spectral geometry through the optimizer. For example, per-example clipping in DP training distorts the singular-vector geometry that Muon relies on (2610.02738, 2026-10-01; marginal). — [arXiv listing](https://arxiv.org/search/?query=%22singular+vectors%22+training&searchtype=abstract&order=-announced_date_first) (listing summary only)

### Inferences
- **Mechanistic story for PRISM's saturation.** If spectra and top directions stabilize early (fast timescale), a PRISM-initialized student can skip the fast phase. The slow phase, shape differentiation and the persistent α depth gradient, still has to be learned, and once the teacher has converged its directions offer nothing beyond what the student's own fast phase would provide. Under this reading PRISM buys the fast phase, so gains plateau. (This is a hypothesis; no paper tests it.)
- **Testable hypothesis.** Transfer efficacy should peak for teacher checkpoints mined around the point where the teacher's U/V subspace overlap between successive checkpoints first plateaus. PRISM's probe-scale infrastructure can measure this directly. It would also be the first direct measurement of singular-*vector* stabilization timing (the literature above mostly tracks singular *values*).
- The tail-aware LoRA result (faster to target, same final score) has the same *shape* as PRISM's effect, which suggests "geometry priors accelerate but don't raise the ceiling" may be general.
- **Q/K vs V/O.** Q/K carry the depth-dependent dynamics while V/O are uniform. This predicts that PRISM's transplant benefit should concentrate in Q/K directions, and that V/O spectra should be the most "universal" of PRISM's 5 groups. PRISM can check this by ablating per group.

### Gaps
- No in-window paper measures singular-vector (U/V) subspace stabilization timing in LLM pretraining with open checkpoints (Pythia/OLMo). This is a clear gap PRISM's data could fill.
- I did not verify 2605.26489's exact phase boundaries or model sizes beyond the abstract and snippets.

---

## Q3. Aligning or mapping subspaces across models of different widths or architectures (permutation/rotation alignment, stitching, weight-space learning, hypernetworks)

### Takeaway
This was the most active area in the window. The Maron group alone posted three graph-metanetwork papers. **CrossGMN (2610.01649) is the closest competitor to PRISM's open cross-size bet:** it learns to map a trained source network into a refined initialization of a different-architecture target. It has only been shown at small scale and still relies on distillation. **Transferable GMNs (2610.00420)** show that cross-width weight comparability is strongest under μP, a practical hint for PRISM's cross-size projection.

### Cited Findings
- **CrossGMN (2026-10-01).** A graph metanetwork that takes (trained source network, target-network initialization) and refines the target init using source information. It is invariant to source permutations and equivariant to target permutations. Use case: compression (large→small). It accelerates subsequent knowledge distillation by up to 8.89×, transfers across datasets without retraining (3.78×), and one model handles heterogeneous source architectures. Tested on INRs and on MLP, CNN and ViT image classifiers; no parameter counts given. The authors prove universality for continuous cross-architecture operators. — [arXiv 2610.01649](https://arxiv.org/abs/2610.01649)
- **Transferable Graph Metanetworks (2026-09-30).** Modifies GMNs to be invariant to how networks of different widths represent the same function and to be continuous. "Our modifications significantly improve size generalization on every task we consider." Performance is strongest for μP-trained inputs and robust up to 42× the training width. The authors prove size-generalization guarantees for μP via infinite-width limits and explain failure under other parameterizations. — [arXiv 2610.00420](https://arxiv.org/abs/2610.00420)
- **Graph Metanetworks for Transformer-Based LMs (2026-10-13).** Formalizes the full symmetry group of decoder-only LMs: neuron permutations, *continuous general-linear (GL) symmetries of attention matrices*, and head permutations. Gives an invariant graph construction that works across depths and widths. Introduces two transformer-population datasets and outperforms baselines on **training-corpus prediction** and **fine-tuning detection** from weights. Model sizes are not stated on the abstract page. — [arXiv 2610.18471](https://arxiv.org/abs/2610.18471)
- **Interfaces and Modules / Universal Weight Encoder (2026-10-15; Kozyra & Bronstein).** A compiler turns supported PyTorch models into a typed graph of "interfaces" (coordinate spaces permuted together) and "modules." On a new zoo of 1,200 transformers across 100 architectures, it outperforms baselines at ranking networks within *unseen* architectures, including larger ones, with less memory. — [arXiv 2610.21995](https://arxiv.org/abs/2610.21995)
- **Platonic Task Arithmetic (2026-10-01/02).** Transfers task effects across models via architecture-independent descriptors, either with a least-squares linear operator folded into the target's last layer or with a LoRA on the target's encoder. It keeps 74–80% of the gain. — [arXiv 2610.00929](https://arxiv.org/abs/2610.00929)
- **LinMorph (2026-10-13).** Learns a cross-parameter weight morphism to initialize linear-attention video DiTs from pretrained softmax-attention weights. — [arXiv listing](https://arxiv.org/search/?query=%22weight+space%22&searchtype=abstract&order=-announced_date_first) (listing summary only; abstract page not fetched)
- **Context, older cross-size init baselines.** Weight selection initializes small models from a subset of a larger pretrained model's weights — [arXiv 2311.18823](https://arxiv.org/abs/2311.18823). Weight subcloning — [arXiv 2312.09299](https://arxiv.org/pdf/2312.09299). Learngene/WAVE-style templates for variable-sized init — [arXiv 2605.08209](https://arxiv.org/abs/2605.08209), [arXiv 2601.19694](https://arxiv.org/pdf/2601.19694).
- **Context (May 2026).** A truncated-SVD LLM pretraining paper lists as *future work* whether orthonormal matrices learned in smaller models can initialize larger ones (small→large, the opposite direction from PRISM's bet). — [arXiv 2605.28573](https://arxiv.org/html/2605.28573)

### Inferences
- **Competitive positioning.** CrossGMN is a *learned*, symmetry-aware large→small init operator with demonstrated distillation speedups. PRISM's cross-size bet is an *analytic* projection (no meta-training, no model zoo). PRISM's advantages are zero meta-training cost and applicability to LLM scale, where nobody has built a model zoo large enough for a metanetwork. Its disadvantage is that CrossGMN's universality theorem and empirical speedups set a bar. A head-to-head at small scale (PRISM projection vs CrossGMN init, both followed by plain training with no distillation) would be the most informative comparison. **Composable:** PRISM's projected init could serve as CrossGMN's "target initialization" input.
- **μP matters for cross-width comparability.** 2610.00420 proves width-transfer guarantees only under μP and explains failure elsewhere. PRISM's cross-size experiments should test whether the teacher and/or student being μP-parameterized makes projected subspaces transfer better. This is cheap and directly actionable.
- **Gauge issue for per-matrix SVD.** 2610.18471 highlights continuous GL symmetries in attention: W_Q and W_K can be reparameterized by A and A^{-T} without changing the function. Per-matrix SVDs of W_Q and W_K separately are therefore *not* invariants of the function; the products W_Q W_K^T and W_V W_O are. Two observations point the same way: 2609.26165 reports that product-targeted (QK) factorizations reach higher held-out tail-subspace fractions than independent factor surgery, and 2604.22778 reports that the depth-dependent dynamics live in Q/K. Together they suggest PRISM may gain by mining and transplanting *QK and OV circuit* SVDs rather than per-matrix ones. (This is an inference, not tested anywhere.)

### Gaps
- No in-window paper does cross-*size* subspace projection for LLM *pretraining* initialization (large teacher → small student, then train from scratch). PRISM's open bet appears unaddressed at LLM scale.
- Model scales for the GMN papers (2610.18471, 2610.00420) are not on the abstract pages; full papers not reviewed.
- ICLR 2027 OpenReview submissions: not found or indexed by my searches.

---

## Q4. Model fingerprinting, weight provenance, lineage detection — and does PRISM-init leave or evade a lineage signature?

### Takeaway
I found **no new weight-based LLM lineage/fingerprinting paper first published in the window**; the in-window fingerprinting papers I found are behavioral or text-based. The field's leading weight-based method (Aug 2026, just pre-window) uses *the same two-part decomposition as PRISM*: a spectral-energy signature for coarse family identity and subspace alignment for fine lineage. That validates PRISM's fingerprint as provenance-relevant, but it also means a PRISM-initialized model may carry a detectable teacher signature, which is an IP consideration. In-window work adds: post-training preserves block-level geometry (derived models stay detectable), metanetworks can detect fine-tuning from weights, and function-preserving null-space moves are a potential evasion vector.

### Cited Findings
- **In-window, Block-Level Weight-Space Structure Persists Under Post-Training (announced Sept 2026†; submitted 23 Aug 2026; ICML 2026 Weight-Space Symmetries workshop).** Covers 8 post-trained configurations across Qwen2.5, Llama-3.1/3.2, Mistral and Gemma-2 (~1B–7B).
  - None of 291–339 tensors stays byte-identical.
  - Block-level mean cosine similarity is above 0.99 and relative Frobenius distance below 0.13.
  - Exception: Qwen2.5-Coder vs its base has cosine ≈ 0.64, and some specializations sit in disconnected regions.
  - The stated application is memory sharing (18–48% GPU savings), not provenance.

  — [arXiv 2609.26147](https://arxiv.org/abs/2609.26147)
- **In-window, Graph Metanetworks for Transformer LMs (2026-10-13).** Learned weight-space models outperform baselines at **fine-tuning detection** and **training-corpus prediction** directly from weights. — [arXiv 2610.18471](https://arxiv.org/abs/2610.18471)
- **In-window, Hessian Null Space Continuation (2026-09-29).** Function-preserving moves through weight space produce internals that differ more than independently trained models do. — [arXiv 2609.38081](https://arxiv.org/abs/2609.38081)
- **In-window, Universal Weight Encoder (2026-10-15).** A single encoder generalizes across 100 transformer architectures. — [arXiv 2610.21995](https://arxiv.org/abs/2610.21995)
- **In-window, non-weight fingerprinting.** Listing summaries only; abstract pages not fetched:
  - "Every Model Leaves a Trace: Multi-Resolution LLM Fingerprinting" (2610.28434, 2026-10-20) attributes outputs to source models across families, distilled models and LoRA adaptations from text.
  - DIB-Bench (2610.21135, 2026-10-14) adds behavioral fingerprinting plus manifest-hash checks.
  - An agent-identity framework (2610.17126, 2026-10-12).

  — [arXiv listing, "fingerprinting models"](https://arxiv.org/search/?query=fingerprinting+models&searchtype=abstract&order=-announced_date_first)
- **In-window, adjacent.** A transmitted "subliminal" trait can be recovered as a weight-space direction from parameter updates alone and projected out (2610.13536, 2026-10-09). — [arXiv listing](https://arxiv.org/search/?query=%22weight+space%22&searchtype=abstract&order=-announced_date_first) (listing summary only)
- **Context (2026-08-07, COLM 2026), "Who Built This Model?"** Weight-space lineage without input data via (i) spectral energy from singular-value distributions, which separates independent-origin models and families, and (ii) subspace alignment, which separates same-series and shared-base relatives, including differences in data scale and post-training. Tested on 110+ open-weight LLM pairs. The abstract does not address models *initialized from* another model's weights or adversarial rotation/permutation. — [arXiv 2608.07786](https://arxiv.org/abs/2608.07786); code: [OPTML-Group/LLM-Biometrics](https://github.com/OPTML-Group/LLM-Biometrics)
- **Context (2026-07-12), modelDNA.** Calibrated lineage verification from *sampled* weight bytes read over HTTP, with AUROC 1.0 on a 15-model benchmark (per search summary). Also recovers merge mixture weights by least squares and returns 8 verdict classes with abstention. — [arXiv 2607.10617](https://arxiv.org/pdf/2607.10617)
- **Context (Dec 2025), SELF.** A singular-value and eigenvalue LLM fingerprint, reported robust to quantization, pruning and fine-tuning. — [arXiv 2512.03620](https://www.arxiv.org/pdf/2512.03620)
- **Context, behavioral provenance.** TokenPrint (2608.08139) — [arXiv](https://arxiv.org/pdf/2608.08139); Stemma (2607.25880; 0.967 AUC on 770 pairs) and One-Token fingerprinting (2607.10252) — [arXiv listing](https://arxiv.org/search/?query=fingerprint+lineage+weights&searchtype=abstract&order=-announced_date_first).

### Inferences
- **PRISM's fingerprint matches the state-of-the-art lineage decomposition.** PRISM's "spectrum (universal, compressible) + directions (specific, carry the effect)" maps onto 2608.07786's "spectral energy (coarse family) + subspace alignment (fine lineage)." A commercial provenance angle is credible, but on this evidence the *directions*, not the 40-number spectrum, would be the strong IP signal. PRISM's own finding that GPT-2's spectrum substitutes for a native teacher's suggests the 40 numbers alone are too universal to identify a specific model. They might still identify a family; PRISM data can test this.
- **Does PRISM-init leave a lineage signature? Likely yes, at first, but how long it lasts is unknown.** A PRISM student starts with the teacher's U/V. If those directions persist through training, which 2609.26147's ">0.99 cosine after post-training" result makes plausible for *light* updates but not necessarily for full pretraining from a transplanted init, then subspace-alignment detectors like 2608.07786 could flag the student as "shared-base" with the teacher. This has two consequences:
  - **IP/licensing risk:** check the teacher's license on derivative works before mining from non-permissive models.
  - **Positive use:** a deliberately mined fingerprint could act as a provenance watermark.
- **Evasion vector.** HNC shows function-preserving weight moves can produce internals farther apart than independent models. In principle this could scrub weight-based lineage signals, including PRISM-derived ones, without changing behavior. Weight-space provenance therefore likely needs pairing with behavioral fingerprints (TokenPrint, Stemma, 2610.28434).
- **Gap PRISM fills.** No paper measures *how fast an init-transplanted lineage signature decays during from-scratch training*. PRISM can measure this cheaply: track student↔teacher subspace overlap and student↔independent-seed overlap over training. That number decides both the IP-risk and the watermark story.

### Gaps
- I found no in-window weight-based LLM lineage paper. Coverage limits: arXiv cs.CR listings were only partly read (truncated), and Hugging Face Papers and alphaXiv were not browsed directly.
- Whether 2608.07786's method handles "initialized-from-but-retrained" models or rotation/GL-gauge evasion is not stated in its abstract. The full paper was not reviewed.
- modelDNA's AUROC 1.0 figure comes from a search summary and was not verified on the paper page.

---

## Q5. What do the spectrum vs. the singular vectors each encode (spectral-only reparameterizations, spectral pruning and compression)?

### Takeaway
The evidence converges. **The spectrum encodes "how much and how well-trained"** (quality, compressibility, over-fitting, architecture capacity) in a few parameters. **The singular vectors encode "what"** (learned function). The single most PRISM-relevant result is pre-window but independent: correct spectra with random directions train *42.7% worse* than standard init, and the author lists "Spectral Warmup with direction transfer," which is essentially PRISM, as open future work. In-window, 2609.26165 confirms the same asymmetry by ablation on trained LMs.

### Cited Findings
- **Context (2026-04-03), Spectral Warmup negative result.** Initializing each weight matrix as W₀ = U_rand · diag(s·σ*) · V_randᵀ, where σ* are the trained reference model's singular values, trains far worse than standard init. On the D8 model (30.4M), validation loss at 5K steps was 5.307 vs 3.720 for standard init, a "42.7% gap." The paper's stated conclusion: "directions ≫ magnitudes: correct spectral shape with random directions is far worse than random init with incorrect spectra." Its Future Work section lists "Spectral Warmup with direction transfer." — [arXiv 2604.22778](https://arxiv.org/abs/2604.22778) (verified in PDF text, Sec. 7.3 and App. F, Table 8)
- **In-window.** Learned-tail edits are more damaging than spectrum-matched random-direction (Haar) controls in 20/20 cells, so the functional content of the spectral tail is in its directions. — [arXiv 2609.26165](https://arxiv.org/abs/2609.26165)
- **In-window.** The spectrum's *shape* alone, via HTMP heaviness, predicts model quality and over-fitting, data-free, from final-layer weights. — [arXiv 2610.13123](https://arxiv.org/abs/2610.13123)
- **In-window, Spectral Weight Decay (2026-10-08).** A decoupled nuclear-norm post-step shrinks singular values *additively*. On LLaMA 124M–500M it lowers effective rank and improves SVD-LLM compression at matched validation loss: 1.89× compression and 1.18× GPU speedup at 500M with a 4% distortion budget, vs 1.14× and 1.01× for standard weight decay. With 60% label noise it improves clean-test accuracy by up to 17.8 points on MNIST and 4.6 points across four BERT-base tasks. Code: github.com/brain-lab-research/SpectralWD. — [arXiv 2610.11730](https://arxiv.org/abs/2610.11730)
- **In-window, Neural Spectral Capacity (submitted Sat 2026-09-19; borderline).** A closed-form capacity score built from each weight matrix's singular-value spectrum. Via Marchenko–Pastur it is computed from the architecture alone, under standard random init, with no data. It ranks architectures better than #Params or #FLOPs (FlexiBERT τ = 0.505 vs 0.082) and prunes LLaMA-7B to 5.7B with no calibration data. — [arXiv 2609.23087](https://arxiv.org/abs/2609.23087)
- **In-window, The Sign Code (2026-10-16).** In overparameterized vision classifiers, the binary pattern of *neuron* signs dominates: a sign-only kNN matches full-model accuracy, randomizing magnitudes while keeping signs barely hurts ResNets, and flipping 20% of signs drops accuracy to chance. This concerns activation signs, not weight SVDs, so it is adjacent. — [arXiv 2610.23616](https://arxiv.org/abs/2610.23616)
- **Context (2026-04-03).** Spectral-guided pruning, which uses per-layer α to predict layer importance (ρ = 0.69–0.84), beats Last-N heuristics by 1.1×–3.6× on GPT-2 124M–774M and Pythia 160M–1B. — [arXiv 2604.22778](https://arxiv.org/abs/2604.22778)

### Inferences
- **Strong external validation of PRISM's central negative result, plus an explicit invitation.** PRISM's finding that random and analytic directions fail matches 2604.22778's Spectral Warmup (30M scale, similar to PRISM's 10.65M probe). PRISM's positive result, that mined directions work, is exactly the "direction transfer" that paper leaves open. This is PRISM's clearest claim to novelty in the current literature. The report writer should compare magnitudes carefully: PRISM calls spectrum-alone "insufficient," while 2604.22778 finds it actively *harmful* (42.7% worse). PRISM's data can say whether spectrum-only init hurts or is neutral at its scale.
- **The literature supports a division of labor:** the spectrum is a "maturity/quality" statistic (HTMP heaviness, α, NSC capacity, compressibility), and the directions are the "organization." That matches PRISM's thesis that geometry encodes organization separately from knowledge, with the refinement that *organization lives mostly in U/V* and the spectrum sets the energy budget.
- **Composable.** Spectral weight decay or HTMP-optimal heaviness could serve as a *training-time* regularizer that keeps a PRISM student's spectrum near the universal shape while its directions evolve. NSC shows architecture-only spectral priors are informative at init, a cheaper baseline PRISM should beat to justify mining a teacher.

### Gaps
- No in-window paper tests spectral-only *reparameterization during training* (freeze U/V, learn σ, or the reverse) in LLM pretraining. The universal-subspace work's "optimize only scalar coefficients in a shared subspace" use case is reported by secondary summaries, and I did not verify it in the v3 text.
- 2610.13123 does not name model scales on its abstract page, and the comparison of its HTMP heaviness with PRISM's DCT coefficients is untested.
- Transformer Circuits, LessWrong/Alignment Forum and WeightWatcher blog posts in the window were not checked.
