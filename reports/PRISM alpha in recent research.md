# Rivals Validate PRISM's Mechanism, Leaving Bets Open

PRISM's alpha has moved. Between 2026-09-21 and 2026-10-21, other groups independently confirmed PRISM's central mechanism. Trained singular *directions* carry the effect, and the singular-value spectrum does not. That holds both for the init speedup and for finetune retention. The confirmations came from Subspace Tuning (SST), a PEFT spectral-intervention study, and spectrum-matched random-direction ablations. The result validates PRISM, but it also turns PRISM's headline finding into a citation. None of PRISM's forward-looking bets was scooped. No paper in the window transplanted rotated SVD directions into a fresh model, measured tokens-to-target for a geometry-only init at 410M–1B, or projected an LM's SVD subspaces across sizes. None reported an anchor that blocks fact injection while validation loss looks best, and none replaced KL-to-reference with a weight-space anchor in RL. Every one of those bets now has a close neighbor, though:

- **Cross-size:** the learngene group's SPA-FRONT claims **58.7% fewer LM pretraining FLOPs** from cross-size, DCT-compressed weights.
- **Retention:** SST cuts SFT forgetting on Qwen3-8B from **−25.5 to −1.0 points**.
- **RL:** the major open RL stacks now default to **no KL at all**, so a zero-forward-pass anchor replaces a cost few of them still pay.
- **Prior fusion:** The COUNT gets **≈57% compute-equivalent at 1B** from a count-based n-gram prior on the input side. It also found that logit-level fusion during pretraining does no better than applying it only at inference. That is the most direct threat to Prior-Fused PRISM.

Demand sits in post-training. Every major platform that shipped in the window accepts custom losses, none sells a retention or injection control, and OpenAI's fine-tuning wind-down forces customers to re-tune onto open bases. Nobody yet buys geometry-transplant pretraining as a product. The highest-alpha next moves, in order:

1. Run the injection dial head-to-head against SST.
2. Run the pre-registered 410M–1B bet against a tuned same-codebase baseline, with statistics-matched and Muon controls.
3. Publish cross-size projection before the learngene group gets there.
4. Point the RL test at drift-prone regimes, where an anchor has something to fix.

## Validation landed on the mechanism, while the bets stay unclaimed

PRISM makes several kinds of claim ([PRISM README](https://github.com/timepointai/PRISM)):

- **Init transfer.** It blends a fresh model's per-matrix SVD directions toward a same-architecture teacher's and imprints a spectrum compressed to **40 numbers**. On a 10.65M-parameter probe, the student reaches baseline quality **3.3× faster (2.9–3.9)** on modern web text and **2.90×** faster on enwik8, and matches from-scratch quality at **42% of the GPU time**.
- **Retention.** Pointed backward as a per-step lerp toward the model's own pre-finetune weights, the same machinery gives **up to ~10× less forgetting**.
- **The dial.** At strength 0.01 the anchor blocks new facts (**2–3% closed-book recall**) while posting the best validation loss. A weak band at 0.0025 injects at parity with **3× less forgetting**.
- **Prior fusion.** A single-seed probe fusing an n-gram prior into the logits reports a **30×** speedup.
- **Pre-registered bets.** The core bet is a Pythia/OLMo test at 410M–1B, which fails if the advantage fades toward 1×. A GRPO test, specced at ~$500 on a 9B model with prime-rl, would use the anchor in place of KL-to-reference.

Here is how each claim stands after the 30-day window.

| PRISM claim or bet | 30-day verdict | Decisive window evidence | What is still PRISM's |
|---|---|---|---|
| Directions, not spectra, carry init and retention effects | **Validated, no longer novel** | SST; PEFT spectral restoration; spectrum-matched Haar ablations | The *positive* direction-transfer result; a soft, SVD-free mechanism |
| EigenTransfer + 40-number Spectral Imprint (same size) | **Unscooped; spectrum half pre-empted** | SAHyper (teacher spectrum + new bases); operator transfer in ViTs; 108-statistic init | Rotating fresh directions toward a teacher, then training |
| Core bet: tokens-to-target at 410M–1B | **Open; the base rate is shrinkage** | Conflicting persistence studies; pruned-init edge narrows | The first geometry-only measurement at scale, publishable either way |
| Cross-size subspace projection | **Open sub-cell, flanked** | SPA-FRONT, CrossGMN, Config2Weights | SVD-space (basis-level) projection for decoder LMs |
| Finetune retention anchor | **Validated in kind; crowded** | SST at 1.7B–32B; EWC still works at 8B | O(params) cost, no SVD; a single scalar dial |
| Injection↔retention dial | **Unscooped** | No window paper; August papers corroborate the eval pitfall | The phenomenon and the "loss is not an injection metric" warning |
| Anchor replaces KL in RL | **Unscooped; need questioned** | Stacks default to KL off; Task Vector Descent's GRPO null | Drift control at zero forward-pass cost in drift-prone RL |
| Prior fusion at the logits | **Threatened** | COUNT: input side wins; logit fusion in pretraining ≈ inference-only | Time-to-target, the geometry stack, byte-level |

### Independent results agree that directions, not spectra, carry the effect

The convergence is striking. **SST** (15 Oct, with Keutzer, Niekum, Dhillon and Khatri among the authors) starts from the observation that on-policy methods "avoid updating the principal singular directions." It restricts SFT updates to the bottom singular subspace. On Qwen3-8B, that holds a five-benchmark general-capability average at **80.0 vs 55.5 for plain SFT (base 81.0)** ([arXiv 2610.22293](https://arxiv.org/abs/2610.22293)).

A systematic PEFT study (6 Oct) found that **PiSSA, which trains the top singular components, forgets most**: on Llama-3.1-8B coding, general-text ΔNLL is 43.00 for PiSSA vs 10.73 for LoRA. Restoring *dominant* components cut forgetting most, while restoring only the pretrained singular *values* left the task intact ([arXiv 2610.09122](https://arxiv.org/abs/2610.09122)). That is a near-exact external replication of PRISM's finding that "anchoring only the spectrum does nothing."

Spectral-tail ablations on trained LMs found learned-direction edits more damaging than spectrum-matched random (Haar) controls in **20 of 20 cells** ([arXiv 2609.26165](https://arxiv.org/abs/2609.26165)). That paper was submitted in August and first announced in September.

The closest pre-window match is an April 2026 paper. Correct spectra with random directions trained **42.7% worse than standard init** (validation loss 5.307 vs 3.720 at 30.4M parameters), and the authors list "Spectral Warmup with direction transfer," which is essentially PRISM, as future work ([arXiv 2604.22778](https://arxiv.org/abs/2604.22778)). In July, spectral-profile matching across 11 GPT-2-style checkpoints gave no measurable gain ([arXiv 2607.09204](https://arxiv.org/abs/2607.09204)). In vision, refactoring a warmed-up ViT's QK/OV *operators* into fresh projections kept **103–121%** of the full transfer gain ([OpenReview II0OaQcnmX](https://openreview.net/forum?id=II0OaQcnmX)).

### What the convergence costs PRISM, and what it leaves

PRISM can no longer lead with "directions matter." It can lead with a narrower claim that nobody has made: **mined directions, transplanted at init, accelerate a fresh model**. Every validating paper above either ablates directions in an already-trained model or protects them during finetuning. None transfers them into a new one.

The 40-number fingerprint is the least defensible piece of alpha as a speed lever. In-window theory puts trained spectra in a heavy-tailed family indexed by a single "heaviness" parameter that predicts model quality ([arXiv 2610.13123](https://arxiv.org/abs/2610.13123)). That plausibly explains why PRISM's 8 DCT coefficients per group compress so well, and why GPT-2's spectrum substitutes for a native teacher's: the coefficients may be a redundant parameterization of one or two numbers. This is an inference and still untested. Its more interesting future may be provenance. The leading weight-lineage method (August) separates families by spectral energy and close relatives by subspace alignment ([arXiv 2608.07786](https://arxiv.org/abs/2608.07786)). That is the same decomposition PRISM uses.

### The new threats are to PRISM's controls, not its method

"A Lasting Bias of Initialization" showed that imposing **108 scalar layerwise statistics** on a random ViT init matches or beats procedural warm-up gains ([OpenReview DzLzZYD84Y](https://openreview.net/forum?id=DzLzZYD84Y)). PRISM's spectrum-only and analytic-DCT controls do not match activation statistics such as attention entropy, LayerNorm stats or residual write ratios. "Rethinking Pre-pretraining" erased a 5.4% pre-pretraining advantage simply by changing the baseline init, which improved the no-pre-pretraining baseline **9.0%** ([OpenReview 73o2bc3VP2](https://openreview.net/forum?id=73o2bc3VP2)). A 124M QK-init study found a broad low-loss plateau down to σ_qk = 10⁻⁵ ([OpenReview 419qaAMpme](https://openreview.net/forum?id=419qaAMpme)). Together these make PRISM's from-scratch baseline more vulnerable than its method.

There is also a subtler gauge problem. Attention has continuous GL symmetries: W_Q and W_K can be reparameterized without changing the function ([arXiv 2610.18471](https://arxiv.org/abs/2610.18471)). Per-matrix SVDs of W_Q and W_K are therefore not function invariants, while the QK and OV circuit products are. That suggests, untested, that PRISM should try mining circuit-level SVDs.

## The scale and cross-size bets are open but flanked by faster-moving neighbors

### Shrinkage is the base rate for the 410M–1B bet

No in-window paper measured tokens-to-target for an SVD- or spectrum-based init from a trained teacher anywhere between 100M and 7B. PRISM's pre-registered result will be a first data point whether it holds or fades. The neighboring evidence says to expect fading.

The persistence literature conflicts:

- Synthetic pre-pretraining saves "**at least 21B PT tokens at the 3B scale**" across 500M–7B ([arXiv 2609.39827](https://arxiv.org/abs/2609.39827)).
- A compute-matched study finds it "never outperforms" natural-language training at 881M/50B tokens, with additive gains that "vanish by 1B natural-language tokens" ([OpenReview 73o2bc3VP2](https://openreview.net/forum?id=73o2bc3VP2)).
- Pruned-Llama-3.1-8B inits beat random init, but "the advantage narrows as the training token budget grows" ([OpenReview a8SnZ5mFCw](https://openreview.net/forum?id=a8SnZ5mFCw)).
- Optimizer speedups follow the same arc. Muon's ~40% advantage below 0.5B shrinks to **~10% at 1.2B** under fair tuning ([arXiv 2509.02046](https://arxiv.org/pdf/2509.02046)).
- Curriculum effects narrow by 410M ([arXiv 2601.21698](https://arxiv.org/html/2601.21698v2)).

PRISM's own failure condition, "fades toward 1×," is the prior to beat, not a tail risk.

What PRISM must beat is set by levers that already report gains at scale:

| Lever | Reported gain | Scale | Date / status |
|---|---|---|---|
| Untied-Grow (model growth) | **1.55× compute efficiency**, improved scaling *exponent* ([OpenReview 9YVNlDplEi](https://openreview.net/forum?id=9YVNlDplEi)) | 10¹⁸–10²⁰ FLOPs | ICLR'27 submission, public 3 Oct |
| SPA-FRONT (cross-size DCT transfer) | **up to 58.7% fewer LM pretraining FLOPs** ([OpenReview A04pxSDaxO](https://openreview.net/forum?id=A04pxSDaxO)) | LM sizes undisclosed | ICLR'27 submission, public 3 Oct |
| The COUNT (count-prior input embedding) | **≈57% more compute** equivalent ([arXiv 2610.20456](https://arxiv.org/abs/2610.20456)) | 1B | 14 Oct |
| G_stack (depth stacking) | 7B loss in 194B vs 300B tokens ([arXiv 2405.15319](https://arxiv.org/abs/2405.15319)) | 7B | 2024 context |
| SAMuon (static spectral step-size prior) | **13.3–24.0% fewer tokens** than Muon ([arXiv 2608.25990](https://arxiv.org/abs/2608.25990)) | 124M–1B | Aug 2026 context |
| PRISM (EigenTransfer + Spectral Imprint) | 2.9–3.9× to baseline quality; 42% of GPU time ([PRISM README](https://github.com/timepointai/PRISM)) | 10.65M | probe scale |

A PRISM multiplier that held at **≥1.5× at 410M+** against a tuned baseline would sit at or above growth's best reported numbers without copying any weights. At ~1.1× it would land in optimizer-tweak territory.

Two benchmark details matter. First, Pythia's published curves come from a 2023 AdamW recipe (300B tokens, 2M-token batches, LR 3e-4) ([EleutherAI/pythia](https://github.com/eleutherai/pythia)). A modern re-run would itself beat Pythia's logged loss in fewer tokens, so the honest comparator is a **same-codebase tuned random-init run**, not Pythia's curve. Second, nothing in the window refreshed dense small-model baselines: Ai2's in-window release was the MoE framework Olmo-core 3 ([Ai2](https://allenai.org/blog/olmocore3)).

### Muon is the likely spoiler for the spectrum half

In-window work shows that Muon's full-spectrum updates "keep the parameter spectrum flat for free." Warmup *hurts* Muon (3.280→3.309) and *helps* AdamW (3.468→3.396), because an orthogonalized update "writes a full-rank matrix in a single step" ([arXiv 2610.10965](https://arxiv.org/html/2610.10965)). Muon-trained teachers also carry **6–14× higher stable rank** than Adam-trained ones ([OpenReview YMkHmZjJ6B](https://openreview.net/forum?id=YMkHmZjJ6B)).

The likely consequence is that PRISM's decaying, imprinted spectrum gets overwritten faster under Muon. Part of PRISM's AdamW gain may be doing AdamW's slow rank-shaping for it during warmup. Directions may fare better: Muon preserves the gradient's singular directions ([arXiv 2606.08388](https://arxiv.org/html/2606.08388v1)). Measuring principal-angle overlap between the student's top subspace and the transplanted U/V under each optimizer is an unclaimed, publishable measurement.

There is also a composition opening, still speculative. SAMuon finds per-direction optimal step sizes are "anisotropic but stable" over training ([arXiv 2608.25990](https://arxiv.org/abs/2608.25990)), which makes a transferred spectrum a candidate for its static prior.

One more point favors PRISM's no-copy design. Growth methods that reset the LR schedule discard **45–50% of the source's annealing progress** ([OpenReview VQ3NV9KF1P](https://openreview.net/forum?id=VQ3NV9KF1P)). PRISM starts a fresh model on a fresh schedule, so it should not pay that tax. No paper tests this directly.

### The learngene group is one paper away from the cross-size cell

The cross-size cell is no longer empty, but the specific sub-cell is.

- **FRONT** (Mar 2026) applies a 3D-DCT to trained weights stacked across layers, then truncates or pads in frequency space to the target size. It reports **31.0% FLOPs savings** for GPT-2 Base→Small, with only scratch and distillation as language baselines ([arXiv 2603.07523](https://arxiv.org/abs/2603.07523)).
- **SPA-FRONT** (public 3 Oct) adds permutation alignment and claims up to 58.7% LM-FLOPs savings ([OpenReview A04pxSDaxO](https://openreview.net/forum?id=A04pxSDaxO)). The same group posted Knowledge Diversion on 17 Oct ([arXiv 2610.24664](https://arxiv.org/abs/2610.24664)).
- **CrossGMN** learns a cross-architecture init operator, but only up to ~210k parameters. It fails exactly where the problem is hardest: **0.74×** on attention-dimension reduction and **0.86×** on head-count reduction ([arXiv 2610.01649](https://arxiv.org/abs/2610.01649)).
- **Config2Weights** generates inits for unseen LLM configurations from a model zoo ([OpenReview XubgOIuvFd](https://openreview.net/forum?id=XubgOIuvFd)).

Nobody has projected a large LM's *SVD subspaces* into a smaller LM and trained it at ≥100M. PRISM's structural advantage is that FRONT works in weight-index space, which is why it needs permutation alignment, while SVD directions are basis-level objects. PRISM still has to solve the ambient-coordinate mapping between residual streams of different widths. Two in-window results help:

- Weight-space metanetworks generalize **up to 42× in width under μP** and fail under other parameterizations ([arXiv 2610.00420](https://arxiv.org/abs/2610.00420)). Cross-size runs should therefore be μP-native.
- Procrustes-aligned PCA subspaces already move capabilities across heterogeneous architectures ([arXiv 2610.15322](https://arxiv.org/abs/2610.15322)).

The learngene group publishes fast, and an "SVD-domain learngene" would land on PRISM's bet. That makes this the one bet with a clock on it.

### COUNT took the input side of prior fusion and cast doubt on the logit side

**The COUNT** (14 Oct; Magnusson, Smith, Koh) builds parameter-free n-gram embeddings from counts of orders 1–6, held in host memory, and adds them at the input. Its results:

- At 1B, it improves validation loss "about as much as a model with **57% more compute**," and 46–89% across a 60M–1B ladder ([arXiv 2610.20456](https://arxiv.org/abs/2610.20456)).
- It tested logit-level product-of-experts fusion during pretraining, following the canonical residual-over-n-gram method ([arXiv 2210.14431](https://arxiv.org/pdf/2210.14431)). At 190M, that fusion "**does not improve on applying fusion only at inference**" ([arXiv 2610.20456 HTML](https://arxiv.org/html/2610.20456)).

That null is measured on final loss at a fixed budget, not on PRISM's time-to-baseline metric. It comes from a single scale. The figures were extracted from the paper's HTML and should be spot-checked against the PDF. Even so, it sharply narrows Prior-Fused PRISM's claim. What remains is logit-side fusion, measured by tokens-to-target, stacked on geometric init. The 30× number is single-seed at 10.65M.

COUNT also helps PRISM in one place. Six n-gram orders beat three "at every budget," which supports PRISM's planned move from a context-3 to a context-4+ prior ([arXiv 2610.20456 HTML](https://arxiv.org/html/2610.20456)).

Two other in-window papers matter for that move:

- **Risk (Forking).** Models with n-gram memory branches show sudden train/validation separation at epoch boundaries under replay, driven by **low-frequency contexts** ([arXiv 2610.00394](https://arxiv.org/abs/2610.00394)). High-order sparse priors on multi-epoch Shakespeare-scale data are exactly that regime.
- **Help (Cascade).** GPU suffix-array construction indexed the 198B-token C4 in 23 minutes, which makes ∞-gram-order priors cheap to build ([arXiv listing](https://arxiv.org/search/?query=infini-gram&searchtype=all&abstracts=show&order=-announced_date_first&size=50)).

Byte-level statistical priors remain an empty cell. The one in-window byte-LM scaling paper does not study priors ([arXiv 2610.05978](https://arxiv.org/abs/2610.05978)), so PRISM's enwik8 gate would be a first data point.

One synergy is worth testing (it is a hypothesis). COUNT's embedding is a count-weighted sum of the model's *own unembedding rows*. A PRISM-initialized unembedding should make COUNT informative from step 0 rather than only after the unembedding trains.

## SST crowds retention, but the injection dial and the RL anchor remain PRISM's

### Retention is now a crowded race PRISM can win only on cost and simplicity

SST's numbers are the new bar ([arXiv 2610.22293 HTML](https://arxiv.org/html/2610.22293v1)):

- **Across four domains:** mean retention of 79.6 vs 37.6 for SFT, against a base of 81.0.
- **By scale:** forgetting falls to −0.9 points at 14B, −3.6 at 4B and −8.0 at 1.7B, so the benefit grows with model size.
- **At 32B** (two epochs): −1.0 vs −15.6 for SFT.
- **Reverse-KL sweep:** a weak KL penalty left the prior capability at about 23 (23.6 with no KL), and a strong one cut it to 10.6. That is the same narrow-useful-band shape PRISM reports for its dial, and it mirrors PRISM's "Pareto-dominates low LR" result.

SST did not test an L2-to-init or lerp baseline, knowledge injection, or RL. It also carries an SVD dependency PRISM avoids: basis refresh adds **5.1–37.6% per step** unsharded.

The window added at least seven more subspace or singular-vector methods: SVC, SOAR, ChainLoRA, NB-LoRA, EoupCT, CGR and a tropical per-unit projector. A Llama-3.1-8B translation study shows classical weight anchors still work: EWC lost **1.7 general points vs 11.0** for standard finetuning ([arXiv 2609.28395](https://arxiv.org/abs/2609.28395)). Self-distillation is becoming the default data-side recipe; iSDFT keeps 73% of evaluations within 0.5 points of base, vs 52% for the strongest baseline ([arXiv 2609.24646](https://arxiv.org/abs/2609.24646)). PRISM's prior-art table also omits a May 2026 *soft* top-k directional penalty, Rotation-Preserving SFT, which is the closest published relative of its anchor ([arXiv 2605.10973](https://arxiv.org/html/2605.10973v1)).

Two window papers suggest part of PRISM's retention may not be directional at all:

- Forgetting concentrates in **output-embedding rows of tokens absent from the new corpus**, amplified by Adam's second-moment normalization. Raising Adam's ε on the output projection alone removes **39.4–67.9% of forgetting**, and post-hoc editing recovers under 5% ([arXiv 2610.09835](https://arxiv.org/abs/2610.09835)).
- Accumulated Adam momentum drives the probability leakage that shows up as forgetting, and resetting it helps ([arXiv 2610.03940](https://arxiv.org/abs/2610.03940)).

PRISM's lerp pulls unembedding rows back toward W₀ but leaves Adam moments untouched. Task Vector Descent shows that interpolating optimizer state alongside weights matters ([arXiv 2610.05402](https://arxiv.org/abs/2610.05402)). Two cheap controls follow: exclude the output embedding from the lerp, and lerp or reset Adam moments.

The composable upside is an *anisotropic* lerp, strong on the top singular directions and weak elsewhere, using SVC-style channel scores ([arXiv 2610.11076](https://arxiv.org/abs/2610.11076)) or SST's basis. It might keep retention while unblocking injection. Stacking it with self-distillation follows the pre-window finding that combined anchors raised 100-task retention from **1.2% to 34.9%** ([arXiv listing, 2609.06986](https://arxiv.org/search/?query=forgetting+fine-tuning&searchtype=abstract&abstracts=show&order=-announced_date_first&size=100)).

### Nobody else has seen an anchor block injection while loss looks best

The window is thin on knowledge injection, and no paper reports PRISM's phenomenon. In PRISM's runs, a retention-grade anchor posts the best validation loss of any arm while closed-book recall collapses to 2–3%, against 98–100% unanchored. A weak band injects at parity with 3× less forgetting ([PRISM README](https://github.com/timepointai/PRISM)).

Three August papers make the claim timely without pre-empting it:

- A QLoRA study finds lower-rank adapters retain more but **learn fewer facts** ([arXiv 2608.25677](https://arxiv.org/html/2608.25677v1)).
- "Knowing but Not Saying" shows models can recognize facts they cannot generate closed-book ([arXiv 2608.20794](https://arxiv.org/abs/2608.20794)).
- SFT-injected facts stay bound to their training question forms ([arXiv 2608.25243](https://arxiv.org/pdf/2608.25243)).

Injection is hard across mechanisms. In-window work on Engram-style n-gram memories finds write coverage falls from **98% at 50 facts to 12.7% at 1,000** ([arXiv listing](https://arxiv.org/search/?query=%22conditional+memory%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50)).

SST ran no injection experiment. That opens the most differentiating test available. If new facts need writes into principal directions or output rows, hard-projection methods like SST should block injection the way PRISM's strong band does. PRISM's single-scalar dial would then be the only retention method in the field with a measured injection setting.

### The RL anchor bet needs drift, because the field already dropped KL

No window paper replaces KL with a weight-space anchor in GRPO, RLVR or RLHF. One paper catalogs **seven failure modes** of KL inside group-relative updates, which validates PRISM's premise that output-space KL is a noisy anchor ([arXiv 2610.12161](https://arxiv.org/abs/2610.12161)).

The market moved first, though:

- **TRL** defaults GRPO to β=0.0 and does not load the reference model ([TRL docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)).
- **prime-rl** has no reference-model KL. Prime Intellect calls the explicit anchor "partly redundant" because on-policy RL stays close to its start ([Prime Intellect](https://primeintellect.ai/rl-glossary/kl-penalty)).
- **verl** defaults `use_kl_loss` to False ([verl docs](https://verl.readthedocs.io/en/latest/examples/config.html)).
- **Tinker** defaults to 0.0 but recommends 0.05 and names catastrophic forgetting as the risk ([Tinker docs](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)).

The saving is also bounded. In one large run the RL generator used **3× the trainer's compute** ([SemiAnalysis](https://newsletter.semianalysis.com/p/rl-systems-mind-the-gap-matching)), so removing a reference forward pass trims a minority line item. PRISM's honest pitch is *restoring drift control at zero forward-pass cost*, not cutting a cost most recipes already removed.

The window adds two cautions:

- Task Vector Descent lerps toward a recent checkpoint. It recovers "substantial non-targeted performance in SFT but **little in GRPO**" ([arXiv 2610.05402](https://arxiv.org/abs/2610.05402)).
- In off-policy RLVR, a fixed reference scored **32.96 vs 46.58** for a refreshed one ([arXiv 2610.20814](https://arxiv.org/abs/2610.20814), listing-only). A fixed-W₀ anchor needs an EMA or refreshed-anchor arm.

Prior art outside the window is also missing from PRISM's table. WARP (2024) uses an EMA policy anchor and interpolates back toward the initialization inside RLHF, while still using KL ([arXiv 2406.16768](https://arxiv.org/abs/2406.16768)). PRISM's claim must narrow to a per-step lerp to fixed W₀ as the *sole* anchor, run head-to-head against KL β-sweeps.

The anchor earns its keep where RL actually drifts:

- **Zero-RL models are fragile under later finetuning.** After a LoRA finetune on IMDB, Eurus-2-7B-PRIME fell from **53.65 to 20.77** on math, while an instruct model barely moved (56.08→55.17) ([arXiv 2609.33898](https://arxiv.org/abs/2609.33898)). PRISM's claim that a PRISM-pretrained base finetunes with ≈0 forgetting speaks to this.
- **Rollout recycling** produced ~34× more KL drift without generalization gains ([arXiv 2610.26416](https://arxiv.org/abs/2610.26416), listing-only).
- **Chained LoRA runs** raised perplexity by one to two orders of magnitude ([arXiv 2610.23918](https://arxiv.org/abs/2610.23918), listing date uncertain).

There is also a clean mechanistic test nobody has run. "The Path Not Taken" names the KL anchor as one of three gates that keep RLVR off the principal directions ([arXiv 2511.08567](https://arxiv.org/abs/2511.08567)). Whether dropping KL opens that gate, and whether a W₀ lerp closes it again, can be measured as principal-direction overlap.

## Demand flows through post-training platforms, not pretraining buyers

The spending gradient favors PRISM's backward use. A SemiAnalysis chart, relayed secondhand, estimates post-training/RL at **53–55% of OpenAI+Anthropic compute by late 2026**, with pretraining falling to 7% ([OfficeChai](https://officechai.com/ai/while-share-of-inference-has-held-steady-over-the-last-2-years-pre-training-compute-has-moved-to-post-training-semi-analysis/)). That is an estimate, not a primary source.

In the window, the platforms converged on "bring your own loss":

- **Tinker** shipped custom RL losses, full fine-tuning in beta, an SDFT recipe for continual learning, and checkpoint lineage tracing ([Thinking Machines](https://thinkingmachines.ai/news/one-year-of-tinker/)).
- **Together Post-Training** launched with custom objectives "computed locally and sent to the Trainer" at an introductory $9/GPU-hr on B200 ([Unite.AI](https://www.unite.ai/together-ai-launches-together-post-training-service-for-custom-models/)).
- **Fireworks** paused managed RFT and moved RL to its Training API ([Fireworks changelog](https://docs.fireworks.ai/updates/changelog)).
- **Cloudflare** launched an RL fine-tuning platform and conceded that fine-tuning can reduce general-purpose performance ([Cloudflare](https://blog.cloudflare.com/clef-decision-models/)). Its HN thread drew 642 points and no discussion of forgetting ([HN](https://news.ycombinator.com/item?id=49923692)).

**None markets a retention dial, an injection control, or a cheaper KL substitute.** A one-line lerp can ship as a recipe on all of them without platform buy-in.

Two forces sharpen the backward pitch. OpenAI's GPT-4o and GPT-4.1 families shut down on **14 January 2027**, and their fine-tunes go with them ([OpenAI deprecations](https://developers.openai.com/api/docs/deprecations)). That forces a re-tuning wave onto open bases. Practitioners are also converging on a rule: fine-tune only when the knowledge "can't come from elsewhere" ([Vertexcover](https://blog.vertexcover.io/fine-tuning-235b-moe-four-gpus)). That makes **knowledge injection without forgetting the use case that justifies fine-tuning at all**, which is exactly PRISM's dial. Enterprise "sovereign AI" is being defined the same way, as continual mid- and post-training of open weights with forgetting safeguards ([arXiv 2608.27147](https://arxiv.org/abs/2608.27147)).

The forward use has interest but no buyer category. Labs market efficiency as "× fewer FLOPs to match a named base model": Microsoft claims MAI-Base-2 is **4.2–12.3×** more pretraining-FLOP-efficient than DeepSeek-V4-Pro-Base, and does not mention initialization or checkpoint reuse ([Microsoft AI](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/)). PRISM should pitch itself as a stackable lever inside such recipes, not as a standalone multiplier.

Three signals sharpen the forward case:

- **Compute prices.** Transacted GPU prices rose: B200 to **$9.40/hr (+21.92% month over month)** and H100 to $3.09 (+10.75%) ([Ornn](https://data.ornn.com/)). Each saved FLOP is worth more.
- **Sovereign programs.** South Korea is adding a ₩4.7T frontier track, and an academic voice said "It may actually be faster to upgrade existing models rather than develop something new from scratch" ([Korea Times](https://www.koreatimes.co.kr/business/tech-science/20261011/cover-story-koreas-2-track-ai-strategy-casts-uncertainty-over-sovereign-model-project)). Geometry transfer maps onto that tension.
- **Lineage cuts both ways.** A summary of the EU GPAI guidelines says counted training compute includes "compute for weight merging or **initialization using pre-trained models**" ([FLI summary](https://artificialintelligenceact.eu/gpai-guidelines-overview/)). That wording is unverified against the official text. Separately, weight-lineage detectors use PRISM's own spectrum-plus-subspace decomposition ([arXiv 2608.07786](https://arxiv.org/abs/2608.07786)). A PRISM student plausibly carries a detectable teacher signature, which is an IP risk and a watermark opportunity. Nobody has measured how fast that signature decays during from-scratch training.

The audience will demand the scale result before it believes the claim. On HN, an unbenchmarked continual-learning launch drew "over-promising on avoiding catastrophic forgetting" ([HN](https://news.ycombinator.com/item?id=49783133)). PRISM's pre-registered failure condition fits that audience, but probe-scale numbers will be discounted until 410M–1B lands.

## Six moves, ranked by alpha per dollar and urgency

The ranking puts the cheapest, least-contested differentiator first and the time-limited priority claim close behind.

| Rank | Move | What it settles | Comparators that must be in the run |
|---|---|---|---|
| 1 | Injection-dial head-to-head on closed-book recall with paraphrase splits | Whether PRISM alone owns the retention↔injection frontier | SST, Rotation-Preserving SFT, EWC, LoRA, iSDFT; output-embedding-excluded and Adam-moment-lerped PRISM arms |
| 2 | Hardened 410M–1B core bet | Whether geometry-at-init is a fading offset or a durable multiplier | Tuned same-codebase random init; small-σ QK; mimetic; 108-statistic-matched init; spectrum-only / directions-only / both; AdamW and Muon arms; ≥3 scales |
| 3 | Cross-size SVD projection preprint | Priority on the open cell | FRONT/SPA-FRONT, Weight Selection, mimetic; CrossGMN at small scale; μP teacher and student; attention-width and head reduction |
| 4 | RL anchor in drift-prone regimes | Whether the anchor fixes a real problem | KL β-sweep, β=0, EMA anchor, WARP; async prime-rl, recycled rollouts, zero-RL fine-tunability test |
| 5 | Prior-fusion COUNT control | Whether logit fusion helps on time-to-target | Train-with-fusion vs inference-only fusion; COUNT input embedding; COUNT + PRISM init; epoch-boundary validation |
| 6 | Packaging and posture | Adoption and IP | Anchor as a custom-loss recipe on Tinker, Together and prime-rl; lineage-decay curve; updated prior-art table |

**Move 1** comes first because it is the cheapest and runs on PRISM's existing facts rig. It answers SST on the one axis SST never tested, and it maps directly onto the re-tuning demand.

**Move 2** is the decisive bet, and it should follow the hardened accounting the field now expects:

- a tuned baseline;
- a token-equivalent multiplier measured at end of training against a fitted baseline curve;
- donor compute stated;
- downstream evaluation alongside loss, since init-like interventions can change capabilities at equal perplexity ([arXiv 2609.38764](https://arxiv.org/abs/2609.38764));
- the teacher's optimizer recorded.

COUNT's OLMo-core ladder (60M–1B, Dolma 3) is a ready-made harness that also puts Move 5 on shared footing. modded-nanoGPT's Track 3 fixes architecture, data and batch and minimizes steps, which makes it the cleanest public venue for an init-only change ([modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)).

**Move 3** carries the clock. Even a probe-scale preprint of the SVD-space projection method, tested on attention-width reduction where CrossGMN fails, would establish priority.

**Move 4** fits the existing ~$500 prime-rl spec, since prime-rl is fully asynchronous. It should report:

- the reference-pass saving as a share of total step cost;
- principal-direction overlap with and without KL;
- an EMA-anchor arm;
- Adam-moment handling.

**Move 6** includes updating the prior-art table, which today omits WARP, Rotation-Preserving SFT, SST, COUNT, FRONT, the 2022 residual-over-n-gram paper and the Spectral Warmup null.

The composition partners, in order of expected value, are:

- SST's basis or SVC's channel scores for an anisotropic anchor;
- self-distillation as a data-side anchor;
- COUNT's input-side prior on top of PRISM's unembedding;
- spectrum-aware Muon variants (SAMuon, Musec) as hosts for the imprint;
- growth ladders seeded or extended by PRISM geometry;
- PRISM projections as the target-init input to CrossGMN-style metanetworks.

## Conclusion

The past month changed what PRISM is. A month ago it was a project claiming an unusual mechanism. It is now a project whose mechanism is consensus and whose value depends on two things nobody else has measured. The first is whether transplanted directions buy a *durable* multiplier at 410M–1B. The second is whether a single scalar anchor can trace the retention↔injection frontier better than projection methods that need SVDs. The second is the more defensible asset. It is unscooped, cheap to test, aligned with the platforms where post-training money is going, and it carries a warning ("loss is not an injection metric") that the field's evaluation habits currently miss. The RL bet is best read as insurance for async, recycled and multi-stage pipelines rather than a cost cut, because the industry already deleted the cost. Prior fusion is now PRISM's weakest arm and must be re-earned on time-to-target.

Two less obvious implications follow. First, PRISM's spectrum-plus-directions decomposition is the same one weight-lineage forensics uses. The 40-number fingerprint's best long-run use may therefore be provenance and compute attribution rather than speed, and the lineage-decay curve may matter more to sovereign and regulated buyers than the speedup does. Second, the strongest scale evidence in the window came from levers that change scaling *exponents* (growth, count priors), not starting points. If PRISM's effect is only an offset, the base rate says it fades. If it composes with an exponent-changing lever, the "3× to way" composition thesis is where the alpha lives. Several of this report's threat assessments rest on abstract-level or listing-only evidence: ICLR 2027 submissions were visible only as abstracts, and several arXiv items were seen only in listings. The verdicts on SPA-FRONT, Gibbs-Gap and COUNT's late-fusion null should be re-checked against full texts before PRISM commits compute against them.
