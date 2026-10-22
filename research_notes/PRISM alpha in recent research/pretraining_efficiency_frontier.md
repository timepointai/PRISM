# Pretraining-efficiency frontier (window 2026-09-21 → 2026-10-21) and its interaction with initialization and weight spectra

Scope note: In-window = first posted or substantially updated 2026-09-21 to 2026-10-21. Anything earlier is marked **[context, date]**. The search index had thin coverage of in-window arXiv postings. Many 2609.*/2610.* papers probably exist that these notes do not capture (see Gaps). Absence of evidence here is weak evidence.

---

## 1. What new pretraining-efficiency results appeared in the window, by which levers, at what speedups and scales?

### Takeaway
The biggest in-window result is the modded-nanoGPT wall-clock record #92: 39.9 s vs 73.9 s, 1.85×, merged 2026-09-28. Most of that gain comes from systems work, the loss function, and the embedding table, not the optimizer. The optimizer share was about 5 s of the 34 s saved. In-window papers add modest optimizer engineering (AF-Muon, Spectrally Targeted Muon) and a tokenizer-level gain (CoBPE, ~30% shorter sequences at 780M–1.3B). None of them reports a clean, tuned-baseline "X× fewer tokens" at ≥410M that approaches PRISM's probe-scale 3×.

### Cited Findings
**Speedrun leaderboard (in-window merge)**
- modded-nanoGPT PR #360 "New Record: 0.665 minutes (39.9 seconds): ANVIL2, Sampled-softmax, New Embedding Table, Full-stack fp8", by devenpzak. Opened 2026-08-31, **merged 2026-09-28**. Record 39.914 ± 0.120 s (n=18), val CE 3.27731. Prior record 73.889 ± 0.137 s (n=9) on the same machine. Δ −33.98 s, −46%, 1.85×. Hardware 8×H100 — [PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360)
- Ablation of seconds saved in PR #360: sampled-softmax training loss +8.40; hashed n-gram embedding table +7.05; ANVIL optimizer +5.08 (update rule + anneal 4.16, graph capture 0.92); full-stack FP8 +4.96; mixed-width attention +3.82; training-step graph capture +2.99; sparse grad comms/dataloader +1.20; MUDDformer residual mix +0.48 — [PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360)
- ANVIL is described only as "an improvement of 21 millinats at roughly equal wall clock, compared to record 89's shipper optimizer". The PR does not define its update rule. The new embedding table is zero-initialized and learned entirely within the timed run. Validation uses the full 50304-way softmax, while training uses sampled softmax — [PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360)
- A LinkedIn post (Marin Luet) expands ANVIL as "Averaged Normalized Velocity with Isotropic Lanes". It describes "tighter whitening than standard Newton-Schulz", dual fast/slow momentum, and row normalization. **Unverified, single social-media source.** Its attribution does not obviously match the PR author — [search result: LinkedIn](https://fr.linkedin.com/in/marin-luet)
- The modded-nanoGPT README lists record #92 as 0.665 min dated **08/30/26** (run date) and shows no records dated Sep/Oct 2026. The target is 3.28 val CE on FineWeb, timed on 8×H100 (PrimeIntellect machines). Track 2 (GPT-2 Medium, 350M, target 2.92) was last at #18, 17.35 min (12/31/25). Track 3 ("optimization": minimize steps with fixed architecture, data and batch size) is linked, but no records were visible — [modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt)
- Record #91 (08/06/26, context) was "Mask logits for infeasible token continuations during validation", a validation-side change — [modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt)

**Optimizer papers (in-window)**
- **AF-Muon** (arXiv 2610.01395, submitted 2026-10-01; Lagzian, Halvachi, Zhang, Lin, Liu). Removes the auxiliary AdamW from Muon for tied-embedding models. It uses a support-aware finite-cap LMO for tied vocab tables and RMS-normalized updates for 1-D params. Tested on decoder-only LMs from 124M to 1B plus several other modalities. Claims better mean val loss and perplexity than Hybrid Muon and a SCION-style Sign endpoint. **No numeric loss gains in the abstract.** About 20% optimizer-state memory saving and about 1% step-time overhead. An earlier version is an OPT 2026 (NeurIPS workshop) poster — [arXiv 2610.01395](https://arxiv.org/abs/2610.01395)
- **Spectrally Targeted Muon** (arXiv 2610.10965v1, 2026-10-07; Goyal & Ramkumar, Stanford). Orthogonalizes only singular values above or below a threshold τ, interpolating between normalized SGD and Muon. NanoGPT setup: 12L, d=768, 0.80B FineWeb tokens. Val loss at step 1530: Muon 3.280±0.008, AdamW 3.468±0.008, AdamW+warmup 3.396±0.004, Muon+warmup 3.309±0.001. Reports **no wall-clock speedup claim** — [arXiv 2610.10965](https://arxiv.org/html/2610.10965)

**Tokenization / data lever (in-window)**
- **CoBPE** "More Than Words: Compositional Tokenization for Efficient Language Models" (arXiv 2610.05597, 2026-10-04; Reif, Kaplan, Schwartz; listed COLM 2026). Pretrained from scratch at 780M and 1.3B. About 30% shorter sequences than BPE and +1.2 points average downstream at matched training compute. No tokens-to-loss figure in the abstract — [arXiv 2610.05597](https://arxiv.org/abs/2610.05597)

**Infrastructure (in-window)**
- **Olmo-core 3** (Ai2, 2026-10-01) is a training framework, not new model weights. 47B MoE on 8×B300: 52,000 vs 19,400 tokens/s/GPU (~2.7×). MXFP8 is ~21% faster than BF16. A 1.2T-param config peaked at 858 TFLOP/s/GPU on 512 GPUs, with random routing (system benchmark, not a trained model). The next Olmo "will use an MoE architecture." No small dense reference runs. Optimizer not specified — [Ai2 blog](https://allenai.org/blog/olmocore3); [SiliconANGLE 2026-10-02](https://siliconangle.com/2026/10/02/ai2-releases-olmo-core-3-to-make-developing-large-mixture-of-experts-llms-more-efficient/)

**Immediately-before-window context**
- **[context, 2026-08-26] Spectral Allocation / SAMuon** (arXiv 2608.25990; Wu, Yu, Zhang, Woodland). Per-singular-direction loss-optimal step sizes are "anisotropic but stable": a volatile Edge-of-Stability "head" needs small steps and a tolerant "bulk" accepts larger ones. SAMuon keeps the head at Muon scale and amplifies the bulk via a **static spectral prior**. On modded-nanoGPT models from 124M to 1B it needs **13.3–24.0% fewer training tokens than Muon** (Scion impl.) to reach the same val loss and beats tuned AdamW in all configs (no AdamW number given). SAMuon-lite has near-zero overhead — [arXiv 2608.25990](https://arxiv.org/abs/2608.25990)
- **[context, 2026-09-10/11] Musec** (arXiv 2609.11655; Liu, Wang, Luo). Clips momentum singular values above a threshold instead of flattening them all to 1. Claims stability where Muon variants diverge and parity under well-tuned configs. No scales given in the abstract — [arXiv 2609.11655](https://arxiv.org/abs/2609.11655)
- **[context, 2026-05] Token-Superposition Training** (arXiv 2605.06546, Nous Research). An early phase trains on averaged multi-token "bags", then reverts to standard training. Claims ~2–3× wall-clock speedup at fixed FLOPs, or the same loss at ~half the compute — [Nous Research](https://nousresearch.com/token-superposition); [arXiv 2605.06546](https://arxiv.org/abs/2605.06546)
- **[context, 2026] Moonshot researcher claim**: "many model arch / optimizer improvements can save you >20% flops needed for the same loss". Social-media claim, not a paper — [Digg summary](https://digg.com/ai/6ysfampi)

### Inferences
- The leaderboard's 1.85× is a **wall-clock** gain that combines systems work (FP8, graph capture, comms), a training-loss change (sampled softmax) and parameterization (hashed n-gram embeddings). It is not a token-efficiency result, so it is not the right yardstick for PRISM's "fewer tokens" claim. The optimizer's share (~5 s of 34 s, ~21 millinats at equal wall clock) is a small, steady increment.
- In-window token-efficiency gains at 0.1–1.3B are in the **~10–30% range** (SAMuon vs Muon just before the window; CoBPE's 30% shorter sequences). Data and tokenization levers (Token Superposition, BETR in the context below) are the ones claiming ≥2×. A PRISM init-only lever that kept even ~1.3× at 410M–1B against a *tuned* baseline would sit within or above the optimizer-lever range.
- The record #92 embedding table is zero-initialized and the speedrun's output projections start at zero (Q2). Speedrun recipes treat init as a small, fixed detail rather than a lever, so a weight-geometry init is underexplored in that venue. Track 3 (fixed architecture, data and batch; minimize steps) is structurally the right place to test an init-only change.

### Gaps
- No in-window Dion, SOAP or Shampoo paper was found. The nearest is **[context, 2026-07] "SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales"** (arXiv 2607.20548), whose numbers I did not retrieve — [arXiv 2607.20548](https://arxiv.org/pdf/2607.20548)
- I could not verify the contents of arXiv 2610.06387 "Scaling Down the Scaling Laws" (Joe Dwyer) or 2609.10702 "Data-Efficient Language Modeling". Only titles or reference lists surfaced — [2610.06387](https://arxiv.org/abs/2610.06387v1); [2609.10702](https://arxiv.org/pdf/2609.10702)
- ANVIL's mechanism, and whether it orthogonalizes like Muon, is unverified.
- No in-window Kimi, DeepSeek or Essential AI efficiency posts were found.
- I did not systematically browse Hugging Face daily paper pages or OpenReview ICLR 2027 submissions for the window, so coverage is incomplete.

---

## 2. New work on optimizer × weight-spectrum interaction: would an imprinted init spectrum/subspace be preserved, erased or amplified under Muon vs AdamW?

### Takeaway
The in-window evidence (Spectrally Targeted Muon) points one way: Muon-style full-spectrum updates actively **keep the parameter spectrum flat**. AdamW lowers effective rank and builds rank slowly from zero-init layers. Muon writes "a full-rank matrix in a single step". So a **decaying (non-flat) imprinted singular-value spectrum is likely to be flattened or overwritten faster under Muon than under AdamW**. Muon does, however, keep the gradient's singular *directions*, so an imprinted *subspace* could survive if gradients align with it. That is untested. A July 2026 negative result (spectral-profile init gives no gain) is the closest prior art to PRISM's spectrum component.

### Cited Findings
**In-window**
- Spectrally Targeted Muon: "full-spectrum updates keep the spectrum flat for free"; "shrinking the largest singular values is what keeps the parameter spectrum flat" — [arXiv 2610.10965](https://arxiv.org/html/2610.10965)
- Same paper: AdamW substantially reduces parameter effective rank (consistent with Wang et al. 2025), while Muon keeps richer rank. AdamW's output projections "take the whole run to build rank and are still rising at the end". Top-targeted variants lower rank the most, and their zero-initialized output projections "never gain much rank at all" — [arXiv 2610.10965](https://arxiv.org/html/2610.10965)
- Same paper on init: random init gives "approximately dynamically isometric weight matrices" (citing Pennington et al. 2017), and "training must move the parameters away from this state". It also says "keeping the spectrum flat is cheap, but isn't what drives performance" — [arXiv 2610.10965](https://arxiv.org/html/2610.10965)
- Same paper: "an orthogonalized update writes a full-rank matrix in a single step", so for Muon "warmup only throws away full-rate steps". Warmup hurt Muon (3.280 → 3.309) and helped AdamW (3.468 → 3.396). Gradient matrices are poorly conditioned, with effective rank below ~280 for most of training. Muon reaches high-effective-rank regions faster than AdamW — [arXiv 2610.10965](https://arxiv.org/html/2610.10965)

**Context (pre-window, 2026)**
- **[2026-09-10] Musec**: Muon replaces the momentum's nonzero singular values with 1. The paper links this "spectral flattening" to loss spikes and unbounded weight growth. Clipping only the top singular values "keeps the momentum matrix's spectral structure" — [arXiv 2609.11655](https://arxiv.org/abs/2609.11655)
- **[2026-08-26] SAMuon**: per-direction optimal step sizes are anisotropic but *stable* over training. Uniform Muon scaling underuses the bulk. A static spectral prior on step sizes gives 13.3–24.0% token savings vs Muon (124M–1B) — [arXiv 2608.25990](https://arxiv.org/abs/2608.25990)
- **[2026-06] "The Spectral Dynamics and Noise Geometry of Muon"** (arXiv 2606.08388): the polar update keeps the gradient's singular directions but flattens the update spectrum. Decoupled weight decay scales all singular values down without changing spectral shape. In small NanoGPT, Muon preserves stable rank and beats AdamW, while a matched small-ViT control reverses the ranking — [arXiv 2606.08388](https://arxiv.org/html/2606.08388v1)
- **[2026-05] "Can Muon Fine-tune Adam-Pretrained Models?"** (arXiv 2605.10468): Muon-trained weights keep higher stable rank and spectral entropy throughout NanoChat pretraining — [arXiv 2605.10468](https://arxiv.org/pdf/2605.10468)
- **[2026-05] "The Stability of Singular Distribution"** (arXiv 2605.26489): proposes a two-phase spectral view of LM pretraining. Weight decay suppresses weight-norm growth, which allows larger changes to the singular distribution — [arXiv 2605.26489](https://arxiv.org/html/2605.26489)
- **[2026-07-10] Complexity-Guided Component-wise Initialization** (arXiv 2607.09204; Garbers & Oh). Measured Frobenius norm and effective-rank entropy across 11 pretrained GPT-2-style checkpoints and built initializers that imitate per-component magnitudes and **spectral profiles**. These changed the models' spectral patterns but gave "no measurable performance gain", while reusing pretrained weights "remains competitive". The authors conclude that "coarse spectral matching alone is not a reliable optimization strategy" and that reuse "likely requires preserving richer information than component-wise scale and singular-value shape". The abstract does not say whether singular vectors were transferred; my reading is that they were not — [arXiv 2607.09204](https://arxiv.org/abs/2607.09204)

### Inferences
- **Erasure risk under Muon (magnitudes).** Each Muon step adds a matrix with flat singular values and a fixed spectral norm. Spectrally Targeted Muon shows these updates drive the parameter spectrum flat, and that Muon makes full-rank writes from step 1 with no warmup benefit. PRISM's DCT-compressed *decaying* spectrum is therefore likely to be overwritten faster under Muon than under AdamW. Decoupled weight decay also shrinks the init component geometrically at rate ηλ per step under either optimizer (2606.08388). This is inference; no paper tests an imprinted spectrum under Muon.
- **Possible preservation or amplification (directions).** Muon keeps the gradient's singular directions (2606.08388). If PRISM's imprinted U and V align with early gradient subspaces, which is plausible because they come from a trained model, Muon's updates would lie in that subspace and reinforce it. This would show up as persistent principal-angle overlap between W_t's top subspace and the imprinted U and V. It is untested and is the key measurement for the "composable?" question.
- **AdamW may be the friendlier host.** AdamW builds rank slowly and tends to lower effective rank (2610.10965). An imprinted low-effective-rank, decaying spectrum is closer to where AdamW drifts anyway. That may partly explain why PRISM shows large gains under AdamW. It also means part of PRISM's AdamW advantage may be "doing AdamW's slow rank-shaping for it", which Muon already does differently. This predicts a **smaller PRISM gain on top of Muon** and should be pre-registered as the null.
- **Underexploited edge: spectrum-aware Muon variants.** Musec (clip, don't flatten), Spectrally Targeted Muon (threshold) and SAMuon (static per-direction spectral prior, stable over training) all move away from full flattening. They are natural composition partners for an imprinted spectrum. SAMuon's finding that per-direction step-size profiles are *stable* over training suggests a transferred spectrum (PRISM's 8 DCT coefficients) could serve as SAMuon's static prior. This is speculative, but cheap to test at probe scale.
- **The 2607.09204 null result sets a clear ablation PRISM must pass.** Spectrum-only imprinting reportedly does nothing. PRISM's distinguishing ingredient is the U,V *directions*, so the scale-up should report spectrum-only, directions-only and both. Otherwise reviewers will read PRISM's gain against 2607.09204.

### Gaps
- No paper in or near the window directly measures what happens to a non-random, structured init spectrum or subspace under Muon vs AdamW. This is an open question and a publishable PRISM-side contribution.
- I did not verify whether 2607.09204 transferred singular vectors (abstract only), or the model scales it used.
- The Spectrally Targeted Muon paper does not give a parameter count for its 12L/768d NanoGPT, and it is a single small setup.

---

## 3. Early-training dynamics and init's lasting effect at scale (critical periods, warmup, early window; does init matter at 1B+?)

### Takeaway
In-window evidence is thin. The one concrete in-window datum is that warmup has **opposite effects** on Muon and AdamW: early steps matter differently by optimizer. Pre-window 2026 work argues init *scale* still matters for LLM loss, and that spectral-profile init alone doesn't help. I found no in-window study showing init effects persisting at 1B+.

### Cited Findings
- In-window: warmup hurts Muon (3.280 → 3.309) and helps AdamW (3.468 → 3.396) at step 1530 in a 0.8B-token NanoGPT run. The proposed mechanism is that an orthogonalized update writes a full-rank matrix in a single step. Muon drops loss faster than AdamW over the first ~100 steps — [arXiv 2610.10965](https://arxiv.org/html/2610.10965)
- **[context, 2026-06] "Small Initialization Matters for Large Language Models"** (arXiv 2606.17945): reducing init scale consistently improves pretraining loss and downstream performance in dense and MoE LLMs. Arbitrarily small init is not beneficial, and the authors propose a γ=1 threshold. This argues against the view that init effects wash out in large-scale training — [arXiv 2606.17945](https://arxiv.org/html/2606.17945)
- **[context, 2026-07] 2607.09204**: spectral-profile-matching init changes spectral structure but not performance (GPT-2-style) — [arXiv 2607.09204](https://arxiv.org/abs/2607.09204)
- **[context, 2026-01] Curriculum-learning analysis** (arXiv 2601.21698): effects are strongest at small scale and the differences narrow by 410M. This is a fade-with-scale pattern for an early-training lever — [arXiv 2601.21698](https://arxiv.org/html/2601.21698v2)
- **[context, 2024] HyperCloning** (arXiv 2409.12903): function-preserving expansion of a smaller pretrained model into a larger one reports 2–4× training speedup. This is the closest "inherit structure at init" lever with weights copied — [arXiv 2409.12903](https://arxiv.org/html/2409.12903v2)
- **[context, in-window paper using Pythia]** arXiv 2609.29494 sweeps Pythia-410M over 34 checkpoints from initialization to step 143,000 and an OLMo 2 1B model over 32 checkpoints. This shows the public step-0/early checkpoints are usable for early-window comparisons. The paper's topic is unrelated (machine self-report) — [arXiv 2609.29494](https://arxiv.org/pdf/2609.29494)

### Inferences
- If Muon makes the first ~100 steps maximally aggressive (full-rank writes, no warmup), the window in which a PRISM imprint can shape the trajectory before being overwritten is likely *shorter* under Muon. PRISM's 3× under AdamW+warmup may partly come from the warmup phase, when AdamW moves slowly and the imprint dominates. The scale-up should report curves with and without warmup.
- The 2601.21698 curriculum result (gains narrow by 410M) is the exact failure pattern PRISM pre-registered ("advantage fades toward 1× with scale"). Analogous early-training levers have shown this fade, so a fade is the prior to beat, not an unlikely outcome.

### Gaps
- No in-window paper on critical periods or the lasting effect of init at 1B+. No in-window paper on "loss-curve early window" diagnostics at scale.
- I did not fetch 2606.17945 in full, so its scales and token counts are unverified.

---

## 4. Current reproducible baselines for Pythia/OLMo-class runs at 410M–1B; anything in-window that changes the "published baseline"?

### Takeaway
Nothing in the window changes the published small-model baselines. Ai2's in-window release (Olmo-core 3) is MoE training infrastructure with no new dense 410M–1B models or ladders, and the next Olmo is to be MoE. Pythia (AdamW, 300B tokens, 2M-token batch, LR 3e-4 at 410M/1B) and OLMo 2 1B remain the trusted public references. They are **AdamW baselines with older recipes**, which raises the bar for honest comparison (see Q5).

### Cited Findings
- Pythia: each model saw ~299.9B (~300B) tokens. The 410M and 1B models share a batch size of 2M tokens and LR 3.0e-4 (per the EleutherAI GitHub page, via search summary) — [EleutherAI/pythia](https://github.com/eleutherai/pythia); [Pythia paper](https://arxiv.org/pdf/2304.01373)
- Olmo-core 3 (2026-10-01) is framework-only, with no new model weights and no small dense reference runs, tokens-to-loss figures or ladder experiments. The next-generation Olmo "will use an MoE architecture" — [Ai2 blog](https://allenai.org/blog/olmocore3)
- Ai2's October newsletter mentions a companion AstaBrief 8B release (scientific report generation), not a pretraining baseline — [Ai2 Newsletter Oct 2026](https://allenai.org/newsletters/2026-10-newsletter)
- An alternative public speedrun baseline is modded-nanoGPT Track 2 (GPT-2 Medium ~350M, target val loss 2.92 from Karpathy's llm.c baseline), whose latest record is 17.35 min (12/31/25). Track 3 fixes architecture, data and batch and minimizes steps — [modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt)
- **[context, 2026-06] Hyperball II**: at 1.2B, MuonW gives ≈10% token-equivalent speedup over the AdamW scaling law, and MuonH sustains 20–30%, with the gap growing with training duration (same authors as Fantastic Optimizers I; 8B comparison used manually chosen HPs) — [arXiv 2606.16899](https://arxiv.org/html/2606.16899v1)

### Inferences
- "Beat Pythia-410M/1B's published loss in fewer tokens" is a weaker bar than it looks. Pythia's recipe (2023, AdamW, older data) is not tuned by 2026 standards, and a modern tuned AdamW or Muon re-run at the same architecture would itself reach Pythia's loss in fewer tokens. A credible scale-up should compare PRISM against a **same-codebase re-run** (random init, same data and tokens, tuned LR and schedule), not only against Pythia's logged curve. PRISM's donor should also not be the very model whose curve is the target.
- Because the next Olmo is MoE and Olmo-core 3 is MoE-focused, OLMo-class dense 1B baselines are unlikely to be refreshed soon. OLMo 2 1B (context, pre-window) and Pythia remain the reference points.
- modded-nanoGPT Track 2 (350M, 2.92) and Track 3 (steps-to-target with fixed arch/data/batch) are cheap public venues where a PRISM init would face strong, heavily tuned Muon baselines.

### Gaps
- I did not retrieve a published tokens-to-target-loss table for Pythia-410M/1B or OLMo 2 1B. These must be read from public checkpoint logs.
- I did not find any in-window OLMo/Pythia small-model release, revised scaling ladder, or EleutherAI post.
- I could not check OLMo 3 small-model availability or recipe details within this tool budget (pre-window).

---

## 5. Methodologies and standards for honestly measuring "X× fewer tokens" claims

### Takeaway
The standing standard comes from pre-window 2025–2026 work, and nothing in the window overturns it. Tune the baseline as hard as the method. Measure at the end of training, not mid-curve. Report a token- or compute-equivalent multiplier against a scaling-law fit. Show the trend across ≥3 scales, because claimed optimizer speedups of 1.4–2× at <0.5B have shrunk to ~1.1× at 1.2B. In-window evidence adds that warmup and schedule choices interact with the optimizer, and that leaderboard gains can come from validation-side or loss-function changes.

### Cited Findings
- **[context, 2025-09] Fantastic Pretraining Optimizers I** (arXiv 2509.02046): over 4,000 models; "2× speedups over AdamW? Unlikely. Beware under-tuned baseline or limited scale! E.g. Muon: ~40% speedups <0.5B & only 10% at 1.2B (8× Chinchilla)". Many claimed speedups come from insufficient tuning, and fair sweeps remove most of the advantage — [K. Wen on X](https://x.com/wen_kaiyue/status/1963633867140526319); [arXiv 2509.02046](https://arxiv.org/pdf/2509.02046)
- **[context, 2026-06] Hyperball II** reports results as "token-equivalent speedup over the AdamW scaling law" (≈10% MuonW, 20–30% MuonH at 1.2B) — [arXiv 2606.16899](https://arxiv.org/html/2606.16899v1)
- **[context, 2026-08] Optimizer survey** (arXiv 2608.28557): recommends decomposing any speedup claim into a ratio of baseline to method resources, and flags baselines that use transferred rather than tuned hyperparameters — [arXiv 2608.28557](https://arxiv.org/pdf/2608.28557)
- **[context, 2026-07] OmniOpt** (arXiv 2607.04033): a recurring theme in the literature is revisiting claimed speedups under stricter HP search, end-of-training evaluation, HP transfer and scaling conditions — [arXiv 2607.04033](https://arxiv.org/pdf/2607.04033)
- **[context, 2025] Compute-multiplier definition** (BETR): "A compute multiplier of X between methods A and B indicates that method A requires only 1/X of the compute needed by method B to reach the same performance under compute-optimal training". BETR reports 1.8–2.8× across 1e19–1e22 FLOPs, from over 500 models — [arXiv 2507.12466](https://arxiv.org/html/2507.12466v1); definition quoted as used in [arXiv 2604.15706](https://arxiv.org/pdf/2604.15706)
- In-window: warmup flips sign between Muon and AdamW (2610.10965). The baseline's schedule choice alone can move the comparison by ~0.07 nats at this scale — [arXiv 2610.10965](https://arxiv.org/html/2610.10965)
- In-window leaderboard hygiene: record #92 trains with sampled softmax but validates with the full 50304-way softmax. Record #91 changed validation itself (masking infeasible continuations). The leaderboard requires n-run means with std (n=18 for #92) — [PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360); [README](https://github.com/KellerJordan/modded-nanogpt)
- **[context, 2025-06] Automated LLM Speedrunning Benchmark** (arXiv 2506.22419) uses modded-nanoGPT records as a reproducibility benchmark — [arXiv 2506.22419](https://arxiv.org/pdf/2506.22419)

### Inferences
Checklist for PRISM's scale-up, derived from the above:
1. **Tuned same-codebase baseline.** Give the random-init baseline an LR and warmup sweep at least as wide as PRISM's. Report PRISM's sensitivity to LR, since an imprinted init may shift the optimal LR.
2. **Token-equivalent multiplier vs a fitted baseline curve or scaling law**, measured at end of training (and at the target loss), not "time to the baseline's best mid-run loss" alone.
3. **≥3 scales (e.g., ~125M, 410M, 1B)** with the multiplier's trend plotted. This directly tests the pre-registered "fades toward 1×" failure condition. The Fantastic-Optimizers pattern (1.4× → 1.1×) is the prior.
4. **Two optimizer arms (AdamW and Muon)**, each tuned, to settle "orthogonal vs composable". A spectrum-aware Muon arm (Musec, SAMuon or Spectrally Targeted Muon) would be a third, high-information arm.
5. **Ablations against the known null.** Spectrum-only vs U,V-only vs both, so the result is not read against 2607.09204's spectral-init null result.
6. **Donor accounting.** State the donor model's training compute and whether the donor saw the evaluation distribution. A PRISM speedup that does not amortize the donor's cost is a transfer result, not a pretraining-efficiency result.
7. **Validation hygiene.** Same tokenizer, same validation set and full-softmax eval for both arms. Seeds ≥3 with std.

### Gaps
- I found no in-window paper specifically critiquing speedup claims or proposing new measurement standards. The standards above are pre-window.
- I could not verify whether ICLR 2027 OpenReview submissions (likely in or near the window) include relevant benchmark or critique papers.
