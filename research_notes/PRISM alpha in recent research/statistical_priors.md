# Statistical / n-gram / lookup-table priors combined with neural LMs (window: 2026-09-21 → 2026-10-21)

Scope note: These notes cover the relation to PRISM's "Prior-Fused PRISM" arm. That arm fuses a fixed, shared n-gram prior into the logits during training as a product of experts, so the network learns only the residual, and it stacks PRISM's SVD-spectrum geometric init on top. The headline single-seed Shakespeare probe (10.65M params) showed PRISM alone 15× faster to baseline, the n-gram prior alone 3.8×, and the hybrid 30×. A context-3 prior caps at baseline quality, and the stated path forward is a context-4+ sparse prior. The enwik8 gate is a prior floor of 1.9061 bpb.

Method: I swept arXiv search sorted by announcement date, using these query families: "n-gram language model", "conditional memory", n-gram embedding/memory, Engram/over-encoding/hash embedding, infini-gram, kNN-LM, product of experts, n-gram speculative decoding, byte-level/tokenizer-free + n-gram, unigram/bigram/token-frequency prior, and nonparametric/datastore. I then fetched the abstract and HTML pages of every hit. Dates are arXiv v1 submission dates as shown on the abs/listing pages.

Caveat: many numbers were extracted through a summarizing fetch tool from arXiv abs or HTML pages. The figures for The COUNT (the central item) should be spot-checked against the PDF before external use.

Window boundary: SwitchSD (Sep 17), DeepSeek-V4.1-Flash (Sep 17), the Inference Pareto Atlas (Sep 15), Lngram v2 (Sep 3) and Qwen3.8-Next (Aug 31) all fall just before the window. They are labeled as context.

## Q1. Has anyone in the window published training-time fusion of n-gram/count priors into LM logits (PoE / residual learning), or n-gram-guided pretraining/curricula, and with what speedups at what scales?

### Takeaway
No in-window paper scoops logit-level, training-time product-of-experts fusion of an n-gram prior aimed at tokens-to-target speedup. The single most important in-window paper is **The COUNT** (arXiv 2610.20456, Oct 14, Magnusson / Smith / Koh). Its count-based n-gram prior, fed into pretraining **on the input side**, is worth ~57% more compute at 1B params (46–89% across a 60M–1B ladder). It also directly tested **logit-level multiplicative (PoE) and additive fusion during pretraining** and found **no gain over applying the same fusion only at inference** (tested at 190M). That result both validates "count priors in pretraining" and threatens the specific logit-fusion mechanism PRISM uses, at least on the final-loss-at-fixed-budget metric.

### Cited Findings
- **The COUNT: Counting N-grams Unlocks Scaling on Host Memory and Training-Free Adaptation.** arXiv 2610.20456, v1 submitted 14 Oct 2026, by Ian Magnusson, Noah A. Smith and Pang Wei Koh (cs.CL/cs.LG). It builds n-gram embeddings with no new learned parameters from precomputed n-gram counts held in host (CPU) memory. It weights the LM's own output-unembedding rows for the tokens that most often follow the current n-gram history. "At a fixed compute budget, adding COUNT as an extra input embedding for pretraining 1B-parameter LMs improves validation loss about as much as a model with 57% more compute." — [arXiv abs](https://arxiv.org/abs/2610.20456)
- **COUNT mechanism and scale details:**
  - Orders 1–6 are counted over Dolma 3, and the table stores the top K=32 next tokens with stupid-backoff scores. Training uses K=8 and J=6 orders.
  - The embedding is a score-weighted sum of unembedding rows, averaged over matched orders, added to the token embedding at the first block, with "embedding amplification."
  - Backbones are 60M, 100M, 190M, 370M, 600M, 760M and 1B on the OLMo-core ladder, at D/N ∈ {10, 20, 40}, trained on `allenai/dolma3_mix-6T-1025-7B`, with the same corpus used for counts.
  - Body text: COUNT "improves LM loss as much as using 46–89% more compute" and task loss "comparably to a 6–56% increase in compute."
  - The main table is 797.5 GB (orders 1–6, histories seen ≥265 times). Smaller tables are 459.2, 142.0, 68.5 and 17.1 GB, and the 17.1 GB table is used for 60M–190M.
  - — [arXiv HTML](https://arxiv.org/html/2610.20456)
- **COUNT's late-fusion ablation (the key threat to Prior-Fused PRISM):**
  - Appendix B.1 defines additive fusion (interpolation with weight λ) and **multiplicative fusion, "a normalized product at the logit level, following Li et al. (2022)"**, with an ε=10⁻⁴ floor on mass for non-candidate tokens.
  - Pretraining with late fusion (190M, scalar weight initialized at 0.1, 17 GB table) "does not improve on applying fusion only at inference." Section 4.2: "activating late fusion during pretraining does not improve on purely inference-time late fusion." Section 7.1: late fusion "does not benefit from inclusion in pretraining" and improves generalist performance less than COUNT. Figure 9: "COUNT achieves better generalist performance than either additive or multiplicative late fusion."
  - The visible text gives no numeric values for these figures. It is also unclear from the extracted text whether both fusion variants, or only one, were pretrained.
  - — [arXiv HTML](https://arxiv.org/html/2610.20456)
- **COUNT supports higher-order priors.** Appendix B.5: "training with six n-gram orders gives lower LM and task loss than three orders at every budget (190M)." — [arXiv HTML](https://arxiv.org/html/2610.20456)
- **COUNT's late-fusion pretraining result is a single-scale claim.** The pretrained-fusion comparison is reported only at 190M, and other scales use inference-only fusion. So the scaling behavior of pretrained fusion is not established. — [arXiv HTML](https://arxiv.org/html/2610.20456)
- **Secondary coverage and author social posts** repeat the 57%-more-compute-at-1B claim. The affiliations are not shown on the arXiv abs page, and I did not verify them independently. — [st-hakky news summary](https://book.st-hakky.com/en/news/n-gram-count-enables-host-memory-scaling); [Digg aggregation](https://digg.com/ai/ybcsshvs)
- **No other in-window paper does logit-level residual learning over a fixed n-gram/count prior.**
  - My arXiv sweeps for "n-gram pretraining transformer", "product of experts language model", and "unigram/bigram/token frequency prior logits training" turned up no such work.
  - The only in-window "product of experts" hit is PoE-Fuse (2609.37485, Sep 27), which does feature-level fusion of frozen vision/language experts for satellite change detection and is not relevant.
  - — [arXiv search: n-gram pretraining](https://arxiv.org/search/?query=n-gram+pretraining+transformer&searchtype=all&abstracts=show&order=-announced_date_first&size=50); [arXiv search: product of experts](https://arxiv.org/search/?query=%22product+of+experts%22+language+model&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **Not a match: ORL (2610.20687, ~Oct 2026).** "On-Policy Residual Learning for Diffusion Language Models" learns "residual context" across denoising steps in diffusion LMs. It is not residual learning over a fixed prior for AR pretraining. — [arXiv 2610.20687](https://arxiv.org/html/2610.20687v1)
- **Tangential, in window: "The Geometry of Language Model Logits" (2610.18520, Oct 13).** Ordered logits follow a log-linear rank law, with slope α ≈ 1.4–1.7 in trained models versus ≈ 0.2 at initialization. This is a possible diagnostic for how far PRISM's geometric init plus prior moves the step-0 logit geometry toward the trained regime. — [arXiv search listing](https://arxiv.org/search/?query=%22unigram%22+OR+%22bigram%22+OR+%22token+frequency%22+prior+logits+training+language+model&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **Context, pre-window:**
  - Li et al. 2022, "N-gram Is Back: Residual Learning of Neural Text Generation with n-gram Language Model" (arXiv 2210.14431), is the canonical logit-level residual-over-n-gram method. COUNT's multiplicative fusion follows it. A secondary summary reports ~0.9 PPL improvement on WikiText-103 with GPT-2, and the paper reports +0.15 BLEU on WMT14 En-De. — [arXiv 2210.14431](https://arxiv.org/pdf/2210.14431); [Lacuna summary](https://lacuna.tiptreesystems.com/work/residual-learning-of-neural-text-generation-with-n)

### Window inventory at a glance (all questions)
Verdict key: **V** validates · **T** threatens · **S** scoops · **C** composable · **G** leaves a gap.

| Date (v1) | arXiv | Title (short) | Core claim | Scale | Verdict vs Prior-Fused PRISM |
|---|---|---|---|---|---|
| 2026-10-14 | [2610.20456](https://arxiv.org/abs/2610.20456) | The COUNT | Parameter-free count-based n-gram embedding (orders 1–6) in pretraining ≈ 57% more compute at 1B (46–89% LM loss, 6–56% task loss across ladder). Logit-level PoE/additive fusion in pretraining is no better than inference-only fusion (190M). | 60M–1B, D/N 10–40, Dolma 3, 17–797 GB tables | **V + T + C**; partial **S** of "count prior in pretraining" (input side); **G** on tokens-to-target |
| 2026-09-28 | [2609.35578](https://arxiv.org/abs/2609.35578) | FactorEngram | Factorized n-gram memory: sparse coefficients over shared basis, per-basis gating; best inserted before attention in middle layers | 340M, 1B | **C** (learned memory; no numbers in abstract) |
| 2026-10-07 | [2610.10533](https://arxiv.org/abs/2610.10533) | EngramEdit | Edit facts by updating shared n-gram embeddings. CounterFact after 2,000 edits: 99.5 efficacy / 93.9 utility | LongCat-Flash-Lite | Neutral (editing; shows n-gram tables hold knowledge) |
| 2026-10-10 | [2610.14921](https://arxiv.org/abs/2610.14921) | LoRA vs N-gram Memory drift | N-gram-memory-only updates drift base reps 26%/15% vs LoRA 64%/35% | 48-layer MoE; Llama-3-8B | Neutral / weak **C** |
| 2026-10-15 | [2610.21810](https://arxiv.org/abs/2610.21810) | N-Gram Memories at Two Bits | 2-bit table codes; output KL −41–63% vs RTN; DeepSeek table 203 GB → 51 GB | Production tables | **C** (storage for large priors) |
| 2026-10-18 | [2610.25317](https://arxiv.org/search/?query=%22conditional+memory%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50) | Atoms, Not Knowledge | Hash-addressed memory writes; write coverage falls 98% → 12.7% from 50 → 1,000 facts | 124M–7B; Qwen3.8-Flash-Next-FP8 (176.95B, 320M FP8 rows) | Neutral |
| 2026-10-15 | [2610.21553](https://arxiv.org/search/?query=%22conditional+memory%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50) | Writability and the Pathway | Knowledge injection into addressed memory; strict answer scores stay at 0 | GPT-2, Qwen | Neutral |
| 2026-09-22 | [2609.25853](https://arxiv.org/search/?query=Engram+OR+%22over-encoding%22+OR+%22over-tokenized%22+OR+%22hash+embedding%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50) | MemoryAthena | Engram retrieval as anchor plus routed generated memory; QA avg 37.65 → 39.28 | ~201M memory side | Neutral |
| 2026-10-08 | [2610.12599](https://arxiv.org/search/?query=Engram+OR+%22over-encoding%22+OR+%22over-tokenized%22+OR+%22hash+embedding%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50) | STAPLE | Token-indexed embedding residual pathway upcycling. OLMo-2 1B: +4.27 HumanEval, +2.27 GSM8K vs continued dense training | OLMo-2 1B/7B | **C** (lookup capacity as residual pathway) |
| 2026-09-30 | [2610.00394](https://arxiv.org/abs/2610.00394) | Forking | N-gram memory branches cause sudden train/val separation at epoch boundaries under replay, dominated by low-frequency contexts | NanoGPT; DeepSeek-style Engram | **T** (risk for high-order sparse priors on multi-epoch small data) |
| 2026-10-10 | [2610.15254](https://arxiv.org/search/?query=infini-gram&searchtype=all&abstracts=show&order=-announced_date_first&size=50) | Cascade | GPU out-of-core suffix arrays 3.0–8.2× faster than infini-gram's pipeline; C4 (198B tokens) indexed in 23 min | Corpus infra | **C** (cheap context-4+ / ∞-gram priors) |
| 2026-10-06 | [2610.07780](https://arxiv.org/search/?query=n-gram+speculative+decoding&searchtype=all&abstracts=show&order=-announced_date_first&size=50) | APEX | Learned router over EAGLE-3 / n-gram / draft model; up to 5.24× speedup; −41% wasted tokens vs fixed n-gram k=16 | Qwen3-8B, vLLM | Framing (inference-only n-gram use) |
| 2026-10-05 | [2610.05774](https://arxiv.org/abs/2610.05774) | AdaSpark | Online scheduler pricing n-gram fill vs drafted candidates; 1.5–3.1× over llama.cpp DSpark | 3 dense + 1 MoE target | Framing |
| 2026-10-05 | [2610.05978](https://arxiv.org/abs/2610.05978) | Byte Language Models | Byte Transformers with hash embeddings + token-superposition beat subword with scale; 3.4× more accepted spec-decode tokens | Sizes not in abstract | **G** / **C** (no statistical-prior study) |
| 2026-10-08 | [2610.12806](https://arxiv.org/search/?query=byte-level+OR+tokenizer-free+n-gram&searchtype=all&abstracts=show&order=-announced_date_first&size=50) | Quantum-native LMs | Byte-level quantum LMs still trail classical n-gram baselines | Tiny | Tangential |

### Inferences
- **The "open cell" is narrower than "n-gram priors in training."** COUNT now occupies "fixed, parameter-free count prior injected during pretraining, scaled to 1B with a compute-equivalent number," and it reports a positive result. Prior-Fused PRISM's remaining distinct claims are:
  - (a) the injection point is the **logits**, as a PoE residual;
  - (b) the metric is **tokens-to-target / time-to-baseline**, not final loss at fixed D/N;
  - (c) the prior is stacked with **geometric (SVD-spectrum) init**.
  - COUNT tests none of (b) or (c).
- **COUNT's late-fusion null result is the most direct threat.** At 190M and D/N ≥ 10, a model pretrained under PoE fusion ends no better than the same architecture with fusion applied only at inference. This does not contradict PRISM's 3.8× / 30× time-to-baseline numbers. Early-training acceleration and end-of-budget gains are different quantities, and PRISM measures the former on a 10.65M model. Still, PRISM must now run COUNT's control on its own metric: train under fusion vs. train without fusion and add the prior post hoc, compared on tokens-to-target and on final loss.
- **COUNT's 6-orders-beat-3 result (190M) independently supports PRISM's plan** to move from a context-3 to a context-4+ sparse prior.
- **The prior can enter on the input side, the logit side, or both, and these are composable.** A COUNT-style input embedding (which reuses the model's own unembedding rows, so it interacts naturally with PRISM's SVD-injected output geometry) plus a logit-side PoE prior plus geometric init is an untested stack. It is a plausible "composition gate" experiment.
- **COUNT's mechanism weights unembedding rows by counts, so its effectiveness depends on unembedding quality.** That suggests a specific synergy hypothesis: PRISM's SVD-initialized unembedding would make COUNT-style embeddings informative from step 0, rather than only after the unembedding has trained. This is untested.

### Gaps
- I could not see numeric values in COUNT's Figures 9–10 (late-fusion vs COUNT) or the per-size mapping of the 46–89% and 6–56% compute-equivalent ranges. Whether COUNT's gain grows or shrinks with size cannot be stated from the extracted text.
- It is unclear whether COUNT pretrained with the multiplicative (PoE) variant, the additive variant, or both in the 190M late-fusion-pretraining ablation.
- I found no in-window paper reporting **tokens-to-target-loss / early-training speedup** from any statistical prior, so no external number is directly comparable to PRISM's 3.8× / 15× / 30×.
- OpenReview ICLR 2027 submissions were not searchable. The ICLR blog says papers become public only at the end of review, so anonymous submissions in this cell may exist but be invisible. — [ICLR 2027 blog](https://blog.iclr.cc/category/iclr-2027/); [ICLR 2027 submission policies](https://blog.iclr.cc/2026/09/02/submission-policies-for-iclr-2027/)
- The Hugging Face Papers search page did not render listings via fetch, so HF-only items (no arXiv) may be missed.

## Q2. What's new in conditional memory / lookup / hashed-embedding layers (Engram-style, memory layers, product-key, over-tokenized/n-gram embeddings), and do they report pretraining-efficiency gains that overlap with Prior-Fused PRISM's claim?

### Takeaway
The window brought a burst of Engram-family follow-ups: factorized memory, quantization, editing, continual-learning drift, write/injection studies, upcycling, and failure modes. Production MoEs (DeepSeek, Qwen3.8-Next, LongCat) now ship n-gram tables of 51–203 GB. However, **only The COUNT reports a pretraining compute-equivalent gain in the window**, and it is the parameter-free count variant. The learned-memory papers in the window report architecture or quality gains without compute-equivalent or tokens-to-target numbers. One of them (Forking) documents an overfitting failure mode concentrated in low-frequency n-gram contexts.

### Cited Findings
- **FactorEngram** (2609.35578, 28 Sep 2026; Yang, Zhou, Miao, Wu). It replaces Engram's per-slot hashed embeddings, each with a single scalar gate, with sparsity-regularized coefficients over a shared basis dictionary. Each basis component is gated by the backbone hidden state. It covers single tokens and multi-token n-grams, and works best when inserted before the attention sublayer in middle layers. It improves LM and downstream performance on 340M and 1B backbones; the abstract gives no numeric deltas. — [arXiv abs](https://arxiv.org/abs/2609.35578)
- **COUNT vs learned n-gram embeddings:**
  - When GPU memory is limited, giving that memory to backbone parameters and using COUNT outperforms splitting it with learned n-gram embeddings.
  - COUNT is on the Pareto frontier against Over-Encoding (as implemented in LongCat) for learned-parameter budgets of 60M–1B.
  - Embedding amplification helps COUNT but has mixed effects on a standard transformer, raising loss at 760M and 1B.
  - — [arXiv abs](https://arxiv.org/abs/2610.20456); [arXiv HTML](https://arxiv.org/html/2610.20456)
- **N-Gram Memories at Two Bits** (2610.21810, v1 15 Oct, v2 16 Oct; Han et al.). The Conditional Program code reconstructs only the needed rows. The 2-bit code lowers output KL by 41–63% vs 2-bit round-to-nearest. The DeepSeek n-gram table shrinks from 203 GB to 51 GB. — [arXiv abs](https://arxiv.org/abs/2610.21810)
- **EngramEdit** (2610.10533, 7 Oct). Experiments use LongCat-Flash-Lite.
  - CounterFact after 2,000 sequential edits: Efficacy 99.5, Generalization 97.0, Specificity 85.2, Utility 93.9, vs the best baseline utility of 76.6 (MoEEdit).
  - Over 92% of new errors on unrelated queries involve updated shared embeddings.
  - It cites DeepSeek-V4.1-Flash (2609.19969) and Qwen3.8-Next (2608.30320) as using similar n-gram designs.
  - — [arXiv HTML](https://arxiv.org/html/2610.10533)
- **Representation Drift in Continual Learning: LoRA versus N-gram Memory** (2610.14921, 10 Oct; Hao Wu). On a 48-layer MoE, updating only the n-gram memory changes base representations by 26% (math) / 15% (code) relative RMS, vs 64%/35% for merged LoRA and 97%/39% for shared LoRA. Forgetting correlates with drift (Spearman ρ=0.85; 0.71 on Llama-3-8B). — [arXiv search listing](https://arxiv.org/search/?query=%22n-gram+embedding%22+OR+%22n-gram+embeddings%22+OR+%22n-gram+memory%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **Forking: Sudden Overfitting Under Replay** (2610.00394, 30 Sep; Yu, Guo, Zhao, Yu, Ziming Liu).
  - Models with an over-encoding n-gram memory branch show "a sharp separation of training and validation loss at epoch boundaries" under data replay, in NanoGPT and in a DeepSeek-style Engram model.
  - Low-frequency contexts contribute most of the gap.
  - The n-gram module creates "weakly interacting context-specific subspaces" that amplify it.
  - Larger training budgets and heavily crowded tables suppress it.
  - — [arXiv abs](https://arxiv.org/abs/2610.00394)
- **STAPLE** (2610.12599, 8 Oct; Sadhukhan … Beidi Chen). Adds token-indexed embeddings as a residual pathway alongside FFN up-projections when upcycling a pretrained checkpoint. On OLMo-2 1B vs continued dense training: +4.27 HumanEval, +2.80 MBPP, +2.27 GSM8K, +1.65 DROP, and gains persist at 7B. It decodes ~2× faster than an MoE-upcycling baseline. — [arXiv search listing](https://arxiv.org/search/?query=Engram+OR+%22over-encoding%22+OR+%22over-tokenized%22+OR+%22hash+embedding%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **MemoryAthena** (2609.25853, 22 Sep). Uses Engram retrieval as an anchor plus routed generated memory. The 5-task QA average rises from 37.65 to 39.28, and the 6-task general NLP average from 76.73 to 79.13. The memory side has ~201M params. — [arXiv search listing](https://arxiv.org/search/?query=Engram+OR+%22over-encoding%22+OR+%22over-tokenized%22+OR+%22hash+embedding%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **Atoms, Not Knowledge** (2610.25317, 18 Oct) and **Writability and the Pathway** (2610.21553, 15 Oct), both from Gao, Tian, Yang, Yin.
  - On Qwen3.8-Flash-Next-FP8 (176.95B params, 320,001,536 FP8 n-gram rows), write coverage falls from 98% at 50 facts to 12.7% at 1,000 facts.
  - In the frozen GPT-2 study, strict answer scores stay at zero.
  - — [arXiv search listing](https://arxiv.org/search/?query=%22conditional+memory%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **Context, pre-window:**
  - Engram, "Conditional Memory via Scalable Lookup" (arXiv 2601.07372, Jan 2026; ACL 2026) frames conditional memory as a sparsity axis alongside MoE. — [arXiv 2601.07372](https://arxiv.org/html/2601.07372v1)
  - Qwen3.8-Next (2608.30320, 31 Aug 2026) is a 125B-A6B MoE with **51B n-gram embedding parameters held off-accelerator**. It leads its 397B-A17B predecessor on 8/14 benchmarks with ~1/3 the activated params, ~1/3 the tokens and ~1/9 the training FLOPs. That gain is not attributable to the n-gram tables alone. — [arXiv search listing](https://arxiv.org/search/?query=%22n-gram+embedding%22+OR+%22n-gram+embeddings%22+OR+%22n-gram+memory%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50); [arXiv 2608.30320](https://arxiv.org/pdf/2608.30320)
  - Lngram v2 (2609.03426, 3 Sep) scales latent n-gram memory to a 30B VLM.
  - DeepSeek-V4.1-Flash (2609.19969, 17 Sep) is a 552B-backbone MoE with 16B decode-activated params on 45T tokens; its abstract does not mention Engram. — [arXiv abs](https://arxiv.org/abs/2609.19969)
  - Other COUNT-cited 2026 relatives: LongCat / Over-Encoding 2601.21204, L³ Large Lookup Layers 2601.21461, MeKi 2602.03359, Tensorizing Engram 2606.08347, and NGM (training-free inference-time memory) 2605.16893. — [arXiv HTML 2610.20456](https://arxiv.org/html/2610.20456)

### Inferences
- **Learned n-gram tables are the field's mainstream "statistical memory" axis.** They are now standard in production MoEs, and they are trained jointly from scratch. Prior-Fused PRISM's distinctive bet is the opposite end: a **fixed** statistical prior that the network does not have to learn. COUNT is the only window paper on that end, and it beat learned tables under GPU-memory constraints. That supports the "don't learn what you can count" thesis.
- **Forking is a concrete risk for PRISM's planned context-4+ sparse prior on Shakespeare-scale, multi-epoch data.** High-order contexts are mostly low-frequency, and a count prior built from the training split is maximally sharp on seen continuations. The "below the n-gram's own floor" claim should be checked on held-out val across epoch boundaries. Forking's findings that crowded tables and larger budgets suppress the effect also suggest that backoff and smoothing strength matter.
- **FactorEngram's shared-basis factorization parallels PRISM's SVD-direction injection** (low-rank shared structure across lookup rows). A factorized prior table initialized from a trained model's spectrum is a composition idea, not an observed result.

### Gaps
- FactorEngram's numeric loss/perplexity gains vs Engram, and its training token counts, were not in the abstract.
- No in-window learned-memory paper reports tokens-to-target or compute-equivalent gains, so overlap with PRISM's speed claim cannot be quantified beyond COUNT.
- I found no in-window product-key memory or Meta-style "memory layers" pretraining papers. The arXiv "memory layer" search was flooded by agent-memory papers, and I checked only the first ~38 results.
- Whether the DeepSeek-V4.1-Flash paper body describes Engram tables (the 203 GB figure in 2610.21810 suggests DeepSeek ships one) was not verified.

## Q3. What's new in inference-time n-gram/datastore interpolation (∞-gram successors, n-gram speculative drafters, kNN-LM revivals) that frames the comparison?

### Takeaway
Inference-time n-gram use in the window is mostly engineering: speculative-decoding drafters and routers, and faster suffix-array construction. COUNT supplies the one quality-side result, training-free domain adaptation by swapping count tables at λ=0.25. I found no in-window kNN-LM revival. The framing for PRISM is that the field treats count/n-gram signals as cheap **inference** add-ons, and COUNT's evidence says logit-level fusion adds nothing when moved into pretraining. PRISM's training-time PoE claim therefore has to rest on time-to-target, not on final quality.

### Cited Findings
- **COUNT training-free adaptation.**
  - Domain tables: math (83.4 GB, 10.66B tokens), biomedical (81.7 GB, 95.39B tokens), law (80.2 GB, 12.34B tokens).
  - Interpolating at λ=0.25 lowers in-domain task loss on all three, with math and law improving more than biomedical, which is mixed.
  - Shrinking the 1B model's table to 17.1 GB still beats the standard transformer without retraining.
  - Changing K or J far from the trained setting loses most of the gain.
  - — [arXiv HTML](https://arxiv.org/html/2610.20456)
- **Cascade: GPU-Accelerated Out-of-Core Suffix Array Construction for LLM Corpora** (2610.15254, 10 Oct; Khan, Pandey). 3.0–8.2× faster end-to-end than infini-gram's pipeline; indexes the 198B-token C4 in 23 minutes. — [arXiv search listing](https://arxiv.org/search/?query=infini-gram&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **APEX: Speculate smarter, not deeper** (2610.07780, 6 Oct). A learned per-request router over EAGLE-3, n-gram and draft-model speculation, plus adaptive depth, in vLLM with Qwen3-8B across 6 workloads. Up to 5.24× speedup (APEX-S average 4.27×), and APEX-B cuts wasted tokens by 41.0% vs fixed n-gram speculation at k=16. — [arXiv search listing](https://arxiv.org/search/?query=n-gram+speculative+decoding&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **AdaSpark** (2610.05774, 5 Oct; Liu, Zhang, Xu). An online scheduler that prices n-gram continuations of the request's own text alongside drafted candidates in one best-first verify order. 1.5–3.1× faster than llama.cpp's DSpark with the same drafters, and 1.17–1.52× from the scheduler alone. The abstract has no n-gram-only ablation. — [arXiv abs](https://arxiv.org/abs/2610.05774)
- **Just before the window:**
  - SwitchSD (2609.20186, 17 Sep) uses internal-state probes to detect copy intent (AUC > 0.99) and switch between EAGLE drafting and context copying, for up to +15% throughput over EAGLE3.
  - The Inference Engineering Pareto Atlas (2609.17863, 15 Sep) finds n-gram speculative decoding ran at 0.90–0.98× baseline, i.e., no benefit, for Qwen2.5-7B on vLLM 0.12.
  - — [arXiv search listing](https://arxiv.org/search/?query=n-gram+speculative+decoding&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **kNN-LM and datastore interpolation.** The arXiv "kNN-LM" search found no in-window LM paper; the only in-window hit is a DNA-barcode MAE. A "nonparametric/datastore" search found no in-window paper interpolating a datastore distribution with LM next-token probabilities. The closest are CopyGen (2610.28393, 20 Oct), a copy-constrained logit plugin for frozen LLMs, and FuseRAG (2610.22818, 15 Oct), which injects latent evidence. That search's visible results only covered 15–20 Oct. — [arXiv search: kNN-LM](https://arxiv.org/search/?query=kNN-LM&searchtype=all&abstracts=show&order=-announced_date_first&size=50); [arXiv search: nonparametric/datastore](https://arxiv.org/search/?query=nonparametric+OR+datastore+OR+%22retrieval-augmented+language+model%22+next-token&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **∞-gram successors:** no new in-window ∞-gram-style LM-interpolation paper. The most recent infini-gram-tagged items before the window are infrastructure: SoftMatcha 2 (2602.10908, Feb 2026) and Infini-News (2605.18337, May 2026). — [arXiv search: infini-gram](https://arxiv.org/search/?query=infini-gram&searchtype=all&abstracts=show&order=-announced_date_first&size=50)

### Inferences
- **Inference-time n-gram signals are being commoditized** (drafters, routers, tables built in minutes). Count-based priors are no longer an exotic dependency, which lowers the engineering cost of PRISM's context-4+ prior. Cascade-style suffix arrays make ∞-gram-order priors buildable for Pythia/OLMo-scale corpora.
- **COUNT's swap-the-table adaptation is a capability PRISM's fixed shared prior could inherit for free.** Because the prior is outside the weights, a PoE-trained residual model could in principle take a domain prior at inference. Whether a residual network trained against one prior tolerates a swapped prior is untested. COUNT's B.4 finding, that large departures from the trained K and J lose most of the gain, suggests sensitivity.

### Gaps
- No in-window kNN-LM or ∞-gram-interpolation quality numbers (perplexity deltas) were found to anchor the inference-time comparison. The latest such numbers predate the window.
- APEX and AdaSpark acceptance lengths and n-gram-only contributions are not in the abstracts.

## Q4. Any results on byte-level / tokenizer-free models where a statistical prior helps early training?

### Takeaway
I found nothing in the window. One in-window byte-LM scaling paper uses hash embeddings and shows byte Transformers overtaking subword models with scale, but it does not study statistical priors or early-training speed. "Statistical prior for byte-level early training" remains an open cell, which matters for PRISM's enwik8 prior-floor gate (1.9061 bpb), where no external comparator exists.

### Cited Findings
- **Byte Language Models: Scaling, Emergent Abstractions, and Information Allocation** (2610.05978, 5 Oct; Wang, Luo, Zhang, Wu).
  - Byte Transformers with token-superposition training and hash embeddings "consistently outperform subword Transformers as model size scales."
  - Restricting up to 25% of intermediate layers to local representations preserves downstream performance.
  - Exploiting boundary-concentrated uncertainty gives 3.4× more accepted speculative tokens than subword models.
  - The abstract gives no model sizes, token budgets, n-gram-hash details, or early-training/prior claims.
  - — [arXiv abs](https://arxiv.org/abs/2610.05978)
- **Trainable Quantum-Native Language Models** (2610.12806, 8 Oct). Byte-level quantum-register LMs improve test NLL by 0.20 ± 0.03 nats/byte over controls at context length 12, but "still trail classical n-gram baselines." This is tangential, but it shows n-gram baselines remain the reference at small scale and short context. — [arXiv search listing](https://arxiv.org/search/?query=byte-level+OR+tokenizer-free+n-gram&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **Context, pre-window:** Chang et al., "Characterizing Learning Curves During Language Model Pre-Training" (arXiv 2308.15419; TACL 2024), finds that pretraining learns n-gram statistics first, then gradually refines low-probability predictions using longer context. This is a mechanism-level reason a fixed n-gram prior should shortcut early training. — [arXiv 2308.15419](https://arxiv.org/html/2308.15419v2)

### Inferences
- Because byte-level models must first learn the byte→sub-word n-gram structure that tokenizers otherwise supply, a fixed byte n-gram prior plausibly gives a larger early-training speedup there than at token level. No window paper tests this, so PRISM's enwik8 work would be a first data point rather than a replication.

### Gaps
- I found no byte-level or tokenizer-free paper in the window reporting bpb-vs-tokens with a count/n-gram prior, and none reporting enwik8 numbers comparable to the 1.9061 bpb floor.
- Whether 2610.05978's hash embeddings are n-gram hashes (BLT-style) was not stated in the abstract.

## Q5. Is there evidence on whether prior-fusion gains persist or vanish with scale?

### Takeaway
The only in-window scaling evidence is COUNT's. **Input-side** count priors keep a large compute-equivalent gain across 60M–1B (46–89% on LM loss, 57% at 1B), so that gain does not vanish by 1B. **Logit-side** fusion showed no benefit from being present during pretraining at the single scale tested (190M). For logit-level PoE, the evidence leans toward "final-loss gains collapse to the inference-only effect," while early-training acceleration (PRISM's metric) is unmeasured at any scale above PRISM's own 10.65M.

### Cited Findings
- COUNT improves LM loss as much as 46–89% more compute, and task loss as much as 6–56% more compute, across the 60M–1B ladder at D/N ∈ {10, 20, 40}. The abstract headline is 57% at 1B. — [arXiv abs](https://arxiv.org/abs/2610.20456); [arXiv HTML](https://arxiv.org/html/2610.20456)
- COUNT's late-fusion pretraining test was run only at 190M. The scaling of pretrained fusion "is not established," and other scales used inference-only fusion. — [arXiv HTML](https://arxiv.org/html/2610.20456)
- Embedding amplification, which COUNT needs, increases loss for the plain transformer at 760M and 1B. Part of COUNT's measured margin at the top of the ladder is therefore a COUNT-plus-amplification vs no-amplification comparison. — [arXiv HTML](https://arxiv.org/html/2610.20456)
- Forking: "larger training budgets and heavily crowded tables suppress" the n-gram-memory overfitting gap. Failure modes of n-gram components shrink with budget, not only with model size. — [arXiv abs](https://arxiv.org/abs/2610.00394)
- **Production-scale signal, pre-window:** Qwen3.8-Next (Aug 31) carries 51B off-accelerator n-gram embedding params in a 125B-A6B MoE and reports ~1/9 the training FLOPs of its predecessor at comparable quality. This is confounded with other architecture changes. — [arXiv search listing](https://arxiv.org/search/?query=%22n-gram+embedding%22+OR+%22n-gram+embeddings%22+OR+%22n-gram+memory%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50)

### Inferences
- **The scale evidence splits by injection point.** Input-side count embeddings persist to 1B (COUNT), and learned n-gram tables persist to production scale (Qwen/DeepSeek/LongCat). Logit-side PoE during training has a null result on final loss at 190M. If PRISM's 30× hybrid result is mostly an early-training effect (consistent with "networks learn n-gram statistics first"), it may hold on the tokens-to-target metric while vanishing on final-loss-at-budget. That should be stated explicitly in the pre-registered core bet for Pythia/OLMo.
- **COUNT's ladder (OLMo-core, 60M–1B, Dolma 3) is a ready-made external harness.** Reproducing Prior-Fused PRISM on that ladder, with COUNT as a baseline arm, would place PRISM directly against the newest positive result rather than against older kNN-LM and ∞-gram work.

### Gaps
- There is no per-size breakdown of COUNT's 46–89% range in the extracted text, so the trend direction from 60M to 1B is unknown.
- There is no in-window evidence above 1B for any fixed (non-learned) statistical prior used in pretraining.
- There is no in-window evidence on whether **logit-level** fusion during pretraining changes **time-to-target** at any scale. This is the cell PRISM still uniquely occupies, but it is now flanked by COUNT's null on final loss.
