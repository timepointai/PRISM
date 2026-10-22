# Forgetting, retention, knowledge injection and KL alternatives in LLM finetuning / RL post-training (window 2026-09-21 → 2026-10-21)

Scope note: window = items first submitted or substantially updated 2026-09-21 to 2026-10-21. Dates below come from the arXiv abstract page when I opened it ("abs-verified"). Items I saw only in arXiv search listings are marked "listing-only". For those, the date and summary come from arXiv's own search-result abstract text and I did not open the paper. A few listing dates don't fit their arXiv ID sequence and are flagged. Framework docs (TRL, verl, prime-rl, Tinker) are "current as accessed 2026-10-21" and are not necessarily window-dated. Verdict tags relative to PRISM: **VALIDATES / THREATENS / SCOOPS / COMPOSABLE / GAP-PRISM-FILLS**.

Headline: I found **no window paper that proposes or tests a weight-space anchor (L2-to-init, lerp-to-base, EMA-to-reference, parameter-space trust region) as a replacement for KL-to-reference in GRPO/RLVR/RLHF**. I also found **no window paper reporting that an anchor blocks fact injection while validation loss looks best**. Both PRISM bets look unscooped in the window. The closest competitor to PRISM's *mechanism* claim (singular directions do the retention work) is **Subspace Tuning (SST), arXiv 2610.22293 (15 Oct 2026)**. It strongly validates the directional story at 1.7B–32B scale for SFT, and that also makes it the main novelty threat.

---

## Q1. Has anyone in the window proposed or evaluated weight-space anchors as a replacement for KL-to-reference in RL post-training? What does dropping KL do to forgetting, and is there an opening for a cheap anchor?

### Takeaway
No window paper replaces KL with a weight-space anchor in RL. The nearest window item is **Task Vector Descent (2610.05402)**, which lerps toward the *previous checkpoint* (not the base) at sequence boundaries and was tested in GRPO. Its authors report that the lerp recovers "little" non-targeted performance in GRPO, versus "substantial" recovery in SFT. That is a caution flag for PRISM's RL bet. Mainstream stacks now default to **no KL at all** (TRL β=0.0, prime-rl has no reference KL, verl `use_kl_loss=False`, Tinker `kl_penalty_coef=0.0`), so drift control in RL is largely unfilled. A zero-forward-pass anchor has an opening there. The open question is whether on-policy RL forgets enough to need one. The best window evidence that it does is indirect: KL-free "zero-RL" models are much more fragile under later finetuning (2609.33898).

### Cited Findings
**Window items: RL drift control / KL**
- **2610.12161, "When KL Regularization Misfires in Group Policy Optimization"** (F. Ding, sole author; abs-verified, submitted 8 Oct 2026, 27 pp.). Asks why *removing* reference-KL sometimes improves group policy optimization. Identifies seven failure modes of KL inside group-relative updates:
  - k1 estimator sampling noise when folded into rewards
  - residual KL updates after reward clipping, after gradient cancellation, and in identical-reward groups
  - KL growth with response length and imbalance in KL's relative contribution
  - KL concentration on a few tokens

  Proposes ZCPO, which calibrates within-group reward coefficients by conditional-KL relative drift. Math-reasoning experiments only; no forgetting/retention evaluation and no model sizes or numbers on the abs page. **Verdict: VALIDATES** PRISM's premise that output-space KL is a noisy, awkward anchor in GRPO, and **leaves a GAP** because it stays in KL-space and never measures retention. — [arXiv 2610.12161](https://arxiv.org/abs/2610.12161)
- The same paper cites prior PAPO results: removing reference-KL from GRPO raised Qwen2.5-VL-3B/7B multimodal reasoning avg@8 from 47.92%→50.18% and 58.78%→61.30%. It notes DAPO and GMPO omit reference-KL. — [arXiv 2610.12161 HTML](https://arxiv.org/html/2610.12161v1)
- **2610.05402, "Task Vector Descent: Learning from Non-IID Batches"** (Baumann, Hübotter, Akata, Krause; abs-verified, 4 Oct 2026).
  - Method: run L local AdamW steps from checkpoint θ_t, then set θ_{t+1} = θ_t + λ(θ_t(L) − θ_t). This is a lerp toward the locally adapted endpoint, applied once per sequence boundary, not per step. The optimizer state is interpolated with the same λ (first moment linearly, second moment in sqrt space).
  - The anchor is the *most recent integrated checkpoint, not the pretrained model*.
  - GRPO experiment: Qwen2.5-3B-Instruct on Reasoning Gym (Countdown, Mini-Sudoku, N-Queens, String Manipulation). Best average gain at λ=0.6 for L=256 and L=1024. KL use not visible.
  - Authors state scaling "recovers substantial non-targeted performance in SFT but little in GRPO".
  - Other numbers: reasoning SFT Best@4 gain 30.43→41.53 at λ=0.4; continual pretraining (GPT-2 124M) BPT gain 0.386→0.463 at λ=0.6.
  - L2-SP, EMA and KL penalties are not discussed.

  **Verdict: closest window relative (same lerp algebra). Partially THREATENS the RL bet** (an interpolation anchor did little for retention in GRPO). **Not a scoop**: different anchor target (recent checkpoint vs W₀), different cadence, and no KL replacement. Also **COMPOSABLE**: its finding that the optimizer state should be interpolated too is directly relevant to PRISM's lerp, which (as specced) moves W but not Adam moments. — [arXiv 2610.05402](https://arxiv.org/abs/2610.05402); [HTML](https://arxiv.org/html/2610.05402v1)
- **2609.33898, "No Free Efficiency: Revisiting the Trade-off Between Training Efficiency and Model Vulnerability"** (Liu, Sakuma, Backes, Wen; abs-verified, 27 Sep 2026).
  - Section 3.1 says zero-RL recipes omit explicit KL regularization. Section 6 offers this as a "plausible optimization-level explanation" for sharper geometry, but the paper does not test it directly.
  - Forgetting test: LoRA-finetune on IMDB, then measure average math accuracy over GSM8K, MAWPS, Minerva Math, MATH500, OlympiadBench, AIME24 and AMC23:
    - Qwen2.5-Math-7B-Instruct: 56.08→55.17
    - Qwen2.5-7B-SimpleRL-Zoo: 53.86→47.52
    - Eurus-2-7B-PRIME: 53.65→**20.77**
  - Sharpness on GSM8K: 0.0615 (Instruct) vs 0.9120 (SimpleRL) vs 0.5565 (PRIME).

  **Verdict: GAP-PRISM-FILLS / VALIDATES the pain point.** KL-free RL models are fragile for downstream finetuning, and PRISM reports that a PRISM-pretrained base finetunes with ≈0 forgetting. Caveat: the comparison is confounded by pipeline differences, and the KL explanation is the authors' hypothesis. — [arXiv 2609.33898](https://arxiv.org/abs/2609.33898); [HTML](https://arxiv.org/html/2609.33898v1)
- **2610.20814, "Gibbs-Gap Policy Optimization for Off-Policy RL with Verifiable Rewards"** (listing-only, 14 Oct 2026). Reference refresh matters: a fixed-reference ablation scores 32.96 vs 46.58 with the default refresh schedule on a five-benchmark average. **Verdict: THREATENS a naive fixed-W₀ anchor in off-policy RL** (a fixed reference cost reward), and suggests an EMA/refreshed anchor variant. — [arXiv 2610.20814](https://arxiv.org/abs/2610.20814); [arXiv search listing](https://arxiv.org/search/?query=KL+reference+policy+GRPO&searchtype=abstract&abstracts=show&order=-announced_date_first&size=100)
- **2610.18738, "Enabling Token-wise Importance Sampling in Off-Policy RLVR with Proximal KL Regularization"** (listing-only, 13 Oct 2026). Separates a proximal reference from the sampling policy and replaces PPO clipping with bounded future-KL credit; reported up to +2.58% accuracy over GRPO. Stays in KL-space; no retention eval. **Verdict: leaves GAP.** — [arXiv 2610.18738](https://arxiv.org/abs/2610.18738)
- **2610.23085, "Rethinking GRPO for Large Reasoning Model Safety Alignment"** (listing-only, 16 Oct 2026). "Beacon" applies KL anchoring only on benign prompts to limit over-refusal; reports up to 32.83 pp lower average safety error. Shows practitioners are moving toward *selective* anchoring. **Verdict: COMPOSABLE.** — [arXiv 2610.23085](https://arxiv.org/abs/2610.23085)
- **2610.26416, "Co-Evolving Policy and Reward for Code Generation: A Mechanistic Study of Single-Adapter Drift"** (listing-only, 19 Oct 2026). Rollout recycling gives ~34× more KL from the reference than execution-only GRPO, and the gains don't generalize to held-out problems. Drift is a live failure mode in off-policy/recycled RL. — [arXiv 2610.26416](https://arxiv.org/abs/2610.26416)
- **2610.24137, "Continual Test-Time Learning with Evolution Strategies (ESRL)"** (listing-only, 16 Oct 2026). Perturbs an anchor policy that is refreshed from the student at block boundaries (not the base); up to +31.7% accuracy over base with "substantially reduced forgetting". — [arXiv 2610.24137](https://arxiv.org/abs/2610.24137)
- **2610.24236, "Preserving Reasoning Diversity in RLVR via Adaptive Replay (TAR)"** (Li, Galhotra, Cardie; abs-verified, 16 Oct 2026). Memory bank of base-model solutions, replayed only when they are being suppressed; two Qwen2.5 scales, six math benchmarks. Claims substantially reduced forgetting of correct base solutions; search-snippet text says ~0–4% vs severe for GRPO (not verified on the abs page). Output-space replay, no weight anchor. **Verdict: COMPOSABLE** (it fights mode loss; PRISM fights parameter drift). — [arXiv 2610.24236](https://arxiv.org/abs/2610.24236)

**Pre-window context (date-labeled), important prior art for the RL bet not in PRISM's "known relatives" list**
- **WARP: "On the Benefits of Weight Averaged Rewarded Policies"** (arXiv 2406.16768, June 2024). An RLHF method that uses an EMA of the policy as a dynamic KL anchor, and linearly interpolates merged policies *back toward the initialization* (LITI). This is the closest published "weight-space interpolation toward init inside RLHF" relative. It still uses KL during training. **PRISM must cite and differentiate**: PRISM uses a per-step lerp to fixed W₀ as the *only* anchor with no KL forward pass, versus WARP's post-hoc/periodic interpolation plus KL. — [arXiv 2406.16768](https://arxiv.org/abs/2406.16768) (via [arXiv search](https://arxiv.org/search/?query=%22weight+space%22+regularization+language+model+fine-tuning&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50))
- **2605.30148, "Overcoming Forgetting in LLM Fine-Tuning with Evolution Strategies"** (May 2026). Introduces "Anchored Weight Decay", which pulls toward the initial parameters during ES fine-tuning. This is an explicit L2-to-init anchor for LLM finetuning, but in ES rather than gradient RL. — [arXiv 2605.30148](https://arxiv.org/html/2605.30148v1)
- **2602.18037** (Feb 2026) proposes *gradient regularization* to replace the KL penalty against reward hacking in RLHF/RLVR. It also reports that a smaller KL coefficient from the start does not work, and discusses periodic reference resets. — [arXiv 2602.18037](https://arxiv.org/pdf/2602.18037)
- **2510.00237, "Debunk the Myth of SFT Generalization"** (Oct 2025) tested an L2 anchor to base parameters and token-level KL to the base as proximity controls in finetuning. It notes RL-tuned models stay closer to base checkpoints than SFT-tuned ones. — [arXiv 2510.00237](https://arxiv.org/pdf/2510.00237)
- **SALSA** (arXiv 2411.01798, Nov 2024) keeps KL but swaps the reference for a weight-space *soup* of two SFT models, allowing larger deviation while staying stable. — [arXiv 2411.01798](https://arxiv.org/html/2411.01798v1)

### Inferences
- **RL-anchor bet status: OPEN (not scooped in window).** Prior art exists outside the window (WARP 2024 LITI/EMA, Elastic Reset, ES Anchored Weight Decay 2605.30148, the L2-to-base controls in 2510.00237). PRISM's claim has to be narrow: a *per-step lerp to fixed W₀ as a drop-in replacement for KL-to-reference, removing the reference forward pass, evaluated head-to-head against KL β-sweeps on retention and reward*. No one in the window has run that comparison.
- **The biggest risk is not a scoop but low need.** Window and recent evidence (SST's motivation, Path-Not-Taken, Task Vector Descent's GRPO null, flat MMLU/IFEval in many GRPO papers; see Q2/Q5) says on-policy RL already forgets little. The pre-registered GRPO experiment should therefore pick settings where RL *does* drift:
  - off-policy/async RL (prime-rl is fully async)
  - rollout recycling (2610.26416)
  - long runs where ProRL-style resets are used
  - zero-RL from base (2609.33898 fragility)
  - multi-stage SFT→RL→SFT pipelines
- 2610.20814's fixed-vs-refreshed reference result suggests PRISM's RL spec should include an **EMA-of-policy anchor variant** (W₀ replaced by a slow EMA), as well as fixed W₀. That is a cheap ablation that pre-empts the obvious reviewer objection.
- Task Vector Descent's result (interpolating the optimizer state with λ matters) implies PRISM should report whether its lerp also scales Adam moments. A pure weight lerp leaves momentum pointing away from W₀, which 2610.03940 (Q2) links to forgetting.

### Gaps
- No window paper measures *general-capability forgetting* (MMLU/IFEval/etc.) for GRPO with β=0 vs β>0 on the same base. I could not find a controlled "drop KL → X points more forgetting" number in the window. 2609.33898 is confounded and measures fragility under *later* finetuning, not forgetting during RL.
- I could not open the GRPO appendix of 2610.05402 (truncated), so I don't know the exact GRPO stability numbers, or whether KL was on.
- 2610.20814, 2610.18738, 2610.23085, 2610.26416 and 2610.24137 are listing-only; their numbers come from arXiv search-result abstracts.
- OpenReview ICLR 2027 submissions were not indexed by my search tools. I found no ICLR-2027-specific items and can't rule out relevant anonymous submissions.

---

## Q2. New findings on why RL forgets less than SFT, and what they imply for a directional weight anchor

### Takeaway
The window consolidates a **geometric explanation**: on-policy RL/self-distillation update *off* the principal singular directions, while SFT concentrates updates *on* them. The strongest window paper, **SST (2610.22293)**, turns this into an SFT method that projects updates away from the top singular subspace and nearly eliminates forgetting at 8B–32B. This is strong outside **validation** of PRISM's attribution result (pinning singular directions does the work). It is also the main **novelty threat**: a well-credentialed group shows the directional mechanism at scale for SFT. Second-order window findings point to **optimizer-level causes** (Adam ε / second-moment amplification on absent-token output embeddings; momentum history), which PRISM's lerp may partly counteract but should test explicitly.

### Cited Findings
- **2610.22293, "Towards Continual Learning: Mitigating Forgetting With Subspace Tuning (SST)"** (Chavan, Sareen, Tiwari, Chandra, Obando-Ceron, Keutzer, Niekum, Dhillon, Khatri; abs-verified, 15 Oct 2026). — [arXiv 2610.22293](https://arxiv.org/abs/2610.22293); [HTML](https://arxiv.org/html/2610.22293v1)
  - **Claim:** On-policy methods "avoid updating the principal singular directions" and forget less; SFT concentrates updates along them. SST restricts each update to the *bottom* singular subspace (projects out the top-k directions of W_ref), optionally refreshing the basis every K steps. Optimizer-agnostic; composes with SFT and self-distillation. No RL experiments of its own; the RL claim cites prior work (Path-Not-Taken, RL's Razor).
  - **Scale:** Qwen3 1.7B/4B/8B/14B (5-epoch Science), 32B (2-epoch). Also Qwen2.5-7B-Instruct and Olmo-3-7B-Think.
  - **Retention, Qwen3-8B "LM avg" over GPQA-D, MMLU-Pro, GSM8K, HumanEval and IFEval (base 81.0):**
    - SFT 55.5 (−25.5) → SFT+SST 80.0 (−1.0); SFT+SST+R10 80.2
    - SDFT 57.0 (−24.0) → SDFT+SST 79.5
    - SFT+Muon 59.5; ReInvoke (post-hoc KL repair) 70.3; OPLoRA 79.2
  - **Mean over four domains:** SFT 37.6, Sculpting 54.9, OPLoRA 63.1, SFT+SST 79.6 (base 81.0).
  - **By scale:** SFT loses 30–38 pts at every size. SST loses 0.9 at 14B, 3.6 at 4B, 8.0 at 1.7B (benefit grows with scale). 32B two-epoch: SFT −15.6 vs SST −1.0.
  - **Instruct model, less clean:** Tool Use −11.3→−5.6; Medical −7.1→−4.7.
  - **Reverse-KL baseline (App. B.10, FinQA prior capability):** λ=0: 23.6; 0.001: 23.2; 0.01: 21.3; 0.1: 10.6. Weak KL leaves forgetting unchanged and strong KL suppresses task learning; meanwhile protecting rank recovers prior capability "almost completely" and in-domain accuracy *rises*.
  - **KL drift** on WikiText-2: 2–4× lower than unconstrained recipes.
  - **Cost on Qwen3-8B (8×L40S):** projection +26.2 ms/step (1.5%); basis refresh +5.1% (R100) / +37.6% (R10), or 0.6%/4.7% when SVD is split across 8 GPUs. No inference cost.
  - **Not tested:** LoRA/MiLoRA/PiSSA/EWC/L2-to-init baselines, knowledge injection, RL.
  - **Verdict: VALIDATES** PRISM's attribution, and its KL-dial result mirrors PRISM's "Pareto-dominates lower LR". **THREATENS** novelty of "directional preservation beats KL/LR for SFT retention". **Leaves GAP:** no injection test, no RL test, no soft-anchor/lerp comparison, and an SVD dependency PRISM avoids.
- **2511.08567, "The Path Not Taken: RLVR Provably Learns Off the Principals"** (Zhu et al., Nov 2025; context, pre-window). Proposes a "three-gate" theory: KL anchor; model geometry steering updates off principal directions into low-curvature, spectrum-preserving subspaces; and bf16 precision hiding micro-updates, so the off-principal bias *looks like* sparsity. Finds RLVR shows minimal spectral drift and reduced principal-subspace rotation, while SFT targets principal weights and distorts the spectrum. Argues SFT-era PEFT is mismatched to RL. — [arXiv 2511.08567](https://arxiv.org/abs/2511.08567)
- **2610.05200, "Learning without Overwriting: A Theory of Self-Distillation and SFT in Continual Reasoning"** (Uemura, Suzuki; abs-verified, 4 Oct 2026). Models reasoning as DAG search. Hinted on-policy self-distillation induces "sparse yet effective" updates that avoid forgetting; SFT on correct paths makes dense updates that overwrite. Theory only on the abs page, with no numbers. **Verdict: VALIDATES** the "SFT drift is dense/structured" framing. — [arXiv 2610.05200](https://arxiv.org/abs/2610.05200)
- **2610.25651, "OPDArena: Understanding Learning and Forgetting in On-Policy Distillation"** (Xu, Ye, Sun, Wang, Dong, Wei / Microsoft Research; abs-verified 19 Oct 2026, listing shows 18 Oct). Five pretrained models compared under SFT, RLVR and OPD. Students of RLVR-trained teachers learn while retaining; students of SFT-trained teachers lose more. *Out-of-domain* distribution change is the more consistent predictor of forgetting. No numbers on the abs page. — [arXiv 2610.25651](https://arxiv.org/abs/2610.25651)
- **2610.02437, "Learning Style, Forgetting Semantics: A Case Study of SFT and RFT on Classification Tasks"** (listing-only, 1 Oct 2026). SFT with a nonuniform teacher develops style drift with lower-bounded forgetting; RFT retains zero semantic error in simulations. — [arXiv 2610.02437](https://arxiv.org/abs/2610.02437)
- **2610.04158, "How RL Reshapes LLM Reasoning: Transferability, Coverage, and Scaling Laws"** (listing-only, 2 Oct 2026). Reports both cross-domain gains *and* forgetting after RL, attributed to shifts in strategy selection. This is evidence that RL is not forgetting-free. — [arXiv 2610.04158](https://arxiv.org/abs/2610.04158)
- **2610.17726, "TSRL: Trigger-based Supervised RL for Continual Agent Training"** (listing-only, 13 Oct 2026). RL backbone with SFT only on failure states; retains up to 9.6 pts more prior-task capability than SFT→RL. — [arXiv 2610.17726](https://arxiv.org/abs/2610.17726)
- **2609.35259, "On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics"** (listing-only, 28 Sep 2026). Llama3 and Qwen2.5. Forward KL is robust to the rollout policy while reverse KL is more sensitive; *learning rate governs forgetting and update sparsity*. — [arXiv 2609.35259](https://arxiv.org/abs/2609.35259)
- **2610.09835, "A Deafening Silence: Catastrophic Forgetting Lives in the Output Embeddings of Tokens the Data Never Speaks"** (Han, Song, Park; abs-verified, 7 Oct 2026). Freezing studies (5 settings up to 1.4B) show forgetting concentrates in *output-embedding rows of tokens absent from the new corpus*. Absent tokens get persistent one-sided softmax gradients, amplified by Adam's second-moment normalization. Fix: raise Adam ε for the output projection only. Removes 39.4–67.9% of forgetting across 7 stable configs (8 settings, 160M–12B, four families) without hurting target learning. With replay it reaches 79.8% (Qwen/Korean). It rescues released-head LoRA from a 23× forgetting surge. Post-hoc editing of drifted rows recovers <5%, so the intervention must happen during training. **Verdict: COMPOSABLE / partial THREAT.** It is a near-free optimizer-level alternative mechanism, and suggests PRISM's retention may partly come from pulling unembedding rows back. — [arXiv 2610.09835](https://arxiv.org/abs/2610.09835)
- **2610.03940, "A Step Towards Forgetting: Optimiser History and the Loss of Answer Mass"** (Ruscio, Nabarro, Thompson; abs-verified, 2 Oct 2026). Three LM families. Old-task loss rises mainly from probability *leaking outside the answer set*, not confusion among old answers. Accumulated Adam momentum from *older* new-task gradients drives the leakage. Resetting momentum (norm-matched) lowers final old-task loss. No anchoring discussed. **Verdict: COMPOSABLE.** — [arXiv 2610.03940](https://arxiv.org/abs/2610.03940)
- **2609.16537, "What Does Layer-Importance Reveal About Transformers and SSMs?"** (listing-only, 14 Sep 2026, pre-window). Residual transformers up to 14B: necessity and plasticity anti-align, and concentrating updates in plastic layers increases forgetting. — [arXiv search listing](https://arxiv.org/search/?query=forgetting+fine-tuning&searchtype=abstract&abstracts=show&order=-announced_date_first&size=100)

### Inferences
- The "RL forgets less because it avoids principal directions" story is now mainstream, cited by SST and others. PRISM's attribution result (pinning singular directions does the work; spectrum-only does nothing) is **consistent with, not ahead of**, the field. The claim PRISM can still own is the *mechanism of a soft, SVD-free isotropic lerp that de facto protects the principal directions*, plus the dial and injection behavior.
- **Direct implication for the RL bet:** if RLVR already avoids principal directions, a W₀-anchor's marginal benefit in *on-policy* GRPO should be small, consistent with Task Vector Descent's GRPO null. Expected value concentrates in: (a) the SFT/cold-start stages before RL; (b) off-policy/async RL; (c) the reference-forward-pass *cost saving* at equal retention, rather than a retention gain. The pre-registration should state which of these it bets on.
- SST's reverse-KL sweep (weak = useless, strong = blocks learning) is the same failure shape PRISM reports for LR and for strong anchors. That makes "anchor/KL strength is a dial with a narrow useful band" a robust cross-method phenomenon. PRISM's weak band (s≈0.0025) is the analogue of a protected-rank sweet spot.
- 2610.09835 and 2610.03940 suggest two cheap controls for PRISM's attribution experiments: (i) exclude the output embedding from the lerp; (ii) reset or lerp Adam moments. These tell you whether PRISM's retention is "directional" or partly "unembedding-row" and "momentum" repair.

### Gaps
- No window paper directly measures principal-direction overlap of updates under GRPO *with vs without* KL. Whether dropping KL changes the off-principal property is untested; Path-Not-Taken lists the KL anchor as one of its three gates.
- OPDArena, 2610.05200 and SVC (Q4) gave no numbers on the abs pages; full texts not read.
- SST's rank-ablation exact values weren't extractable (figure-only).

---

## Q3. New results on knowledge injection via finetuning, "learns less / forgets less" tradeoffs, and evaluation pitfalls. Does anyone else report anchors blocking injection?

### Takeaway
The window is **thin on knowledge injection**. I found **no window paper reporting that a regularizer or anchor blocks closed-book fact acquisition while validation loss looks best**, so PRISM's injection-dial finding appears **unscooped**. The best supporting evidence is from **August 2026 (pre-window)**:
- a QLoRA acquisition–retention frontier (lower rank retains more but learns fewer facts)
- a "knowing but not saying" pitfall (models recognize facts they can't generate closed-book)
- a finding that SFT-injected facts are bound to training question forms

Together these back PRISM's "loss looks best while recall is 2–3%" warning as a real evaluation pitfall.

### Cited Findings
**Window**
- **2610.22381, "Binding-Specific Relearning Savings for Test-Time Continual Learning"** (listing-only; listing says submitted 2 Oct 2026, which is **inconsistent with its ID order** — 2610.22293 is 15 Oct — so treat the date as uncertain). Qwen2.5-1.5B: previously learned facts speed relearning, cutting required optimizer updates by ~18%. — [arXiv 2610.22381](https://arxiv.org/abs/2610.22381)
- **2610.17607, "Facts Don't Speak Louder than Words: The Behavioral Essence of Long-CoT Distillation"** (listing-only, 12 Oct 2026). Argues long-CoT distillation mainly transfers reasoning *behavior* rather than factual knowledge. — [arXiv 2610.17607](https://arxiv.org/abs/2610.17607)
- **2610.22293 (SST)** explicitly has *no* knowledge-injection or fact-learning experiments. Its "preserving target-task performance" claims are reasoning/QA-domain accuracy, not new-fact recall. — [arXiv 2610.22293 HTML](https://arxiv.org/html/2610.22293v1)
- arXiv abstract searches for "knowledge injection fine-tuning" and "new facts fine-tuning" sorted by date returned no window paper reporting anchoring/regularization blocking fact acquisition, or validation-loss vs exact-match disagreement. — [arXiv search: knowledge injection](https://arxiv.org/search/?query=knowledge+injection+fine-tuning&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50); [arXiv search: new facts](https://arxiv.org/search/?query=new+facts+fine-tuning+language+models+memorization&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **2609.17346, "Where Should a Document Live?"** (listing-only, 15 Sep 2026, just pre-window). KV-based "cartridges" beat parametric methods by 10 pts in the oracle setting but caused a 6% control-benchmark drop (13% in coding). Non-parametric injection also has a retention cost. — [arXiv search listing](https://arxiv.org/search/?query=forgetting+fine-tuning&searchtype=abstract&abstracts=show&order=-announced_date_first&size=100)

**Pre-window context (August 2026 unless noted)**
- **2608.25677, "Learning New Facts with QLoRA: An Acquisition-Retention Frontier"** (Aug 2026). Compares full FT vs QLoRA at several ranks on an OpenStreetMap-derived anonymized geographic-fact benchmark. Lower-rank adapters keep OOD performance but *learn fewer facts*; higher ranks generalize better to rephrasings at a cost to unrelated benchmarks. **Verdict: VALIDATES** PRISM's dial shape (learns-less/forgets-less frontier) in the fact-injection setting. — [arXiv 2608.25677](https://arxiv.org/html/2608.25677v1)
- **2608.20794, "Knowing but Not Saying: Preventing Factual Access Failures in LLM SFT via Recall-Anchored Distillation"** (Aug 2026). After domain SFT, models can still recognize/rank the correct answer under constrained evaluation but fail to produce it in closed-book generation. **Verdict: VALIDATES** PRISM's evaluation-pitfall point: likelihood-style metrics overstate usable knowledge. — [arXiv 2608.20794](https://arxiv.org/abs/2608.20794)
- **2608.25243, "From Memorization to Absorption: Mixed-Policy RL for Continual Knowledge Injection"** (Aug 2026). Supervised-injected facts are bound to their training question forms and can't be retrieved under different phrasings or question types. — [arXiv 2608.25243](https://arxiv.org/pdf/2608.25243)
- **2605.16865, "MixSD: Mixed Contextual Self-Distillation for Knowledge Injection"** (May 2026). Retains up to 100% of base held-out capability vs as little as 1% for standard SFT. — [arXiv search listing](https://arxiv.org/search/?query=new+facts+fine-tuning+language+models+memorization&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **2609.08067, "Popular Knowledge Propagates More Errors in LLM Knowledge Updating" (PopAnchor)** (7 Sep 2026, pre-window). Facts tied to highly connected entities are more likely to be corrupted by neighboring finetuning updates; rehearsal on popular facts reduces forgetting. — [arXiv search listing](https://arxiv.org/search/?query=new+facts+fine-tuning+language+models+memorization&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)

### Inferences
- **Injection-dial bet status: OPEN.** Nobody in the window reports the specific PRISM phenomenon: a strong anchor gives the best validation loss and best retention while closed-book recall collapses to 2–3%, and a weak band injects at parity with 3× less forgetting. The August QLoRA frontier and "knowing but not saying" papers make the claim *credible and timely* without scooping it. PRISM should cite them as converging evidence.
- The self-distillation family (MixSD, iSDFT, SynthSelf; see Q4) is becoming the default "inject without forgetting" recipe. The natural head-to-head for PRISM's weak-band claim is **PRISM-lerp vs self-distillation at equal closed-book recall**, which nobody has run. A combination may also be composable.
- SST and similar subspace methods are untested on fact injection. Since new facts may need writes into principal directions (or, per 2610.09835, into output-embedding rows), there is a plausible prediction: hard-projection methods like SST will *block* injection like PRISM's strong band does. Testing SST on PRISM's injection task would be a high-value, differentiating experiment. (Inference; untested.)

### Gaps
- No window source found reporting validation loss vs exact-match disagreement for fact injection.
- I did not find window knowledge-injection work at scale (≥7B) with closed-book exact-match and paraphrase splits. arXiv search coverage of mid-October may also be incomplete.

---

## Q4. New continual-learning methods for LLMs (model merging, subspace-constrained finetuning, orthogonal / singular-vector-preserving finetuning, replay), especially any that preserve singular directions

### Takeaway
The window is crowded with **subspace/singular-vector methods**: SST (bottom-subspace projection), SVC (singular-vector channel selection), SOAR (Stiefel-constrained LoRA), ChainLoRA (SVD+Procrustes merging), NB-LoRA (null-basis LoRA), EoupCT (orthogonalize against estimated pretraining gradients), and a tropical-geometry per-unit projector. One systematic PEFT study (2610.09122) finds LoRA forgets least and **PiSSA, which trains the top singular components, forgets most**. Restoring *dominant* singular components gives the largest drop in general-text NLL, while restoring only singular *values* preserves the task. That is a near-exact external replication of PRISM's "directions matter, spectrum doesn't" attribution. All of these are heavier than PRISM's lerp (SVD, projection, extra modules, or data). None tests RL with the anchor replacing KL, and none tests closed-book fact injection.

### Cited Findings
- **2610.09122, "Are Parameter-Efficient Fine-tuning Methods Really Different?"** (Li, Lu, Liu; abs-verified, 6 Oct 2026). — [arXiv 2610.09122](https://arxiv.org/abs/2610.09122); [HTML](https://arxiv.org/html/2610.09122v1)
  - **Setup:** LoRA, DoRA, PiSSA, MiLoRA, OFT and HRA on Qwen2.5-7B and Llama-3.1-8B (MetaMath, CodeFeedback; 20k examples) plus FLUX.2-klein-4B.
  - **Coding forgetting (general-text ΔNLL ×10⁻³):**

    | Model | LoRA | PiSSA |
    |---|---|---|
    | Qwen2.5-7B | 1.60 | 4.25 |
    | Llama-3.1-8B | 10.73 | 43.00 |

    Task scores are comparable; PiSSA forgets most in every complete LLM comparison.
  - **Spectral interventions:**
    - restoring pretrained singular values while keeping adapted bases largely preserves task performance
    - restoring *dominant* components gives the largest mean reduction in general-text NLL for all six methods
    - Llama: dominant restoration costs LoRA/DoRA ~27 pp accuracy, and PiSSA nearly all

  **Verdict: VALIDATES** PRISM's attribution: dominant directions carry retention, and the spectrum alone isn't the lever. It also corroborates that touching principal directions (PiSSA) maximizes forgetting.
- **2610.22293 (SST)**: see Q2. Bottom-subspace projection; Qwen3-8B Science −25.5→−1.0 pts; 1.5–5% overhead. **VALIDATES + THREATENS.** — [arXiv 2610.22293](https://arxiv.org/abs/2610.22293)
- **2610.11076, "Stability-Plasticity Balance via Singular-Vector Selection in LLM Continual Learning (SVC)"** (Wang, Zhang, Pang, Zheng, Zheng; abs-verified, 8 Oct 2026). Treats each singular-vector channel as a unit and scores it before finetuning: domain data estimates adaptation benefit, and a fixed general corpus proxies forgetting cost. Selects trainable channels via knee screening, Pareto filtering and Otsu thresholding; the rest stay fixed to preserve pretrained capability. Four LLM families, eight tasks; better retention than "existing PEFT baselines". No numbers on the abs page. **Verdict: VALIDATES** the singular-direction unit of analysis. **COMPOSABLE**, since SVC's channel scores could set per-direction anchor strength in PRISM. — [arXiv 2610.11076](https://arxiv.org/abs/2610.11076)
- **2610.04670, "A Tropical Geometry View of Forgetting: A Per-Unit Projector for Knowledge-Preserving Fine-Tuning"** (Zhang, Chen, Ren, Zhang; abs-verified, 3 Oct 2026). Gate-aware per-unit projector. OPT-1.3b: forgets less than Adam-NSCL at all six matched budgets, gap 1.1×→4.3×; with 1/5.5 the directions it halves Adam-NSCL's forgetting at GPM's threshold. OPT-6.7b: matches Adam-NSCL forgetting while learning more. Activation-subspace, not pretrained-weight anchoring. — [arXiv 2610.04670](https://arxiv.org/abs/2610.04670)
- **2610.16664, "Making LoRA Identifiable: Orthogonal Alignment for Continual Learning (SOAR)"** (listing-only, 12 Oct 2026). Constrains LoRA's A on the Stiefel manifold and aligns B via Procrustes; no per-task memory growth. — [arXiv 2610.16664](https://arxiv.org/abs/2610.16664)
- **2610.00431, "ChainLoRA: Geometry-Preserving Task Vector Merging for Continual Learning in LLMs"** (listing-only, 30 Sep 2026). Replay-free LoRA merging with adaptive SVD and Procrustes alignment; SOTA among replay-free methods on the Large and SuperNI benchmarks. — [arXiv 2610.00431](https://arxiv.org/abs/2610.00431)
- **2609.25618, "Reasoning-Preserving Fine-Tuning of Post-RL LLMs with Null-Basis LoRA (NB-LoRA)"** (listing-only, 21 Sep 2026). Constrains LoRA to an approximate null basis of reasoning activations; matches LoRA adaptation while keeping post-RL reasoning near pre-finetune accuracy. Targets the SFT-after-RL use case, PRISM's "backward" niche. — [arXiv 2609.25618](https://arxiv.org/abs/2609.25618)
- **2609.30935, "EoupCT: Estimating and Orthogonalizing Unknown Pre-training Gradients"** (listing-only, 25 Sep 2026). Estimates pretraining gradients via a learnable soft prompt and orthogonalizes new-task updates to them. — [arXiv 2609.30935](https://arxiv.org/abs/2609.30935)
- **2610.17713, "Confine to the Task, Anchor the Rest: Concept-Guided Retention Regularizer for PEFT (CGR)"** (Kowsar, Medri, Meng, Liu; abs-verified, 13 Oct 2026). A frozen SAE splits concepts. The weight channel softly confines updates to the task-concept subspace; the activation channel anchors general concepts to the frozen model. Improves retention at comparable accuracy (CLIP panel; world knowledge under Llama-3.1-8B instruction tuning). No numbers on the abs page. — [arXiv 2610.17713](https://arxiv.org/abs/2610.17713)
- **2609.32771, "Continual Learning via Self-Probe Gradients (CPLUS)"** (listing-only, 26 Sep 2026). Self-generated probes scale down conflicting updates; less forgetting than baselines across five LMs and four benchmarks, especially with scarce past data. — [arXiv 2609.32771](https://arxiv.org/abs/2609.32771)
- **2609.32818, "When Can First-Order Models of Fine-Tuning Bound Forgetting?"** (listing-only, 26 Sep 2026). LoRA FT of 0.6B–14B. The complete Freedman bound held in all 112 conditions; the simplified bound was violated in 14. — [arXiv 2609.32818](https://arxiv.org/abs/2609.32818)
- **2609.28395, "Fine-Tuning LLMs for Translation: General Forgetting Mitigation…"** (listing-only, 23 Sep 2026). Llama 3.1 8B: EWC lost 1.7 pts on general benchmarks vs 11.0 for standard FT, but only data mixing preserved MT-specific instruction controls. An up-to-date EWC (weight-space importance-weighted anchor) data point at 8B: classical weight-space anchors still work. **VALIDATES.** — [arXiv 2609.28395](https://arxiv.org/abs/2609.28395)
- **Self-distillation / on-policy-target family (window):**
  - **iSDFT 2609.24646** (Khamis…Bou-Ammar / Huawei Noah's Ark; abs-verified v1 21 Sep, v2 28 Sep 2026). Budgeted teacher targets plus "anchoring to the frozen base policy" (mechanism not specified on the abs page). 73% of evaluations within 0.5 pts of base vs 52% for the strongest baseline; beats SDFT in 7/8 settings, four backbones. — [arXiv 2609.24646](https://arxiv.org/abs/2609.24646)
  - **SynthSelf 2610.13113** (listing-only, 9 Oct). Beats six methods across 14 retention benchmarks for small models. — [arXiv 2610.13113](https://arxiv.org/abs/2610.13113)
  - **Projected Distillation 2610.27467** (Deshmukh et al.; abs-verified, 20 Oct 2026). Minimal edits of the student's failed attempts as targets; Qwen3-8B and OLMo-2-7B, five domains. Best specialization–retention frontier area in 8/9 cells vs GRPO/OPD; retention falls 0.916→0.744 when minimal corrections are swapped for gold solutions. — [arXiv 2610.27467](https://arxiv.org/abs/2610.27467)
  - **SAPD 2610.27847** (listing-only, 20 Oct). Offline on-policy distillation; less forgetting than SFT, ≥4.7× faster per step than online OPD. — [arXiv 2610.27847](https://arxiv.org/abs/2610.27847)
  - **NADS 2610.25751** (listing-only, 19 Oct). Navigator-guided synthetic data selection plus distillation of pretrained behavior. — [arXiv 2610.25751](https://arxiv.org/abs/2610.25751)
- **2610.11132, "SFT-as-Context Mitigates Forgetting in Supervised Fine-Tuning"** (Tang, Hua, Qian, Tiwari, Qin; abs-verified, 8 Oct 2026). Training-free: the parent model reads the SFT model's response as context. Within 2.2 pts of the parent on general capabilities across 19 parent–SFT pairs and 11 benchmarks, and within ~2 pts of the SFT model on tuned tasks (AIME24 2.2, LiveCodeBench 2.1). Inference cost is two models, not quantified. **Verdict: alternative (inference-time), not composable in weights.** — [arXiv 2610.11132](https://arxiv.org/abs/2610.11132)
- **2610.23918, "Cross-Modal Forgetting and Transfer in Unified Multimodal Post-Training"** (listing-only; listing date 5 Oct 2026 is **inconsistent with the ID**, so date uncertain). Janus-Pro-7B LoRA: a single run shows no forgetting, but chaining three runs destroys earlier capability and raises general-text perplexity by 1–2 orders of magnitude. Shows that *sequential/multi-stage* finetuning is where forgetting bites. — [arXiv 2610.23918](https://arxiv.org/abs/2610.23918)
- Pre-window context: **2510.17776, "Mapping Post-Training Forgetting in Language Models at Scale"** (Oct 2025) found two-checkpoint linear interpolation/merging does not reliably mitigate forgetting (Qwen2.5 Base/Coder). — [arXiv 2510.17776](https://arxiv.org/abs/2510.17776)
- Pre-window context: **2605.10973, "Rotation-Preserving Supervised Fine-Tuning"** (May 2026) penalizes changes to each matrix's pretrained top-k singular-vector block; reports better ID/OOD tradeoffs than SFT on math and a stronger starting point for RL. This is a *soft directional penalty*, the closest pre-window method to PRISM's attribution-guided anchor. — [arXiv 2605.10973](https://arxiv.org/html/2605.10973v1)
- Pre-window context: **2609.06986, "Continual Learning Mechanisms Compose for Long-Horizon Memorization"** (listing-only, 6 Sep 2026). Across 100 sequential tasks, combining data, function and *weight* anchors with merged LoRA raised final retention from 1.2% to 34.9% (28×). — [arXiv search listing](https://arxiv.org/search/?query=forgetting+fine-tuning&searchtype=abstract&abstracts=show&order=-announced_date_first&size=100)

### Inferences
- PRISM's attribution claim is now corroborated from three independent angles in or near the window: SST (projection), 2610.09122 (dominant-component restoration; PiSSA worst), and Rotation-Preserving SFT (May 2026, soft top-k penalty). **Directional preservation is no longer a novel headline.** PRISM's distinct contributions are:
  1. an SVD-free, per-step lerp that costs O(params) with no projection, refresh or SVD (SST's refresh costs 5–38% unsharded)
  2. a single scalar dial that maps the retention↔injection frontier
  3. the injection-blocking finding
  4. the KL-replacement-in-RL claim
- **Composable directions:**
  - use SVC-style per-channel scores or SST's top-k basis to make the PRISM lerp *anisotropic* (strong on top singular directions, weak elsewhere); this could keep retention while unblocking injection
  - combine PRISM with self-distillation (iSDFT/SDFT) as data-side plus weight-side anchors, consistent with 2609.06986's "anchors compose" result
- Head-to-head priority for PRISM at its next scale: **PRISM-lerp vs SST vs Rotation-Preserving SFT vs EWC vs LoRA**, on (i) general retention, (ii) closed-book fact injection with paraphrase splits, (iii) cost. Nobody has published (ii) for any of these.

### Gaps
- SVC, SOAR, ChainLoRA, NB-LoRA, EoupCT and CGR: numbers not extracted (listing/abs only).
- I did not find a window MiLoRA/PiSSA-for-retention method beyond 2610.09122's comparison. MiLoRA's retention numbers vs LoRA weren't in the extracted text.
- No window paper combines a weight-space anchor with self-distillation and measures fact injection.

---

## Q5. What do practitioners (prime-rl, verl, OpenRLHF, TRL, lab blogs) use for drift control, and what does the reference-model forward pass cost them?

### Takeaway
Current defaults across major open post-training stacks **turn the reference-KL off**:
- **TRL GRPO:** β=0.0, and the reference model isn't loaded.
- **prime-rl:** no reference-model KL for RL; the optional `kl_tau` anchors to the *behavior* policy and is off by default.
- **verl:** `use_kl_loss=False` by default (GRPO examples turn it on at 0.001, `low_var_kl`).
- **Tinker:** `kl_penalty_coef=0.0` by default, but recommends 0.05 and names catastrophic forgetting as the risk.

The stated reasons are that KL is "not essential" for GRPO and that dropping it saves memory and a forward pass per batch. Drift control in practice is mostly implicit (on-policy sampling, clipping), plus occasional reference resets (ProRL). This leaves a **zero-cost anchor gap**. I found little practitioner evidence of acute forgetting pain from on-policy RL itself; the pain shows up in SFT stages, multi-stage chains, and downstream fine-tunability.

### Cited Findings
- **TRL GRPOConfig docstring** (current docs): "beta … defaults to `0.0`. KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving training speed." It notes DeepSeek-R1 used 0.001. The rationale cites Open-Reasoner-Zero, "Understanding R1-Zero-Like Training" (Dr. GRPO) and DAPO showing KL "is not essential". Reference-sync options exist (`sync_ref_model=False`, `ref_model_mixup_alpha=0.6`, `ref_model_sync_steps=512`), i.e., a built-in weight-space EMA/mixup of the *reference*, off by default. — [TRL GRPO Trainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- **Prime Intellect RL glossary** (undated page): a reference-model KL penalty "requires an extra forward pass through the reference model on every batch". prime-rl "has no reference-model KL penalty for RL". Its optional `kl_tau` term in the default loss anchors to the behavior policy and is off by default. A reference model appears only in distillation modes (`opd`, `opsd`). The page notes Dr. GRPO, DAPO and CISPO omit the reference penalty, and ProRL periodically resets the reference to a recent snapshot. It says on-policy RL stays close to its start in KL even at β=0, so "the explicit anchor is partly redundant". — [Prime Intellect: KL penalties and reference models](https://primeintellect.ai/rl-glossary/kl-penalty)
- **prime-rl README:** describes "fully asynchronous RL" and the AIPO loss; no KL/reference or drift-control features mentioned; Releases section empty as accessed. — [prime-rl GitHub](https://github.com/PrimeIntellect-ai/prime-rl)
- **verl config docs:**
  - `actor.use_kl_loss` default False (comment: True is used for GRPO)
  - `kl_loss_coef` 0.001; `kl_loss_type` `low_var_kl` (k3)
  - `algorithm.use_kl_in_reward` False; the reference worker is enabled only if either is True
  - the docs recommend ref param offload for models >7B (memory pressure); the page is internally inconsistent on `kl_ctrl.kl_coef` (YAML 0.005 vs prose 0.001)

  — [verl config docs](https://verl.readthedocs.io/en/latest/examples/config.html)
- **Tinker (Thinking Machines) RL hyperparameters doc:** `kl_penalty_coef` default 0.0; 0.01 light, 0.05 "good starting point", 0.1+ "may slow reward improvement". The reference is the base model via `kl_reference_config`. Frames the KL as guarding against reward hacking, *catastrophic forgetting* and mode collapse. — [Tinker docs: RL hyperparams](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)
- **Soup framework PR #1249** (third-party GitHub): a KL-free run "as in the DAPO and Dr. GRPO recipes" couldn't be configured yet, and "every GRPO run pays for the reference forward". Anecdotal evidence that the reference pass is seen as a cost. — [GitHub PR](https://github.com/MakazhanAlpamys/Soup/pull/1249)
- **Practitioner pain evidence (mixed).** Recent GRPO papers that check general benchmarks mostly report flat MMLU/IFEval after narrow-domain GRPO (MDP-GRPO 2606.06058; SMEPO 2605.25198; Typhoon-S 2601.18129), while SFT lowers MMLU more visibly. — [arXiv 2606.06058](https://arxiv.org/pdf/2606.06058); [arXiv 2605.25198](https://arxiv.org/pdf/2605.25198); [arXiv 2601.18129](https://arxiv.org/pdf/2601.18129). Contrast: zero-RL (KL-free) models are fragile under later finetuning (PRIME 53.65→20.77). — [arXiv 2609.33898 HTML](https://arxiv.org/html/2609.33898v1). And SFT stages lose 25–38 pts of general capability at 1.7B–14B. — [arXiv 2610.22293 HTML](https://arxiv.org/html/2610.22293v1)

#### Master inventory of window items (2026-09-21 → 2026-10-21)

| Date | arXiv | Title (short) | Core claim / numbers | Scale | Verdict vs PRISM |
|---|---|---|---|---|---|
| 21 Sep (v2 28 Sep) | [2609.24646](https://arxiv.org/abs/2609.24646) | iSDFT | Budgeted self-distillation + base-policy anchor; 73% vs 52% evals within 0.5 pt of base | 4 backbones | COMPOSABLE |
| 21 Sep* | [2609.25618](https://arxiv.org/abs/2609.25618) | NB-LoRA | LoRA in null basis of reasoning activations preserves post-RL reasoning | n/r | COMPOSABLE / competitor for "backward" use |
| 23 Sep* | [2609.28395](https://arxiv.org/abs/2609.28395) | MT forgetting mitigation | EWC −1.7 vs FT −11.0 general pts | Llama 3.1 8B | VALIDATES (weight anchors work at 8B) |
| 25 Sep* | [2609.30935](https://arxiv.org/abs/2609.30935) | EoupCT | Orthogonalize to estimated pretrain gradients | multiple LLMs | COMPOSABLE |
| 26 Sep* | [2609.32771](https://arxiv.org/abs/2609.32771) | CPLUS | Self-probe gradients scale down conflicting updates | 5 LMs | COMPOSABLE |
| 26 Sep* | [2609.32818](https://arxiv.org/abs/2609.32818) | First-order forgetting bounds | Freedman bound held in 112/112 | 0.6–14B LoRA | context |
| 27 Sep | [2609.33898](https://arxiv.org/abs/2609.33898) | No Free Efficiency | Zero-RL (KL-free) models fragile: PRIME 53.65→20.77 after IMDB LoRA | 7B (32B appx) | GAP-PRISM-FILLS / VALIDATES pain |
| 28 Sep* | [2609.35259](https://arxiv.org/abs/2609.35259) | On/off-policy distillation dynamics | LR governs forgetting and update sparsity | Llama3, Qwen2.5 | VALIDATES (LR as lever) |
| 30 Sep* | [2610.00431](https://arxiv.org/abs/2610.00431) | ChainLoRA | SVD+Procrustes LoRA merging, replay-free SOTA | n/r | COMPOSABLE |
| 1 Oct* | [2610.02437](https://arxiv.org/abs/2610.02437) | Style vs semantics, SFT vs RFT | SFT style drift → lower-bounded forgetting; RFT none (sim) | small | context (RL<SFT) |
| 2 Oct | [2610.03940](https://arxiv.org/abs/2610.03940) | Optimiser history & answer mass | Old Adam momentum drives leakage; momentum reset helps | 3 LM families | COMPOSABLE (lerp Adam state) |
| 2 Oct* | [2610.04158](https://arxiv.org/abs/2610.04158) | How RL reshapes reasoning | RL causes some forgetting via strategy shifts | n/r | VALIDATES need in RL (weak) |
| 3 Oct | [2610.04670](https://arxiv.org/abs/2610.04670) | Tropical per-unit projector | 1.1–4.3× less forgetting vs Adam-NSCL | OPT-1.3b/6.7b | COMPOSABLE |
| 4 Oct | [2610.05200](https://arxiv.org/abs/2610.05200) | Theory: OPSD vs SFT | OPSD sparse updates avoid forgetting; SFT dense updates overwrite | theory | VALIDATES framing |
| 4 Oct | [2610.05402](https://arxiv.org/abs/2610.05402) | Task Vector Descent | Lerp θ+λΔ (λ≈0.4–0.6) + optimizer-state lerp; GRPO stability gain "little" | 124M–4B; GRPO on Qwen2.5-3B | Closest relative; partial THREAT to RL bet; COMPOSABLE |
| 6 Oct | [2610.09122](https://arxiv.org/abs/2610.09122) | PEFT really different? | LoRA forgets least, PiSSA most (Llama coding ΔNLL 10.73 vs 43.00); dominant-component restore cuts NLL most | Qwen2.5-7B, Llama-3.1-8B | VALIDATES attribution |
| 7 Oct | [2610.09835](https://arxiv.org/abs/2610.09835) | Deafening Silence | Forgetting in absent-token output embeddings; Adam-ε fix removes 39.4–67.9% | 160M–12B | COMPOSABLE / partial THREAT |
| 8 Oct | [2610.11076](https://arxiv.org/abs/2610.11076) | SVC | Singular-vector channel selection for stability/plasticity | 4 families, 8 tasks | VALIDATES; COMPOSABLE |
| 8 Oct | [2610.11132](https://arxiv.org/abs/2610.11132) | SFT-as-Context | Training-free; within 2.2 pts of parent, 19 pairs | various | alternative |
| 8 Oct | [2610.12161](https://arxiv.org/abs/2610.12161) | KL misfires in GPO | 7 KL failure modes in GRPO; ZCPO | n/r | VALIDATES KL pain; GAP |
| 9 Oct* | [2610.13113](https://arxiv.org/abs/2610.13113) | SynthSelf | Synthetic self-distillation beats 6 methods on 14 retention benchmarks | small models | COMPOSABLE |
| 12 Oct* | [2610.16664](https://arxiv.org/abs/2610.16664) | SOAR | Stiefel-constrained LoRA + Procrustes | n/r | COMPOSABLE |
| 13 Oct* | [2610.17726](https://arxiv.org/abs/2610.17726) | TSRL | SFT only on failures; +9.6 pts retention vs SFT→RL | agents | context |
| 13 Oct | [2610.17713](https://arxiv.org/abs/2610.17713) | CGR (SAE anchor) | SAE-concept weight confinement + activation anchor | Llama-3.1-8B, CLIP | COMPOSABLE |
| 13 Oct* | [2610.18738](https://arxiv.org/abs/2610.18738) | Proximal KL off-policy RLVR | Proximal reference; +2.58% vs GRPO | n/r | GAP (KL-space only) |
| 14 Oct* | [2610.20814](https://arxiv.org/abs/2610.20814) | Gibbs-Gap PO | Fixed reference 32.96 vs refreshed 46.58 | n/r | THREAT to fixed-W₀ anchor in off-policy RL |
| 15 Oct | [2610.22293](https://arxiv.org/abs/2610.22293) | **SST** | Bottom-subspace projection; Qwen3-8B −25.5→−1.0; KL dial fails; 1.5–5% overhead | 1.7B–32B | **VALIDATES + THREATENS (closest competitor)** |
| 16 Oct* | [2610.23085](https://arxiv.org/abs/2610.23085) | Beacon (GRPO safety) | Benign-only KL anchoring; −32.83 pp safety error | LRMs | COMPOSABLE |
| 16 Oct | [2610.24236](https://arxiv.org/abs/2610.24236) | TAR adaptive replay | Base-solution replay in RLVR reduces forgetting | 2 Qwen2.5 scales | COMPOSABLE |
| 16 Oct* | [2610.24137](https://arxiv.org/abs/2610.24137) | ESRL | Anchor-policy ES, refreshed anchor; +31.7% w/ less forgetting | n/r | context |
| 19 Oct (abs) | [2610.25651](https://arxiv.org/abs/2610.25651) | OPDArena | RLVR-teacher students retain; OOD shift predicts forgetting | 5 models | VALIDATES (RL<SFT) |
| 19 Oct* | [2610.26416](https://arxiv.org/abs/2610.26416) | Single-adapter drift | Rollout recycling → ~34× KL drift, no generalization | code RL | VALIDATES pain in off-policy RL |
| 20 Oct | [2610.27467](https://arxiv.org/abs/2610.27467) | Projected Distillation | Best frontier 8/9 vs GRPO/OPD; retention 0.916 vs 0.744 | Qwen3-8B, OLMo-2-7B | COMPOSABLE |
| 20 Oct* | [2610.27847](https://arxiv.org/abs/2610.27847) | SAPD | Offline OPD; less forgetting than SFT; ≥4.7× faster | agents | COMPOSABLE |
| ? (listed 5 Oct†) | [2610.23918](https://arxiv.org/abs/2610.23918) | Cross-modal forgetting | Chained LoRA runs → PPL ↑ 10–100× | Janus-Pro-7B | VALIDATES multi-stage pain |
| ? (listed 2 Oct†) | [2610.22381](https://arxiv.org/abs/2610.22381) | Relearning savings | ~18% fewer updates to relearn facts | Qwen2.5-1.5B | context (injection) |

\* listing-only date (arXiv search listing; abs page not opened). † listing date inconsistent with arXiv ID order; treat as uncertain. n/r = not reported on the page I read.

### Inferences
- The field has **removed** the explicit anchor rather than replaced it. So PRISM's RL pitch is best framed as *restoring drift control at zero forward-pass cost*, not *replacing an expensive KL people currently pay for*. The framing matters because most open GRPO recipes no longer pay that cost.
- Rough cost of what PRISM would remove where KL is still used (verl GRPO examples, Tinker with KL, DeepSeek-R1-style β=0.001):
  - one extra reference forward per batch: roughly ⅓ of a policy fwd+bwd in FLOPs (an estimate from the standard 1:2 forward:backward FLOP ratio, not a measured figure from these sources)
  - a full second copy of weights, which verl suggests offloading above 7B

  PRISM's lerp needs a stored W₀ copy too, but no forward pass. Memory parity is a wash unless W₀ is kept in lower precision or offloaded. (Inference.)
- Built-in precedents for *weight-space reference handling* already ship in TRL (`sync_ref_model` with `ref_model_mixup_alpha`), and ProRL uses reference resets. An "EMA/mixup in weight space" idea is familiar to practitioners, which lowers adoption friction. It also means reviewers will ask how PRISM differs from "just set β=0 and lerp the policy toward the ref".
- Where pain is felt most, based on the evidence gathered:
  1. SFT/domain finetuning stages, which lose 25–38 pts at 1.7B–14B (SST)
  2. multi-stage/chained pipelines (2610.23918; NB-LoRA's post-RL SFT use case)
  3. downstream fine-tunability of KL-free RL models (2609.33898)
  4. off-policy/recycled/async RL drift (2610.26416; 2610.20814)

  Narrow-domain on-policy GRPO shows the least pain.

### Gaps
- No primary-source measurement found of the reference-forward-pass share of GRPO step time in any framework, and no OpenRLHF defaults checked (not fetched).
- No window-dated lab blog posts (Prime Intellect, Ai2 Tülu/OLMo, Nous, Thinking Machines) on drift control/forgetting surfaced in search. A Thinking Machines search found nothing dated October 2026.
- prime-rl's `docs/algorithms.md` (loss details, `kl_tau`) was not read directly; the claim rests on Prime Intellect's own glossary page.
- Practitioner threads (X/Reddit/Discord) on RL forgetting in the window were not found by search; pain-point ranking rests on papers, not practitioner testimony.
