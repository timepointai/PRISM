# Alpha moves: development brief

> **For the human handing this off:** set the values in §0, then give the agent
> this whole file as its task, or tell it: *"Read docs/ALPHA-MOVES-BRIEF.md and
> execute it end to end."* Everything below is addressed to the agent.

You are a development agent working in `timepointai/PRISM`. Your job is to
implement the six ranked moves in
[`reports/PRISM alpha in recent research.md`](../reports/PRISM%20alpha%20in%20recent%20research.md)
(section "Six moves, ranked by alpha per dollar and urgency") in a single
autonomous session. Run every experiment the budget below authorizes, with this
repo's full rigor. Leave every larger experiment fully prepared, so that it can be
launched with one command. A clean negative result is a deliverable. A
flattering number without an artifact is a defect.

Written 2026-10-21. Facts below about external tools (TRL, prime-rl, Tinker,
Together, Pythia's data and checkpoints, `torch.optim.Muon`) are as understood
on that date. Verify each against current docs before you build on it.

---

## 0. Session parameters (the human edits these)

| Parameter | Default | Meaning |
|---|---|---|
| `TIER_A_BUDGET_USD` | **300** | Total cloud spend you may incur without asking (§6, Tier A). |
| `PER_RUN_CAP_USD` | **40** | Any single launch estimated above this is Tier B, whatever its purpose. |
| `AUTHORIZE_TIER_B` | **no** | `yes` lets you launch the Tier B gates in §6, in gate order, within `TIER_B_BUDGET_USD`. |
| `TIER_B_BUDGET_USD` | **0** | Ceiling for Tier B when authorized. |
| Tier C | **prepare only** | Never launched in this session. |
| `MERGE_POLICY` | **pr** | `pr`: open one PR `alpha-moves → main` and stop. `merge`: merge into `main` yourself with `merge:` commits, as the repo history does. |
| Compute | **Modal** | The CLI is installed and authenticated on the user's machine. If you find it isn't, see §3.4. |

---

## 1. Definition of done

- [ ] The golden-run check (§3.5) passes: every existing default produces identical eval lines.
- [ ] `cd src && python prism_selftest.py` passes, with new checks for every new math path.
- [ ] W0–W6 (§5) are implemented. Every new knob defaults to off, folds into run keys or tags, and is recorded in artifacts.
- [ ] Every Tier A run has executed. Anything called a result has 3 seeds. Artifacts are fetched, committed, and written up in `RESULTS.md` (new sections from §18) and its run ledger.
- [ ] Every Tier B and Tier C run is prepared: runner script, config, a passing smoke or dry run, a cost estimate, a pre-registration committed before any launch, and one launch command recorded in the status doc.
- [ ] The README mechanism map and prior art are updated, along with "Honest bounds" and "What's next". No number appears in prose without an artifact, unless it is an external number labeled as reported.
- [ ] `docs/ALPHA-MOVES-STATUS.md` exists and is current. It covers per-move status, verdicts, the spend ledger (estimated and actual), the decisions log, and open questions for the user.
- [ ] Branches are pushed to `timepointai/PRISM`, and a PR is opened (or the merge is done, per `MERGE_POLICY`).

---

## 2. Read first, in this order

1. `CLAUDE.md`: commit and safety rules (restated in §3.1).
2. `README.md`: "Start here (humans and agents)", "Honest bounds on the claims", "Where PRISM sits: the mechanism map", "What's next".
3. `docs/NEXT-EXPERIMENTS.md`, the whole file. These parts bind you:
   - the core-bet decision rule;
   - the three-outcome ladders, with their gates, costs and kill criteria;
   - the two-act framing spec and its banned phrases.

   You may add pre-registrations. You may never weaken one.
4. `reports/PRISM alpha in recent research.md`, which is the source of the moves. For each external paper's numbers, quotes and links, use `research_notes/PRISM alpha in recent research/*.md`.
5. `RESULTS.md` §7 and §10–§17, plus `results/README.md`.
6. `docs/FINETUNE-RETENTION.md`, `docs/PRIOR-FUSED-PRISM.md`, `docs/UNIFIED-ARC.md`.
7. Code:
   - `src/train.py`, all of it;
   - in `src/prism_eval.py`: `main`, `method_args_for`, `compute_score`, `run_key`;
   - `src/prism_finetune_eval.py`;
   - `src/prism_init.py`;
   - `src/prism_prior_eval.py`;
   - `src/ngram_floor.py`;
   - `src/fact_recall.py`;
   - `src/prism_selftest.py`;
   - `prism_modal_facts.py`, the runner template.

---

## 3. Rules that override everything else

### 3.1 Repo, commits, safety
- Push only to `timepointai/PRISM`. Never push to `karpathy/nanoGPT` or open a PR against it. Always run `gh pr create --repo timepointai/PRISM --base main`.
- **No `Co-Authored-By` trailers, no "Generated with …" lines, no AI attribution or tool branding.** That applies to commits, PR bodies, code, comments and docs, even if your tooling adds them by default. Credit Timepoint Labs / Timepoint AI only.
- Commit messages follow the repo's style: `feat:`, `results:`, `docs:`, `merge:` or `chore:`, then a short summary after an em dash. Make logical commits. Artifacts go in their own `results:` commits.
- Never rewrite published history, never force-push `main`, and never edit or delete committed `results/*.json` or any data file an existing artifact trained on. The repo is append-only.
- Leave `LICENSE` and `NOTICE` alone, except to add attributions required by code you vendor. Vendor only MIT, BSD or Apache code, never GPL or AGPL.
- Commit no secrets. Prefer ungated models and datasets (Pythia, Qwen, OLMo).

### 3.2 Evidence
- Evidence means a committed `results/*.json`. A **result** has 3 seeds (1337, 1338, 1339), matched schedules, and a committed artifact. A single seed is a **probe**, and is labeled as one everywhere it appears.
- **Tuning parity:** tune any arm, ours or a competitor's, on **seed 1336 only**. Give every method the same budget (at most 3 settings unless a workstream says otherwise). Choose by a criterion you write down before looking. Report numbers from seeds 1337–1339.
- Partial or crashed runs raise; they are never scored. Every new harness keeps this property.
- New knobs default to off and fold into the run key. In `prism_finetune_eval.py`, `run_key` does **not** hash arms. Whenever an arm's meaning changes, use a new `--tag` and a fresh arm name, or banked arms will false-resume.
- Keep warmup much smaller than the step count in probes. Change one variable at a time against the plain configuration on an otherwise identical rig.

### 3.3 Honesty and framing
- **Pre-register.** Before launching any new experiment, commit a short pre-registration to the relevant doc: hypothesis, arms, primary metric, decision rule, kill condition. The launch must come after that commit.
- External numbers are written "reported by <paper>", with a link. They never sit in our tables as if measured in our rig. Cross-rig comparisons are "positioning, not head-to-head."
- A re-implementation of an external method is labeled "our re-implementation of <method> per <paper, section>; not the authors' code". List deviations in its docstring and in the artifact's `arm_specs`. If official code exists under a compatible license, prefer it and record its commit hash.
- Before trusting a re-implementation, check that it reproduces the paper's qualitative signature on our rig. For example, SST should forget less than plain SFT. If it doesn't, debug before concluding anything.
- Report negative and null results with the same prominence as wins. When a pre-registered kill condition fires, say so in `RESULTS.md`, README "Honest bounds", and the status doc. Do not reframe it.
- The banned phrases in `docs/NEXT-EXPERIMENTS.md` stay banned everywhere: "reached at init", "unbounded speedup", "already sits at the baseline's final loss".
- The report rests three threat verdicts on abstracts or listings: SPA-FRONT, Gibbs-Gap, and COUNT's late-fusion null. Read the full text before you build a comparison on any of them, and record what you verified.

### 3.4 Spend
- Before every launch, estimate its cost (GPU type × hours × current Modal price). Record the estimate and the actual cost in the status doc's spend ledger.
- Never exceed `TIER_A_BUDGET_USD` or `PER_RUN_CAP_USD` without the matching authorization in §0. Tier C is prepare-only.
- If Modal or Hugging Face isn't authenticated, don't work around it. Implement, smoke-test on CPU, and mark the runs "prepared, not launched".
- Watch detached runs. Stop a run that diverges or is misconfigured (`modal app stop`) instead of paying for it to finish.

### 3.5 Don't break what exists
- **Golden-run check.** Do this first, on unmodified `main`, before editing anything.
  1. Run these CPU smokes with fixed seeds and save their stdout under `.prism_runs/golden/`:
     - the README "Start here" tiny `prism_eval.py` smoke, for `--method=dirs_only` and for `--method=recipe`;
     - a tiny `prism_finetune_eval.py` smoke with arms `base,plain,raw_weak`;
     - a tiny `prism_prior_eval.py` smoke with arms `baseline,prior`.
  2. Run each one twice, to measure CPU run-to-run variation.
  3. After your changes, and before every merge, rerun them and diff the eval lines. They must be identical, or within the variation you measured on `main`.
- `prism_selftest.py` stays green. Add the tests listed in each workstream.

### 3.6 Autonomy
- Work end to end without asking. The exceptions are spend beyond your authorization, any destructive git operation, and credentials. For everything else, decide, then log the decision and the reason in the status doc's **Decisions** section.
- When a workstream blocks (a paper is unavailable, a dependency won't install, numbers go unstable), write down what blocked it and try one alternative. Then move on to the next workstream.
- Commit `docs/ALPHA-MOVES-STATUS.md` after every workstream. It is your memory if your context resets.

---

## 4. Codebase facts you would otherwise spend an hour finding

**Model**
- `src/model.py` is nanoGPT. `transformer.wte.weight` *is* `lm_head.weight` (tied), so `named_parameters()` yields only `transformer.wte.weight`. Excluding "the output embedding" from an anchor therefore frees the input embedding too. If you need them separated, use row-level handling, or untie them and document that the model changed.
- Attention uses a fused `c_attn`. Its weight is `(3·d, d)`, with Q, K and V in contiguous row blocks `[0:d]`, `[d:2d]`, `[2d:3d]`. Each head is a slice of `d / n_head` rows within a block. GPT-NeoX lays this out differently; see W2.
- The probe model is `config/train_shakespeare_char.py`: 6 layers, 6 heads, d=384, block 256, dropout 0.2, beta2 0.99. This is the 10.65M rig.

**Training script**
- `train.py` config values are module-level globals, overridden by `configurator.py`. New knobs must be plain `int`, `float`, `bool` or `str` globals. They are saved into `ckpt['config']` automatically.
- The anchor works like this:
  - `train.py` captures `prism_targets` as CPU tensors, either at init (scratch path) or from the resumed weights (resume path).
  - On the resume path it supports modes `raw`, `spectral` and `shuffled`, with a `prism_anchor_exclude` substring filter.
  - After every optimizer step it applies `param.data.lerp_(target, current_mod)`. Search for the comment `spectral modulation — pull weights`.
  - Targets move from CPU to GPU on every step. That is fine at 10M parameters; fix it before you scale.
- **Resume finetunes load the base's optimizer state**, so Adam moments carry over from pretraining. A "reset moments" arm means skipping `optimizer.load_state_dict` on resume.
- **Prior fusion:** `_prior_logp` gathers from a dense `(V^C, V)` table. The first C−1 positions of every window get 0, which makes them uniform (the "block-edge" problem). Dense tables are infeasible past context 3 at V=65, and impossible at V=256. `src/ngram_floor.py` contains a sparse, interpolated CPU model (orders 0–6). Start the training-time sparse prior from it.

**Harnesses**
- `prism_eval.py`: schema `prism-eval/1`. `prism_score` is the baseline's steps-to-best divided by the method's steps to reach that loss. `compute_score` already handles left-censoring and reached-at-init.
- `prism_finetune_eval.py`: schema `prism-finetune/1`. Arms are defined in the `ARM_SPECS` dict, `run_arm` builds the `train.py` flags, forgetting is the retention climb, and recall goes through `fact_recall.py`.
- Also `prism_prior_eval.py` (`prism-prior/1`) and `prism_arc_eval.py`.
- Eval lines are parsed by regex (`parse_curve`, `parse_retain`). Add new metrics as new line formats. Never change the existing ones.

**Rigs and runners**
- Reuse these probe rigs verbatim:
  - `PROBE` in `prism_modal_modernweb.py` (modern web);
  - `FLAGS` in `prism_modal_enwik8.py`;
  - `FLAGS` in `prism_modal_facts.py` (fact injection);
  - `prism_modal_catalog.py` (the scored task).
- Modal runners clone `BRANCH` from GitHub at run time, use their own Volume, and write artifacts to it.
  - **Push the branch before launching.** The runner prints the commit it ran.
  - Fetch results with `modal volume get <vol> nanogpt-prism/results/<file> ./results/`, then commit them.
  - Copy an existing runner as your template.
- Smoke ritual before any GPU: `cd src && python prism_selftest.py`, then a `--device=cpu` tiny run of your driver that writes a conforming artifact, then Modal.
- README charts (`assets/atlas-*.svg`) are static, self-contained SVGs with a light palette and a `prefers-color-scheme: dark` block. Read one before drawing a new chart, and match it.

---

## 5. Workstreams

Each workstream lists what to build, the arms, the metrics, the runs and their tier, and what counts as done. W0 comes first; everything depends on it.

### W0 — Shared infrastructure

All knobs below are new `train.py` globals unless noted otherwise, and all default to off.

1. **Optimizer choice:** `optimizer = 'adamw' | 'muon'`.
   - Apply Muon to the 2-D hidden matrices (attention and MLP). Use AdamW for embeddings, the head, and gain and bias parameters.
   - Use `torch.optim.Muon` if the pinned torch has it. Otherwise vendor the reference implementation (MIT) and add a `NOTICE` entry.
   - Record the optimizer and its hyperparameters in artifacts.
   - Tune Muon's LR on seed 1336, with at most 3 settings.
2. **Anchor extensions**
   - `prism_anchor_adam = 'keep' | 'reset' | 'lerp'`. `keep` is today's behavior. `reset` starts the finetune with fresh moments. `lerp` pulls `exp_avg` and `exp_avg_sq` toward the base's end-state moments with the same strength as the weights (cf. Task Vector Descent, arXiv 2610.05402).
   - `prism_anchor_rows = '' | 'absent' | 'present'`. For the tied embedding, anchor only the rows of tokens absent from (or present in) the finetune corpus. This targets where arXiv 2610.09835 finds forgetting concentrates.
   - `out_eps`: an Adam ε override for the embedding/head parameter group only (the arXiv 2610.09835 control).
   - `prism_anchor_aniso = 'k:s_top:s_rest'`: an anisotropic pull.
     - Decompose the drift Δ = W − W₀ against W₀'s top-k singular subspaces, computed once at resume. Pull the top component with `s_top` and the rest with `s_rest`.
     - Choose the projector and document it (for example Δ_top = P_U Δ + Δ P_V − P_U Δ P_V).
     - Selftest: `s_top == s_rest` reproduces the plain lerp exactly, and the projector is idempotent.
   - `prism_anchor_target = 'fixed' | 'ema:β' | 'refresh:N'`: fixed W₀, an EMA anchor, or a periodically refreshed anchor (W4, plus WARP-style arms).
   - `prism_anchor_device`: keep targets on the training device when memory allows.
3. **`src/prism_anchor.py`:** a framework-agnostic anchor.
   - Capture W₀, optionally sharded or CPU-offloaded.
   - Attach through `torch.optim.Optimizer.register_step_post_hook`.
   - Support fixed, EMA and refresh targets, exclude filters, and the anisotropic mode.
   - Add a **LoRA-delta mode.** For W = W₀ + (α/r)·B·A, a lerp toward W₀ with strength s equals scaling B by (1 − s) exactly. Expose that form.
   - Selftests:
     - one hook step equals `train.py`'s inline lerp to within 1e-6 on a tiny GPT;
     - the LoRA-delta equivalence is exact.
4. **`src/prism_metrics.py`**
   - (a) Steps-to-target, with `compute_score` semantics.
   - (b) A token-equivalent multiplier.
     - Fit a monotone baseline curve L(t) = a + b·(t + t₀)^(−c) to the baseline's eval points.
     - Report the fit residuals, and refuse to answer if RMSE is above a tolerance you state.
     - Invert the curve at the method's end-of-training loss to get t_equiv. The multiplier is t_equiv / t_method.
   - (c) Donor-compute accounting (teacher FLOPs ≈ 6·N·D), so every multiplier can be quoted both with and without the amortized donor cost.
   - Unit tests on synthetic curves with known answers.
5. **Online geometry tracking:** `prism_track_overlap = N`.
   - Every N steps, log the mean squared cosine of the principal angles between each weight group's current top-k singular subspaces and two references:
     - (a) the init's subspaces;
     - (b) the transplanted teacher U/V (scratch path) or W₀ (resume path).
   - Use a new log-line format, and extend the harness parsers to bank it into artifacts.
   - Selftest: identical subspaces give 1.0, orthogonal ones give 0.
6. **Recall v2**
   - Write a deterministic generator (new seed) for 2–3 additional held-out paraphrase templates per fact. Put them in `data/facts/recall_prompts_v2.tsv`. Training data and existing files stay unchanged.
   - Extend `fact_recall.py` to report accuracy per template kind.
   - Do the same for the catalog task if the format allows.

**Done when:**
- the golden-run check is still identical;
- the selftest is green with at least 12 new checks;
- every new knob has a CPU smoke that writes a conforming artifact.

### W1 — Move 1: the injection-dial head-to-head (top priority after W0)

**Question.** Does one scalar anchor trace the retention ↔ injection frontier better than the methods the field now uses? And do hard-projection methods like SST block injection the way PRISM's strong band does? SST never ran an injection experiment, so this is the most differentiating test available.

**Rigs.** Use both rigs with their schedules unchanged and a new `--tag`:
- the **facts** rig (`prism_modal_facts.py` FLAGS: modern-web base, 2,000 base steps, 1,000 finetune steps);
- the **catalog** rig (`prism_modal_catalog.py`).

**Arms.** Every arm forks the same base per seed, with one change each.

*Existing arms:* `plain`, `lowlr_b`, `raw_weak` (0.0025), `raw_mid` (0.01), `raw_exffn`, `raw_exattn`.

*Open cells:*
- the dial's shape: `raw_0035` and `raw_005`;
- the low-LR × weak-anchor combination: `lowlr_weak`.

*PRISM variants:*
- `raw_weak_exemb` (embedding excluded);
- `raw_weak_rows_absent`;
- `raw_weak_adamreset`;
- `raw_weak_adamlerp`;
- `raw_aniso`: strong on the top-k directions and weak elsewhere. Choose k and both strengths on seed 1336.

*Competitors.* These are re-implementations unless official code exists. Read each paper's method section first. Add them as `ft_method = 'sst' | 'rpsft' | 'ewc' | 'lora' | 'selfdistill'` knobs, implemented in `src/prism_baselines.py` and called from `train.py` at defined hook points. Defaults stay byte-identical.

| Arm | Method | Notes |
|---|---|---|
| `sst` | Subspace Tuning (arXiv 2610.22293) | Restrict updates to the bottom singular subspace, with periodic basis refresh. Match their rank and refresh choices as closely as the paper allows. Log the per-step overhead. |
| `rpsft` | Rotation-Preserving SFT (arXiv 2605.10973) | Soft top-k directional penalty. |
| `ewc` | EWC | Diagonal Fisher from the base, on old-domain data. Pick λ from 3 values on seed 1336. |
| `lora_r4`, `lora_r16` | LoRA | A nanoGPT-native wrapper on attention and MLP linears. Merge before scoring recall. |
| `selfdistill` | Self-distillation | KL to the frozen base on the finetune batches. Label it "a self-distillation baseline in the spirit of SDFT/iSDFT (arXiv 2609.24646), not a reproduction". |
| `out_eps` | Adam ε raised on the embedding/head group only | arXiv 2610.09835. |

**Metrics.**
- Forgetting (the existing old-domain retention climb).
- Recall: seen, unseen, and the v2 paraphrases.
- New-domain val loss.
- The adaptation-censored guard.
- Per-step wall-clock overhead against `plain`.

The primary display is the recall-vs-forgetting frontier, with every arm on both rigs.

**Pre-register these before running.**
1. Does `sst` block injection (seen recall within 10 points of `raw_mid`), or inject at parity (seen recall within 10 points of `plain`)? Unseen-phrasing recall is the secondary readout, because its seed variance is high.
2. Is any arm better than `raw_weak` on both axes, in 3 out of 3 seeds?
3. Do the `exemb`, `rows_absent` or Adam arms change forgetting by 20% or more? That would mean part of the retention is not directional.

**Runs (Tier A).** Estimate first. Split the work into at most 3 runner launches with distinct tags.

**Write-up.**
- `RESULTS.md` §18.
- README Finding 7. Update its chart only if the verdict changes.
- Mechanism-map rows for SST, Rotation-Preserving SFT and iSDFT.

**Scale-up (prepared; the smoke is Tier A, the study is Tier B).** This is Outcome 2 (G1/G2) from `docs/NEXT-EXPERIMENTS.md`, re-aimed at SST. Build `prism_modal_retention1b.py`:
- **Base:** Qwen3-1.7B-Base. SST reports −8.0 points at 1.7B, so this gives a direct reference.
- **Data:** SST's own SFT domain if it is public (check the paper and code), plus a synthetic fact-injection set with paraphrase splits.
- **Arms:**
  - SFT;
  - SST, from official code if released;
  - the anchor dial at 3 strengths, re-locating the weak band for Adam scale;
  - LoRA;
  - self-distillation.
- **Evaluation:** a pinned lm-eval-harness suite (MMLU, ARC-C, HellaSwag, WinoGrande, GSM8K) plus closed-book recall.

The ~$20 1B smoke with a mini dial sweep is Tier A. The full study is Tier B.

### W2 — Move 2: harden the core bet

There are two halves:
- **2a**, probe-scale controls you can run now (Tier A);
- **2b**, the Pythia ladder, built and smoke-tested in this session and launched only per tier.

**2a. Control suite at probe scale.** Run on modern web and enwik8, with the PROBE flags, 3 seeds, under AdamW and under Muon. Add each control as an `init_mode` knob:
- **`tuned_baseline`:** sweep the baseline's LR × warmup × init σ (at most 9 settings, seed 1336). The PRISM arm gets the same sweep budget.
- **`small_qk`:** initialize the Q and K blocks of `c_attn` at a σ far below default. Sweep 3 values, including 1e-5 (OpenReview 419qaAMpme).
- **`mimetic`:** mimetic initialization of attention, using the formulas in arXiv 2305.09828.
- **`statmatch`:** a statistics-matched random init. Impose the teacher's per-layer scalar statistics on a random init, without any directions.
  - Read OpenReview DzLzZYD84Y for its 108-statistic list.
  - At minimum, match per-matrix norm, plus attention entropy, LayerNorm output statistics and residual write ratio on a calibration batch.
  - List exactly which statistics you matched.

Then:
- Rerun the decomposition under the strongest baseline you found. Use the existing arms (`spectral_only`, `dirs_only`, `recipe`). Add a pure-directions arm if none exists: EigenTransfer onto the fresh init's own spectrum, with no imprint.
- Run every arm under AdamW and under Muon, and record the teacher's optimizer. Add a Muon-trained teacher variant.

**Metrics:**
- `prism_score`;
- the token-equivalent multiplier (W0.4);
- the end-of-training gap;
- overlap traces (W0.5). These answer whether Muon overwrites the imprint faster than it erases the transplanted directions, which is unmeasured anywhere.

**Pre-registered rules:**
- If dirs_only's steps-to-target ratio against the strongest control falls below 1.2× on either corpus, write that into README "Honest bounds" **before any scale spend**.
- If the ratio falls below 1.2× under Muon, label the core bet "AdamW-only until shown otherwise".

**Write-up:** `RESULTS.md` §19.

**2b. The Pythia ladder.** Write the G1 code in this session; launch G2 and later only per tier.
- **G1 port:** `src/prism_neox.py` implements PRISM's ops on GPT-NeoX/Pythia.
  - Name map: `attention.query_key_value`, `attention.dense`, `mlp.dense_h_to_4h`, `mlp.dense_4h_to_h`.
  - Embeddings are untied (`embed_in` and `embed_out`). Rotary covers 25% of the head (`rotary_pct` 0.25). The residual is parallel.
  - **Gotcha:** NeoX packs QKV *per head*, viewed as `(n_heads, 3·head_dim)`. That differs from nanoGPT's `[Q;K;V]` blocks.
  - The teacher is the same-size final checkpoint, as G2 pre-registers. Larger checkpoints are W3's cross-size extension.
  - Extract directions and spectra from the published checkpoints. Apply EigenTransfer onto **Pythia's own published step0 init** (revision `step0`), so that "only the init differs" is literally true.
  - Selftests:
    - with align=0, the result reproduces step0 exactly;
    - the QKV split round-trips;
    - group classification is correct;
    - outputs are orthonormal.
- **Training harness:** same architecture, the same token stream (Pythia's pre-shuffled tokenized Pile), and the same schedule as Pythia's published config.
  - Verify all three from EleutherAI's repo and Hugging Face.
  - A **same-codebase random-init baseline from step0** runs alongside. Overlay Pythia's published loss curve as a reproduction check.
  - Report no PRISM ratio until our baseline tracks that published curve, within a tolerance you state in advance. A 2023 recipe re-run in different code is not a fair comparator.
- **Arms per scale:** baseline (step0), PRISM (step0 + EigenTransfer), and from G3 onward, the strongest control from 2a. Use AdamW, matching Pythia's recipe. A Muon arm is a separate, pre-registered extension.
- **Accounting:**
  - steps-to-target;
  - the token-equivalent multiplier at end of training;
  - donor compute stated;
  - Pythia's published downstream suite, via lm-eval-harness at matched tokens;
  - the teacher's optimizer recorded.
- **Gates** (from `docs/NEXT-EXPERIMENTS.md`; never weaken them):
  - G2: 70M, single seed. Kill if the ratio is below 1.2×. Tier B, ~$100.
  - G3: 160M, 3 seeds. Tier B, ~$500.
  - G4: 410M. Tier C, ~$1.5k.

  Build `prism_modal_pythia.py` with one flag per gate. Include a dry-run mode (20 steps on a small GPU, then an artifact check) and per-gate cost estimates. If Tier B is authorized, launch G2 only after 2a's controls have landed.

### W3 — Move 3: cross-size SVD projection, with a draft writeup (time-sensitive)

**Why.** Nobody has projected a large LM's SVD subspaces into a smaller LM and trained it at 100M parameters or more. The learngene group (FRONT, then SPA-FRONT) is one step from it.

**Build `src/prism_crosssize.py`.**
- **Primary operator: residual-stream PCA alignment.**
  1. Run the teacher on a calibration slice.
  2. Take the top-d_S principal axes P (d_T × d_S) of its residual stream.
  3. Form projected targets W̃ = P_outᵀ W_T P_in for each matrix. Handle MLP hidden width by PCA of MLP activations, or by SVD truncation. Handle heads by projecting per-head QK/OV circuits.
  4. Take the SVD of W̃ for U and V, then run the normal EigenTransfer and imprint.
- **Alternative:** Procrustes alignment (precedent: arXiv 2610.15322).
- **Depth mismatch:** select or average layers, and state which.
- **Parameterization:** use μP for teacher and student. Width generalization holds under μP only (arXiv 2610.00420).

**Baselines:**
- scratch;
- Weight Selection (arXiv 2311.18823);
- FRONT (arXiv 2603.07523): 3D-DCT truncate or pad, per the paper;
- mimetic;
- same-size dirs_only, as the upper reference.

**Test cases** (probe scale, modern web, 3 seeds):
1. Width 768 → 384. The teacher is 12 layers at width 768, trained on modern web.
2. Head count 12 → 6 at fixed width.
3. Attention head-dimension reduction. CrossGMN gets 0.74× here; it is the hard case.
4. GPT-2 124M → the 10.65M byte-level student. Internal matrices only, since the vocabularies differ.

**Metrics:** `prism_score` against scratch, the end-of-training gap, overlap traces, and teacher compute stated.

**Pre-register:** success means ≥1.5× steps-to-target against scratch on case 1, and above 1× on case 3, in 3 out of 3 seeds.

**Deliverables:**
- `RESULTS.md` §20.
- `docs/CROSS-SIZE.md`, covering method and results.
- `docs/preprint/cross-size-projection.md`: a preprint-style draft with abstract, method, related work (FRONT/SPA-FRONT, CrossGMN, Weight Selection, LiGO, mimetic), results with artifact links, and limitations. **Do not submit or post it anywhere.** The user decides.

**Prepared (Tier B):** Pythia-1B/2.8B → Pythia-160M/410M projection from published checkpoints, on the W2 harness.

### W4 — Move 4: the RL anchor where RL drifts

**Why.** The major stacks default KL off: TRL GRPO β = 0, verl `use_kl_loss = False`, and prime-rl has no reference KL. The pitch is therefore drift control at zero forward-pass cost in regimes that actually drift, not cutting a cost few people still pay.

**Build** on `prism_anchor.py` (W0.3).
- Integrations: TRL's GRPOTrainer (callback or optimizer post-hook) and prime-rl. For prime-rl, read its trainer and add the smallest faithful hook, kept as a plugin or patch file, not a fork.
- Verify each stack's defaults from current docs, and record versions.

**Pilot** (Tier A as single-seed probes, at most $100 for all of W4; 3-seed confirmations are Tier B):
- TRL GRPO;
- Qwen2.5-0.5B-Instruct or Qwen3-0.6B;
- GSM8K;
- one GPU, about 300 steps.

**Arms:**
- β = 0;
- KL with β ∈ {0.01, 0.04, 0.1}, swept on seed 1336;
- a fixed-W₀ anchor, at 3 strengths including the weak band;
- an EMA anchor;
- a WARP-style arm: EMA anchor plus end interpolation toward init (arXiv 2406.16768);
- the anchor plus the Adam-moment lerp.

**Drift regimes.** Cross each regime with β = 0, the best KL, and the best anchor:
- on-policy, as the reference;
- recycled rollouts, reusing each batch m = 4 times;
- stale or async generation, with a policy lag of k steps;
- **zero-RL fine-tunability:** after RL, LoRA-finetune on IMDB, then re-score GSM8K (arXiv 2609.33898's protocol).

**Metrics:**
- task accuracy;
- retention on a small pinned lm-eval suite;
- KL to init;
- principal-direction overlap between the update and W₀'s top singular directions (read arXiv 2511.08567 for the metric);
- a step-time breakdown, giving the reference forward pass's share of total step cost. Report that share even if it is small.

**Pre-register:** the anchor earns its keep in a regime only if, in 3 out of 3 seeds, it does both of these:
- matches the best KL's retention within noise, at no more reward cost;
- beats β = 0 on retention.

**Prepared (Tier C):** the 9B prime-rl spec from `docs/NEXT-EXPERIMENTS.md`, re-aimed at the drift regimes.

### W5 — Move 5: the COUNT control for prior fusion (re-earn the 30×)

**Prerequisite.** This is the Act 2 blocker in `docs/NEXT-EXPERIMENTS.md`: a training-time **sparse** n-gram prior at context 4 or more, with block-edge backoff in `_prior_logp`, so the first C−1 positions are no longer uniform.
- Build it from `ngram_floor.py`'s sparse interpolated model.
- If gathering on the fly is too slow, precompute per-position prior log-probs for the train and val bins.
- Selftest: it matches `ngram_floor.py`'s scoring on a slice.

**Arms** (enwik8 primary, Shakespeare for continuity, 3 seeds, dense eval):
- baseline;
- dirs_only;
- prior fused during training;
- PRISM + prior fused during training;
- **inference-only fusion:** train the baseline and dirs_only without the prior, then fuse the same prior at eval;
- **`count_embed`:** a COUNT-style, parameter-free input embedding.
  - Use n-gram orders 1–6, and a count-weighted sum of the model's own unembedding rows over observed continuations. Follow the method section of arXiv 2610.20456 (HTML) exactly.
  - With tied embeddings, "own unembedding rows" means `wte` rows.
- `count_embed` + dirs_only.

**Logging:** validation at every epoch boundary, as well as the dense eval. arXiv 2610.00394 reports splits at epoch boundaries in exactly this regime.

**Metrics:** steps-to-target against the baseline (primary), end-of-training loss, and the token-equivalent multiplier. This run also serves as the composition gate for the core bet ("Act 2 on enwik8") in `docs/NEXT-EXPERIMENTS.md`.

**Pre-register:** logit fusion earns its keep only if fusing during training beats inference-only fusion on steps-to-target, in 3 out of 3 seeds, on enwik8. The binding two-act framing applies to everything written about the prior arms.

**Write-up:** `RESULTS.md` §21. Update README Finding 8 and the 30× caveat, whichever way the result lands.

### W6 — Move 6: packaging, lineage decay, prior art

1. **`docs/ANCHOR-RECIPES.md` + `examples/`:** the anchor as a drop-in for HF Trainer, TRL, prime-rl, plain PyTorch, and the hosted platforms (Tinker, Together Post-Training).
   - For each hosted platform, read its current docs and state exactly what its extension surface allows.
   - The anchor is a **post-optimizer-step parameter update**, not a loss over logits. Where a platform exposes only a logit loss, give the closest faithful equivalent and say exactly how it differs:
     - for LoRA: scale B by (1 − s), or decoupled weight decay on B alone with wd = s / lr;
     - for full-parameter training: an L2-to-W₀ penalty is coupled to Adam and is **not** equivalent.
   - Claim no platform support you haven't verified.
2. **The lineage-decay curve.** Using W0.5 tracking, measure how fast the teacher's signature decays during training. The signature is the spectral-energy profile plus subspace alignment, per arXiv 2608.07786's metrics.
   - Compare PRISM-initialized students, baseline students, and an unrelated teacher.
   - Probe scale, 3 seeds. Write it up in `RESULTS.md` §22.
   - Phrase IP implications as open questions, not legal conclusions.
3. **README "Where PRISM sits":** add one row each for:
   - WARP;
   - Rotation-Preserving SFT;
   - SST;
   - The COUNT;
   - FRONT/SPA-FRONT;
   - the residual-over-n-gram paper (arXiv 2210.14431);
   - the Spectral Warmup null (arXiv 2604.22778);
   - CrossGMN;
   - the 108-statistic init.

   Each row is one line on how it relates to PRISM. Keep "positioning, not head-to-head."

### W7 — Cheap mechanism probes (only if budget and time remain)

- **Circuit-level mining:** EigenTransfer on per-head QK and OV circuit products, which are gauge-invariant (arXiv 2610.18471), against per-matrix SVDs.
- **Fingerprint redundancy:** fit a 1–2-parameter heavy-tailed spectrum family (arXiv 2610.13123) and imprint it, against the 8-coefficient DCT.
- **SAMuon composition:** use the transferred spectrum as SAMuon's static per-direction step-size prior (arXiv 2608.25990). Label it speculative.

---

## 6. Spend tiers

| Tier | What | Rough cost | Default |
|---|---|---|---|
| **A** | All probe-scale runs in W1, W2a, W3 (cases 1–4), W5, W6.2, W7; the W1 1B smoke; the W4 single-seed pilot (≤ $100) | ~$250–300 total | Authorized up to `TIER_A_BUDGET_USD` |
| **B** | Pythia G2, 70M (~$100) and G3, 160M (~$500); the 1B retention study (~$300–500); W4 3-seed confirmations, or an RL pilot at 1.5–3B (~$500); cross-size from Pythia-1B | per item | Prepare and dry-run. Launch only if `AUTHORIZE_TIER_B = yes`, in gate order, within `TIER_B_BUDGET_USD` |
| **C** | Pythia G4, 410M (~$1.5k) and beyond; 9B prime-rl ($1.5–3k) | per item | Prepare only |

Costs come from `docs/NEXT-EXPERIMENTS.md` and from probe timings (a 3-seed 10.65M `prism_eval` run is about an L4-hour). Re-estimate each one with current prices and your measured step times before launching.

---

## 7. Execution order and branching

1. Read §2. Capture the golden runs. Commit a skeleton `docs/ALPHA-MOVES-STATUS.md`.
2. Build W0, with tests.
3. Pre-register and launch W1. While it runs, build W2a and launch it. Then build W5's sparse prior and launch it. Then build W3 and launch it. Then the W4 pilot. Then W6 and W7.
4. As each run lands: fetch the artifact, commit it, and write its `RESULTS.md` section.
5. Prepare the Tier B and C runners, each with a dry run.
6. Do a final docs pass, update the status doc, open the PR.

**Branching**
- Create the integration branch `alpha-moves` from `main`.
- Runner branches are `exp/<name>`, cut from `alpha-moves`, with new names (don't reuse an existing `exp/*` branch). Modal clones these, so push before every launch.
- Merge results back into `alpha-moves` with `merge:` commits.
- Keep `train.py` edits small and additive, one knob at a time, so parallel workstreams don't conflict.
- If your environment supports parallel workers or worktrees, W1, W2a, W3 and W5 are independent once W0 is done.

---

## 8. Write-up and deliverables

- **`RESULTS.md`**
  - New sections: §18 (W1), §19 (W2a), §20 (W3), §21 (W5), §22 (W6.2), §23 (W4 pilot, labeled probe).
  - Ledger rows continue after `AA`: `AB`, `AC`, and so on.
  - Each section states its verdict in its title, as the existing sections do.
- **`README.md`**
  - Update the ledger, any findings whose verdict changed, the mechanism map, "Honest bounds" and "What's next".
  - If a headline claim changes, also update the machine-readable summary at the bottom of `docs/how-prism-works.html`.
- **`docs/NEXT-EXPERIMENTS.md`:** add a dated "Status update" block at the top, in the style of the existing ones. Append new pre-registrations. Weaken nothing.
- **`results/README.md`:** one entry per new artifact, plus any schema additions. Additions only: keep `prism-*/1` readers working.
- **`docs/ALPHA-MOVES-STATUS.md`:**
  - for each move: done, partial or blocked; its artifacts; its verdict in one sentence; and what is prepared, with launch command and cost;
  - the **Decisions** log;
  - the **Spend ledger**;
  - **Open questions for the user.**
- **PR body:** a summary by move, verdicts with artifact links, what is gated and what it costs, and total spend. No attribution lines.

## 9. Your final message

Start with one line per move giving its outcome. Then report total spend against budget. Then list what needs the user's decision: each Tier B launch, its cost, and the result that would justify it. End with the PR link.
