# Alpha moves — status

*Session started 2026-10-21. Brief: [`ALPHA-MOVES-BRIEF.md`](ALPHA-MOVES-BRIEF.md).
Session parameters: `TIER_A_BUDGET_USD=300`, `PER_RUN_CAP_USD=40`, `AUTHORIZE_TIER_B=no`,
`MERGE_POLICY=pr`, compute = Modal (profile `realityinspector`, authenticated).
This file is the agent's memory: updated after every workstream.*

## Per-move status

| Move | Workstream | Status | Verdict (one sentence) | Artifacts |
|---|---|---|---|---|
| — | W0 shared infrastructure | **done** (code), golden check identical, selftest 67/67 | Anchor object, Muon/out_eps/μP optimizer builder, metrics, overlap tracker, recall v2, competitor hooks — all default-off. | — |
| 1 | W1 injection-dial head-to-head | catalog 3-seed **done** (RESULTS §18); facts 3-seed running | Catalog: SST (paper default) blocks injection like the 0.01 anchor (3%); embedding/Adam controls move forgetting <20% (directional); **self-distillation beats the weak anchor on both axes in 3/3 seeds** — the dial does not own the frontier on this rig. | `finetune_20261022T043049Z.json`; tuning `…T024927Z` (catalog), `…T024917Z` (facts) |
| 2a | W2a core-bet controls (probe scale) | **done** except small_qk/mimetic enwik8 (running); RESULTS §19 | Against the tuned baseline (LR 2e-3, warmup 300): dirs_only 3.3× (modernweb) / 3.1× (enwik8) under AdamW — the geometry survives tuning; under Muon the ratio falls to 1.4× / 1.5× (above the 1.2× line, barely); the no-imprint arm keeps 2.45×; statmatch and spectral_only sit below the baseline (1 seed so far). | `sweep_20261022T*`, `dirs_only_20261022T15{3244,3444,3457,3515}Z`, `dirs_pure_20261022T153518Z` |
| 2b | W2b Pythia ladder (G1 port, runner) | **built + merged**; dry run attempt 1 OOM'd on the L4 at micro-batch 32 (fixed: micro-batch 4 for the dry gate); attempt 2 running; G2+ Tier B (prepared only) | — | — |
| 3 | W3 cross-size SVD projection | cases 1–4 + FRONT + Weight Selection case 1 **done**; WS cases 2–3 running; preprint DRAFT not posted (RESULTS §20) | **Pre-registered success FAILS on case 1**: width 768→384 gives an early head start but no endpoint gain (1.0×); GPT-2→bytes null; FRONT far below scratch; **Weight Selection (index-space copy) wins the width cell at 4.0×**, so the claim narrows to the head cells: heads 12→6 transfers fully (3.7×, −0.11 nats), head-dim 64→32 gives 1.28× in 3/3 (CrossGMN's failure cell). | `dirs_only_20261022T0{35754,40349,40448,43832}Z`, `init_state_20261022T041102Z` |
| 4 | W4 RL anchor pilot | default-LR wave: 3 recycle arms done (no drift at all: KL(init) 0.005–0.007; the KL reference forward adds 33% to the step, the anchor 0.1%); amended 10× LR wave (6 arms) running | low need at default LR (pre-registered honest verdict); drift regime pending | `rl_*_recycle4_*` |
| 5 | W5 sparse prior / COUNT control | run 1 **VOID**; run 2 (`w5b`) 2/3 seeds in: fused init 2.02 bpb (the Act-1 dot), train-time fusion ends 1.87 bpb vs inference-only 2.02 vs dirs_only 1.90; dirs_prior 1.845 (composition saturates, +0.015 nats over prior); COUNT embedding worsens the endpoint | (final on seed 3) | `prior_20261022T035740Z.json` (void, kept as record) |
| 6 | W6 packaging, lineage decay, prior art | recipes + examples done; README mechanism-map rows done; lineage decay §22 written from enwik8 (modernweb + unrelated-teacher control running) | PRISM students keep ~0.8 overlap with the teacher's directions over 1,500 steps (chance 0.03); baseline students converge to 0.2–0.3 on their own | `dirs_only_20261022T172455Z.json` |
| 7 | W7 cheap mechanism probes | not started | — | — |

## Golden-run check (§3.5)

Captured on unmodified `main` (commit `4f71f6c`), CPU, fixed seeds, each smoke
run twice from a clean state; logs under `.prism_runs/golden/` (gitignored).

| smoke | command | run-to-run variation on main | after W0 |
|---|---|---|---|
| `prism_eval --method=dirs_only` (t10/s10/e5, seed 1337, bs 4) | README tiny smoke | identical (8 eval lines) | **identical** |
| `prism_eval --method=recipe` | same | identical | **identical** |
| `prism_finetune_eval` arms `base,plain,raw_weak` (b10/f10, 2L/64d) | tag `golden` | identical (8 eval lines) | **identical** |
| `prism_prior_eval` arms `baseline,prior` (c3, t10/s10) | tag `golden` | identical (6 eval lines) | **identical** |

CPU runs are bit-reproducible here, so the acceptance bar is exact equality.

## Decisions

- **D1 (2026-10-21)** The brief's knob name `optimizer` collides with train.py's
  optimizer object; the knob is `optim` (`'adamw' | 'muon'`).
- **D2** The anchor object replaces the inline lerp on *every* path (scratch mod
  wheel, resume raw/spectral/shuffled). Justification: one code path; the golden
  check proves the default is bit-identical.
- **D3** `out_eps` applies to the single tied `wte`/`lm_head` table (nanoGPT ties
  them), which is exactly arXiv 2610.09835's tied-embedding case; `wpe` shares the
  group. Documented in `prism_optim.py`.
- **D4** Checkpoints written by non-default optimizers carry an extra key
  `optimizer_param_names` so a resume can map Adam moments by parameter identity
  across groupings; plain checkpoints are unchanged.
- **D5** Catalog recall v2 skipped: the catalog prompt is a fixed structured line
  with no paraphrase axis (its `unseen` split is already novel attribute
  combinations). Facts recall v2 (3 new held-out templates × 120 facts) added.
- **D6** Competitor hyper-parameters: each gets three settings on seed 1336 under
  the pre-registered criterion in `docs/NEXT-EXPERIMENTS.md` (W1 block); the bare
  arm name is set to the winner before the 3-seed launch.
- **D7** `ft_method_cfg` / `count_embed_cfg` use `key:value` (nanoGPT's
  configurator splits every flag on `=`, so `k=v` inside a value breaks it).
- **D8** `torch.optim.Muon` runs its Newton–Schulz iterations in bfloat16, which
  has no fast CPU kernels: ~50 s/step for the 10.65M model on this laptop. CPU
  smokes with Muon are therefore slow but correct; GPU is unaffected. Kept the
  torch implementation (the brief's first choice) rather than vendoring one.
- **D10** W5 run 1 is void by its own pre-registered kill condition (the `prior` arm never
  reached baseline-best and its step-0 loss sat above uniform). Root cause: the sparse
  prior's in-window gather built each position's context from the symbols BEFORE it,
  predicting the current byte. The standalone sequential scorer was correct, which is
  why the build check passed. Fixed (`prism_prior.py`), three alignment selftests
  added, identical run relaunched as `w5b`. The void artifact stays committed as a record.
- **D11** The W2a sweep driver shared one run dir between the 6-point and the 2-point
  grids, so the second invocation's artifact overwrote the first's; the committed sweep
  artifacts were reconstructed from the banked stage files (noted inside each).
- **D12** The seed-1336 tuning showed the committed baseline recipe was already within
  0.01 nats of its grid's best; the tuned schedule (LR 2e-3, warmup 300) is used for
  every W2a control arm. statmatch (70 teacher statistics on a random init) is WORSE
  than the baseline on both corpora at 1 seed; the strongest control is therefore the
  tuned baseline or small_qk (σ=1e-5 beat it by 0.008 nats on 1336), whose 3-seed run
  decides the cross-artifact ratio.
- **D9** W2b/W3/W5 are built by parallel workers in their own worktrees
  (branches `exp/alpha-w2b`, `exp/alpha-w3`, `exp/alpha-w5`, all cut from
  `alpha-moves` at 630f181), prepare-only; every GPU launch goes through me.

## CPU smokes (every new knob writes a conforming artifact)

Finetune harness, tag `smokeknobs` (modernweb base, facts corpus, recall v1+v2,
`--track_overlap=5`): arms plain, raw_weak, raw_weak_rows_absent,
raw_weak_adamreset, raw_weak_adamlerp, raw_aniso_t1, out_eps, sst_t2, rpsft_t2,
ewc_t2, lora_r4_t2, selfdistill_t2, lowlr_weak, raw_weak_exemb, plain_adamreset —
all complete; artifact carries `step_ms`, `ftm_overhead_ms`, `overlap`,
`recall_v2a/b/c`, `step_overhead_vs_plain`. From-scratch harness: `--optim=muon`,
`dirs_pure`, `small_qk`, `mimetic`, `statmatch`, `tuned` (init_std both arms +
method), `--teacher_optim=muon`, `--track_overlap` — all rc 0 with artifacts;
`train.py --mup_base_width=32` runs. `prism_sweep.py` tiny grid — artifact
`prism-sweep/1`. Smoke artifacts were deleted (never evidence).

## W1 tuning picks (seed 1336, pre-registered rule: lowest forgetting with seen-recall ≥ 0.90, else highest seen)

Catalog rig (`results/finetune_20261022T024927Z.json`):

| method | t1 | t2 | t3 | chosen | note |
|---|---|---|---|---|---|
| sst (r) | 0.5: seen 0.20, forget 1.12 | 0.9: 0.12, 0.96 | 0.99: 0.05, 0.95 | **t1 (r=0.5)** | no setting reaches 0.90 — SST blocks the catalog task even with half the spectrum free |
| rpsft (λ) | 0.1: 0.98, 2.51 | 1.0: 0.78, 1.89 | 10: 0.12, 1.06 | **t1 (λ=0.1)** | the only setting at parity; forgets ~like plain (2.82) |
| ewc (λ, Fisher normalised to mean 1) | 0.1: 0.13, 0.37 | 1.0: 0.08, 0.33 | 10: 0.00, 0.29 | **t1 (λ=0.1)** | blocked at every grid point — the grid sat above EWC's useful band; a limitation of the 3-setting budget, stated as such |
| lora_r4 (lr) | 3e-4: 0.15, 1.67 | 1e-3: 0.35, 2.22 | 3e-3: 0.35, 2.61 | **t2 (lr 1e-3)** | LoRA r=4 learns the task poorly in 1,000 steps; lr reused for r=16 |
| selfdistill (α) | 0.3: 0.98, 0.55 | 1.0: 0.98, 0.38 | 3.0: 0.00, 0.30 | **t2 (α=1.0)** | beats raw_weak (0.93, 0.89) on BOTH axes on this seed — the 3-seed run decides |
| raw_aniso | 32:0.01:0.0025 → 0.90, 0.57 | 32:0.02:0 → 0.95, 1.60 | 96:0.01:0.0025 → 0.85, 0.45 | **t1** | better forgetting than raw_weak at the parity boundary |

Facts rig: pending (relaunched after a container preemption).

## Spend ledger

| launch | tier | GPU | est. hours | est. USD | actual USD | status |
|---|---|---|---|---|---|---|
| W1 tuning, facts rig, seed 1336, 21 arms (`prism_modal_alpha.py --rig facts --slot a --tag w1tune`, commit 630f181) | A | L4 | 1.9 | 1.55 | pending | running (ap-nayF5Hz8sMFe4G44aKTnHC) |
| W1 tuning, catalog rig, seed 1336, 21 arms (`--rig catalog --slot b --tag w1tune`) | A | L4 | 1.9 | 1.55 | pending | running (ap-vRlsFc7SZGjYXLZXJLNAp9) |
| W1 tuning facts — relaunch after a container preemption at arm lora_r4_t1 (resumes banked arms; runner now clears stale locks) | A | L4 | ~1.2 (remaining) | ~1.0 | pending | running (fc-01M5GVZF7XFXR6PDMWY4H3X1P2) |
| W2a sweeps (seed 1336): modernweb baseline (ev-a), modernweb dirs_only (ev-b), enwik8 baseline (ev-c), enwik8 dirs_only (ev-d), Muon muon_lr for the baseline on modernweb (ev-e) and enwik8 (ev-f) | A | L4 | 0.9+1.1+0.9+1.1+0.3+0.3 = 4.6 | 3.66 | pending | running |
| W5 enwik8 prior fusion, 8 arms × 3 seeds (`prism_modal_alpha_w5.py`, volume prism-alpha-w5) | A | L4 | 3.8 | 3.02 | pending | running (fc-01M5GW3EN2D0D5J0ZR9XQHZ59P) |
| W3 wave 1: case 1 xsize (w3-a), case 2 xsize (w3-b), case 3 xsize (w3-c), case 4 xsize (w3-d), case 4 front (w3-e), all `--track_overlap=50` | A | L4 | 4.6+1.6+1.3+1.1+1.1 = 9.7 | 7.80 | pending | running |
| W2b Pythia dry run (70M, 2 arms × 20 iters, L4; data fetch 84 MB) | A | L4 | ~0.3 | <1 | pending | launched (ap-KG24YiLp5Azd2dUYtZpuuQ) |
| W1 catalog 3-seed result run, 23 arms + base, tuned overrides, `--track_overlap=50` (slot b, tag w1) | A | L4 | 9.7 (conservative) | 7.76 | pending | running (fc-01M5GXB5EKYZCR46YTQY6847RX) |

**Budget note (2026-10-21 evening).** The user's Modal panel showed $4.48 used and
about **$38 of workspace usage remaining** at that point — below the brief's $300
Tier A ceiling. Committed launches above total ≈ $27 estimated; the facts 3-seed
run (≈ $8), W2a's 3-seed arms (≈ $15) and the W4 pilot (≈ $15–30) will exceed
the remaining balance unless it is raised. Priority order if funds run short:
W1 facts 3-seed → W2a core arms (dirs_only, tuned baseline, statmatch, small_qk
on modernweb) → W4 minimal (β=0 / best KL / anchor in the recycle regime) → the
rest. Every launch stays under the $40 per-run cap.

Modal prices used for estimates (2026-10-21, modal.com/pricing): L4 $0.80/h,
A10 $1.10/h, L40S $1.95/h, A100-40 $2.10/h, A100-80 $2.50/h, H100 $3.95/h.

## Open questions for the user

- **Modal balance.** Your panel showed ~$38 of workspace usage left. The full Tier A
  plan needs roughly $80–100 more than that (see the ledger). Top up, or accept the
  priority order in the budget note.
- **EWC grid.** All three pre-registered λ values blocked injection on the catalog
  rig; the tuning-parity budget forbids a fourth. Shall EWC get a second,
  lower grid (λ ∈ {0.001, 0.003, 0.01}) as a separately pre-registered follow-up?

**D13 (2026-10-22, user decision relayed by the planning session).** PRISM GPU spend is
capped at $100 total through 2026-11-21. Finish the runs already in flight, open the PR,
and launch nothing else: no Tier B (Pythia G2, 1B retention study, W4 3-seed
confirmations) and no 124M replication before 2026-11-21.
