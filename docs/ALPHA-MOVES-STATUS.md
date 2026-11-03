# Alpha moves — status

*Session started 2026-10-21. Brief: [`ALPHA-MOVES-BRIEF.md`](ALPHA-MOVES-BRIEF.md).
Session parameters: `TIER_A_BUDGET_USD=300`, `PER_RUN_CAP_USD=40`, `AUTHORIZE_TIER_B=no`,
`MERGE_POLICY=pr`, compute = Modal (profile `realityinspector`, authenticated).
This file is the agent's memory: updated after every workstream.*

## Per-move status

| Move | Workstream | Status | Verdict (one sentence) | Artifacts |
|---|---|---|---|---|
| — | W0 shared infrastructure | **done** (code), golden check identical, selftest **117/117** (2026-11-02) | Anchor object, Muon/out_eps/μP optimizer builder, metrics, overlap tracker, recall v2, competitor hooks — all default-off. | — |
| 1 | W1 injection-dial head-to-head | catalog and facts rigs **done, 3 seeds each** (RESULTS §18) | Our re-implementations of SST, EWC and LoRA r=4 block injection on both rigs (catalog SST paper default is 11.7 points *below* the strong band, outside rule 1's "within 10" reading). Self-distillation beats the weak anchor on both axes on catalog (3/3) but not on facts (seen recall lower in 2/3), so the "drop the frontier claim" consequence (needs both rigs) does not fire. Embedding exclusion moves facts-rig forgetting +20.3% (crosses rule 3's bar); catalog all under. The 2-seed held-out-recall claims are withdrawn (D15). | `finetune_20261022T043049Z.json`, `finetune_20261102T225917Z.json` (supersedes the partial `finetune_20261022T143128Z.json`) |
| 2a | W2a core-bet controls (probe scale) | **done** (RESULTS §19) | Strongest control by the pre-registered rule: `small_qk` on modern web (below the tuned baseline in 3/3), the tuned baseline on enwik8 (small_qk untuned there, mimetic and statmatch above it). `dirs_only` vs the strongest control: **3.2× (modern web, recomputed cross-artifact) / 3.1× (enwik8)** under AdamW; 1.4× / 1.5× under Muon (vs the Muon baseline). All above the 1.2× line. The no-imprint arm keeps 2.45×. The token-equivalent multiplier is refused on every seed by design (its tolerance sits below eval noise: a design flaw, recorded). | `sweep_20261022T*`, `dirs_only_20261022T15{3244,3444,3457,3515}Z`, `small_qk_20261102T215708Z`, `small_qk_20261102T224802Z`, `mimetic_20261102T233902Z`, `dirs_pure_20261022T153518Z` |
| 2b | W2b Pythia ladder (G1 port, runner) | **dry gate passed** on GPU (2026-11-02); G2+ not launched (Tier B, D13) | Both 70M arms run 20 real iterations on an L4 and write conforming `prism-pythia/1` artifacts; plumbing only, no ratio. | `pythia_20261022T143230Z.json`, `pythia_20261022T143231Z.json` |
| 3 | W3 cross-size SVD projection | cases 1–4 + FRONT + Weight Selection cases 1–3 **done**; preprint DRAFT not posted (RESULTS §20) | Pre-registered criterion FAILS on case 1 (width + depth halving, 1.0×) and passes on case 3 (width-only halving, 1.28×). Case 2 changes no parameter shape (same-size effect, 3.7×). Our Weight Selection re-implementation beats the projection in both size-changing cells (4.0×, 8.2×); its case-2 49× is a degenerate upper bound. | `dirs_only_20261022T0{35754,40349,40448,43832}Z`, `init_state_20261022T041102Z`, `init_state_20261022T170647Z`, `init_state_20261102T214129Z`, `init_state_20261102T214752Z` |
| 4 | W4 RL anchor pilot | LR 1e-6 recycle:4 wave and the LR 1e-5 wave (6 arms) **done**, single-seed probes (RESULTS §23); on-policy arms not run (D14) | At LR 1e-6 nothing drifts; at 1e-5 KL to init is 0.32–0.58 with no retention loss in any arm, so the anchor cannot beat β0 (low need). The anchor callback costs 6–7 ms per step. The earlier 33% reference-forward figure was a GPU-type confound and is withdrawn. | `rl_*_s1337_w4_*.json`, `rl_*_s1337_w4lr1e5_*.json` |
| 5 | W5 sparse prior / COUNT control | run 1 **VOID**; run 2 **3 seeds, exploratory** (RESULTS §21) | Run 2 fails the pre-registered prior-fidelity check (1.9271 vs 1.9061 ± 0.02 bpb, by 0.0010). The decision rule and the composition gate are not evaluable as written (every fused arm starts below baseline-best). Exploratory: train-time fusion ends 0.10–0.15 nats below inference-only fusion in 3/3 seeds; dirs_prior is only 0.016–0.018 below prior alone; COUNT embedding 0.07–0.085 above dirs_only. | `prior_20261102T214708Z.json` (supersedes the partial `prior_20261022T170423Z.json`) |
| 6 | W6 packaging, lineage decay, prior art | recipes + examples done (prime-rl patch replaced by a manual-edit note, FSDP unverified); lineage decay §22 **done** (enwik8, modern web, unrelated-teacher control) | Our re-implementation of the subspace metric separates a PRISM student from a same-data baseline (by 0.31–0.51 per non-embedding group) and from an unrelated teacher in every seed at the 1,500-step horizon (0.75× the teacher's tokens). A shared random-init seed alone lifts overlap to 0.07–0.25 (seed-1337 confound of the control, stated). | `dirs_only_20261022T172455Z.json`, `dirs_only_20261022T181750Z.json`, `dirs_only_20261102T222126Z.json` |
| 7 | W7 cheap mechanism probes | CPU half of W7.2 **done**; GPU halves listed, not run (D13) | The imprinted DCT-8 curve's tail rises to its maximum (basis pinned at t→π); a 2-parameter stretched exponential reconstructs the teacher's spectra within 3–14%. | `w7_specfit_20261102T214437Z.json`, `docs/alpha/W7-NOTES.md` |

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

Facts rig (`results/finetune_20261022T024917Z.json`, seed 1336; re-derived 2026-11-02
from the artifact by the same rule; every pick matches `config.arm_overrides` of the
3-seed run):

| method | t1 | t2 | t3 | chosen | note |
|---|---|---|---|---|---|
| sst (r) | 0.5: seen 0.06, forget 1.44 | 0.9: 0.03, 1.53 | 0.99: 0.02, 1.58 | **t1 (r=0.5)** | none reaches 0.90 → highest seen |
| rpsft (λ) | 0.1: 1.00, 2.96 | 1.0: 1.00, 2.43 | 10: 0.07, 1.46 | **t2 (λ=1.0)** | lowest forgetting at parity |
| ewc (λ) | 0.1: 0.06, 0.55 | 1.0: 0.02, 0.44 | 10: 0.00, 0.36 | **t1 (λ=0.1)** | blocked at every grid point, as on catalog |
| lora_r4 (lr) | 3e-4: 0.05, 2.01 | 1e-3: 0.06, 2.55 | 3e-3: 0.20, 2.84 | **t3 (lr 3e-3)** | none reaches 0.90; lr reused for r=16 |
| selfdistill (α) | 0.3: 1.00, 0.78 | 1.0: 0.97, 0.56 | 3.0: 0.46, 0.45 | **t2 (α=1.0)** | |
| raw_aniso | 32:0.01:0.0025 → 0.89, 0.69 | 32:0.02:0 → 1.00, 1.73 | 96:0.01:0.0025 → 0.29, 0.57 | **t2** | t1 just misses the 0.90 band |

## Spend ledger

| launch | tier | GPU | est. hours | est. USD | actual USD | status |
|---|---|---|---|---|---|---|
| W1 tuning, facts rig, seed 1336, 21 arms (`prism_modal_alpha.py --rig facts --slot a --tag w1tune`, commit 630f181) | A | L4 | 1.9 | 1.55 | not itemised | done 10-22 (`finetune_20261022T024917Z.json`) |
| W1 tuning, catalog rig, seed 1336, 21 arms (`--rig catalog --slot b --tag w1tune`) | A | L4 | 1.9 | 1.55 | not itemised | done 10-22 (`finetune_20261022T024927Z.json`) |
| W1 tuning facts — relaunch after a container preemption at arm lora_r4_t1 (resumes banked arms; runner now clears stale locks) | A | L4 | ~1.2 (remaining) | ~1.0 | not itemised | done 10-22 (same artifact) |
| W2a sweeps (seed 1336): modernweb baseline (ev-a), modernweb dirs_only (ev-b), enwik8 baseline (ev-c), enwik8 dirs_only (ev-d), Muon muon_lr for the baseline on modernweb (ev-e) and enwik8 (ev-f) | A | L4 | 0.9+1.1+0.9+1.1+0.3+0.3 = 4.6 | 3.66 | not itemised | done 10-22 (`sweep_20261022T*.json`) |
| W5 enwik8 prior fusion, 8 arms × 3 seeds (`prism_modal_alpha_w5.py`, volume prism-alpha-w5) | A | L4 | 3.8 | 3.02 | not itemised | done 10-22; run 1 VOID (`prior_20261022T035740Z.json`) |
| W3 wave 1: case 1 xsize (w3-a), case 2 xsize (w3-b), case 3 xsize (w3-c), case 4 xsize (w3-d), case 4 front (w3-e), all `--track_overlap=50` | A | L4 | 4.6+1.6+1.3+1.1+1.1 = 9.7 | 7.80 | not itemised | done 10-22 (cases 1–4 + FRONT artifacts) |
| W2b Pythia dry run (70M, 2 arms × 20 iters, L4; data fetch 84 MB) | A | L4 | ~0.3 | <1 | not itemised | never ran on 10-22 (L4 shortage, then the spend limit); ran 11-02 |
| W1 catalog 3-seed result run, 23 arms + base, tuned overrides, `--track_overlap=50` (slot b, tag w1) | A | L4 | 9.7 (conservative) | 7.76 | not itemised | done 10-22 (`finetune_20261022T043049Z.json`) |

| W2a 3-seed arms: dirs_only / dirs_pure / recipe / spectral_only / mimetic / statmatch on modernweb, dirs_only / statmatch on enwik8 (AdamW); dirs_only / dirs_pure / spectral_only (Muon); Muon-teacher dirs_only (ev-a..f chains) | A | L4 | ~22 | ~17 | — | **done** (16 artifacts) |
| W2a small_qk (modernweb, enwik8) + mimetic enwik8 chain (ev-e) | A | L4 | 3.7 | 3 | — | never scheduled on 10-22 (spend limit); relaunched and done 11-02 |
| W1 facts 3-seed result run, 23 arms + base, tuned overrides (slot a, tag w1) | A | L4 | 9.7 | 7.8 | — | seeds 1337–1338 + 19/24 arms of 1339 on 10-22; last 5 arms done 11-02 (A10) |
| W3 wave 2: Weight Selection cases 1–3 (w3-a/b/c) | A | L4 | 4.6+1.6+1.3 | 6 | — | case 1 done 10-22; case 2 at 2/3 seeds on 10-22, seed 3 done 11-02; case 3 done 11-02 |
| W5 run 2 (`w5b`, same flags, fixed prior) | A | L4 | 3.8 | 3 | — | seeds 1337–1338 on 10-22; seed 1339 done 11-02 |
| W6.2 lineage chain: enwik8 dirs_only tracked / modernweb dirs_only tracked / modernweb with the enwik8 teacher's directions as reference (ev-g) | A | L4 | 4 | 3.2 | — | run 1 done 10-22; runs 2–3 done 11-02 |
| W4 pilot wave 1: β0 / KL 0.04 / anchor × on-policy / recycle:4, LR 1e-6 (`prism_modal_rl.py`) | A | A100-40 | 6 × 0.7 | ~9 | — | 3 recycle arms done 10-22; the 3 on-policy arms ran 25–205 of 300 steps before the halt killed them and were not relaunched (D14) |
| W4 pilot wave 2 (pre-registered amendment): the same six at LR 1e-5 (tag `w4lr1e5`) | A | A100-40 | 6 × 0.7 | ~9 | — | killed by the spend limit mid-run on 10-22 (no artifacts, ~$6 of A100 time lost); relaunched fresh and done 11-02 |

**Compute halt (2026-10-22, 18:30 UTC).** Every launch after this point returned
`Workspace ... has exceeded its spend limit`; containers already running were stopped
and queued ones never scheduled. The workspace limit was $42.50 with $4.48 used before
this session's GPU work, so **this session spent about $38 on Modal**, against the
brief's $300 Tier A ceiling. The actual figure is on the Modal billing page (the CLI
does not expose it). Nothing further was launched (D13).

**To finish the in-flight work once the workspace limit is raised** (historical, 10-22; executed 11-02, see "Finish-up session" below) (≈ $12 total, all
Tier A, all resuming from banked stages on their volumes; commands run from the repo
root on branch `alpha-moves`):

```
# W1 facts seed 1339, 5 arms left (~1 L4-h)
ARMS=base,plain,lowlr_b,raw_weak,raw_mid,raw_exffn,raw_exattn,raw_0035,raw_005,lowlr_weak,raw_weak_exemb,raw_weak_rows_absent,raw_weak_adamreset,raw_weak_adamlerp,plain_adamreset,raw_aniso,out_eps,sst,sst_paper,rpsft,ewc,lora_r4,lora_r16,selfdistill
OVR=sst:sst_t1,rpsft:rpsft_t2,ewc:ewc_t1,lora_r4:lora_r4_t3,lora_r16:lora_r16_t3,selfdistill:selfdistill_t2,raw_aniso:raw_aniso_t2
modal run --detach prism_modal_alpha.py --rig facts --slot a --tag w1 --seeds 1337,1338,1339 --arms $ARMS --extra "--arm_overrides=$OVR --track_overlap=50"
# W5 run 2 seed 1339 (~1.3 L4-h)
modal run --detach prism_modal_alpha_w5.py --extra "--corpus=enwik8 --sparse_context=6 --teacher_steps=2000 --student_steps=1500 --eval_every=10 --eval_iters=50 --batch_size=32 --block_size=256 --seeds=1337,1338,1339 --arms=baseline,dirs_only,prior,dirs_prior,baseline_inffuse,dirs_inffuse,count,count_dirs --tag=w5b"
# still queued on Modal (resume by themselves when the limit is raised): ev-e small_qk chain, ev-g lineage runs 2–3, w3-c Weight Selection case 3, Pythia dry, 3 on-policy W4 arms
# W4 10× LR wave: relaunch all six (`--tag w4lr1e5`, ~$9) — nothing was banked
```

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

## Open questions for the user (updated 2026-11-02)

- **License mismatch.** `LICENSE` is BSL 1.1 (commit c8d3562), but the README License
  section and `HANDOFF.md` still say "PRISM Community License 1.2", as do two claude.ai
  artifacts. A README diff is prepared and not applied.
- **A lower EWC grid** (λ ∈ {0.001, 0.003, 0.01}) as a separately pre-registered
  follow-up. All three pre-registered λ values blocked injection on both rigs.
- **Isolating cross-size cells** (new compute): width-only from the wider teacher
  (6L/768 → 6L/384) and depth-only (12L/384 → 6L/384), to separate depth from width in
  case 1's failure. Also an unconfounded lineage control (an unrelated teacher with a
  different init seed for every student seed).
- **A pre-registered W5 rerun** with the in-harness prior within 0.02 bpb of Act 1 and a
  target the fused arms do not start below.
- **W7 GPU halves** (≈ $1–2 each; `docs/alpha/W7-NOTES.md`), in particular a faithful
  spectrum imprint against the current DCT-8 curve.
- **Refresh the two claude.ai artifacts** (Results Atlas, Alpha Map). They predate these
  verdicts.
- **Delete the merged branches on origin** (`alpha-moves`, `exp/alpha-*`) now that the
  runs are finished. The agent cannot push `--delete`.

**D13 (2026-10-22, user decision relayed by the planning session).** PRISM GPU spend is
capped at $100 total through 2026-11-21. Finish the runs already in flight, open the PR,
and launch nothing else: no Tier B (Pythia G2, 1B retention study, W4 3-seed
confirmations) and no 124M replication before 2026-11-21.

## Finish-up session (2026-11-02, branch `alpha-moves-finish`)

Scope: finish only the runs in flight or queued at the 2026-10-22 halt, fix the
write-ups, open a PR. D13's cap binds ($100 of PRISM GPU spend 2026-10-22 →
2026-11-21; ≈ $38 was spent on 10-22).

**Modal inventory before any launch (2026-11-02).** 14 apps from 10-22 were still
listed as `ephemeral (detached)` with 0 tasks and no running containers; their logs
had expired. None had resumed after the workspace limit was lifted. All 14 were
stopped (user decision, 2026-11-02) so that no volume could get a second live app
(the runners clear `lock.json` at container start). No banked stage was lost: stages
live on the volumes.

| app | runner | volume / slot | what it was | state on 11-02 | action |
|---|---|---|---|---|---|
| ap-PVFEXhdivOiaC8QkWVfki2 | prism-alpha | prism-alpha-a | W1 facts 3-seed | stopped 10-22 | relaunched (below) |
| ap-iaZkONgiY15HcfOoUeXtJI | prism-alpha-w5 | prism-alpha-w5 | W5 run 2 (`w5b`) | stopped 10-22 | relaunched |
| ap-7g18xr2AppUnNrae0Lt6vp | prism-alpha-eval | ev-e | small_qk + mimetic chain | queued, 0 tasks | stopped, relaunched |
| ap-3UO0oC4yhJjib1GZh2AwLb | prism-alpha-eval | ev-g | lineage runs 2–3 | queued, 0 tasks (run 2 had banked s1337 baseline) | stopped, relaunched |
| ap-GL3nOvZxg0XSIFByA0uMMY | prism-alpha-w3 | w3-b | WS case 2 (seed 1339 in flight at the halt) | queued, 0 tasks | stopped, relaunched |
| ap-NmzjFJioloRQhl08ke0SXb | prism-alpha-w3 | w3-c | WS case 3 | queued, 0 tasks | stopped, relaunched |
| ap-Of6Hsb8KOYdTAEKNyZxp3D | prism-alpha-pythia | prism-alpha-pythia | Pythia 70M dry gate | queued, 0 tasks | stopped, relaunched |
| ap-pWScoE56pZsirkjFI0EfJb, ap-rk7sZuXJIYjQXYyQL5JKnK, ap-TJTnQWJFJQ8uFCYuSBIazB | prism-alpha-rl | prism-alpha-rl | W4 wave 1 on-policy arms (had run ~25–205 of 300 steps when killed; the RL driver does not resume) | queued, 0 tasks | stopped; **not relaunched** (user decision on spend) |
| ap-LBEFuUgatS6qufESjfkV0p, ap-EFO6oLEP2amzi5N1U02cch, ap-uuXKm7fIdVuhMn3mSGMytJ, ap-kMgqXmlFWC98hPVugiIAHX, ap-UE7uAcvIN6kYQ2Vxeld8Iw, ap-63sjcUB3Mw26QksoMecpsm | prism-alpha-rl | prism-alpha-rl | W4 wave 2, LR 1e-5 (nothing banked) | queued, 0 tasks | stopped, relaunched fresh |

**Launch flags.** W1 and W5 use the commands recorded above (the `--arm_overrides`
string matches `config.arm_overrides` in `finetune_20261022T143128Z.json`, and the W5
flags reproduce the run_key of `prior_20261022T170423Z.json`). The other launches were
rebuilt from the committed artifacts' configs, the banked run_keys on each volume and
the runners' own flag builders, not copied from the 10-22 session transcript. Where a
flag could not be recovered it is stated: the lineage unrelated-teacher control uses
the seed-1337 enwik8 teacher's `directions.pt` as the single tracker reference
(`--track_ref=.prism_cache/ev-enwik8_teacher_s1337_t2000/directions.pt`), and both
`small_qk` arms pass `--qk_sigma=1e-5` explicitly. `--extra` values that begin with
`--` must be passed as `--extra=...` (click otherwise parses them as options).

**Code identity.** Every runner resets to its branch head at container start. Heads on
11-02 equal the 10-22 heads (`exp/alpha-w1` 5e5420a, `-w2a` 7a436db, `-w2b` 5419d2d,
`-w3` d37226a, `-w4` 7dbc332, `-w5` 7c3e7f2). The banked partials were produced at
exactly these commits, except the W2a arms (d37226a). The d37226a → 7a436db diff
touches only the overlap tracker (`--track_ref`, the `[trace]` line, an extra `S` key in
`directions.pt`), which does not change any training trajectory, so the new small_qk /
mimetic arms are comparable with the committed W2a arms.

| launch (2026-11-02) | volume | branch @ commit | app | GPU | est. L4/A100-h | est. USD |
|---|---|---|---|---|---|---|
| W1 facts seed 1339, 5 arms left (rpsft, ewc, lora_r4, lora_r16, selfdistill) — first launch; never scheduled in ~2 h of L4 queueing, stopped unrun | prism-alpha-a | exp/alpha-w1 @ 5e5420a | ap-NAgTZZD6EJ1OKa2dQ7Cyjo | L4 | 0 | 0 |
| W1 facts, the same command with `--gpu A10G` (identical flags and run_key; the 19 banked seed-1339 arms reused); done in ~19 min | prism-alpha-a | exp/alpha-w1 @ 5e5420a | ap-n1GAUmYy1TN5wRWU2AYcZP | **A10** | ~0.3 | ~0.4 |
| W5 run 2 seed 1339 (`w5b`) | prism-alpha-w5 | exp/alpha-w5 @ 7c3e7f2 | ap-lCth9uCIuY8Amj8onUlefu | L4 | ~1.3 | ~1.0 |
| W3 Weight Selection case 2, seed 1339 (seeds 1337–1338 banked) | w3-b | exp/alpha-w3 @ d37226a | ap-VWeU2btn6r6VibE1rGLUGb | L4 | ~0.6 | ~0.5 |
| W3 Weight Selection case 3, 3 seeds | w3-c | exp/alpha-w3 @ d37226a | ap-mbeKacVmOSHFMul3yH8LNz | L4 | ~1.3 | ~1.0 |
| W2a small_qk modernweb → small_qk enwik8 → mimetic enwik8 | ev-e | exp/alpha-w2a @ 7a436db | ap-32E69VJOroFKGEmgIDHNre | L4 | ~3 | ~2.4 |
| W6.2 lineage: modernweb tracked → modernweb vs the enwik8 teacher's directions | ev-g | exp/alpha-w2a @ 7a436db | ap-oHgyVsMkL1tAWDAwZKmARh | L4 | ~2 | ~1.6 |
| W2b Pythia 70M dry gate (2 arms × 20 iters, micro-batch 4); a first non-detached launch (ap-t4MQhTow9wGySh92BgnMn1) cancelled its spawned arms when the entrypoint returned, ≈ $0 | prism-alpha-pythia | exp/alpha-w2b @ 5419d2d | ap-5m8iXEXAehLF84LPtXAHab | L4 | ~0.3 | <1 |
| W4 wave 2, LR 1e-5, {β0, KL 0.04, anchor 0.0025} × {recycle:4, stale:4}, tag `w4lr1e5` | prism-alpha-rl | exp/alpha-w4 @ 7dbc332 | ap-5GetjdeQ7dstIqiy8zFZ8D, ap-A1pEXApW5SCyPBJrNgjXg4, ap-3GbwPoUTy46LuQHmK4ZMi4, ap-Ji7pkMgmHdOQ0qKLRnN3WX, ap-jevZpoqaHjFU3T9L3olW5C, ap-KFm9KFWqhR7RsmIuSr1geo | A100-40 | 6 × ~1.2 | ~15 |

Projected session spend ≈ $24 (≈ $9 L4/A10 + ≈ $15 A100); cumulative since 10-22 ≈ $62
of the $100 cap. GPU class: W1's last five seed-1339 arms ran on an A10, not an L4 (their
step-time ratios are excluded from RESULTS §18); four W4 wave-2 arms were placed on
A100-80GB cards although the runner requests A100-40GB (`config.device` in each artifact). Actual figures: from the Modal billing page (the CLI does not expose
billing).

- **D14 (2026-11-02)** W4 on-policy arms (wave 1, LR 1e-6, 3 arms) are **not run**: the
  measured on-policy step is ~36 s on the A100-40 (vs 7.3 s recycled), so the three arms
  cost ≈ $21, not the ≈ $4 the handoff estimated. The user chose the LR 1e-5 wave only.
- **D15 (2026-11-02)** The harnesses' `median` is `sorted(v)[n // 2]`: correct for
  3 seeds, but for 2 seeds it returns the **upper** value. Every number labelled
  "median of 2" that was read from a partial artifact's `summary` (the W1 facts table in
  RESULTS §18 and the WS case-2 49×) was the better seed, not a median. The W5 §21 table
  used means of 2. All partial citations below give per-seed values instead. The code is
  unchanged (the 3-seed artifacts are correct).
- **D16 (2026-11-02)** D12 resolved by the 3-seed runs: the strongest W2a control is
  `small_qk` on modern web (1.4778 vs the tuned baseline's 1.4848, lower in 3/3 seeds) and
  the tuned baseline on enwik8. `dirs_only`'s ratio against small_qk was recomputed
  cross-artifact with `prism_metrics.steps_to_target_ratio` at small_qk's per-seed best:
  3.2× (2.5–3.2). The pre-registered below-1.2× rule does not fire on either corpus.
- **D17 (2026-11-02)** Completed partials are committed under their completion stamp
  when the harness filename collides with a committed partial; the partial stays
  (`results/README.md`, "Naming").
