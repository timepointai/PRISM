# Alpha moves — status

*Session started 2026-10-21. Brief: [`ALPHA-MOVES-BRIEF.md`](ALPHA-MOVES-BRIEF.md).
Session parameters: `TIER_A_BUDGET_USD=300`, `PER_RUN_CAP_USD=40`, `AUTHORIZE_TIER_B=no`,
`MERGE_POLICY=pr`, compute = Modal (profile `realityinspector`, authenticated).
This file is the agent's memory: updated after every workstream.*

## Per-move status

| Move | Workstream | Status | Verdict (one sentence) | Artifacts |
|---|---|---|---|---|
| — | W0 shared infrastructure | **done** (code), golden check identical, selftest 67/67 | Anchor object, Muon/out_eps/μP optimizer builder, metrics, overlap tracker, recall v2, competitor hooks — all default-off. | — |
| 1 | W1 injection-dial head-to-head | **tuning running** (seed 1336, facts + catalog, Modal) | — | — |
| 2a | W2a core-bet controls (probe scale) | sweep driver built; sweeps launching | — | — |
| 2b | W2b Pythia ladder (G1 port, runner) | **built + merged** (prism_neox, pile_mmap, pythia_train, gated runner; 18 selftests); dry run pending; G2+ Tier B | — | — |
| 3 | W3 cross-size SVD projection | **built + merged** (prism_crosssize, WS/FRONT builders, prism_eval knobs, runner, CROSS-SIZE.md, preprint draft); launching cases 1–4 | — | — |
| 4 | W4 RL anchor pilot | not started | — | — |
| 5 | W5 sparse prior / COUNT control | **built + merged** (prism_prior 5.93 GB enwik8 c6 tables reproduce Act 1 to 1e-4; COUNT embedding; 8-arm harness); launching | — | — |
| 6 | W6 packaging, lineage decay, prior art | not started | — | — |
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

## Spend ledger

| launch | tier | GPU | est. hours | est. USD | actual USD | status |
|---|---|---|---|---|---|---|
| W1 tuning, facts rig, seed 1336, 21 arms (`prism_modal_alpha.py --rig facts --slot a --tag w1tune`, commit 630f181) | A | L4 | 1.9 | 1.55 | pending | running (ap-nayF5Hz8sMFe4G44aKTnHC) |
| W1 tuning, catalog rig, seed 1336, 21 arms (`--rig catalog --slot b --tag w1tune`) | A | L4 | 1.9 | 1.55 | pending | running (ap-vRlsFc7SZGjYXLZXJLNAp9) |
| W1 tuning facts — relaunch after a container preemption at arm lora_r4_t1 (resumes banked arms; runner now clears stale locks) | A | L4 | ~1.2 (remaining) | ~1.0 | pending | running (fc-01M5GVZF7XFXR6PDMWY4H3X1P2) |
| W2a sweeps (seed 1336): modernweb baseline (ev-a), modernweb dirs_only (ev-b), enwik8 baseline (ev-c), enwik8 dirs_only (ev-d), Muon muon_lr for the baseline on modernweb (ev-e) and enwik8 (ev-f) | A | L4 | 0.9+1.1+0.9+1.1+0.3+0.3 = 4.6 | 3.66 | pending | running |

Modal prices used for estimates (2026-10-21, modal.com/pricing): L4 $0.80/h,
A10 $1.10/h, L40S $1.95/h, A100-40 $2.10/h, A100-80 $2.50/h, H100 $3.95/h.

## Open questions for the user

- (none yet)
