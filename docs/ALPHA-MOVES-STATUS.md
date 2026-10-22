# Alpha moves — status

*Session started 2026-10-21. Brief: [`ALPHA-MOVES-BRIEF.md`](ALPHA-MOVES-BRIEF.md).
Session parameters: `TIER_A_BUDGET_USD=300`, `PER_RUN_CAP_USD=40`, `AUTHORIZE_TIER_B=no`,
`MERGE_POLICY=pr`, compute = Modal (profile `realityinspector`, authenticated).
This file is the agent's memory: updated after every workstream.*

## Per-move status

| Move | Workstream | Status | Verdict (one sentence) | Artifacts |
|---|---|---|---|---|
| — | W0 shared infrastructure | **done** (code), golden check identical, selftest 67/67 | Anchor object, Muon/out_eps/μP optimizer builder, metrics, overlap tracker, recall v2, competitor hooks — all default-off. | — |
| 1 | W1 injection-dial head-to-head | building → tuning launch pending | — | — |
| 2a | W2a core-bet controls (probe scale) | not started | — | — |
| 2b | W2b Pythia ladder (G1 port, runner) | not started | — | — |
| 3 | W3 cross-size SVD projection | not started | — | — |
| 4 | W4 RL anchor pilot | not started | — | — |
| 5 | W5 sparse prior / COUNT control | not started | — | — |
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

## Spend ledger

| launch | tier | GPU | est. hours | est. USD | actual USD | status |
|---|---|---|---|---|---|---|
| (none yet) | | | | | | |

Modal prices used for estimates (2026-10-21, modal.com/pricing): L4 $0.80/h,
A10 $1.10/h, L40S $1.95/h, A100-40 $2.10/h, A100-80 $2.50/h, H100 $3.95/h.

## Open questions for the user

- (none yet)
