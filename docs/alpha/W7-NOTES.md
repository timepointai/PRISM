# W7 — cheap mechanism probes (notes, 2026-11-02)

The brief lists three probes "only if budget and time remain". D13 forbids new GPU
work without the user's OK, so this session ran only the CPU-only half of one probe
and lists the GPU halves with their cost.

## W7.2 fingerprint redundancy, CPU half: how well does the stored imprint describe the spectrum?

`w7_spectrum_fit.py` (repo root; analysis only, nothing under `src/` changed) takes a
fingerprint's per-matrix singular values, averages them per group exactly as
`prism_extract.py` does, and compares three descriptions of each group's normalised
curve:

- the stored 8-coefficient DCT, expanded with `prism_init.dct_expand`, the function
  the Spectral Imprint uses (`prism_init.py:453`);
- a 2-parameter power law, (1 + i/τ)^(−α);
- a 2-parameter stretched exponential, exp(−(i/λ)^β).

These are the 2,000-step teachers of the W2a/lineage rig, seed 1337, enwik8 and modern
web. Artifact: [`results/w7_specfit_20261102T214437Z.json`](../../results/w7_specfit_20261102T214437Z.json).

| group | relative L2 error, DCT-8 as imprinted (enwik8 / modern web) | power law (2 params) | stretched exp. (2 params) |
|---|---|---|---|
| attention | 0.74 / 0.74 | 0.11 / 0.11 | **0.03 / 0.04** |
| attn_proj | 0.28 / 0.36 | 0.06 / 0.10 | **0.03 / 0.04** |
| embedding | 1.12 / 1.17 | 0.21 / 0.25 | **0.10 / 0.14** |
| ffn_up | 0.95 / 0.94 | 0.22 / 0.22 | **0.12 / 0.12** |
| ffn_down | 0.42 / 0.38 | **0.04 / 0.04** | 0.05 / 0.08 |

**What this shows (reconstruction only, high confidence: it is deterministic):**

1. **The imprinted spectrum is not the teacher's spectrum.** The DCT-II basis is
   sampled at t ∈ [0, π), and every basis function cos((k+½)t) goes to 0 as t → π.
   So the expansion's tail is pinned near softplus(0) whatever the coefficients are,
   while the true tail is ~0.01–0.07 of the top value. After `dct_expand`
   renormalises to its maximum, the **smallest** singular values become the
   **largest** in three of five groups (attention, embedding and ffn_up have their
   maximum at the last index), and the head is flattened (embedding: 0.79 at index 0
   against a true 1.0). Both teachers show the same argmax pattern. The imprint
   blends this curve with a flat one (`flat + λ·(shaped − flat)`), and `train.py`
   calls it with λ = 1.0, so the student receives the full curve, rising tail
   included, rescaled to the fresh init's Frobenius norm.
2. **Two parameters describe the actual curve better than the eight stored ones.**
   A stretched exponential is within 3–14% everywhere; the power law fits ffn_down
   best.
3. **This does not say the imprint is useless.** RESULTS §19 measures the imprint
   as implemented (`dirs_only` 3.3× vs `dirs_pure` 2.45× on modern web, AdamW). Whatever
   that gain is, it comes from *this* curve, rising tail included, not from the
   teacher's literal spectrum. Claims that "the spectrum" transfers (README Finding 5)
   are about the DCT-8 imprint as implemented. Whether a faithful spectrum would
   help more, less or not at all is untested.

## GPU halves (not run; each needs the user's OK under D13)

| probe | what would run | est. cost |
|---|---|---|
| W7.2 training half | `dirs_only` with (a) a stretched-exponential imprint and (b) a tail-corrected DCT imprint (sample t ∈ [0, π·(n−1)/n] with a boundary term, or fit in log space), 3 seeds, modern web, PROBE rig, tuned baseline in-run. Needs a default-off knob in `prism_init.py` plus the golden check. | ≈ 2 arms × 0.7 L4-h ≈ $1.2 (+ teacher reuse on the ev slot) |
| W7.1 circuit-level mining | EigenTransfer on per-head QK/OV circuit products (gauge-invariant, arXiv 2610.18471) against per-matrix SVD at the same size. `src/prism_crosssize.py` already factors circuits; a same-size path is new code. | ≈ 0.7 L4-h ≈ $0.6 per 3-seed arm |
| W7.3 SAMuon composition | the transferred spectrum as SAMuon's static per-direction step-size prior (arXiv 2608.25990). Speculative; needs a Muon variant. | ≈ $1–2 probe, plus implementation |
