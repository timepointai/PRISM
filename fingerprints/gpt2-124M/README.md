# The GPT-2 fingerprint

`spectra.json` is the spectral fingerprint of **OpenAI's GPT-2 124M** — the
group-averaged, DCT-8-compressed singular-value distributions of its trained
weight matrices: 5 weight groups × 8 coefficients = **40 numbers (~160 bytes as
float32)**. This is the *entire* payload the `gpt2` arm of the modern-web
experiment transplants into a fresh model.

- Extracted 2026-07-27 with `src/prism_extract.py --hf gpt2` (transformers
  5.9.0, HF checkpoint `gpt2`), spectra file copied verbatim; the directions
  (U/Vᵀ) are deliberately **not** kept — they don't cross model sizes, and the
  point of this fingerprint is that the spectrum alone might.
- Use: `prism_eval.py --method=spectral_only --fingerprint=../fingerprints/gpt2-124M`
  (spectrum imprint + Mod Wheel only; no EigenTransfer, no teacher training).
- GPT-2 weights: © OpenAI, released under a modified MIT license. What is
  redistributed here is 40 aggregate statistics of those weights, for
  reproducibility of the committed results.
