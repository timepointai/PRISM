# cold-dct — the never-trained fingerprint

The cold-assembly control: a full PRISM fingerprint in which **no component
was trained on anything relevant** —

- `spectra.json`: GPT-2 124M's universal spectrum (40 numbers, copied verbatim
  from [`../gpt2-124M/`](../gpt2-124M/) — trained, but never on this corpus,
  this tokenizer, or this model size; measured at parity with a native
  teacher's spectrum in RESULTS §11).
- `directions.pt`: **analytic orthonormal DCT-II bases** — pure trigonometry,
  low-frequency-first, generated on demand by
  `src/gen_cold_directions.py --out ../fingerprints/cold-dct` (~85MB, not
  committed; deterministic with no seed).

Use: `prism_eval.py --corpus=modernweb --method=dirs_only --fingerprint=../fingerprints/cold-dct`

The question it answers: is the load-bearing property of the directions
(Run R: 3.3× + better-than-baseline convergence) that they were *trained*,
or merely that they are *well-formed*? Run S (~random directions) is the
unstructured control; this is the structured-but-never-trained cell.
