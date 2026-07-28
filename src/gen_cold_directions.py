"""gen_cold_directions.py — synthesize NEVER-TRAINED directions: analytic
orthonormal DCT-II bases, low-frequency-first, for the exact student
architecture. No model, no data, no gradients touched these — pure structure.

The cold-assembly control (RESULTS §11/§13 context): trained directions carry
the from-scratch transfer (Run R); ~random directions don't (Run S). This file
builds the third cell — directions that are *structured but never trained* —
to test whether the load-bearing property is training or mere well-formedness.
Leading singular vectors of trained transformers are empirically smooth;
DCT columns are the canonical smooth orthonormal family, ordered smoothest
first, so index-paired EigenTransfer blends fresh leading directions toward
smooth analytic ones.

    python gen_cold_directions.py --out ../fingerprints/cold-dct

Writes directions.pt (~85MB — gitignored, regenerated on demand; deterministic
with no seed, it is pure trigonometry). Pair with a spectra.json (e.g. the
committed GPT-2 fingerprint) and run:

    python prism_eval.py --corpus=modernweb --method=dirs_only \
        --fingerprint=../fingerprints/cold-dct ...
"""
import argparse
import math
import os

import torch

from model import GPTConfig, GPT
from prism_init import classify_nanogpt_param


def dct_basis(n):
    """Orthonormal DCT-II basis matrix (n×n), columns ordered by frequency
    (column 0 = constant). Columns are orthonormal by construction."""
    i = torch.arange(n, dtype=torch.float64).unsqueeze(1)      # sample index
    j = torch.arange(n, dtype=torch.float64).unsqueeze(0)      # frequency
    B = torch.cos(math.pi * (i + 0.5) * j / n)
    B[:, 0] *= math.sqrt(1.0 / n)
    B[:, 1:] *= math.sqrt(2.0 / n)
    return B


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_head', type=int, default=6)
    p.add_argument('--n_embd', type=int, default=384)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--vocab_size', type=int, default=256)
    args = p.parse_args()

    out_pt = os.path.join(args.out, 'directions.pt')
    if os.path.exists(out_pt):
        print(f'{out_pt} already exists — nothing to do.')
        return

    model = GPT(GPTConfig(n_layer=args.n_layer, n_head=args.n_head,
                          n_embd=args.n_embd, block_size=args.block_size,
                          vocab_size=args.vocab_size, dropout=0.0, bias=False))
    bases = {}

    def basis(n):
        if n not in bases:
            bases[n] = dct_basis(n)
        return bases[n]

    directions = {}
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        group = classify_nanogpt_param(name)
        if group is None:
            continue
        m, n = param.shape
        k = min(m, n)
        U = basis(m)[:, :k].to(torch.float32).clone()        # (m, k) orthonormal cols
        Vt = basis(n)[:, :k].T.to(torch.float32).clone()     # (k, n) orthonormal rows
        directions[name] = {'U': U, 'Vt': Vt, 'group': group,
                            'shape': [int(m), int(n)]}
        print(f'  {name:45s} [{m:>5},{n:>5}] {group:>10s}  (analytic DCT)')

    os.makedirs(args.out, exist_ok=True)
    torch.save(directions, out_pt)
    print(f'Saved {len(directions)} never-trained direction frames → {out_pt}')


if __name__ == '__main__':
    main()
