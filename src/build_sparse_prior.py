"""build_sparse_prior.py — count the sparse interpolated n-gram prior
(prism_prior.py) over a token file and save it as a .pt.

    python build_sparse_prior.py --train data/shakespeare_eval/train.bin \
        --meta data/shakespeare_char/meta.pkl --context 6 --out .prism_cache/sparse/c6.pt

The last --dev_bytes of the counting data are held out of the counts and used
to tune the interpolation weights μ_k greedily (ngram_floor.py's procedure)
unless --mus is given: 'act1' reuses the enwik8 weights tuned for Act 1
(results/ngram_floor_20260728T202028Z.json), or a comma list. Reports the
prior's bits/symbol on the dev slice and, with --val, sequentially on a
validation file (full left context, the Act-1 protocol) — the prior's own
number, never a PRISM result.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

from prism_prior import build_tables, tune_mus, save_prior, SparsePrior

SRC = os.path.dirname(os.path.abspath(__file__))
ACT1 = os.path.join(os.path.dirname(SRC), 'results', 'ngram_floor_20260728T202028Z.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, help='uint16 .bin to count over')
    ap.add_argument('--meta', required=True, help='meta.pkl (vocab_size)')
    ap.add_argument('--context', type=int, default=6, help='max order C (orders 0..C)')
    ap.add_argument('--c_min', type=int, default=1, help='drop contexts seen < c_min times (1 = exact)')
    ap.add_argument('--row_dtype', default='float16', choices=['float16', 'float32'])
    ap.add_argument('--dev_bytes', type=int, default=1_000_000)
    ap.add_argument('--mus', default='auto', help="'auto' (tune on dev) | 'act1' | 'm0,m1,...'")
    ap.add_argument('--val', default='', help='optional val .bin for a sequential bits/symbol number')
    ap.add_argument('--val_max', type=int, default=5_000_000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    V = int(pickle.load(open(args.meta, 'rb'))['vocab_size'])
    data = np.array(np.memmap(args.train, dtype=np.uint16, mode='r'))
    assert data.max() < V
    dev_bytes = min(args.dev_bytes, len(data) // 10)
    count_end = len(data) - dev_bytes
    counting = data[:count_end]
    C = args.context
    print(f'counting on {count_end:,} symbols (V={V}, orders 0..{C}, c_min={args.c_min}); '
          f'dev {dev_bytes:,} held out')
    t0 = time.time()
    tables = build_tables(counting, C, V, c_min=args.c_min, row_dtype=getattr(np, args.row_dtype))
    print(f'  counted in {time.time() - t0:.1f}s')

    dev_pos = np.arange(count_end + C, len(data))
    if args.mus == 'auto':
        mus, dev_bpb = tune_mus(tables, data, dev_pos, C, V)
        mus_src = 'tuned on dev (greedy, ngram_floor.py grid)'
    else:
        if args.mus == 'act1':
            mus = json.load(open(ACT1))['tuned_mus']
            mus_src = f'Act 1 enwik8 weights ({os.path.basename(ACT1)})'
        else:
            mus = [float(x) for x in args.mus.split(',')]
            mus_src = 'given'
        assert len(mus) >= C + 1, f'need {C + 1} mus, got {len(mus)}'
        mus = mus[:C + 1]
        dev_bpb = None
    meta = {'train': args.train, 'counting_symbols': int(count_end), 'dev_symbols': int(dev_bytes),
            'c_min': args.c_min, 'row_dtype': args.row_dtype, 'mus_source': mus_src,
            'rows_mb': sum(tables[k][1].nbytes for k in tables) / 1e6,
            'keys_mb': sum(tables[k][0].nbytes for k in tables) / 1e6}
    save_prior(args.out, tables, mus, C, V, meta)
    prior = SparsePrior(__import__('torch').load(args.out, map_location='cpu', weights_only=False), 'cpu')
    if dev_bpb is None:
        dev_bpb, _ = prior.score_sequential(data, start=count_end + C)
    meta['dev_bits_per_symbol'] = round(float(dev_bpb), 4)
    print(f'  mus {mus} ({mus_src}) · dev {dev_bpb:.4f} bits/symbol')
    if args.val:
        val = np.array(np.memmap(args.val, dtype=np.uint16, mode='r'))[:args.val_max]
        vb, n = prior.score_sequential(val)
        meta['val_bits_per_symbol'] = round(vb, 4)
        meta['val_symbols'] = n
        print(f'  VAL (sequential, full left context) {vb:.4f} bits/symbol over {n:,} — the PRIOR\'s number')
    # rewrite with the scores in meta
    save_prior(args.out, tables, mus, C, V, meta)
    print(f'  saved {args.out}: {prior.describe()}')


if __name__ == '__main__':
    main()
