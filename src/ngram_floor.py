"""ngram_floor.py — Act 1 of the reach-at-init experiment: the prior's number.

Measures what a STATIC sparse byte n-gram, counted over enwik8's 90M training
bytes, achieves on the held-out val slice — the free CPU audition that gates
any GPU spend on Act 2 (docs/NEXT-EXPERIMENTS.md, binding framing spec).
This number belongs to the n-gram, not to PRISM.

Method: maximum-likelihood byte n-grams of order 0..K, combined by
Jelinek–Mercer interpolation (p_k = mu_k*ML_k + (1-mu_k)*p_{k-1} where the
order-k context was seen; p_{k-1} otherwise). The mu_k are tuned greedily on
a dev slice held out of the COUNTING data (last 1M bytes of train), then the
tuned mixture is scored once on the val slice (bytes 90–95M) that no tuning
touched. Scoring is sequential with full left context — the training-time
windowed prior additionally needs the block-edge backoff, which this ceiling
does not include.

Everything is vectorized numpy; the whole pass runs in minutes on a laptop.
Writes results/ngram_floor_<stamp>.json. Commit it — Act 1's number is
evidence like any other.

    python ngram_floor.py --max_order 6
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(SRC, 'data', 'enwik8', 'enwik8')
RESULTS = os.path.join(os.path.dirname(SRC), 'results')


def keys_for(arr, k, pos):
    """uint64 key of the k bytes ENDING at pos-1 (the order-k context of the
    byte at pos), vectorized. k=0 returns zeros."""
    out = np.zeros(len(pos), dtype=np.uint64)
    for j in range(k):
        out = out * np.uint64(256) + arr[pos - k + j].astype(np.uint64)
    return out


def count_table(arr, k):
    """Sorted unique (context,next) keys + counts, and context keys + counts,
    for order k over the whole counting array."""
    pos = np.arange(k, len(arr))
    gram = keys_for(arr, k, pos) * np.uint64(256) + arr[pos].astype(np.uint64)
    gk, gc = np.unique(gram, return_counts=True)
    ck = gk // np.uint64(256) if k > 0 else np.zeros(1, dtype=np.uint64)
    if k > 0:
        cku, idx = np.unique(ck, return_inverse=True)
        cc = np.bincount(idx, weights=gc).astype(np.int64)
    else:
        cku, cc = np.zeros(1, dtype=np.uint64), np.array([gc.sum()])
    return gk, gc.astype(np.int64), cku, cc


def ml_probs(arr, k, pos, tables):
    """ML prob of arr[pos] under order-k counts, 0 where context unseen (plus
    a seen-mask). Vectorized double searchsorted."""
    gk, gc, cku, cc = tables[k]
    gram = keys_for(arr, k, pos) * np.uint64(256) + arr[pos].astype(np.uint64)
    ctx = gram // np.uint64(256) if k > 0 else np.zeros(len(pos), dtype=np.uint64)
    gi = np.searchsorted(gk, gram)
    gi = np.clip(gi, 0, len(gk) - 1)
    ghit = gk[gi] == gram
    num = np.where(ghit, gc[gi], 0).astype(np.float64)
    ci = np.searchsorted(cku, ctx)
    ci = np.clip(ci, 0, len(cku) - 1)
    chit = cku[ci] == ctx
    den = np.where(chit, cc[ci], 1).astype(np.float64)
    return np.where(chit, num / den, 0.0), chit


def mix(mls, hits, mus):
    """Interpolate order 0..K given per-position ML arrays and seen-masks."""
    p = np.full(len(mls[0]), 1.0 / 256)          # order -1: uniform
    for k in range(len(mls)):
        pk = mus[k] * mls[k] + (1 - mus[k]) * p
        p = np.where(hits[k], pk, p)
    return p


def bpb(p):
    return float(-np.log2(np.clip(p, 1e-12, None)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_order', type=int, default=6)
    ap.add_argument('--dev_bytes', type=int, default=1_000_000)
    args = ap.parse_args()
    K = args.max_order

    data = np.frombuffer(open(RAW, 'rb').read(), dtype=np.uint8)
    assert len(data) == 100_000_000
    count_end = 90_000_000 - args.dev_bytes      # counting data excludes dev
    counting = data[:count_end]
    print(f'counting on {count_end:,} bytes · dev {args.dev_bytes:,} · '
          f'val 5,000,000 (bytes 90–95M, untouched by tuning)')

    tables, sizes = {}, {}
    for k in range(K + 1):
        tables[k] = count_table(counting, k)
        sizes[k] = int(len(tables[k][0]))
        print(f'  order {k}: {sizes[k]:,} distinct (k+1)-grams')

    dev_pos = np.arange(90_000_000 - args.dev_bytes + K, 90_000_000)
    val_pos = np.arange(90_000_000 + K, 95_000_000)

    dev_ml, dev_hit, val_ml, val_hit = [], [], [], []
    for k in range(K + 1):
        m, h = ml_probs(data, k, dev_pos, tables)
        dev_ml.append(m); dev_hit.append(h)
        m, h = ml_probs(data, k, val_pos, tables)
        val_ml.append(m); val_hit.append(h)

    # Greedy bottom-up mu tuning on dev only.
    mus = [0.5] * (K + 1)
    grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    for _ in range(3):                            # a few sweeps to settle
        for k in range(K + 1):
            best = min(grid, key=lambda g: bpb(
                mix(dev_ml, dev_hit, mus[:k] + [g] + mus[k + 1:])))
            mus[k] = best
    dev_bpb = bpb(mix(dev_ml, dev_hit, mus))

    per_order = {}
    for k in range(K + 1):
        per_order[k] = round(bpb(mix(dev_ml[:k + 1], dev_hit[:k + 1],
                                     mus[:k + 1])), 4)
    val_bpb = bpb(mix(val_ml, val_hit, mus))

    print(f'\n  tuned mus: {mus}')
    print(f'  dev bpb (tuning slice): {dev_bpb:.4f}')
    for k, v in per_order.items():
        print(f'  dev bpb through order {k}: {v:.4f}')
    print(f'\n  ACT 1 — VAL bits/byte (order 0..{K} interpolated): {val_bpb:.4f}')
    print('  This number belongs to the n-gram (see the framing spec).')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    art = {
        'schema': 'prism-ngram-floor/1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'corpus': 'enwik8',
        'protocol': {'counting_bytes': int(count_end),
                     'dev': 'last 1M of train (tuning only)',
                     'val': 'bytes 90-95M, untouched by tuning',
                     'scoring': 'sequential, full left context',
                     'note': 'Act 1 of the reach-at-init framing spec: this is '
                             'the PRIOR\'s number, not a PRISM result.'},
        'max_order': K,
        'distinct_ngrams_per_order': sizes,
        'tuned_mus': mus,
        'dev_bpb': round(dev_bpb, 4),
        'dev_bpb_through_order': per_order,
        'val_bpb': round(val_bpb, 4),
        'references_bpb': {'gzip': 2.92, 'bzip2': 2.32, 'zstd19': 2.2,
                           'xz': 1.99, 'baseline_1500_steps_val': 1.9315,
                           'dirs_only_1500_steps_val': 1.7796,
                           'note': 'compressor figures are whole-file '
                                   'published numbers — different protocol'},
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f'ngram_floor_{stamp}.json')
    json.dump(art, open(path, 'w'), indent=2)
    print(f'\n  Artifact: results/{os.path.basename(path)} — COMMIT THIS FILE.')


if __name__ == '__main__':
    main()
