"""prism_prior.py — a training-time SPARSE interpolated n-gram prior, and the
COUNT-style parameter-free input embedding built on it (alpha-moves W5).

The prior. Orders 0..C of maximum-likelihood next-token distributions counted
over a corpus, combined by Jelinek–Mercer interpolation exactly as
`ngram_floor.py` does (p_k = μ_k·ML_k + (1−μ_k)·p_{k−1} where the order-k
context was seen; p_{k−1} otherwise; order −1 is uniform). Storage per order:
sorted int64 context keys (base-V, oldest byte first) and an (n_ctx, V) fp16
row table of ML probabilities — exact, no hashing. `SparsePrior.logp(X)`
gathers (B,T,V) log-probs on the training device from IN-WINDOW context only:
at position t inside a window, orders k > t have no context and count as
unseen, so the mixture backs off to the orders that fit (position 0 gets the
unigram). That is the block-edge backoff `train.py`'s dense `_prior_logp`
lacked — the first C−1 positions are no longer uniform.

The COUNT embedding (arXiv 2610.20456 §2, our re-implementation — not the
authors' code; see docs/alpha/W5-NOTES.md for every deviation):

    f_t = (1/J) Σ_j Σ_{k ≤ K} S(w_k | h_{t,j}) · B[w_k]

B is the LM's unembedding (nanoGPT's tied `wte`), h_{t,j} the longest STORED
suffix of the context at the j-th order, S the stupid-backoff score (count
ratio for observed continuations, 0.4·S(w|h[1:]) otherwise; the empty history
scores unigram frequency), kept for the top-K continuations and used without
renormalisation. h^(0)_t = f_t + e_t, then embedding amplification
√d·LayerNorm(·) (the paper's §3.3 formula; we use a parameter-free LayerNorm so
the arm adds zero parameters and the checkpoint schema is unchanged).
Installed by replacing `model.forward` (model.py and the state_dict untouched).
"""
import json
import math
import os
import types

import numpy as np
import torch
import torch.nn.functional as F


# ── keys ──

def keys_for(arr, k, pos, V):
    """int64 key of the k symbols ENDING at pos-1 (the order-k context of the
    symbol at pos), oldest first: Σ_j arr[pos-k+j]·V^(k-1-j). k=0 → zeros."""
    out = np.zeros(len(pos), dtype=np.int64)
    for j in range(k):
        out = out * np.int64(V) + arr[pos - k + j].astype(np.int64)
    return out


# ── building ──

def build_tables(arr, C, V, c_min=1, log=print, row_dtype=np.float16):
    """Per-order sorted context keys + fp16 ML row tables from a uint8/uint16
    array. Returns dict order -> (keys int64[n_ctx], rows fp16[n_ctx, V], n_ctx,
    n_grams, total_count)."""
    arr = np.asarray(arr)
    tables = {}
    for k in range(C + 1):
        pos = np.arange(k, len(arr))
        gram = keys_for(arr, k, pos, V) * np.int64(V) + arr[pos].astype(np.int64)
        gk, gc = np.unique(gram, return_counts=True)
        ctx = gk // np.int64(V)
        nxt = (gk % np.int64(V)).astype(np.int64)
        cku, inv = np.unique(ctx, return_inverse=True)
        cc = np.bincount(inv, weights=gc)                     # context totals
        if c_min > 1:
            keep_ctx = cc >= c_min
            remap = -np.ones(len(cku), dtype=np.int64)
            remap[keep_ctx] = np.arange(int(keep_ctx.sum()))
            keep = keep_ctx[inv]
            inv = remap[inv[keep]]
            gc, nxt, cc, cku = gc[keep], nxt[keep], cc[keep_ctx], cku[keep_ctx]
        rows = np.zeros((len(cku), V), dtype=row_dtype)
        rows[inv, nxt] = (gc / cc[inv]).astype(row_dtype)
        tables[k] = (cku.astype(np.int64), rows, int(len(cku)), int(len(gk)), int(gc.sum()))
        mb = rows.nbytes / 1e6 + cku.nbytes / 1e6
        log(f'  order {k}: {len(cku):,} contexts, {len(gk):,} distinct (k+1)-grams, '
            f'{mb:,.1f} MB ({np.dtype(row_dtype).name} rows + int64 keys)')
    return tables


def _lookup_np(tables, arr, k, pos, V):
    """(ML prob of arr[pos] under order k, seen mask) — numpy, sequential scoring."""
    keys, rows = tables[k][0], tables[k][1]
    ctx = keys_for(arr, k, pos, V)
    i = np.clip(np.searchsorted(keys, ctx), 0, len(keys) - 1)
    hit = keys[i] == ctx
    p = rows[i, arr[pos].astype(np.int64)].astype(np.float64)
    return np.where(hit, p, 0.0), hit


def mix_np(mls, hits, mus, V):
    p = np.full(len(mls[0]), 1.0 / V)
    for k in range(len(mls)):
        pk = mus[k] * mls[k] + (1 - mus[k]) * p
        p = np.where(hits[k], pk, p)
    return p


def tune_mus(tables, arr, dev_pos, C, V, grid=None, sweeps=3, log=print):
    """Greedy bottom-up μ tuning on dev positions (ngram_floor.py's procedure)."""
    grid = grid or [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    mls, hits = [], []
    for k in range(C + 1):
        m, h = _lookup_np(tables, arr, k, dev_pos, V)
        mls.append(m)
        hits.append(h)

    def bpb(mus):
        return float(-np.log2(np.clip(mix_np(mls, hits, mus, V), 1e-12, None)).mean())
    mus = [0.5] * (C + 1)
    for _ in range(sweeps):
        for k in range(C + 1):
            mus[k] = min(grid, key=lambda g: bpb(mus[:k] + [g] + mus[k + 1:]))
    log(f'  tuned mus {mus} · dev bits/symbol {bpb(mus):.4f}')
    return mus, bpb(mus)


def save_prior(path, tables, mus, C, V, meta):
    obj = {'schema': 'prism-sparse-prior/1', 'C': C, 'V': V, 'mus': list(map(float, mus)),
           'meta': meta,
           'keys': {str(k): torch.from_numpy(tables[k][0]) for k in range(C + 1)},
           'rows': {str(k): torch.from_numpy(tables[k][1]) for k in range(C + 1)},
           'n_ctx': {str(k): tables[k][2] for k in range(C + 1)},
           'n_grams': {str(k): tables[k][3] for k in range(C + 1)}}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(obj, path)
    with open(path + '.json', 'w') as f:
        json.dump({k: v for k, v in obj.items() if k not in ('keys', 'rows')}, f, indent=2)


# ── the runtime object ──

class SparsePrior:
    def __init__(self, obj, device, vocab_size=None):
        self.C, self.V = int(obj['C']), int(obj['V'])
        if vocab_size is not None:
            assert vocab_size == self.V, f'prior vocab {self.V} != model vocab {vocab_size}'
        self.mus = [float(m) for m in obj['mus']]
        self.meta = obj.get('meta', {})
        self.device = device
        self.keys = {k: obj['keys'][str(k)].to(device) for k in range(self.C + 1)}
        self.rows = {k: obj['rows'][str(k)].to(device) for k in range(self.C + 1)}
        self.n_ctx = {k: int(obj['n_ctx'][str(k)]) for k in range(self.C + 1)}
        self.bytes = sum(t.numel() * t.element_size() for d in (self.keys, self.rows) for t in d.values())

    def describe(self):
        return (f"orders 0..{self.C}, V={self.V}, contexts " +
                ', '.join(f'{k}:{self.n_ctx[k]:,}' for k in range(1, self.C + 1)) +
                f", {self.bytes / 1e9:.2f} GB on {self.device}, mus={self.mus}" +
                (f", c_min={self.meta.get('c_min')}" if self.meta.get('c_min', 1) > 1 else ''))

    @torch.no_grad()
    def _rows_hits(self, X):
        """Per-order (ML rows (B,T,V) float32, hit mask (B,T)) from IN-WINDOW context.
        Orders longer than the available in-window context are misses."""
        B, T = X.shape
        X = X.to(self.device).long()
        out = []
        key = torch.zeros(B, T, dtype=torch.long, device=self.device)
        pw = 1
        # train.py's convention (the dense `_prior_logp`): the distribution at position t
        # predicts Y[t] = X[t+1] from the k symbols ENDING AT t, X[t-k+1..t] — the current
        # symbol is the newest context symbol. (A shift by k instead of k-1 here would
        # predict the CURRENT symbol, which is what the voided first W5 run did.)
        for k in range(self.C + 1):
            if k > 0:
                sh = k - 1
                shifted = X if sh == 0 else torch.cat(
                    [torch.zeros(B, sh, dtype=X.dtype, device=X.device), X[:, :T - sh]], dim=1)
                key = key + shifted * pw               # oldest symbol gets the highest weight
                pw *= self.V
            keys = self.keys[k]
            i = torch.searchsorted(keys, key).clamp_(0, keys.numel() - 1)
            hit = keys[i] == key
            if k > 1:
                hit[:, :k - 1] = False                # block edge: fewer than k symbols seen so far
            rows = self.rows[k][i].float()           # (B,T,V)
            out.append((rows, hit))
        return out

    @torch.no_grad()
    def probs(self, X):
        """(B,T,V) interpolated next-symbol probabilities (float32, rows sum to 1)."""
        rh = self._rows_hits(X)
        B, T = X.shape
        p = torch.full((B, T, self.V), 1.0 / self.V, device=self.device)
        for k, (rows, hit) in enumerate(rh):
            pk = self.mus[k] * rows + (1.0 - self.mus[k]) * p
            p = torch.where(hit.unsqueeze(-1), pk, p)
        return p

    @torch.no_grad()
    def logp(self, X):
        return torch.log(self.probs(X).clamp_min(1e-12))

    @torch.no_grad()
    def backoff_scores(self, X, orders):
        """Stupid-backoff scores S_n(w|h) for each requested order n, with the
        'longest stored suffix' rule: on a miss at order n the scores of the
        longest stored shorter suffix are used unscaled; on a hit, observed
        continuations score C(h,w)/C(h) and unobserved ones 0.4·S(w|h[1:]).
        Returns {n: (B,T,V)}."""
        rh = self._rows_hits(X)
        S = rh[0][0]                                  # order 0: unigram frequencies
        out = {}
        if 0 in orders:
            out[0] = S
        for n in range(1, self.C + 1):
            rows, hit = rh[n]
            backed = torch.where(rows > 0, rows, 0.4 * S)
            S = torch.where(hit.unsqueeze(-1), backed, S)
            if n in orders:
                out[n] = S
        return out

    # ── sequential (full left context) scoring, for Act-1-style numbers ──
    def score_sequential(self, arr, start=None, end=None):
        """bits/symbol of arr[start:end] with full left context (numpy, CPU),
        ngram_floor.py's protocol. Returns (bpb, n_scored)."""
        arr = np.asarray(arr)
        start = self.C if start is None else max(start, self.C)
        end = len(arr) if end is None else end
        pos = np.arange(start, end)
        tables = {k: (self.keys[k].cpu().numpy(), self.rows[k].cpu().numpy()) for k in range(self.C + 1)}
        mls, hits = [], []
        for k in range(self.C + 1):
            m, h = _lookup_np(tables, arr, k, pos, self.V)
            mls.append(m)
            hits.append(h)
        p = mix_np(mls, hits, self.mus, self.V)
        return float(-np.log2(np.clip(p, 1e-12, None)).mean()), int(len(pos))


def load_sparse_prior(path, device, vocab_size=None):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    assert obj.get('schema') == 'prism-sparse-prior/1', f'not a sparse prior: {path}'
    return SparsePrior(obj, device, vocab_size=vocab_size)


# ── COUNT-style input embedding ──

def _parse_cfg(spec):
    out = {}
    for part in (spec or '').split(','):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(':', 1)
        out[k.strip()] = v.strip()
    return out


def count_embedding(prior, X, B_rows, K=8, orders=(1, 2, 3, 4, 5, 6)):
    """f_t per Eq. 1: mean over orders of the top-K stupid-backoff-weighted sum of
    unembedding rows. B_rows: (V, d) — the (tied) unembedding; gradient flows
    through it (the paper does not detach; see W5-NOTES)."""
    orders = [n for n in orders if n <= prior.C]
    S_by = prior.backoff_scores(X, set(orders))
    f = None
    for n in orders:
        S = S_by[n]                                            # (B,T,V) float32
        if K < S.shape[-1]:
            topv, topi = S.topk(K, dim=-1)
            Sk = torch.zeros_like(S).scatter_(-1, topi, topv)
        else:
            Sk = S
        fj = Sk.to(B_rows.dtype) @ B_rows                      # (B,T,d)
        f = fj if f is None else f + fj
    return f / len(orders)


def install_count_embedding(model, prior, cfg_spec=''):
    """Replace model.forward with nanoGPT's forward plus the COUNT embedding on
    the input side. No new parameters; state_dict unchanged. Config
    'K:8,amp:sqrt_d,orders:1-6' (amp: sqrt_d | none)."""
    cfg = _parse_cfg(cfg_spec)
    K = int(cfg.get('K', 8))
    amp = cfg.get('amp', 'sqrt_d')
    o = cfg.get('orders', '1-6')
    lo, hi = (int(x) for x in o.split('-')) if '-' in o else (int(o), int(o))
    orders = tuple(range(lo, hi + 1))
    assert amp in ('sqrt_d', 'none'), f'count amp must be sqrt_d|none, got {amp}'
    d = model.config.n_embd

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        f = count_embedding(prior, idx, self.lm_head.weight, K=K, orders=orders)
        h0 = tok_emb + f.to(tok_emb.dtype)
        if amp == 'sqrt_d':
            h0 = math.sqrt(d) * F.layer_norm(h0.float(), (d,)).to(tok_emb.dtype)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(h0 + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    model.forward = types.MethodType(forward, model)
    model._count_embed = {'K': K, 'amp': amp, 'orders': orders}
    return model
