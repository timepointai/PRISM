"""prism_neox.py — PRISM's operations on the GPT-NeoX / Pythia family (W2b, G1 port).

Pythia (EleutherAI) publishes architecture, exact data order, schedules and 154
checkpoints per model — including `step0`, the exact initialization each run
started from. That makes the core bet unusually clean: the PRISM arm is Pythia's
OWN published step0 init with EigenTransfer (and optionally the Spectral Imprint)
applied from the same-size final checkpoint, trained on the same token stream with
the same schedule. Only the initialization differs.

Name map (HF `GPTNeoXForCausalLM`, verified against transformers 5.9.0):

    gpt_neox.layers.<i>.attention.query_key_value.weight  (3·d, d)   group 'attention'
    gpt_neox.layers.<i>.attention.dense.weight            (d, d)     group 'attn_proj'
    gpt_neox.layers.<i>.mlp.dense_h_to_4h.weight          (4d, d)    group 'ffn_up'
    gpt_neox.layers.<i>.mlp.dense_4h_to_h.weight          (d, 4d)    group 'ffn_down'
    gpt_neox.embed_in.weight / embed_out.weight           (V, d)     group 'embedding' (UNTIED)
    biases, LayerNorms                                                skipped (1-D)

**The QKV gotcha.** NeoX packs the fused projection PER HEAD: the (3·d, d) weight
is viewed as (n_heads, 3·head_size, d) and head h's q, k, v are the contiguous
row slices [h·3hs, h·3hs+hs), [+hs, +2hs), [+2hs, +3hs)
(`qkv.view(*, -1, 3*head_size)` then `chunk(3, dim=-1)` in modeling_gpt_neox.py).
nanoGPT's c_attn is [Q; K; V] in three d-row blocks. `qkv_unpack` / `qkv_repack`
convert between the two layouts (a pure row permutation, exact round-trip); all
SVD work is done in nanoGPT's block layout so fingerprints share one convention.

Other architectural facts recorded for the reader: rotary embeddings cover
`rotary_pct` = 25% of each head's dims (position-dependent roles inside the q/k
sub-blocks — EigenTransfer treats the whole matrix, as it does for nanoGPT); the
residual is parallel (x + attn(ln1 x) + mlp(ln2 x)); embeddings are untied so
`embed_in` and `embed_out` are anchored/transferred separately.

Exactness guarantee: with align=0 and imprint=False `apply_prism_neox` touches
nothing (no SVD round-trip), so the baseline arm IS the published step0 bit for bit.
"""
import json
import math
import os
import re

import numpy as np
import torch

from prism_init import dct_expand, align_directions

GROUPS = ('attention', 'attn_proj', 'ffn_up', 'ffn_down', 'embedding')


def classify_neox_param(name):
    """Spectrum group for a GPTNeoX parameter name; None for anything PRISM skips."""
    if name.endswith('.bias') or 'layernorm' in name or 'layer_norm' in name:
        return None
    if 'attention.query_key_value' in name:
        return 'attention'
    if 'attention.dense' in name:
        return 'attn_proj'
    if 'mlp.dense_h_to_4h' in name:
        return 'ffn_up'
    if 'mlp.dense_4h_to_h' in name:
        return 'ffn_down'
    if name.endswith('embed_in.weight') or name.endswith('embed_out.weight'):
        return 'embedding'
    return None


def layer_index(name):
    m = re.search(r'\.layers\.(\d+)\.', name)
    return int(m.group(1)) if m else None


# ── QKV layout ──

def qkv_unpack(W, n_heads):
    """NeoX per-head packed (3d, d) -> nanoGPT [Q; K; V] block layout (3d, d)."""
    d3, d = W.shape
    assert d3 == 3 * d, f'expected (3d, d), got {tuple(W.shape)}'
    hs = d // n_heads
    Wh = W.reshape(n_heads, 3, hs, d)                 # head, (q|k|v), head_dim, in
    return Wh.permute(1, 0, 2, 3).reshape(3 * d, d).contiguous()


def qkv_repack(W_blocks, n_heads):
    """nanoGPT [Q; K; V] block layout (3d, d) -> NeoX per-head packed (3d, d)."""
    d3, d = W_blocks.shape
    assert d3 == 3 * d
    hs = d // n_heads
    Wb = W_blocks.reshape(3, n_heads, hs, d)
    return Wb.permute(1, 0, 2, 3).reshape(3 * d, d).contiguous()


def qkv_unpack_bias(b, n_heads):
    d = b.shape[0] // 3
    hs = d // n_heads
    return b.reshape(n_heads, 3, hs).permute(1, 0, 2).reshape(3 * d).contiguous()


# ── model access ──

def neox_matrices(model):
    """[(name, param, group)] for every PRISM-relevant 2-D weight."""
    out = []
    for name, p in model.named_parameters():
        g = classify_neox_param(name)
        if g is not None and p.dim() == 2:
            out.append((name, p, g))
    return out


def canonical_weight(name, W, n_heads):
    """The matrix PRISM analyses: block layout for QKV, as-is otherwise."""
    return qkv_unpack(W, n_heads) if 'query_key_value' in name else W


def from_canonical(name, W, n_heads):
    return qkv_repack(W, n_heads) if 'query_key_value' in name else W


# ── fingerprint extraction ──

def _dct_fit(sv, n_dct):
    """prism_extract._dct_fit (duplicated to keep this module importable without model.py):
    fit n_dct DCT coefficients to a normalised singular-value curve."""
    clipped = np.clip(np.asarray(sv, dtype=float), 0.01, None)
    target = np.log(np.exp(clipped) - 1.0 + 1e-10)
    n = len(target)
    t = np.linspace(0, np.pi, n, endpoint=False)
    basis = np.zeros((n, n_dct))
    for i in range(n_dct):
        basis[:, i] = np.cos((i + 0.5) * t)
    coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
    return coeffs.tolist()


@torch.no_grad()
def extract_neox_fingerprint(model, n_dct=8, verbose=False):
    """Per-matrix U/Vt (canonical layout) + group-averaged DCT spectra from a
    trained GPTNeoX model. Returns {'spectra': {group: coeffs},
    'directions': {name: {'U', 'Vt', 'group', 'shape'}}, 'meta': {...}}."""
    n_heads = model.config.num_attention_heads
    group_svs = {}
    directions = {}
    for name, p, g in neox_matrices(model):
        W = canonical_weight(name, p.data.float(), n_heads)
        U, s, Vt = torch.linalg.svd(W, full_matrices=False)
        s_norm = s / s.max() if s.max() > 0 else s
        group_svs.setdefault(g, []).append(s_norm.cpu().numpy())
        directions[name] = {'U': U.cpu(), 'Vt': Vt.cpu(), 'group': g, 'shape': list(W.shape)}
        if verbose:
            print(f'  {name:60s} {str(list(W.shape)):>16s} {g:>10s} top3=[{s_norm[0]:.3f},{s_norm[1]:.3f},{s_norm[2]:.3f}]')
    spectra = {}
    for g, svs in group_svs.items():
        max_len = max(len(s) for s in svs)
        interp = [np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(s)), s) for s in svs]
        spectra[g] = _dct_fit(np.mean(interp, axis=0), n_dct)
    meta = {'family': 'gpt_neox', 'hidden_size': model.config.hidden_size,
            'num_hidden_layers': model.config.num_hidden_layers,
            'num_attention_heads': n_heads, 'vocab_size': model.config.vocab_size,
            'n_dct': n_dct, 'qkv_layout': 'canonical [Q;K;V] blocks (qkv_unpack)',
            'n_matrices': len(directions)}
    return {'spectra': spectra, 'directions': directions, 'meta': meta}


def save_fingerprint(fp, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json.dump(fp['spectra'], open(os.path.join(out_dir, 'spectra.json'), 'w'), indent=2)
    json.dump(fp['meta'], open(os.path.join(out_dir, 'meta.json'), 'w'), indent=2)
    torch.save(fp['directions'], os.path.join(out_dir, 'directions.pt'))
    return out_dir


def load_fingerprint(path):
    if isinstance(path, dict):
        return path
    return {'spectra': json.load(open(os.path.join(path, 'spectra.json'))),
            'directions': torch.load(os.path.join(path, 'directions.pt'), map_location='cpu',
                                     weights_only=False),
            'meta': json.load(open(os.path.join(path, 'meta.json')))
            if os.path.exists(os.path.join(path, 'meta.json')) else {}}


# ── application ──

@torch.no_grad()
def apply_prism_neox(model, fingerprint, align=0.75, imprint=True, lam=1.0,
                     align_mode='linear', align_topk=0, groups=None, verbose=True):
    """EigenTransfer (+ Spectral Imprint) onto a GPTNeoX model in place.

    For each 2-D weight in a transferred group: W = U_f s_f V_fᵀ (canonical layout);
    U, V are rotated `align` of the way toward the fingerprint's directions
    (prism_init.align_directions — index-paired blend + re-orthogonalisation by
    default); the spectrum is the fingerprint's group DCT spectrum rescaled to
    W's Frobenius norm when imprint=True, else W's own singular values.

    align == 0 and imprint == False is an exact no-op (nothing is touched), so
    the baseline arm is the published step0 bit for bit. Returns the count of
    shaped matrices."""
    fp = load_fingerprint(fingerprint)
    spectra, dirs = fp['spectra'], fp['directions']
    n_heads = model.config.num_attention_heads
    groups = set(groups) if groups else set(GROUPS)
    if align <= 0 and not imprint:
        if verbose:
            print('[prism-neox] align=0, imprint=False: no-op (published init kept exactly)')
        return 0
    n_shaped = 0
    for name, p, g in neox_matrices(model):
        if g not in groups or g not in spectra:
            continue
        W = canonical_weight(name, p.data.float(), n_heads)
        U_f, s_f, Vt_f = torch.linalg.svd(W, full_matrices=False)
        if imprint:
            frob = torch.norm(W, 'fro').item()
            target = torch.tensor(dct_expand(np.array(spectra[g]), len(s_f)),
                                  dtype=s_f.dtype, device=s_f.device)
            flat = torch.ones_like(s_f)
            blended = torch.clamp(flat + lam * (torch.clamp(target, min=0.01) - flat), min=0.01)
            s_new = blended * (frob / torch.norm(blended).item())
        else:
            s_new = s_f
        U_use, Vt_use = U_f, Vt_f
        if align > 0 and name in dirs:
            U_pre, Vt_pre = dirs[name]['U'].to(W.device), dirs[name]['Vt'].to(W.device)
            if Vt_pre.shape == Vt_f.shape:
                Vt_use = align_directions(Vt_f, Vt_pre, align, mode=align_mode, topk=align_topk, rows=True)
            if U_pre.shape == U_f.shape:
                U_use = align_directions(U_f, U_pre, align, mode=align_mode, topk=align_topk, rows=False)
        W_new = U_use @ torch.diag(s_new) @ Vt_use
        p.data.copy_(from_canonical(name, W_new, n_heads).to(p.dtype))
        n_shaped += 1
    if verbose:
        print(f'[prism-neox] shaped {n_shaped} matrices: EigenTransfer align={align}'
              + (f', Spectral Imprint lam={lam}' if imprint else ', NO imprint (own spectrum kept)')
              + f', groups={sorted(groups)}')
    return n_shaped


# ── checkpoints ──

def load_neox(path_or_repo, revision=None, dtype=torch.float32, device='cpu'):
    """Load a GPTNeoXForCausalLM (local dir or HF repo id + revision)."""
    from transformers import GPTNeoXForCausalLM
    kw = dict(torch_dtype=dtype)
    if revision:
        kw['revision'] = revision
    m = GPTNeoXForCausalLM.from_pretrained(path_or_repo, **kw)
    return m.to(device)


def synthetic_neox_config(hidden=64, layers=2, heads=4, vocab=128, ctx=64):
    from transformers import GPTNeoXConfig
    return GPTNeoXConfig(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                         intermediate_size=4 * hidden, vocab_size=vocab,
                         max_position_embeddings=ctx, rotary_pct=0.25,
                         use_parallel_residual=True, tie_word_embeddings=False)
