"""prism_crosssize.py — cross-size SVD projection (alpha-moves W3).

Nobody has projected a large LM's SVD subspaces into a smaller LM and trained it.
This module does the ambient-coordinate step that EigenTransfer needs when the
teacher and student have different widths, depths or head layouts, and writes a
FINGERPRINT DIR (spectra.json + directions.pt keyed by the STUDENT's nanoGPT
parameter names) that `prism_eval.py --fingerprint=<dir> --method=dirs_only`
consumes unchanged.

Primary operator — residual-stream PCA alignment (`align='pca'`)
  1. Run the teacher on a calibration slice; pool the residual-stream vectors at
     every block input and at the final LayerNorm's input; take the top-d_S
     principal axes P ∈ R^{d_T×d_S} (orthonormal columns).
  2. Project every matrix into the student's coordinates with P on the side(s)
     that touch the residual stream (y = x Wᵀ ⇒ W̃ = W P for residual-reading
     matrices; W̃ = Pᵀ W for residual-writing ones; W̃ = P_outᵀ W P_in in general):
       • MLP hidden width via PCA of the post-GELU hidden activations, P_h per
         layer (`hidden='pca'`), or SVD truncation of the projected c_fc
         (`hidden='svd'`): W̃_fc = P_hᵀ W_fc P, W̃_proj = Pᵀ W_proj P_h.
       • Attention heads via the gauge-invariant per-head circuit products
         (arXiv 2610.18471): M_h = W_q_hᵀ W_k_h and O_h = W_v_hᵀ W_o_hᵀ, both
         d_T×d_T; projected PᵀM_hP, PᵀO_hP; re-factored at the student head dim by
         a rank-hd_S SVD (W̃_q = Σ^½U_kᵀ, W̃_k = Σ^½V_kᵀ; W̃_v = Σ^½U_kᵀ,
         W̃_o[:,h] = V_kΣ^½). QK gets the √(hd_S/hd_T) attention-scale factor.
         When the head count shrinks, the heads with the largest projected
         circuit energy (‖PᵀM_hP‖_F · ‖PᵀO_hP‖_F) are kept.
       • Embeddings (same vocab only): wte P, wpe[:T_S] P.
  3. Depth mismatch: uniform layer selection (teacher layer ⌊i·L_T/L_S⌋ for
     student layer i; `layer_select='average'` averages the L_T/L_S-layer groups).
  4. SVD each W̃ for U/Vt → directions.pt; the spectra are the TEACHER's
     group-averaged singular-value profiles (the imprint is size-agnostic), or
     the projected matrices' with `spectra='projected'`.

Alternative operator — `align='procrustes'` (precedent arXiv 2610.15322): the
teacher's PCA coordinates are rotated onto a fresh STUDENT's residual PCA
coordinates on the same tokens by orthogonal Procrustes (R = UVᵀ of c_Tᵀc_S);
the ambient map becomes P = P_T R P_Sᵀ and everything else is unchanged.
(Requires the same tokenisation; the GPT-2 → byte-level case uses 'pca'.)

Baseline init builders (write a state_dict .pt for train.py's `init_state`):
  weight_selection — arXiv 2311.18823: first-N layers, evenly spaced element
      selection per dimension (stride ⌊t/s⌋ from index 0, the paper's worked
      example), the SAME index set for every tensor sharing that dimension
      (consistency), embeddings copied when the vocab matches, LayerNorm gains
      selected. Our re-implementation; no rescaling (none in the paper).
  front — arXiv 2603.07523 §3: stack each weight type across the teacher's
      layers into an L×out×in tensor, orthonormal 3D-DCT (scipy.fft.dctn,
      norm='ortho'), keep the low-frequency learngene block (ratio r, default
      1.0 = everything), TRUNCATE the high-frequency side to L_S×out_S×in_S (or
      zero-pad to grow), inverse DCT. No rescaling (none in the paper). Modules
      the paper leaves unselected (embeddings, LayerNorms) stay at the
      student's fresh init. Our re-implementation of the operator; the FRONT+
      spectral-regularised refinement is NOT implemented.

Teachers: a nanoGPT checkpoint (`--teacher_ckpt`) or HF `gpt2` (`--teacher_hf`,
Conv1D weights transposed as in prism_init.extract_directions; internal
matrices only, since the vocabulary differs from the byte-level student).

    python prism_crosssize.py --teacher_ckpt out-eval-teacher-s1337/ckpt.pt \
        --student_n_layer 6 --student_n_embd 384 --student_n_head 6 \
        --calib data/shakespeare_teacher --out .prism_cache/xsize --align pca
    python prism_crosssize.py --teacher_hf gpt2 --calib_text data/modernweb/modern.txt ...
    python prism_crosssize.py ... --builder front --out_state init_front.pt
"""
import argparse
import json
import math
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from model import GPTConfig, GPT
from prism_init import classify_nanogpt_param
from prism_extract import _dct_fit

GROUPS = ['attention', 'attn_proj', 'ffn_up', 'ffn_down', 'embedding']


# ══════════════════════════════════════════════════════════════════════════
# Teachers — a uniform view: per-layer matrices in nn.Linear (out, in) layout
# ══════════════════════════════════════════════════════════════════════════

class Teacher:
    """Holds a teacher's matrices in nanoGPT layout plus the modules needed for
    calibration hooks. `layers[l]` = dict(c_attn, attn_proj, c_fc, mlp_proj, ln_1,
    ln_2); `wte`, `wpe`, `ln_f`; `n_head`, `n_embd`, `n_layer`, `vocab_size`."""

    def __init__(self, layers, wte, wpe, ln_f, n_head, n_embd, model=None,
                 blocks=None, ln_f_mod=None, gelus=None, source=''):
        self.layers, self.wte, self.wpe, self.ln_f = layers, wte, wpe, ln_f
        self.n_head, self.n_embd, self.n_layer = n_head, n_embd, len(layers)
        self.vocab_size = None if wte is None else wte.shape[0]
        self.model, self.blocks, self.ln_f_mod, self.gelus = model, blocks, ln_f_mod, gelus
        self.source = source

    @property
    def head_dim(self):
        return self.n_embd // self.n_head


def teacher_from_nanogpt(ckpt_path, device='cpu'):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    conf = GPTConfig(**ck['model_args'])
    m = GPT(conf)
    m.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()}, strict=False)
    m.to(device).eval()
    return teacher_from_model(m, source=ckpt_path)


def teacher_from_model(m, source='nanogpt'):
    sd = {k: v.detach().float().cpu() for k, v in m.state_dict().items()}
    layers = []
    for l in range(m.config.n_layer):
        p = f'transformer.h.{l}.'
        layers.append({'c_attn': sd[p + 'attn.c_attn.weight'], 'attn_proj': sd[p + 'attn.c_proj.weight'],
                       'c_fc': sd[p + 'mlp.c_fc.weight'], 'mlp_proj': sd[p + 'mlp.c_proj.weight'],
                       'ln_1': sd[p + 'ln_1.weight'], 'ln_2': sd[p + 'ln_2.weight']})
    return Teacher(layers, sd['transformer.wte.weight'], sd['transformer.wpe.weight'],
                   sd['transformer.ln_f.weight'], m.config.n_head, m.config.n_embd, model=m,
                   blocks=list(m.transformer.h), ln_f_mod=m.transformer.ln_f,
                   gelus=[b.mlp.gelu for b in m.transformer.h], source=source)


def teacher_from_hf(name='gpt2', device='cpu'):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained(name)
    m.to(device).eval()
    cfg = m.config
    layers = []
    for l, blk in enumerate(m.transformer.h):
        # HF Conv1D stores (in, out): transpose to nn.Linear's (out, in)
        layers.append({'c_attn': blk.attn.c_attn.weight.detach().float().cpu().T.contiguous(),
                       'attn_proj': blk.attn.c_proj.weight.detach().float().cpu().T.contiguous(),
                       'c_fc': blk.mlp.c_fc.weight.detach().float().cpu().T.contiguous(),
                       'mlp_proj': blk.mlp.c_proj.weight.detach().float().cpu().T.contiguous(),
                       'ln_1': blk.ln_1.weight.detach().float().cpu(),
                       'ln_2': blk.ln_2.weight.detach().float().cpu()})
    return Teacher(layers, m.transformer.wte.weight.detach().float().cpu(),
                   m.transformer.wpe.weight.detach().float().cpu(),
                   m.transformer.ln_f.weight.detach().float().cpu(), cfg.n_head, cfg.n_embd,
                   model=m, blocks=list(m.transformer.h), ln_f_mod=m.transformer.ln_f,
                   gelus=[b.mlp.act for b in m.transformer.h], source=f'hf:{name}')


# ══════════════════════════════════════════════════════════════════════════
# Calibration data and activation statistics
# ══════════════════════════════════════════════════════════════════════════

def calib_batches_from_bin(data_dir, block_size, n_batches=8, batch_size=8, seed=20261021):
    """Deterministic token batches from a nanoGPT train.bin."""
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_batches):
        ix = rng.integers(0, len(data) - block_size, size=batch_size)
        out.append(torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix]))
    return out


def calib_batches_from_text(path, tokenizer_name, block_size, n_batches=8, batch_size=8, seed=20261021):
    """Deterministic batches of a text file tokenised with an HF tokenizer (GPT-2 teacher)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    text = open(path, 'rb').read().decode('utf-8', errors='ignore')
    ids = tok(text[:2_000_000])['input_ids']
    ids = torch.tensor(ids, dtype=torch.long)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_batches):
        ix = rng.integers(0, len(ids) - block_size, size=batch_size)
        out.append(torch.stack([ids[i:i + block_size] for i in ix]))
    return out


@torch.no_grad()
def collect_activations(teacher, batches, device='cpu'):
    """Residual-stream samples per block input (+ ln_f input) and post-GELU MLP
    hidden samples per layer. Returns (resid: list[L+1] of (N, d), hidden: list[L] of (N, 4d))."""
    resid = [[] for _ in range(teacher.n_layer + 1)]
    hidden = [[] for _ in range(teacher.n_layer)]
    handles = []
    for i, blk in enumerate(teacher.blocks):
        handles.append(blk.register_forward_pre_hook(
            lambda mod, inp, i=i: resid[i].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu())))
        handles.append(teacher.gelus[i].register_forward_hook(
            lambda mod, inp, out, i=i: hidden[i].append(out.detach().float().reshape(-1, out.shape[-1]).cpu())))
    handles.append(teacher.ln_f_mod.register_forward_pre_hook(
        lambda mod, inp: resid[teacher.n_layer].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu())))
    for X in batches:
        teacher.model(X.to(device))
    for h in handles:
        h.remove()
    return [torch.cat(r) for r in resid], [torch.cat(h) for h in hidden]


def pca_axes(X, k):
    """Top-k principal axes (d×k, orthonormal columns) of the rows of X, plus the
    mean and the per-axis variances. When k == d no reduction is needed and the
    IDENTITY is returned: a full-rank rotation of the residual basis would change
    the function (LayerNorm is elementwise, not rotation-invariant) for no gain,
    and it is what makes the same-size case reproduce the teacher's directions."""
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    d = X.shape[1]
    if k >= d:
        var = Xc.var(0, unbiased=True)
        return torch.eye(d), mu.squeeze(0), var
    # covariance eigendecomposition via SVD of the centred data (stable, no d×d inverse)
    _, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T.contiguous(), mu.squeeze(0), (S[:k] ** 2) / max(1, Xc.shape[0] - 1)


def procrustes(A, B):
    """Orthogonal R minimising ‖A R − B‖_F for paired rows (A: N×k, B: N×k)."""
    U, _, Vt = torch.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vt


# ══════════════════════════════════════════════════════════════════════════
# Layer selection
# ══════════════════════════════════════════════════════════════════════════

def select_layers(L_T, L_S, mode='uniform'):
    """Teacher layer indices (or index groups) per student layer."""
    if mode == 'uniform':
        return [[int(i * L_T / L_S)] for i in range(L_S)]
    if mode == 'average':
        assert L_T % L_S == 0, 'average needs an integer depth ratio'
        r = L_T // L_S
        return [list(range(i * r, (i + 1) * r)) for i in range(L_S)]
    if mode == 'first':
        return [[i] for i in range(min(L_S, L_T))]
    raise ValueError(mode)


def _avg(mats):
    return mats[0] if len(mats) == 1 else torch.stack(mats).mean(0)


# ══════════════════════════════════════════════════════════════════════════
# The projection
# ══════════════════════════════════════════════════════════════════════════

def _factor_rank(Mt, k, scale=1.0):
    """Rank-k symmetric-free factorisation M ≈ A B with A = Σ^½ U_kᵀ (k×d), B = Σ^½ V_kᵀ (k×d),
    i.e. M ≈ Aᵀ B. Returns (A, B, energy)."""
    U, S, Vt = torch.linalg.svd(Mt, full_matrices=False)
    k = min(k, S.shape[0])
    s = (S[:k] * scale).clamp_min(0).sqrt()
    A = s[:, None] * U[:, :k].T
    B = s[:, None] * Vt[:k]
    return A.contiguous(), B.contiguous(), float(torch.linalg.norm(Mt))


def project_layer(layer, P, P_h, n_head_T, n_head_S, d_S, scale_qk=True):
    """Project one teacher layer's four matrices into student coordinates.
    layer: dict with c_attn (3d_T,d_T), attn_proj (d_T,d_T), c_fc (4d_T,d_T), mlp_proj (d_T,4d_T).
    P: (d_T, d_S); P_h: (4d_T, 4d_S). Returns dict(c_attn, attn_proj, c_fc, mlp_proj, heads_kept)."""
    d_T = P.shape[0]
    hd_T = d_T // n_head_T
    hd_S = d_S // n_head_S
    W = layer['c_attn']
    Wo = layer['attn_proj']
    circuits = []
    for h in range(n_head_T):
        sl = slice(h * hd_T, (h + 1) * hd_T)
        Wq, Wk, Wv = W[0 * d_T:1 * d_T][sl], W[1 * d_T:2 * d_T][sl], W[2 * d_T:3 * d_T][sl]
        Woh = Wo[:, sl]                                  # (d_T, hd_T)
        M = Wq.T @ Wk                                    # (d_T, d_T): scores = x M xᵀ
        O = Wv.T @ Woh.T                                 # (d_T, d_T): out = x O
        Mt, Ot = P.T @ M @ P, P.T @ O @ P
        circuits.append((h, Mt, Ot, float(torch.linalg.norm(Mt) * torch.linalg.norm(Ot))))
    if n_head_S > n_head_T:
        raise ValueError('student has more heads than the teacher; head growth is not implemented')
    kept = sorted(sorted(circuits, key=lambda c: -c[3])[:n_head_S], key=lambda c: c[0])
    c_attn = torch.zeros(3 * d_S, d_S)
    attn_proj = torch.zeros(d_S, d_S)
    qk_scale = math.sqrt(hd_S / hd_T) if scale_qk else 1.0   # keep x M xᵀ/√hd invariant
    for j, (h, Mt, Ot, _) in enumerate(kept):
        sl = slice(j * hd_S, (j + 1) * hd_S)
        A, B, _ = _factor_rank(Mt, hd_S, scale=qk_scale)
        c_attn[0 * d_S:1 * d_S][sl] = A                  # W̃_q_h
        c_attn[1 * d_S:2 * d_S][sl] = B                  # W̃_k_h
        A, B, _ = _factor_rank(Ot, hd_S)
        c_attn[2 * d_S:3 * d_S][sl] = A                  # W̃_v_h
        attn_proj[:, sl] = B.T                           # W̃_o[:, h]
    c_fc = P_h.T @ layer['c_fc'] @ P                     # (4d_S, d_S)
    mlp_proj = P.T @ layer['mlp_proj'] @ P_h             # (d_S, 4d_S)
    return {'c_attn': c_attn, 'attn_proj': attn_proj, 'c_fc': c_fc, 'mlp_proj': mlp_proj,
            'heads_kept': [c[0] for c in kept]}


def hidden_axes(H, W_fc_proj, k, mode='pca'):
    """P_h ∈ R^{4d_T × k}: PCA of post-GELU hidden samples, or the top-k left singular
    vectors of the residual-projected c_fc (SVD truncation)."""
    if mode == 'pca':
        return pca_axes(H, k)[0]
    U, _, _ = torch.linalg.svd(W_fc_proj, full_matrices=False)
    return U[:, :k].contiguous()


@torch.no_grad()
def project_teacher(teacher, student_cfg, batches, align='pca', hidden='pca', layer_select='uniform',
                    device='cpu', student_seed=1337, include_embeddings=None, verbose=True):
    """Project every teacher matrix into a student of `student_cfg`. Returns
    (projected: {student_param_name: W̃}, info dict)."""
    d_T, d_S = teacher.n_embd, student_cfg.n_embd
    L_T, L_S = teacher.n_layer, student_cfg.n_layer
    resid, hidden_acts = collect_activations(teacher, batches, device)
    pooled = torch.cat(resid)
    P_T, mu_T, var_T = pca_axes(pooled, d_S)
    explained = float(var_T.sum() / (pooled - pooled.mean(0)).var(0, unbiased=True).sum())
    info = {'align': align, 'hidden': hidden, 'layer_select': layer_select,
            'teacher': {'source': teacher.source, 'n_layer': L_T, 'n_embd': d_T, 'n_head': teacher.n_head},
            'student': {'n_layer': L_S, 'n_embd': d_S, 'n_head': student_cfg.n_head},
            'calib_samples': int(pooled.shape[0]), 'pca_explained_variance': explained}
    P = P_T
    if align == 'procrustes':
        # a fresh student's own residual coordinates on the SAME tokens
        torch.manual_seed(student_seed)
        stu = GPT(student_cfg).to(device).eval()
        stu_t = teacher_from_model(stu, source='fresh-student')
        s_resid, _ = collect_activations(stu_t, batches, device)
        groups = select_layers(L_T, L_S, layer_select)
        pairs_T, pairs_S = [], []
        for i, g in enumerate(groups):
            pairs_T.append(resid[g[0]])
            pairs_S.append(s_resid[i])
        pairs_T.append(resid[L_T]); pairs_S.append(s_resid[L_S])
        cT = (torch.cat(pairs_T) - mu_T) @ P_T                       # (N, d_S) teacher PCA coords
        S_all = torch.cat(pairs_S)
        P_S, mu_S, _ = pca_axes(S_all, d_S)
        cS = (S_all - mu_S) @ P_S
        cT = cT / torch.linalg.norm(cT)                                # rotation only: unit scale
        cS = cS / torch.linalg.norm(cS)
        R = procrustes(cT, cS)
        P = P_T @ R @ P_S.T                                            # (d_T, d_S), orthonormal cols
        info['procrustes_residual'] = float(torch.linalg.norm(cT @ R - cS) / torch.linalg.norm(cS))
    groups = select_layers(L_T, L_S, layer_select)
    projected, heads = {}, []
    for i, g in enumerate(groups):
        lay = {k: _avg([teacher.layers[t][k] for t in g]) for k in ('c_attn', 'attn_proj', 'c_fc', 'mlp_proj')}
        H = torch.cat([hidden_acts[t] for t in g])
        P_h = hidden_axes(H, lay['c_fc'] @ P, 4 * d_S, mode=hidden)
        pj = project_layer(lay, P, P_h, teacher.n_head, student_cfg.n_head, d_S)
        heads.append(pj['heads_kept'])
        p = f'transformer.h.{i}.'
        projected[p + 'attn.c_attn.weight'] = pj['c_attn']
        projected[p + 'attn.c_proj.weight'] = pj['attn_proj']
        projected[p + 'mlp.c_fc.weight'] = pj['c_fc']
        projected[p + 'mlp.c_proj.weight'] = pj['mlp_proj']
    same_vocab = teacher.vocab_size == student_cfg.vocab_size
    if include_embeddings is None:
        include_embeddings = same_vocab
    if include_embeddings and same_vocab:
        projected['transformer.wte.weight'] = teacher.wte @ P
        T_S = min(student_cfg.block_size, teacher.wpe.shape[0])
        wpe = torch.zeros(student_cfg.block_size, d_S)
        wpe[:T_S] = teacher.wpe[:T_S] @ P
        projected['transformer.wpe.weight'] = wpe
    info['heads_kept_per_layer'] = heads
    info['layer_groups'] = groups
    info['embeddings_projected'] = bool(include_embeddings and same_vocab)
    if verbose:
        print(f"[xsize] {teacher.source}: {L_T}L/{d_T}d/{teacher.n_head}h → "
              f"{L_S}L/{d_S}d/{student_cfg.n_head}h; align={align} hidden={hidden} "
              f"layers={layer_select} ({groups}); residual PCA explains {explained:.3f} of variance"
              + (f"; procrustes rel. residual {info['procrustes_residual']:.3f}" if align == 'procrustes' else '')
              + f"; embeddings {'projected' if info['embeddings_projected'] else 'left fresh'}")
    return projected, info


# ══════════════════════════════════════════════════════════════════════════
# Fingerprint writer (what prism_eval --fingerprint consumes)
# ══════════════════════════════════════════════════════════════════════════

def group_spectra(mats_by_name, n_dct=8):
    """prism_extract-style group-averaged DCT spectra from {name: W}."""
    group_svs = {}
    for name, W in mats_by_name.items():
        g = classify_nanogpt_param(name)
        if g is None or W.dim() != 2:
            continue
        s = torch.linalg.svdvals(W.float())
        s = s / s.max() if s.max() > 0 else s
        group_svs.setdefault(g, []).append(s.numpy())
    spectra = {}
    for g, svs in group_svs.items():
        max_len = max(len(s) for s in svs)
        interp = [np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(s)), s) for s in svs]
        spectra[g] = _dct_fit(np.mean(interp, axis=0), n_dct)
    return spectra


def teacher_matrices_by_group(teacher):
    """The teacher's own matrices with nanoGPT names (for the group spectra)."""
    out = {}
    for l, lay in enumerate(teacher.layers):
        p = f'transformer.h.{l}.'
        out[p + 'attn.c_attn.weight'] = lay['c_attn']
        out[p + 'attn.c_proj.weight'] = lay['attn_proj']
        out[p + 'mlp.c_fc.weight'] = lay['c_fc']
        out[p + 'mlp.c_proj.weight'] = lay['mlp_proj']
    if teacher.wte is not None:
        out['transformer.wte.weight'] = teacher.wte
        out['transformer.wpe.weight'] = teacher.wpe
    return out


def write_fingerprint(projected, teacher, out_dir, info, spectra_from='teacher', n_dct=8):
    os.makedirs(out_dir, exist_ok=True)
    directions = {}
    for name, W in projected.items():
        U, s, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        directions[name] = {'U': U.contiguous(), 'Vt': Vt.contiguous(),
                            'group': classify_nanogpt_param(name), 'shape': list(W.shape)}
    src = teacher_matrices_by_group(teacher) if spectra_from == 'teacher' else projected
    spectra = group_spectra(src, n_dct)
    json.dump(spectra, open(os.path.join(out_dir, 'spectra.json'), 'w'), indent=2)
    torch.save(directions, os.path.join(out_dir, 'directions.pt'))
    info = dict(info, spectra_from=spectra_from, n_dct=n_dct, n_directions=len(directions))
    json.dump(info, open(os.path.join(out_dir, 'crosssize.json'), 'w'), indent=2)
    print(f"[xsize] fingerprint → {out_dir}: {len(directions)} direction sets, "
          f"spectra for {sorted(spectra)} (from {spectra_from})")
    return out_dir


# ══════════════════════════════════════════════════════════════════════════
# Baseline init builders (state dicts for train.py --init_state)
# ══════════════════════════════════════════════════════════════════════════

def _uniform_idx(t, s):
    """Evenly spaced s indices out of t, stride ⌊t/s⌋ from 0 (Weight Selection's worked example)."""
    if s == t:
        return np.arange(t)
    stride = t // s
    idx = np.arange(s) * stride
    if stride * s > t:  # non-integer ratio fallback
        idx = np.floor(np.arange(s) * t / s).astype(int)
    return idx


def weight_selection(teacher, student_cfg, layer_select='first'):
    """arXiv 2311.18823 — returns a student state dict (only the selected tensors;
    train.py's init_state loader leaves the rest fresh)."""
    d_T, d_S = teacher.n_embd, student_cfg.n_embd
    idx_d = _uniform_idx(d_T, d_S)
    idx_h = _uniform_idx(4 * d_T, 4 * d_S)
    # consistent per-block selection for the fused QKV rows
    idx_qkv = np.concatenate([idx_d, d_T + idx_d, 2 * d_T + idx_d])
    groups = select_layers(teacher.n_layer, student_cfg.n_layer, layer_select)
    sd = {}
    for i, g in enumerate(groups):
        lay = {k: _avg([teacher.layers[t][k] for t in g]) for k in teacher.layers[0]}
        p = f'transformer.h.{i}.'
        sd[p + 'attn.c_attn.weight'] = lay['c_attn'][idx_qkv][:, idx_d].clone()
        sd[p + 'attn.c_proj.weight'] = lay['attn_proj'][idx_d][:, idx_d].clone()
        sd[p + 'mlp.c_fc.weight'] = lay['c_fc'][idx_h][:, idx_d].clone()
        sd[p + 'mlp.c_proj.weight'] = lay['mlp_proj'][idx_d][:, idx_h].clone()
        sd[p + 'ln_1.weight'] = lay['ln_1'][idx_d].clone()
        sd[p + 'ln_2.weight'] = lay['ln_2'][idx_d].clone()
    sd['transformer.ln_f.weight'] = teacher.ln_f[idx_d].clone()
    if teacher.vocab_size == student_cfg.vocab_size:
        sd['transformer.wte.weight'] = teacher.wte[:, idx_d].clone()
        T_S = min(student_cfg.block_size, teacher.wpe.shape[0])
        sd['transformer.wpe.weight'] = torch.zeros(student_cfg.block_size, d_S)
        sd['transformer.wpe.weight'][:T_S] = teacher.wpe[:T_S][:, idx_d]
    return sd


def _dct_resize(stack, target_shape, r=1.0):
    """FRONT's operator on one L×out×in stack: orthonormal 3D-DCT, low-frequency
    learngene mask at ratio r, truncate/zero-pad to target_shape, inverse DCT."""
    from scipy.fft import dctn, idctn
    X = stack.numpy().astype(np.float64)
    Phi = dctn(X, norm='ortho')
    if r < 1.0:
        mask = np.zeros_like(Phi)
        ls, ms, ns = (max(1, int(math.floor(r * n))) for n in Phi.shape)
        mask[:ls, :ms, :ns] = 1.0
        Phi = Phi * mask
    out = np.zeros(target_shape)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(Phi.shape, target_shape))
    out[sl] = Phi[sl]                       # truncate the high-frequency side / zero-pad to grow
    return torch.from_numpy(idctn(out, norm='ortho')).float()


def front(teacher, student_cfg, r=1.0):
    """arXiv 2603.07523 §3 — stacks each weight type across ALL teacher layers and
    resizes in the frequency domain. Embeddings / LayerNorms stay fresh."""
    d_S, L_S = student_cfg.n_embd, student_cfg.n_layer
    types = {'attn.c_attn.weight': ('c_attn', (L_S, 3 * d_S, d_S)),
             'attn.c_proj.weight': ('attn_proj', (L_S, d_S, d_S)),
             'mlp.c_fc.weight': ('c_fc', (L_S, 4 * d_S, d_S)),
             'mlp.c_proj.weight': ('mlp_proj', (L_S, d_S, 4 * d_S))}
    sd = {}
    for suffix, (key, shape) in types.items():
        stack = torch.stack([lay[key] for lay in teacher.layers])
        res = _dct_resize(stack, shape, r=r)
        for i in range(L_S):
            sd[f'transformer.h.{i}.{suffix}'] = res[i].contiguous()
    return sd


BUILDERS = {'weightsel': weight_selection, 'front': front}


def build_init_state(builder, teacher, student_cfg, out_path, **kw):
    sd = BUILDERS[builder](teacher, student_cfg, **kw)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    torch.save({'model': sd, 'builder': builder, 'teacher': teacher.source,
                'student': {'n_layer': student_cfg.n_layer, 'n_embd': student_cfg.n_embd,
                            'n_head': student_cfg.n_head}, **{k: str(v) for k, v in kw.items()}},
               out_path)
    print(f"[xsize] init_state ({builder}) → {out_path}: {len(sd)} tensors")
    return out_path


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def student_config(args_or_dict, vocab_size, block_size):
    a = args_or_dict
    return GPTConfig(n_layer=a['n_layer'], n_head=a['n_head'], n_embd=a['n_embd'],
                     block_size=block_size, vocab_size=vocab_size, dropout=0.0, bias=False)


def main():
    p = argparse.ArgumentParser(description='PRISM cross-size projection / baseline inits')
    p.add_argument('--teacher_ckpt', default=None)
    p.add_argument('--teacher_hf', default=None, help="e.g. gpt2")
    p.add_argument('--student_n_layer', type=int, required=True)
    p.add_argument('--student_n_embd', type=int, required=True)
    p.add_argument('--student_n_head', type=int, required=True)
    p.add_argument('--student_block', type=int, default=256)
    p.add_argument('--student_vocab', type=int, default=None,
                   help='student vocab (default: from --calib meta.pkl)')
    p.add_argument('--calib', default=None, help='nanoGPT data dir with train.bin (+ meta.pkl)')
    p.add_argument('--calib_text', default=None, help='text file for an HF teacher')
    p.add_argument('--calib_batches', type=int, default=8)
    p.add_argument('--calib_batch_size', type=int, default=8)
    p.add_argument('--out', default=None, help='fingerprint dir to write')
    p.add_argument('--align', default='pca', choices=['pca', 'procrustes'])
    p.add_argument('--hidden', default='pca', choices=['pca', 'svd'])
    p.add_argument('--layer_select', default='uniform', choices=['uniform', 'average', 'first'])
    p.add_argument('--spectra_from', default='teacher', choices=['teacher', 'projected'])
    p.add_argument('--builder', default=None, choices=list(BUILDERS))
    p.add_argument('--out_state', default=None, help='init_state .pt to write for --builder')
    p.add_argument('--front_r', type=float, default=1.0)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()

    if a.teacher_hf:
        teacher = teacher_from_hf(a.teacher_hf, a.device)
    else:
        assert a.teacher_ckpt, 'need --teacher_ckpt or --teacher_hf'
        teacher = teacher_from_nanogpt(a.teacher_ckpt, a.device)
    vocab = a.student_vocab
    if vocab is None and a.calib:
        vocab = pickle.load(open(os.path.join(a.calib, 'meta.pkl'), 'rb'))['vocab_size']
    assert vocab, 'need --student_vocab or a --calib dir with meta.pkl'
    cfg = student_config({'n_layer': a.student_n_layer, 'n_embd': a.student_n_embd,
                          'n_head': a.student_n_head}, vocab, a.student_block)
    if a.builder:
        assert a.out_state
        kw = {'r': a.front_r} if a.builder == 'front' else {}
        build_init_state(a.builder, teacher, cfg, a.out_state, **kw)
    if a.out:
        if a.teacher_hf:
            assert a.calib_text, 'an HF teacher needs --calib_text'
            batches = calib_batches_from_text(a.calib_text, a.teacher_hf, min(a.student_block, 1024),
                                              a.calib_batches, a.calib_batch_size)
        else:
            assert a.calib, 'a nanoGPT teacher needs --calib'
            batches = calib_batches_from_bin(a.calib, min(a.student_block, teacher.wpe.shape[0]),
                                             a.calib_batches, a.calib_batch_size)
        projected, info = project_teacher(teacher, cfg, batches, align=a.align, hidden=a.hidden,
                                          layer_select=a.layer_select, device=a.device)
        write_fingerprint(projected, teacher, a.out, info, spectra_from=a.spectra_from)


if __name__ == '__main__':
    main()
