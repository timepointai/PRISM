"""prism_inits.py — baseline-init controls for the core bet (W2a) and the hooks
the cross-size study (W3) loads its inits through. Wired behind train.py's
`init_std`, `init_state`, `init_mode`, `mup_base_width`; all no-ops by default.

Modes (applied to a FRESH nanoGPT model, before any Prism geometry):

  init_std=<σ>         Rescale every ≥2-D weight by σ/0.02. nanoGPT draws every
                       Linear/Embedding at std 0.02 (c_proj at 0.02/√(2L)), so this
                       is exactly "the same draw at a different σ" — one variable.
  mup_base_width=<b>   μP-lite (Tensor Programs V, Adam): hidden-block matrices
                       scaled by √(b/width); train.py adds the logit multiplier
                       b/width and prism_optim.py the hidden LR multiplier b/width.
  init_state=<path>    Load a state_dict .pt into the model (strict on shapes,
                       tolerant of missing keys — a cross-size init that leaves
                       the embeddings fresh is the intended use).
  init_mode=
    small_qk:<σ>       Rescale the Q and K row-blocks of every c_attn to std σ
                       (OpenReview 419qaAMpme: a broad low-loss plateau down to 1e-5).
    mimetic[:a1,b1,a2,b2]
                       Mimetic initialization (Trockman & Kolter, arXiv 2305.09828),
                       per head: W_Qᵀ W_K ≈ a1·Z1 + b1·I and W_Vᵀ W_O ≈ a2·Z2 − b2·I
                       with Z ~ N(0, I/d), factored through a rank-head_dim SVD
                       (W_Q = Σ^½U_kᵀ, W_K = Σ^½V_kᵀ; W_V = Σ^½U_kᵀ, W_O[:,h] = V_kΣ^½).
                       Defaults are the paper's character-level PTB values
                       (a1=0, b1=0.5, a2=b2=0.2). The paper has no MLP init; the
                       MLP stays at nanoGPT's default. Our re-implementation, not
                       the authors' code.
    statmatch          A statistics-matched random init (the control OpenReview
                       DzLzZYD84Y motivates: 108 layerwise scalars on a ViT). We
                       impose the TEACHER's scalar statistics on a random init
                       with NO directions, in this order, on a calibration batch:
                         S1  per-matrix Frobenius norm, every ≥2-D weight
                             (6 layers × 4 + wte + wpe = 26 scalars at 6L);
                         S2  LayerNorm gain mean and std per LN (13 LNs × 2 = 26);
                         S3  mean attention-row entropy per layer, matched by a
                             scalar on the layer's Q/K blocks (6);
                         S4  residual write ratios per layer: rms(attn_out)/rms(x)
                             and rms(mlp_out)/rms(x), matched by scalars on
                             attn.c_proj and mlp.c_proj (12).
                       70 scalars at the probe size. NOT matched (no bias terms
                       in this config): MLP pre-activation mean/negativity.
                       Statistics are measured sequentially, layer by layer, so
                       downstream statistics see upstream changes.

Every function reports what it did; train.py prints it so the artifact's stdout
carries the exact list of matched statistics.
"""
import ast
import math

import torch
import torch.nn.functional as F


def _parse_kv(spec):
    out = {}
    for part in (spec or '').split(','):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(':', 1)
        try:
            out[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            out[k.strip()] = v.strip()
    return out


@torch.no_grad()
def rescale_init_std(model, init_std):
    f = float(init_std) / 0.02
    n = 0
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            p.data.mul_(f)
            n += 1
    print(f"[init] rescaled {n} matrices by {f:.4g} (init_std={init_std})")


@torch.no_grad()
def mup_scale_hidden(model, base_width):
    width = model.config.n_embd
    f = math.sqrt(float(base_width) / width)
    n = 0
    for name, p in model.named_parameters():
        if p.dim() >= 2 and '.h.' in name:
            p.data.mul_(f)
            n += 1
    print(f"[init] μP-lite: {n} hidden matrices scaled by √({base_width}/{width}) = {f:.4g}; "
          f"logit mult {base_width / width:.4g}; hidden LR mult {base_width / width:.4g}")


@torch.no_grad()
def load_init_state(model, path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if 'model' in sd and isinstance(sd['model'], dict):
        sd = sd['model']
    sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    own = model.state_dict()
    bad = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
    if bad:
        raise ValueError(f'init_state shape mismatch on {bad[:3]}')
    res = model.load_state_dict({k: v for k, v in sd.items() if k in own}, strict=False)
    print(f"[init] loaded init_state {path}: {len(sd)} tensors; missing (left fresh): "
          f"{len(res.missing_keys)}; ignored: {len([k for k in sd if k not in own])}")


@torch.no_grad()
def small_qk(model, sigma):
    d = model.config.n_embd
    n = 0
    for block in model.transformer.h:
        W = block.attn.c_attn.weight            # (3d, d): [Q; K; V] row blocks
        W[:2 * d].mul_(float(sigma) / 0.02)
        n += 1
    print(f"[init] small_qk: Q/K blocks of {n} layers rescaled to std {sigma}")


@torch.no_grad()
def mimetic(model, a1=0.0, b1=0.5, a2=0.2, b2=0.2, seed=0):
    d, H = model.config.n_embd, model.config.n_head
    hd = d // H
    g = torch.Generator().manual_seed(seed)
    for block in model.transformer.h:
        W_attn = block.attn.c_attn.weight          # (3d, d)
        W_proj = block.attn.c_proj.weight          # (d, d): y = concat_h(v_h) W_projᵀ
        dev, dt = W_attn.device, W_attn.dtype
        I = torch.eye(d)
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            Z1 = torch.randn(d, d, generator=g) / math.sqrt(d)
            U, S, Vt = torch.linalg.svd(a1 * Z1 + b1 * I)
            s = S[:hd].sqrt()
            W_attn[0 * d + h * hd:0 * d + (h + 1) * hd] = (s[:, None] * U[:, :hd].T).to(dev, dt)   # W_q_h
            W_attn[1 * d + h * hd:1 * d + (h + 1) * hd] = (s[:, None] * Vt[:hd]).to(dev, dt)      # W_k_h
            Z2 = torch.randn(d, d, generator=g) / math.sqrt(d)
            U, S, Vt = torch.linalg.svd(a2 * Z2 - b2 * I)
            s = S[:hd].sqrt()
            W_attn[2 * d + h * hd:2 * d + (h + 1) * hd] = (s[:, None] * U[:, :hd].T).to(dev, dt)   # W_v_h
            W_proj[:, sl] = (Vt[:hd].T * s[None, :]).to(dev, dt)                                   # W_o[:, h]
    print(f"[init] mimetic attention init on {len(model.transformer.h)} layers "
          f"(a1={a1}, b1={b1}, a2={a2}, b2={b2}); MLP left at default")


# ── statmatch ──

class _Probe:
    """Forward hooks collecting the per-layer scalars statmatch matches."""

    def __init__(self, model):
        self.model = model
        self.handles, self.stats = [], {}
        for i, block in enumerate(model.transformer.h):
            self.handles.append(block.register_forward_pre_hook(self._pre(i)))
            self.handles.append(block.attn.register_forward_hook(self._attn(i)))
            self.handles.append(block.mlp.register_forward_hook(self._mlp(i)))

    def _pre(self, i):
        def hook(mod, inputs):
            self.stats[f'x_rms_{i}'] = float(inputs[0].float().pow(2).mean().sqrt())
        return hook

    def _attn(self, i):
        def hook(mod, inputs, output):
            self.stats[f'attn_rms_{i}'] = float(output.float().pow(2).mean().sqrt())
            # attention entropy, recomputed from q/k on the (normalised) block input
            x = inputs[0]
            B, T, C = x.shape
            q, k, _ = mod.c_attn(x).split(C, dim=2)
            k = k.view(B, T, mod.n_head, C // mod.n_head).transpose(1, 2)
            q = q.view(B, T, mod.n_head, C // mod.n_head).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~mask, float('-inf')).float()
            p = F.softmax(att, dim=-1)
            ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)       # (B, H, T)
            self.stats[f'attn_entropy_{i}'] = float(ent.mean())
        return hook

    def _mlp(self, i):
        def hook(mod, inputs, output):
            self.stats[f'mlp_rms_{i}'] = float(output.float().pow(2).mean().sqrt())
        return hook

    def run(self, X):
        self.stats = {}
        was = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.model(X)
        self.model.train(was)
        return dict(self.stats)

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def statmatch(model, teacher_ckpt, X, seed=0):
    from model import GPTConfig, GPT
    ck = torch.load(teacher_ckpt, map_location='cpu', weights_only=False)
    teacher = GPT(GPTConfig(**ck['model_args']))
    teacher.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()},
                            strict=False)
    teacher.to(X.device).eval()
    matched = []
    # S1 per-matrix Frobenius norms (random directions kept)
    tp = dict(teacher.named_parameters())
    for name, p in model.named_parameters():
        if p.dim() >= 2 and name in tp and tp[name].shape == p.shape:
            tn, sn = tp[name].float().norm(), p.data.float().norm()
            if sn > 0:
                p.data.mul_((tn / sn).to(p.dtype))
            matched.append(f'fro:{name}')
    # S2 LayerNorm gain mean/std (random draw with the teacher's two moments)
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if p.dim() == 1 and name in tp and 'ln' in name and name.endswith('weight'):
            t = tp[name].float()
            mu, sd = float(t.mean()), float(t.std())
            z = torch.randn(p.shape, generator=g)
            p.data.copy_((mu + sd * z).to(p.device, p.dtype))
            matched += [f'ln_mean:{name}', f'ln_std:{name}']
    # teacher statistics on the calibration batch
    tprobe = _Probe(teacher)
    ts = tprobe.run(X)
    tprobe.remove()
    sprobe = _Probe(model)
    d = model.config.n_embd
    for i, block in enumerate(model.transformer.h):
        # S3 attention entropy via a scalar on the Q/K blocks (bisection in log-scale;
        # larger scale -> sharper attention -> lower entropy)
        target = ts[f'attn_entropy_{i}']
        lo, hi = -6.0, 6.0
        W = block.attn.c_attn.weight
        base_qk = W[:2 * d].clone()
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            W[:2 * d].copy_(base_qk * math.exp(mid))
            ent = sprobe.run(X)[f'attn_entropy_{i}']
            if ent > target:
                lo = mid          # too flat -> scale up
            else:
                hi = mid
        matched.append(f'attn_entropy:{i}')
        # S4 residual write ratios via scalars on the two projections
        st = sprobe.run(X)
        for key, mod in (('attn', block.attn.c_proj), ('mlp', block.mlp.c_proj)):
            t_ratio = ts[f'{key}_rms_{i}'] / max(ts[f'x_rms_{i}'], 1e-12)
            s_ratio = st[f'{key}_rms_{i}'] / max(st[f'x_rms_{i}'], 1e-12)
            if s_ratio > 0:
                mod.weight.mul_(t_ratio / s_ratio)
            matched.append(f'write_ratio_{key}:{i}')
            st = sprobe.run(X)
    sprobe.remove()
    print(f"[init] statmatch: imposed {len(matched)} teacher statistics on a random init "
          f"(no directions): {sum(m.startswith('fro:') for m in matched)} Frobenius norms, "
          f"{sum(m.startswith('ln_') for m in matched)} LN moments, "
          f"{sum(m.startswith('attn_entropy') for m in matched)} attention entropies, "
          f"{sum(m.startswith('write_ratio') for m in matched)} residual write ratios. "
          f"Not matched: MLP pre-activation moments (no bias terms in this config).")
    return matched


def apply_init_knobs(model, init_std=0.02, init_state='', init_mode='default', teacher_ckpt='',
                     mup_base_width=0, device='cpu', get_batch=None):
    if init_std != 0.02:
        rescale_init_std(model, init_std)
    if mup_base_width:
        mup_scale_hidden(model, mup_base_width)
    if init_state:
        load_init_state(model, init_state)
    mode = init_mode or 'default'
    if mode == 'default':
        return
    kind, _, arg = mode.partition(':')
    if kind == 'small_qk':
        small_qk(model, float(arg))
    elif kind == 'mimetic':
        vals = [float(v) for v in arg.split(',')] if arg else []
        mimetic(model, *vals)
    elif kind == 'statmatch':
        assert teacher_ckpt, 'init_mode=statmatch needs init_mode_teacher=<ckpt.pt>'
        assert get_batch is not None
        X, _ = get_batch()
        statmatch(model, teacher_ckpt, X[:8].to(device))
    else:
        raise ValueError(f'unknown init_mode {init_mode!r}')
