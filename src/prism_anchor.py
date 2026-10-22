"""prism_anchor.py — the weight-space anchor as a framework-agnostic object.

PRISM's retention anchor is one line applied after every optimizer step:

    W  <-  (1 - s)·W + s·W_target          (torch: W.lerp_(target, s))

`train.py` historically inlined that loop over a `prism_targets` dict. This
module is the same operation as an object, so it can be attached to ANY
optimizer (`attach()` uses `torch.optim.Optimizer.register_step_post_hook`),
and it carries the extensions the alpha-moves brief asks for:

  * target modes      — `fixed` (W₀), `ema:β` (a slow EMA of the policy, the
                         WARP-style anchor), `refresh:N` (re-snapshot every N
                         steps, Task-Vector-Descent-style);
  * device            — targets on CPU (default; moved per step, as before) or
                         kept on the training device;
  * row masks         — anchor only selected rows of a matrix (the tied
                         embedding's absent/present-token rows, arXiv 2610.09835);
  * anisotropic pull  — decompose the drift Δ = W − W₀ against W₀'s top-k
                         singular subspaces and pull the top component with
                         `s_top`, the rest with `s_rest` (SST/SVC-flavoured);
  * Adam-moment lerp  — pull `exp_avg` / `exp_avg_sq` toward the base's
                         end-state moments with the same strength
                         (cf. Task Vector Descent, arXiv 2610.05402: first
                         moment linearly, second moment in sqrt space);
  * LoRA-delta form   — for W = W₀ + (α/r)·B·A, lerp toward W₀ with strength s
                         is EXACTLY B ← (1 − s)·B. `lora_delta_pull` exposes it.

Anisotropic projector. With U_k, V_k the top-k left/right singular vectors of
W₀, Q_U = I − U_k U_kᵀ, Q_V = I − V_k V_kᵀ:

    Δ_rest = Q_U Δ Q_V                       (the free-to-free block)
    Δ_top  = Δ − Δ_rest = P_U Δ + Δ P_V − P_U Δ P_V

T(Δ) = Δ − Q_U Δ Q_V is idempotent (T(T(Δ)) = T(Δ)); with s_top == s_rest the
pull reduces to the plain lerp exactly. Both are selftested.

Every default reproduces `train.py`'s historical inline lerp bit-for-bit (the
golden-run check in docs/ALPHA-MOVES-STATUS.md covers it).
"""
import math

import torch


def capture_targets(named_params, exclude=(), min_dim=2):
    """Snapshot {name: tensor} of the parameters an anchor will hold. Mirrors the
    historical `prism_targets` capture (CPU clones of every ≥2-D parameter not
    matched by an exclude substring)."""
    excl = [x for x in exclude if x]
    out = {}
    for name, p in named_params:
        if p.dim() < min_dim or any(x in name for x in excl):
            continue
        out[name] = p.data.clone().cpu()
    return out


def parse_target_mode(spec):
    """'fixed' | 'ema:0.01' | 'refresh:100' -> (mode, value)."""
    spec = (spec or 'fixed').strip()
    if spec == 'fixed':
        return 'fixed', 0.0
    kind, _, val = spec.partition(':')
    if kind == 'ema':
        b = float(val)
        assert 0.0 < b < 1.0, f'ema beta must be in (0,1), got {b}'
        return 'ema', b
    if kind == 'refresh':
        n = int(val)
        assert n > 0, 'refresh period must be > 0'
        return 'refresh', n
    raise ValueError(f'unknown prism_anchor_target {spec!r}')


def parse_aniso(spec):
    """'k:s_top:s_rest' -> (k, s_top, s_rest). k is an int (rank) or a fraction
    in (0,1) of min(dim). Empty -> None."""
    if not spec:
        return None
    k_s, s_top, s_rest = spec.split(':')
    k = float(k_s)
    k = k if k < 1.0 else int(k)
    return k, float(s_top), float(s_rest)


def top_subspaces(W, k):
    """Top-k left/right singular vectors of a 2-D weight (float32 math)."""
    U, _, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    r = U.shape[1]
    kk = int(round(k * r)) if isinstance(k, float) else int(k)
    kk = max(1, min(kk, r))
    return U[:, :kk].contiguous(), Vt[:kk].transpose(0, 1).contiguous()


def project_top(delta, U_k, V_k):
    """Δ_top = Δ − Q_U Δ Q_V with Q = I − P. Idempotent; see module docstring."""
    d = delta.float()
    # Q_U Δ Q_V = Δ − U(UᵀΔ) − (ΔV)Vᵀ + U(UᵀΔV)Vᵀ
    UtD = U_k.transpose(0, 1) @ d
    DV = d @ V_k
    rest = d - U_k @ UtD - DV @ V_k.transpose(0, 1) + U_k @ (UtD @ V_k) @ V_k.transpose(0, 1)
    return (d - rest).to(delta.dtype)


class Anchor:
    """Per-step weight-space anchor over a set of named parameters."""

    def __init__(self, named_params, targets, device=None, target_mode='fixed',
                 aniso=None, row_masks=None):
        params = dict(named_params)
        self.params = {n: params[n] for n in targets if n in params}
        missing = [n for n in targets if n not in params]
        if missing:
            raise KeyError(f'anchor targets without a live parameter: {missing[:3]}')
        self.device = device
        self.targets = {n: (t.to(device) if device is not None else t)
                        for n, t in targets.items() if n in self.params}
        self.mode, self.mode_val = parse_target_mode(target_mode)
        self.row_masks = {}
        if row_masks:
            for n, idx in row_masks.items():
                if n in self.params:
                    self.row_masks[n] = torch.as_tensor(idx, dtype=torch.long)
        self.aniso = None
        if aniso:
            k, s_top, s_rest = aniso
            self.aniso = {'k': k, 's_top': s_top, 's_rest': s_rest, 'proj': {}}
            for n, t in self.targets.items():
                if t.dim() == 2 and n not in self.row_masks:
                    self.aniso['proj'][n] = top_subspaces(t, k)
        self.n_applied = 0

    # ── the step ──
    @torch.no_grad()
    def apply(self, strength, step=None):
        """One anchor step at pull strength `strength` (the current mod)."""
        if strength <= 0:
            return
        for name, p in self.params.items():
            t = self.targets[name]
            tdev = t.to(p.device)
            if self.aniso and name in self.aniso['proj']:
                U_k, V_k = self.aniso['proj'][name]
                U_k, V_k = U_k.to(p.device), V_k.to(p.device)
                delta = p.data - tdev
                d_top = project_top(delta, U_k, V_k)
                d_rest = delta - d_top
                p.data.copy_(tdev + (1.0 - self.aniso['s_top']) * d_top
                             + (1.0 - self.aniso['s_rest']) * d_rest)
            elif name in self.row_masks:
                idx = self.row_masks[name].to(p.device)
                rows = p.data.index_select(0, idx)
                rows.lerp_(tdev.index_select(0, idx), strength)
                p.data.index_copy_(0, idx, rows)
            else:
                p.data.lerp_(tdev, strength)
            # target evolution (after the pull)
            if self.mode == 'ema':
                t.lerp_(p.data.to(t.device), self.mode_val)
            elif (self.mode == 'refresh' and step is not None and step > 0
                  and step % self.mode_val == 0):
                t.copy_(p.data.to(t.device))
        self.n_applied += 1

    @torch.no_grad()
    def refresh_from_model(self, names=None):
        """Re-snapshot targets from the live weights (the scratch path's
        `prism_unfold`). `names` limits which targets are refreshed."""
        for n, p in self.params.items():
            if names is None or n in names:
                self.targets[n] = p.data.clone().to(self.targets[n].device)

    def set_target(self, name, tensor):
        self.targets[name] = tensor.to(self.targets[name].device)

    # ── optimizer hook ──
    def attach(self, optimizer, strength_fn, step_fn=None):
        """Register as a post-step hook. `strength_fn()` returns the current
        pull; `step_fn()` the current step (for refresh targets). Returns the
        handle (call .remove() to detach)."""
        def hook(opt, args, kwargs):
            self.apply(strength_fn(), step_fn() if step_fn else None)
        return optimizer.register_step_post_hook(hook)

    def describe(self):
        bits = [f'{len(self.targets)} targets', f'mode={self.mode}'
                + (f':{self.mode_val}' if self.mode != 'fixed' else '')]
        if self.device is not None:
            bits.append(f'device={self.device}')
        if self.row_masks:
            bits.append('rows=' + ','.join(f'{n.split(".")[-2]}[{len(i)}]'
                                           for n, i in self.row_masks.items()))
        if self.aniso:
            bits.append(f"aniso k={self.aniso['k']} s_top={self.aniso['s_top']} "
                        f"s_rest={self.aniso['s_rest']}")
        return ', '.join(bits)


class AdamMomentAnchor:
    """Lerp Adam's moments toward a base state with the weight anchor's strength.
    First moment linearly; second moment in sqrt space (Task Vector Descent,
    arXiv 2610.05402 §Eq.4). `base_state`: {param_name: {'exp_avg': t,
    'exp_avg_sq': t}} captured from the resumed checkpoint's optimizer."""

    def __init__(self, optimizer, named_params, base_state):
        self.opt = optimizer
        params = dict(named_params)
        self.items = [(params[n], st) for n, st in base_state.items() if n in params]

    @torch.no_grad()
    def apply(self, strength):
        if strength <= 0:
            return
        for p, base in self.items:
            st = self.opt.state.get(p)
            if not st or 'exp_avg' not in st:
                continue
            st['exp_avg'].lerp_(base['exp_avg'].to(st['exp_avg'].device), strength)
            v = st['exp_avg_sq']
            v0 = base['exp_avg_sq'].to(v.device)
            v.copy_(((1.0 - strength) * v.sqrt() + strength * v0.sqrt()) ** 2)


@torch.no_grad()
def lora_delta_pull(lora_B_params, strength):
    """Exact LoRA form of the anchor. For W = W₀ + scale·B·A, lerp(W, W₀, s)
    = W₀ + (1−s)·scale·B·A, i.e. B ← (1−s)·B. In-place on each B."""
    for B in lora_B_params:
        B.data.mul_(1.0 - strength)


def l2_penalty_equivalent_note():
    """Why an L2-to-W₀ *loss* is not the same thing (used in docs): under Adam
    the penalty gradient 2λ(W−W₀) is normalised per coordinate by √v̂, so the
    effective pull is anisotropic and LR-coupled; the post-step lerp is a fixed
    fraction of the gap per step regardless of optimizer state. Decoupled weight
    decay on the *delta* (B alone in LoRA) with wd = s/lr is the equivalent."""
    return 'post-step lerp != L2 penalty under Adam; see docstring'
