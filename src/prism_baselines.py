"""prism_baselines.py — competitor finetune methods for the injection-dial
head-to-head (W1), as re-implementations with their deviations listed. Wired
behind train.py's `ft_method` / `ft_method_cfg` (resume path only) at four hook
points: wrap(model) before the optimizer is built, penalty() added to the loss,
pre_step()/post_step() around the optimizer step, export_state_dict() at save.

Each class states: what it is, the paper and section it follows, and every
deviation. None is the authors' code. Hyper-parameters are tuned on seed 1336
only, at most 3 settings per method, by a criterion written down before
looking (docs/NEXT-EXPERIMENTS.md, W1 pre-registration). `ft_method_cfg` is
'key:value,key:value' (nanoGPT's configurator splits flags on '=').

  sst          Subspace Tuning — our re-implementation of arXiv 2610.22293 §2
               (Algorithm 1): the optimizer runs unmodified; its applied update
               ΔW is projected onto the free-to-free block, W ← W_prev + Q_U ΔW Q_V,
               Q = I − P with P the top-k projector of W_ref; k = ⌊r·min(d_out,d_in)⌋,
               r=0.99 default; W_ref and the bases are refreshed every K steps
               (K=0 → never). Applied to the attention and MLP matrices; the
               embeddings are left to the optimizer, as in Algorithm 1.
               Deviations: bases from a full SVD (the paper uses Gram
               eigendecompositions for speed — same subspace); at d=384 a
               0.99 fraction leaves 4 free directions, so r is tuned over
               {0.5, 0.9, 0.99} on seed 1336 (the paper itself dropped to 0.90
               when the free tail was too small).
  rpsft        Rotation-Preserving SFT — our re-implementation of arXiv
               2605.10973 (Algorithm 1): soft penalty λ Σ_W ‖U_kᵀ(W − W₀)V_k‖²_F
               added to the loss, U_k/V_k the top-k singular vectors of W₀,
               applied to the attention and MLP matrices. Deviations: k is a
               rank for OUR width (default 64 of 384 ≈ 17%, cf. 768 of 4096
               ≈ 19%); λ tuned over {0.1, 1, 10}; penalty applied every step
               (the paper allows amortising every s steps).
  ewc          Elastic Weight Consolidation (Kirkpatrick et al. 2017):
               λ/2 Σ_i F_i (θ_i − θ⁰_i)² with a diagonal empirical Fisher
               estimated from the base model on OLD-domain data (train.py's
               val2 split, n_fisher batches of squared batch-mean gradients).
               Deviations: batch-mean rather than per-example gradients; Fisher
               normalised to mean 1 so λ is comparable across runs; λ tuned
               over three values on seed 1336.
  lora         LoRA (Hu et al. 2021) as a nanoGPT-native wrapper on c_attn,
               attn.c_proj, mlp.c_fc, mlp.c_proj: W + (α/r)·B·A, A ~ N(0, 1/r)
               (Kaiming-style), B = 0, α = 2r, no dropout; every base parameter
               frozen (embeddings and LayerNorms included). Merged into a plain
               state dict at every checkpoint save so recall is scored on the
               merged model. Deviations: LR tuned over three values on 1336
               (LoRA wants a higher LR than full finetuning).
  selfdistill  A self-distillation baseline "in the spirit of SDFT/iSDFT
               (arXiv 2609.24646), not a reproduction": loss = CE + α·KL(p_base ‖ p_θ)
               on the finetune batches, with the frozen pre-finetune base as
               teacher (one extra teacher forward per micro-step). α tuned over
               three values on 1336. No budgeted targets, no synthetic rewrites.

Per-step overhead is timed (pre/post hooks + penalty) and printed by train.py
in the `[timing]` line as `ftm_overhead_ms`.
"""
import ast
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_anchor import top_subspaces, project_top


def _parse_cfg(spec):
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


def hidden_matrices(model):
    """[(name, param)] for the attention and MLP matrices of every block."""
    return [(n, p) for n, p in model.named_parameters() if p.dim() == 2 and '.h.' in n]


def _sync(device):
    if device is not None and str(device).startswith('cuda'):
        torch.cuda.synchronize()


class FTMethod:
    name = 'base'
    wraps_model = False

    def __init__(self, cfg, model, device, old_batch_fn=None, new_batch_fn=None):
        self.cfg, self.device = cfg, device
        self._extra_s, self._steps = 0.0, 0

    def wrap(self, model):
        return model

    def penalty(self, raw_model, X, Y, logits):
        return None

    def pre_step(self, raw_model):
        pass

    def post_step(self, raw_model, step):
        pass

    def export_state_dict(self, raw_model):
        return raw_model.state_dict()

    def overhead_ms(self):
        return 1000.0 * self._extra_s / max(1, self._steps)

    def describe(self):
        return f'{self.name} {self.cfg}'


class SST(FTMethod):
    name = 'sst'

    def __init__(self, cfg, model, device, **_):
        super().__init__(cfg, model, device)
        self.r = float(cfg.get('r', 0.99))
        self.refresh = int(cfg.get('refresh', 0))
        self.mats = hidden_matrices(model)
        self.proj = {}
        self.prev = {}
        with torch.no_grad():
            self._set_ref({n: p.data for n, p in self.mats})
        self.k_of = {n: self.proj[n][0].shape[1] for n, _ in self.mats}

    @torch.no_grad()
    def _set_ref(self, weights):
        for n, _ in self.mats:
            W = weights[n]
            U_k, V_k = top_subspaces(W, self.r)
            self.proj[n] = (U_k.to(W.device), V_k.to(W.device))

    @torch.no_grad()
    def pre_step(self, raw_model):
        t = time.perf_counter()
        for n, p in self.mats:
            self.prev[n] = p.data.clone()
        self._extra_s += time.perf_counter() - t

    @torch.no_grad()
    def post_step(self, raw_model, step):
        t = time.perf_counter()
        for n, p in self.mats:
            U_k, V_k = self.proj[n]
            delta = p.data - self.prev[n]
            rest = delta - project_top(delta, U_k, V_k)      # Q_U Δ Q_V
            p.data.copy_(self.prev[n] + rest)
        if self.refresh > 0 and (step + 1) % self.refresh == 0:
            self._set_ref({n: p.data for n, p in self.mats})
        _sync(self.device)
        self._extra_s += time.perf_counter() - t
        self._steps += 1

    def describe(self):
        ks = sorted(set(self.k_of.values()))
        free = sorted({min(p.shape) - k for (n, p), k in zip(self.mats, self.k_of.values())})
        return (f'SST (our re-implementation of arXiv 2610.22293 Alg.1): r={self.r}, '
                f'protected rank k∈{ks}, free dims∈{free}, refresh every '
                f'{self.refresh or "never"}; {len(self.mats)} matrices')


class RPSFT(FTMethod):
    name = 'rpsft'

    def __init__(self, cfg, model, device, **_):
        super().__init__(cfg, model, device)
        self.k = int(cfg.get('k', 64))
        self.lam = float(cfg.get('lam', 1.0))
        self.mats = hidden_matrices(model)
        self.cache = {}
        with torch.no_grad():
            for n, p in self.mats:
                U_k, V_k = top_subspaces(p.data, self.k)
                U_k, V_k = U_k.to(p.device), V_k.to(p.device)
                S_ref = U_k.T @ p.data.float() @ V_k
                self.cache[n] = (U_k, V_k, S_ref)

    def penalty(self, raw_model, X, Y, logits):
        t = time.perf_counter()
        params = dict(raw_model.named_parameters())
        pen = 0.0
        for n, _ in self.mats:
            U_k, V_k, S_ref = self.cache[n]
            S = U_k.T @ params[n].float() @ V_k
            pen = pen + (S - S_ref).pow(2).sum()
        self._extra_s += time.perf_counter() - t
        self._steps += 1
        return self.lam * pen

    def describe(self):
        return (f'RP-SFT (our re-implementation of arXiv 2605.10973 Alg.1): k={self.k}, '
                f'λ={self.lam}, {len(self.mats)} matrices')


class EWC(FTMethod):
    name = 'ewc'

    def __init__(self, cfg, model, device, old_batch_fn=None, **_):
        super().__init__(cfg, model, device)
        self.lam = float(cfg.get('lam', 1.0))
        n_batches = int(cfg.get('n_fisher', 32))
        assert old_batch_fn is not None, 'EWC needs old-domain data (val2_dir) for the Fisher'
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.theta0 = {n: p.data.clone() for n, p in self.params}
        fisher = {n: torch.zeros_like(p.data) for n, p in self.params}
        was = model.training
        model.eval()
        for _ in range(n_batches):
            X, Y = old_batch_fn()
            model.zero_grad(set_to_none=True)
            _, loss = model(X, Y)
            loss.backward()
            for n, p in self.params:
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        model.zero_grad(set_to_none=True)
        model.train(was)
        tot = sum(f.sum() for f in fisher.values())
        cnt = sum(f.numel() for f in fisher.values())
        scale = cnt / max(float(tot), 1e-30)          # normalise to mean 1
        self.fisher = {n: f * scale for n, f in fisher.items()}
        self.n_fisher = n_batches

    def penalty(self, raw_model, X, Y, logits):
        t = time.perf_counter()
        params = dict(raw_model.named_parameters())
        pen = 0.0
        for n, _ in self.params:
            pen = pen + (self.fisher[n] * (params[n] - self.theta0[n]).pow(2)).sum()
        self._extra_s += time.perf_counter() - t
        self._steps += 1
        return 0.5 * self.lam * pen

    def describe(self):
        return (f'EWC (Kirkpatrick 2017; diagonal empirical Fisher from {self.n_fisher} '
                f'old-domain batches, normalised to mean 1): λ={self.lam}')


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r, alpha):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features) / (r ** 0.5))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale

    @torch.no_grad()
    def merged_weight(self):
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scale


class LoRA(FTMethod):
    name = 'lora'
    wraps_model = True

    def __init__(self, cfg, model, device, **_):
        super().__init__(cfg, model, device)
        self.r = int(cfg.get('r', 4))
        self.alpha = float(cfg.get('alpha', 2 * self.r))
        self.wrapped = []

    def wrap(self, model):
        for p in model.parameters():
            p.requires_grad_(False)
        for i, block in enumerate(model.transformer.h):
            for parent, attr in ((block.attn, 'c_attn'), (block.attn, 'c_proj'),
                                 (block.mlp, 'c_fc'), (block.mlp, 'c_proj')):
                lin = getattr(parent, attr)
                wrapped = LoRALinear(lin, self.r, self.alpha).to(lin.weight.device)
                setattr(parent, attr, wrapped)
                self.wrapped.append(wrapped)
        return model

    def export_state_dict(self, raw_model):
        sd = raw_model.state_dict()
        out = {}
        for k, v in sd.items():
            if '.lora_A' in k or '.lora_B' in k:
                continue
            out[k.replace('.base.weight', '.weight').replace('.base.bias', '.bias')] = v
        # merge
        for name, mod in raw_model.named_modules():
            if isinstance(mod, LoRALinear):
                out[name.removeprefix('_orig_mod.') + '.weight'] = mod.merged_weight().detach().clone()
        return out

    def lora_B_params(self):
        return [m.lora_B for m in self.wrapped]

    def describe(self):
        n = sum(m.lora_A.numel() + m.lora_B.numel() for m in self.wrapped)
        return (f'LoRA r={self.r} α={self.alpha:g} on c_attn/attn.c_proj/c_fc/mlp.c_proj '
                f'({len(self.wrapped)} linears, {n:,} trainable params; all base params frozen; '
                f'merged at save)')


class SelfDistill(FTMethod):
    name = 'selfdistill'

    def __init__(self, cfg, model, device, **_):
        super().__init__(cfg, model, device)
        self.alpha = float(cfg.get('alpha', 0.5))
        self.T = float(cfg.get('T', 1.0))
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def penalty(self, raw_model, X, Y, logits):
        t = time.perf_counter()
        with torch.no_grad():
            t_logits, _ = self.teacher(X, Y)
        log_p_t = F.log_softmax(t_logits.float() / self.T, dim=-1)
        log_p_s = F.log_softmax(logits.float() / self.T, dim=-1)
        kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1).mean() * (self.T ** 2)
        _sync(self.device)
        self._extra_s += time.perf_counter() - t
        self._steps += 1
        return self.alpha * kl

    def describe(self):
        return (f'self-distillation baseline (in the spirit of SDFT/iSDFT, arXiv 2609.24646; '
                f'not a reproduction): loss = CE + {self.alpha}·KL(p_base‖p_θ), T={self.T}')


METHODS = {'sst': SST, 'rpsft': RPSFT, 'ewc': EWC, 'lora': LoRA, 'selfdistill': SelfDistill}


def make_ft_method(name, cfg_spec, model, device, old_batch_fn=None, new_batch_fn=None):
    if name not in METHODS:
        raise ValueError(f'unknown ft_method {name!r}; known: {sorted(METHODS)}')
    return METHODS[name](_parse_cfg(cfg_spec), model, device,
                         old_batch_fn=old_batch_fn, new_batch_fn=new_batch_fn)
