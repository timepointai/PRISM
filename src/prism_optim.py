"""prism_optim.py — optimizer construction for train.py's new knobs, kept out of
model.py (nanoGPT's file) and additive to the default path.

Default (`optim='adamw'`, `out_eps=0`, `mup_base_width=0`) reproduces
`GPT.configure_optimizers` exactly: two AdamW groups (decay: every ≥2-D param;
no-decay: the rest), fused on CUDA. The knobs:

  optim='muon'       torch.optim.Muon on the 2-D hidden matrices (attention and
                     MLP, i.e. every ≥2-D param inside `transformer.h.`), AdamW
                     for embeddings/head and 1-D params. Two optimizers wrapped
                     in `MultiOpt` so the training loop sees one object.
  out_eps>0          Adam ε override for the embedding/head parameter group
                     only (arXiv 2610.09835's control). nanoGPT ties wte and
                     lm_head, so the single shared table gets it — exactly the
                     paper's tied-embedding case.
  mup_base_width>0   μP-lite (Tensor Programs V, Adam): hidden-matrix LR
                     multiplier base/width (init std and logit multiplier are
                     applied in train.py / prism_inits.py).

Every group carries `lr_mult`; the schedule sets group['lr'] = lr·lr_mult, which
is lr·1.0 == lr for the default groups (bit-identical).
"""
import inspect

import torch


def is_hidden_matrix(name, p):
    return p.dim() >= 2 and '.h.' in name


def build_optimizer(model, weight_decay, learning_rate, betas, device_type,
                    optim='adamw', out_eps=0.0, muon_lr=0.02, muon_momentum=0.95,
                    muon_wd=None, muon_adjust_lr=None, mup_base_width=0, n_embd=None):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra = dict(fused=True) if use_fused else dict()

    if optim == 'adamw' and out_eps <= 0 and not mup_base_width:
        # byte-identical to GPT.configure_optimizers
        decay = [p for n, p in named if p.dim() >= 2]
        nodecay = [p for n, p in named if p.dim() < 2]
        groups = [{'params': decay, 'weight_decay': weight_decay, 'lr_mult': 1.0},
                  {'params': nodecay, 'weight_decay': 0.0, 'lr_mult': 1.0}]
        opt = torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)
        print(f"num decayed parameter tensors: {len(decay)}, with "
              f"{sum(p.numel() for p in decay):,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay)}, with "
              f"{sum(p.numel() for p in nodecay):,} parameters")
        print(f"using fused AdamW: {use_fused}")
        return opt

    hidden_mult = 1.0
    if mup_base_width:
        width = n_embd or getattr(getattr(model, 'config', None), 'n_embd', None)
        assert width, 'mup_base_width needs the model width'
        hidden_mult = float(mup_base_width) / float(width)

    emb = [p for n, p in named if p.dim() >= 2 and not is_hidden_matrix(n, p)]
    hidden = [p for n, p in named if is_hidden_matrix(n, p)]
    nodecay = [p for n, p in named if p.dim() < 2]

    adam_groups = []
    emb_group = {'params': emb, 'weight_decay': weight_decay, 'lr_mult': 1.0}
    if out_eps > 0:
        emb_group['eps'] = float(out_eps)
    adam_groups.append(emb_group)
    if optim == 'adamw':
        adam_groups.append({'params': hidden, 'weight_decay': weight_decay,
                            'lr_mult': hidden_mult})
    adam_groups.append({'params': nodecay, 'weight_decay': 0.0, 'lr_mult': 1.0})
    adamw = torch.optim.AdamW(adam_groups, lr=learning_rate, betas=betas, **extra)
    print(f"[optim] AdamW groups: emb/head {sum(p.numel() for p in emb):,}"
          + (f" (eps={out_eps})" if out_eps > 0 else '')
          + (f", hidden {sum(p.numel() for p in hidden):,} (lr_mult={hidden_mult:.4g})"
             if optim == 'adamw' else '')
          + f", 1-D {sum(p.numel() for p in nodecay):,}; fused={use_fused}")
    if optim == 'adamw':
        return adamw

    if optim != 'muon':
        raise ValueError(f'unknown optim {optim!r}')
    if not hasattr(torch.optim, 'Muon'):
        raise RuntimeError('torch.optim.Muon not available in this torch; pin torch>=2.10')
    mkw = dict(lr=muon_lr, weight_decay=weight_decay if muon_wd is None else muon_wd,
               momentum=muon_momentum, nesterov=True)
    if muon_adjust_lr:
        mkw['adjust_lr_fn'] = muon_adjust_lr
    muon = torch.optim.Muon([{'params': hidden, 'lr_mult': muon_lr / learning_rate
                              * hidden_mult}], **mkw)
    print(f"[optim] Muon on {len(hidden)} hidden matrices ({sum(p.numel() for p in hidden):,} "
          f"params): lr={muon_lr}, momentum={muon_momentum}, wd={mkw['weight_decay']}"
          + (f", adjust_lr_fn={muon_adjust_lr}" if muon_adjust_lr else ''))
    return MultiOpt([adamw, muon], names=['adamw', 'muon'])


class MultiOpt:
    """Several torch optimizers behind one step()/zero_grad()/state_dict(). The
    concatenated `param_groups` are the live group dicts, so the LR schedule
    loop in train.py works unchanged."""

    def __init__(self, optimizers, names=None):
        self.optimizers = list(optimizers)
        self.names = names or [type(o).__name__ for o in self.optimizers]
        self._post_hooks = []

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    @property
    def state(self):
        out = {}
        for o in self.optimizers:
            out.update(o.state)
        return out

    def step(self, *args, **kwargs):
        for o in self.optimizers:
            o.step(*args, **kwargs)
        for h in self._post_hooks:
            h(self, args, kwargs)

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def register_step_post_hook(self, hook):
        self._post_hooks.append(hook)

        class _H:
            def remove(_s):
                if hook in self._post_hooks:
                    self._post_hooks.remove(hook)
        return _H()

    def state_dict(self):
        return {'kind': 'multi', 'names': self.names,
                'optimizers': [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd):
        assert sd.get('kind') == 'multi' and len(sd['optimizers']) == len(self.optimizers)
        for o, s in zip(self.optimizers, sd['optimizers']):
            o.load_state_dict(s)

    # GradScaler duck-typing (only used when the scaler is enabled, i.e. fp16 CUDA)
    _step_supports_amp_scaling = False


def saved_param_order(model):
    """The flat parameter order `GPT.configure_optimizers` produced when a
    checkpoint was saved: decay (≥2-D) params first, then the rest, each in
    named_parameters order. Returns [(name, param)]."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return [(n, p) for n, p in named if p.dim() >= 2] + [(n, p) for n, p in named if p.dim() < 2]


def flat_param_names(optimizer, model):
    """Parameter names in the optimizer's flat (state-dict index) order — saved
    beside the optimizer state by train.py so a differently-grouped optimizer can
    be resumed by name. For MultiOpt: a list per member optimizer."""
    name_of = {p: n for n, p in model.named_parameters()}
    def flat(o):
        return [name_of[p] for g in o.param_groups for p in g['params']]
    if isinstance(optimizer, MultiOpt):
        return [flat(o) for o in optimizer.optimizers]
    return flat(optimizer)


def flat_saved_state(saved_sd, model, saved_names=None):
    """{param_name: state_dict} from a saved optimizer state. With `saved_names`
    (train.py's `optimizer_param_names`) the mapping is exact for any grouping;
    without it the saved layout is assumed to be GPT.configure_optimizers' (the
    two-group decay/no-decay order every historical checkpoint used). Multi-
    optimizer saves contribute their AdamW member (first by construction)."""
    if saved_sd.get('kind') == 'multi':
        saved_sd = saved_sd['optimizers'][0]
        if saved_names is not None and saved_names and isinstance(saved_names[0], list):
            saved_names = saved_names[0]
        elif saved_names is None:
            return {}
    if saved_names is not None:
        names = list(saved_names)
    else:
        names = [n for n, _ in saved_param_order(model)]
    n_saved = sum(len(g['params']) for g in saved_sd['param_groups'])
    if n_saved != len(names):
        return {}
    return {names[int(idx)]: st for idx, st in saved_sd['state'].items()}


def load_state_by_name(optimizer, model, saved_sd, device, skip_names=(), saved_names=None):
    """Load a saved optimizer state into a possibly differently-grouped
    optimizer by parameter identity. Hyper-parameters (lr, eps, betas) come
    from the NEW optimizer; only per-parameter moments are restored. Params
    without saved state (e.g. Muon-managed ones, or LoRA) start fresh.
    Returns (n_loaded, n_fresh)."""
    by_name = flat_saved_state(saved_sd, model, saved_names)
    name_of = {p: n for n, p in model.named_parameters()}
    opts = optimizer.optimizers if isinstance(optimizer, MultiOpt) else [optimizer]
    n_loaded = n_fresh = 0
    for o in opts:
        if not isinstance(o, torch.optim.AdamW):
            n_fresh += sum(len(g['params']) for g in o.param_groups)
            continue
        for g in o.param_groups:
            for p in g['params']:
                n = name_of.get(p)
                st = by_name.get(n)
                if st is None or n in skip_names:
                    n_fresh += 1
                    continue
                o.state[p] = {k: (v.to(device) if torch.is_tensor(v) and v.dim() > 0 else
                                  (v.clone() if torch.is_tensor(v) else v))
                              for k, v in st.items()}
                n_loaded += 1
    return n_loaded, n_fresh


def base_moments(saved_sd, model, names, device='cpu', saved_names=None):
    """{name: {'exp_avg', 'exp_avg_sq'}} for the given parameter names, from a
    saved optimizer state — the base's end-state moments for AdamMomentAnchor."""
    by_name = flat_saved_state(saved_sd, model, saved_names)
    out = {}
    for n in names:
        st = by_name.get(n)
        if st and 'exp_avg' in st:
            out[n] = {'exp_avg': st['exp_avg'].detach().clone().to(device),
                      'exp_avg_sq': st['exp_avg_sq'].detach().clone().to(device)}
    return out
