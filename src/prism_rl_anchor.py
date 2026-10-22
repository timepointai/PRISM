"""prism_rl_anchor.py — the weight-space anchor as a drop-in for HF/TRL trainers,
plus the drift metrics the RL pilot (W4) reports.

The anchor is a post-optimizer-step parameter update, never a loss over logits:

    W  <-  (1 - s)·W + s·W_target        after every optimizer step

so it needs ZERO reference-model forward passes. `AnchorCallback` is a plain
`transformers.TrainerCallback`; it captures W₀ at `on_train_begin` and applies the
pull in `on_optimizer_step` (fires after `optimizer.step()`, before `zero_grad`),
which every TRL trainer inherits (GRPOTrainer, DPOTrainer, SFTTrainer). Modes:

  target='fixed'        pull toward the pre-RL weights W₀ (PRISM's anchor)
  target='ema:β'        pull toward a slow EMA of the policy (the WARP-style anchor,
                        arXiv 2406.16768 stage 1 — their EMA is the KL reference;
                        here it is the lerp target, with NO KL)
  liti=η                WARP's stage 3 at the end of training: θ ← (1−η)·θ_init + η·θ
  adam='lerp'           also lerp Adam's moments toward their initial state (TVD-style)
  lora=True             anchor the LoRA delta exactly: B ← (1−s)·B  (prism_anchor.lora_delta_pull)

`drift_metrics(model, w0, k, alpha)` returns the KL-free drift readouts:
  * relative weight drift ‖W−W₀‖/‖W₀‖ per group and overall;
  * `principal_overlap`: the fraction of the update's top-α entries that fall in the
    principal mask of W₀ (top-α |rank-k reconstruction of W₀|) — the mask-counting
    metric of "The Path Not Taken" (arXiv 2511.08567), chance = α;
  * `top_subspace_energy`: the share of ‖Δ‖_F² inside W₀'s top-k singular subspaces
    (our complementary continuous reading; chance ≈ k/d on each side).

Step-time accounting: `StepTimer` logs the mean optimizer-step wall-clock and the
share spent inside the callback, so the anchor's own cost is reported, and the
pilot driver measures the reference forward pass's share for the KL arms by
differencing β=0 vs β>0 step times on the same hardware.
"""
import time

import torch

try:
    from transformers import TrainerCallback
except ImportError:                       # keeps the module importable for selftests
    class TrainerCallback:                # type: ignore
        pass

from prism_anchor import parse_target_mode, lora_delta_pull


def _unwrap(model):
    for attr in ('module', '_orig_mod'):
        while hasattr(model, attr):
            model = getattr(model, attr)
    return model


def anchored_params(model, lora=False, min_dim=2):
    """[(name, param)] the anchor acts on: every trainable ≥2-D weight (full
    finetuning) or the LoRA B matrices (lora=True)."""
    m = _unwrap(model)
    if lora:
        return [(n, p) for n, p in m.named_parameters() if p.requires_grad and 'lora_B' in n]
    return [(n, p) for n, p in m.named_parameters() if p.requires_grad and p.dim() >= min_dim]


class AnchorCallback(TrainerCallback):
    def __init__(self, strength, target='fixed', adam='keep', lora=False, liti=0.0,
                 device='cpu', log_every=50):
        self.s = float(strength)
        self.mode, self.mode_val = parse_target_mode(target)
        self.adam = adam
        self.lora = lora
        self.liti = float(liti)
        self.device = device
        self.log_every = log_every
        self.targets, self.init_weights, self.m0 = {}, {}, {}
        self.params = []
        self.timer = StepTimer()
        self.n_steps = 0

    # ── capture ──
    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kw):
        self.params = anchored_params(model, lora=self.lora)
        if self.lora:
            # the LoRA delta's anchor is B ← (1−s)·B; no target copy needed, but keep
            # the init for LITI (θ_init = base weights, delta 0)
            self.init_weights = {}
        else:
            self.targets = {n: p.detach().to(self.device, copy=True) for n, p in self.params}
            self.init_weights = self.targets if self.mode == 'fixed' else \
                {n: p.detach().to(self.device, copy=True) for n, p in self.params}
        self.optimizer = optimizer
        print(f"[anchor] attached to {len(self.params)} tensors "
              f"({sum(p.numel() for _, p in self.params):,} params): s={self.s} target={self.mode}"
              + (f":{self.mode_val}" if self.mode != 'fixed' else '') + f" adam={self.adam} "
              f"lora={self.lora} liti={self.liti}", flush=True)

    # ── the pull ──
    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, model=None, optimizer=None, **kw):
        t0 = time.perf_counter()
        opt = optimizer or getattr(self, 'optimizer', None)
        if self.s > 0:
            if self.lora:
                lora_delta_pull([p for _, p in self.params], self.s)
            else:
                for n, p in self.params:
                    t = self.targets[n]
                    p.lerp_(t.to(p.device, non_blocking=True), self.s)
                    if self.mode == 'ema':
                        t.lerp_(p.detach().to(t.device), self.mode_val)
            if self.adam == 'lerp' and opt is not None and not self.lora:
                for n, p in self.params:
                    st = opt.state.get(p)
                    if not st or 'exp_avg' not in st:
                        continue
                    if n not in self.m0:        # capture the first moments seen as the base
                        self.m0[n] = (st['exp_avg'].detach().to(self.device, copy=True),
                                      st['exp_avg_sq'].detach().to(self.device, copy=True))
                        continue
                    m_b, v_b = self.m0[n]
                    st['exp_avg'].lerp_(m_b.to(p.device), self.s)
                    v = st['exp_avg_sq']
                    v.copy_(((1 - self.s) * v.sqrt() + self.s * v_b.to(p.device).sqrt()) ** 2)
        if p_dev_sync(self.params):
            torch.cuda.synchronize()
        self.timer.add(time.perf_counter() - t0)
        self.n_steps += 1

    # ── WARP-style end interpolation toward the init ──
    @torch.no_grad()
    def on_train_end(self, args, state, control, model=None, **kw):
        if self.liti > 0 and not self.lora:
            for n, p in self.params:
                w0 = self.init_weights[n].to(p.device)
                p.copy_((1.0 - self.liti) * w0 + self.liti * p)
            print(f"[anchor] LITI applied: θ ← (1−{self.liti})·θ_init + {self.liti}·θ", flush=True)
        elif self.liti > 0 and self.lora:
            lora_delta_pull([p for _, p in self.params], 1.0 - self.liti)
            print(f"[anchor] LITI (LoRA form): B ← {self.liti}·B", flush=True)
        print(f"[anchor] {self.n_steps} steps, callback mean {self.timer.mean_ms():.2f} ms/step", flush=True)


def p_dev_sync(params):
    return bool(params) and params[0][1].is_cuda


class StepTimer:
    def __init__(self):
        self.total, self.n = 0.0, 0

    def add(self, dt):
        self.total += dt
        self.n += 1

    def mean_ms(self):
        return 1000.0 * self.total / max(1, self.n)


class StepTimeCallback(TrainerCallback):
    """Mean wall-clock per optimizer step (on_step_begin → on_step_end), so the
    reference-forward share of a KL arm can be read against the β=0 arm."""

    def __init__(self):
        self.t0, self.timer = None, StepTimer()

    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        if self.t0 is not None:
            self.timer.add(time.perf_counter() - self.t0)

    def mean_ms(self):
        return self.timer.mean_ms()


# ── drift metrics ──

@torch.no_grad()
def principal_mask(W0, k, alpha):
    """Top-α fraction of entries of the rank-k reconstruction of W₀, by |value|
    (arXiv 2511.08567's 'principal weights')."""
    U, S, Vh = torch.linalg.svd(W0.float(), full_matrices=False)
    kk = min(k, S.numel())
    rec = (U[:, :kk] * S[:kk]) @ Vh[:kk]
    n = rec.numel()
    m = max(1, int(round(alpha * n)))
    thr = rec.abs().flatten().kthvalue(n - m + 1).values
    return rec.abs() >= thr


@torch.no_grad()
def update_mask(delta, alpha):
    n = delta.numel()
    m = max(1, int(round(alpha * n)))
    thr = delta.abs().flatten().kthvalue(n - m + 1).values
    return delta.abs() >= thr


@torch.no_grad()
def drift_metrics(model, w0, k=256, alpha=0.05, max_matrices=64, groups=None):
    """w0: {name: tensor} of the pre-RL weights. Returns per-group and overall
    relative drift, principal-mask overlap (chance = alpha) and top-k subspace
    energy share of the update. Samples at most `max_matrices` 2-D matrices,
    evenly across the list, to bound the SVD cost."""
    m = _unwrap(model)
    params = {n: p for n, p in m.named_parameters()}
    names = [n for n in w0 if n in params and params[n].dim() == 2 and 'lora' not in n]
    if len(names) > max_matrices:
        step = len(names) / max_matrices
        names = [names[int(i * step)] for i in range(max_matrices)]
    per_group = {}
    tot_d2 = tot_w2 = 0.0
    for n in names:
        W = params[n].detach().float()
        W0 = w0[n].to(W.device).float()
        if W.shape != W0.shape:
            continue
        delta = W - W0
        d2, w2 = float(delta.pow(2).sum()), float(W0.pow(2).sum())
        tot_d2 += d2
        tot_w2 += w2
        g = (groups(n) if groups else n.split('.')[-2]) or 'other'
        rec = per_group.setdefault(g, {'d2': 0.0, 'w2': 0.0, 'overlap': [], 'energy': []})
        rec['d2'] += d2
        rec['w2'] += w2
        if d2 > 0:
            pm = principal_mask(W0, k, alpha)
            um = update_mask(delta, alpha)
            rec['overlap'].append(float((pm & um).sum()) / float(um.sum()))
            U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
            kk = min(k, S.numel())
            Uk, Vk = U[:, :kk], Vh[:kk].T
            top = Uk @ (Uk.T @ delta) + (delta @ Vk) @ Vk.T - Uk @ (Uk.T @ delta @ Vk) @ Vk.T
            rec['energy'].append(float(top.pow(2).sum()) / d2)
    out = {'overall_rel_drift': (tot_d2 / tot_w2) ** 0.5 if tot_w2 > 0 else 0.0,
           'k': k, 'alpha': alpha, 'n_matrices': len(names), 'groups': {}}
    for g, r in per_group.items():
        out['groups'][g] = {
            'rel_drift': (r['d2'] / r['w2']) ** 0.5 if r['w2'] > 0 else 0.0,
            'principal_overlap': sum(r['overlap']) / len(r['overlap']) if r['overlap'] else None,
            'top_subspace_energy': sum(r['energy']) / len(r['energy']) if r['energy'] else None,
        }
    return out


@torch.no_grad()
def kl_to_reference(policy, reference, batches, device, max_batches=8):
    """Mean per-token KL(π_θ ‖ π_ref) on held-out prompt+completion batches
    (input_ids, attention_mask tensors). Reported for every arm, including the
    anchor arms, which never compute it during training."""
    policy.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(batches):
        if i >= max_batches:
            break
        ids = b['input_ids'].to(device)
        am = b['attention_mask'].to(device)
        lp = torch.log_softmax(policy(input_ids=ids, attention_mask=am).logits.float(), -1)
        lr = torch.log_softmax(reference(input_ids=ids, attention_mask=am).logits.float(), -1)
        kl = (lp.exp() * (lp - lr)).sum(-1)
        mask = am.bool()
        tot += float((kl * mask).sum())
        n += int(mask.sum())
    policy.train()
    return tot / max(1, n)
