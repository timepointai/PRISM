"""prime-rl plugin: attach the PRISM anchor to the trainer's optimizer with one call.
Keep this file outside prime-rl (e.g. on PYTHONPATH) and apply prime_rl_anchor.patch,
which inserts the single `attach_anchor(...)` line after `setup_optimizer(...)` in
src/prime_rl/trainer/rl/train.py. Strength comes from PRISM_ANCHOR_S (0 = off).

Under FSDP every rank sees its own parameter shards; capturing W0 per shard and
lerping per shard is the same operation as on the full tensor.
"""
import os
import torch


def attach_anchor(optimizer, model, strength=None, target="fixed", ema_beta=0.01):
    s = float(os.environ.get("PRISM_ANCHOR_S", "0") if strength is None else strength)
    if s <= 0:
        return None
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    targets = {n: p.detach().clone() for n, p in params}      # W0, on the param's device

    @torch.no_grad()
    def hook(opt, args, kwargs):
        for n, p in params:
            t = targets[n]
            p.lerp_(t, s)
            if target == "ema":
                t.lerp_(p, ema_beta)

    handle = optimizer.register_step_post_hook(hook)
    print(f"[prism anchor] attached to {len(params)} tensors, s={s}, target={target}", flush=True)
    return handle
