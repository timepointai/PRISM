# prime-rl: manual edit to attach the PRISM anchor

This is a manual-edit note, not a patch file. It replaces `prime_rl_anchor.patch`,
which could not be applied: it had a bare `@@` hunk header and a literal `(...)`,
and was not generated against a pinned prime-rl commit.

**File:** `src/prime_rl/trainer/rl/train.py` (read 2026-10-21; no commit pinned,
so re-check the call before editing).

**Where:** directly after the call that builds the optimizer, which on 2026-10-21
read as

```python
optimizer, gradient_manager = setup_optimizer(
    config.optim,
    list(model.named_parameters()),
    parallel_dims,
    # ... remaining keyword arguments unchanged ...
)
```

**Add:**

```python
from prime_rl_anchor import attach_anchor          # PRISM: post-step weight-space anchor (PRISM_ANCHOR_S)
attach_anchor(optimizer, model)
```

with `examples/prime_rl_anchor.py` on `PYTHONPATH`. Strength comes from
`PRISM_ANCHOR_S` (0 or unset = off).

**Not verified:** this edit has not been run inside prime-rl, and behaviour under
FSDP is unverified. In particular, `attach_anchor` only anchors parameters with
`p.dim() >= 2`; under a wrapper that exposes flattened 1-D parameters (FSDP1
`use_orig_params=False`), it would anchor nothing. Check the
`[prism anchor] attached to N tensors` line before trusting a run.
