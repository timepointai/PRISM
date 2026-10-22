# The anchor as a drop-in: recipes per training stack

*Alpha-moves W6.1. Written 2026-10-21 against the platform docs as they read that
day; every "supports / does not expose" statement below names its source. Verify
before you build on it.*

## What the anchor is, exactly

PRISM's retention anchor is a **post-optimizer-step parameter update**:

```
after every optimizer.step():    W  ←  (1 − s)·W  +  s·W_target
```

with `W_target = W₀` (the pre-finetune weights) and `s` the one scalar dial
(measured bands at probe scale: s ≈ 0.0025 injects new material at parity with
plain training and forgets ~3× less; s ≈ 0.01–0.02 is maximum retention and
blocks new-fact storage — [`RESULTS.md`](../RESULTS.md) §12–13). It needs no
reference-model forward pass and no SVD. `src/prism_anchor.py` is the
framework-agnostic object; `src/prism_rl_anchor.py` is the HF/TRL callback.

**It is not a loss over logits**, and it is not an L2 penalty either. Under
Adam an L2-to-W₀ penalty's gradient `2λ(W − W₀)` is divided coordinate-wise by
`√v̂`, so the effective pull is anisotropic and coupled to the learning rate
and the moment estimates; the post-step lerp removes a fixed fraction of the
gap every step regardless of optimizer state. Where a platform exposes only a
logit loss, the closest faithful equivalents are given per stack below, with
the difference stated.

### Two exact identities worth knowing

* **LoRA form.** For `W = W₀ + (α/r)·B·A`, `lerp(W, W₀, s) = W₀ + (1−s)(α/r)·B·A`,
  i.e. **`B ← (1 − s)·B`**. Exact (selftested); `prism_anchor.lora_delta_pull`.
* **Decoupled weight decay on the delta.** AdamW's decoupled decay does
  `θ ← θ − lr·wd·θ` before the step. Applied to **B alone** (A, base frozen) with
  `wd = s/lr`, that is `B ← (1 − s)·B` — the LoRA anchor. On full-parameter
  weights it is *not* the anchor: decay pulls toward 0, not toward W₀.

## Stack by stack

| stack | extension surface (verified) | anchor form | status |
|---|---|---|---|
| plain PyTorch | `optimizer.register_step_post_hook` | exact | `examples/anchor_pytorch.py` |
| HF `Trainer` / TRL (`SFTTrainer`, `DPOTrainer`, `GRPOTrainer`) | `TrainerCallback.on_optimizer_step` (fires after `optimizer.step()`, before `zero_grad`; receives `model`, `optimizer`) — [transformers callbacks](https://huggingface.co/docs/transformers/main/en/main_classes/callback); TRL trainers accept `callbacks=[...]` ([TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)) | exact | `src/prism_rl_anchor.AnchorCallback`, `examples/anchor_trl_grpo.py` |
| prime-rl | no callback or hook mechanism in the trainer (its `docs/algorithms.md` documents custom losses via `import_path`, advantage/filter plugins, and no optimizer hook); `optimizer.step()` is a single call in `src/prime_rl/trainer/rl/train.py` | exact, via a one-line patch registering the post-step hook on the optimizer `setup_optimizer` returns (works under FSDP: the hook sees each rank's shards and W₀ is captured per shard) | `examples/prime_rl_anchor.py` + `examples/prime_rl_anchor.patch` (not a fork) |
| Tinker (Thinking Machines) | `forward_backward` / `optim_step` / `sample` / `save_state` primitives and custom loss functions ([docs](https://tinker-docs.thinkingmachines.ai/)); LoRA is the headline training mode; **no documented read/write of adapter weights between steps** | **not expressible exactly.** Closest: LoRA decoupled weight decay on B with `wd = s/lr` *if* `AdamParams.weight_decay` can be scoped to B — the docs do not say it can; applied to A and B it is a different operator (it also shrinks A). A logit-space KL to the base is NOT equivalent (function-space, needs a reference forward). | unverified; recipe written as a question to the platform |
| Together Post-Training | `sample_batch` / `forward_backward` / `optim_step` / `weights_sync`; custom losses "computed locally and sent to the Trainer" from logprobs; adapters on a frozen base ([docs](https://docs.together.ai/docs/post-training)); **no documented raw-parameter access** | same as Tinker: not exact; closest is B-only decay with `wd = s/lr` if scoping is possible; otherwise a logit-space proximal term, which is a different (function-space) regulariser | unverified |
| verl / OpenRLHF | not read this session | — | not claimed |

**Claim no platform support you haven't verified.** The two hosted platforms
expose a loss surface, not a parameter surface; the honest statement is that
the anchor is a one-line change *for the platform*, and that its faithful form
is unavailable to the customer until the platform exposes a post-step hook or
B-only decay.

## Recipes

### Plain PyTorch — `examples/anchor_pytorch.py`

```python
from prism_anchor import Anchor, capture_targets
anchor = Anchor(model.named_parameters(), capture_targets(model.named_parameters()))
anchor.attach(optimizer, strength_fn=lambda: 0.0025)      # register_step_post_hook
for batch in loader:
    loss = model(**batch).loss; loss.backward(); optimizer.step(); optimizer.zero_grad()
```

### HF Trainer / TRL — `examples/anchor_trl_grpo.py`

```python
from prism_rl_anchor import AnchorCallback
trainer = GRPOTrainer(model=..., reward_funcs=..., args=GRPOConfig(beta=0.0, ...),
                      callbacks=[AnchorCallback(strength=0.0025, target='fixed')])
```

`beta=0.0` means TRL loads no reference model: the anchor is the only drift
control and costs one lerp per parameter per step (`[anchor]` log line gives
the measured ms). `target='ema:0.01'` gives the WARP-style EMA anchor
(arXiv 2406.16768, stage 1, but as the lerp target, not as a KL reference);
`liti=η` applies WARP's end-of-training interpolation toward the init.

### prime-rl — `examples/prime_rl_anchor.py`

```python
# after:  optimizer, gradient_manager = setup_optimizer(config.optim, list(model.named_parameters()), ...)
from prime_rl_anchor import attach_anchor
attach_anchor(optimizer, model, strength=float(os.environ.get("PRISM_ANCHOR_S", "0")))
```

`examples/prime_rl_anchor.patch` is the two-line diff against
`src/prime_rl/trainer/rl/train.py` (read 2026-10-21; re-check the line before
applying). With FSDP each rank captures its own shards of W₀ at attach time
and lerps them after each step, which is the same operation on the full tensor.

### LoRA anywhere — the exact adapter form

```python
from prism_anchor import lora_delta_pull
optimizer.register_step_post_hook(lambda opt, a, k: lora_delta_pull(lora_B_params, s))
```

or, equivalently, put the B matrices in their own AdamW group with
`weight_decay = s / lr` and `weight_decay = 0` everywhere else.

## What the anchor does NOT buy (say it plainly)

* It is a weight-space pull; on-policy RL that already stays near its start
  (the stacks' default β = 0 is a bet on exactly that) gains little from it —
  which is why W4 tests it in regimes that *do* drift (recycled rollouts, stale
  generation, zero-RL fine-tunability).
* A fixed W₀ target can cost reward in long off-policy runs where a refreshed
  reference wins (arXiv 2610.20814, listing-only); the EMA target is the
  variant to compare.
* The dial's bands were measured at 10.65M; the weak band's location shifts
  with Adam scale (NEXT-EXPERIMENTS Outcome 2 G1) and must be re-located per
  model size.
