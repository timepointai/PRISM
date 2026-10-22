"""prism_rl_pilot.py — the RL-anchor pilot (W4): GRPO on GSM8K with TRL, one arm
per run, in drift regimes where RL actually drifts. Writes a `prism-rl/1`
artifact per run; `--score` joins several into one table.

    python prism_rl_pilot.py --model Qwen/Qwen2.5-0.5B-Instruct --arm beta0 --regime onpolicy --steps 300 --seed 1337 --tag w4
    python prism_rl_pilot.py --model ... --arm kl:0.04     ...
    python prism_rl_pilot.py --model ... --arm anchor:0.0025 ...
    python prism_rl_pilot.py --model ... --arm anchor_ema:0.0025:0.01 ...
    python prism_rl_pilot.py --model ... --arm warp:0.0025:0.01:0.7 ...     # EMA anchor + LITI(η)
    python prism_rl_pilot.py --model ... --arm anchor_adam:0.0025 ...        # + Adam-moment lerp
    python prism_rl_pilot.py --model ... --regime recycle:4 | stale:4        # drift regimes
    python prism_rl_pilot.py --score results/rl_*.json --tag w4

Arms (β is TRL's KL coefficient; the anchor arms run with β = 0, i.e. NO
reference model is loaded — zero reference forward passes):
  beta0            GRPO as the stacks ship it (TRL default β = 0.0)
  kl:<β>           GRPO + KL-to-reference at β (the thing the anchor replaces)
  anchor:<s>       per-step lerp toward the pre-RL weights W₀ (prism_rl_anchor.AnchorCallback)
  anchor_ema:<s>:<β_ema>   lerp toward a slow EMA of the policy (WARP-style anchor, no KL)
  warp:<s>:<β_ema>:<η>     EMA anchor + end-of-training LITI θ ← (1−η)θ_init + ηθ
  anchor_adam:<s>  fixed-W₀ anchor + Adam-moment lerp (TVD-style)
Regimes:
  onpolicy         one optimizer step per generation batch (TRL num_iterations=1)
  recycle:<m>      each rollout batch reused m times (TRL num_iterations=m)
  stale:<k>        one generation batch serves k optimizer steps (TRL steps_per_generation=k,
                   so the policy that trains is up to k steps ahead of the one that sampled)
  zero_rl          after RL, LoRA-finetune on IMDB (SFT, 200 steps) and re-score GSM8K
                   (arXiv 2609.33898's fine-tunability protocol); add `--zero_rl` to any arm.

Metrics per run: GSM8K accuracy (greedy, held-out test subset) before and after;
retention on a pinned small lm-eval suite (hellaswag, arc_easy, winogrande;
`--retention none` to skip); KL(π_θ‖π_init) on held-out prompts (computed for
every arm after training — the anchor arms never see it during training);
weight drift / principal-mask overlap / top-k subspace energy (prism_rl_anchor.
drift_metrics, arXiv 2511.08567's metric); mean optimizer-step wall-clock and the
anchor callback's own cost; the reference-forward share is read off β=0 vs β>0
step times in `--score`.

Single-seed runs are PROBES and are labelled so in the artifact.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SRC_DIR), 'results')

SYSTEM = ("Solve the math problem. Think step by step, then give the final numeric answer "
          "on the last line in the form '#### <number>'.")


def _num(s):
    s = s.replace(',', '')
    m = re.findall(r'-?\d+(?:\.\d+)?', s)
    return m[-1] if m else None


def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '')
    return _num(text)


def gsm8k_reward(completions, answer, **kwargs):
    out = []
    for c, a in zip(completions, answer):
        text = c[0]['content'] if isinstance(c, list) else c
        got = extract_answer(text)
        out.append(1.0 if got is not None and got == a else 0.0)
    return out


def load_gsm8k(n_train=None, n_eval=300, seed=0):
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main')
    def prep(ex):
        return {'prompt': [{'role': 'system', 'content': SYSTEM},
                           {'role': 'user', 'content': ex['question']}],
                'answer': ex['answer'].split('####')[-1].strip().replace(',', '')}
    train = ds['train'].shuffle(seed=seed).map(prep, remove_columns=ds['train'].column_names)
    if n_train:
        train = train.select(range(min(n_train, len(train))))
    test = ds['test'].shuffle(seed=seed).select(range(min(n_eval, len(ds['test']))))
    test = test.map(prep, remove_columns=ds['test'].column_names)
    return train, test


@torch.no_grad()
def gsm8k_accuracy(model, tok, test, device, max_new_tokens=256, batch_size=16):
    model.eval()
    correct = 0
    for i in range(0, len(test), batch_size):
        rows = test[i:i + batch_size]
        prompts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                   for p in rows['prompt']]
        enc = tok(prompts, return_tensors='pt', padding=True, padding_side='left').to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        for t, a in zip(texts, rows['answer']):
            correct += int(extract_answer(t) == a)
    model.train()
    return correct / len(test)


def retention_suite(model, tok, tasks, limit, device):
    """lm-eval-harness (pinned 0.4.13) on a small suite; returns {task: acc}."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16, device=device)
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=limit, num_fewshot=0,
                                  log_samples=False)
    out = {}
    for t, r in res['results'].items():
        acc = r.get('acc_norm,none', r.get('acc,none'))
        out[t] = float(acc) if acc is not None else None
    return out


def parse_arm(spec):
    kind, _, rest = spec.partition(':')
    vals = [float(v) for v in rest.split(':') if v] if rest else []
    cfg = {'arm': spec, 'kind': kind, 'beta': 0.0, 'anchor': None}
    if kind == 'beta0':
        pass
    elif kind == 'kl':
        cfg['beta'] = vals[0]
    elif kind == 'anchor':
        cfg['anchor'] = dict(strength=vals[0], target='fixed')
    elif kind == 'anchor_ema':
        cfg['anchor'] = dict(strength=vals[0], target=f'ema:{vals[1]}')
    elif kind == 'warp':
        cfg['anchor'] = dict(strength=vals[0], target=f'ema:{vals[1]}', liti=vals[2])
    elif kind == 'anchor_adam':
        cfg['anchor'] = dict(strength=vals[0], target='fixed', adam='lerp')
    else:
        raise SystemExit(f'unknown arm {spec}')
    return cfg


def parse_regime(spec):
    kind, _, rest = spec.partition(':')
    if kind == 'onpolicy':
        return {'regime': spec, 'num_iterations': 1, 'steps_per_generation': 1}
    if kind == 'recycle':
        return {'regime': spec, 'num_iterations': int(rest), 'steps_per_generation': 1}
    if kind == 'stale':
        return {'regime': spec, 'num_iterations': 1, 'steps_per_generation': int(rest)}
    raise SystemExit(f'unknown regime {spec}')


def imdb_lora_finetune(model, tok, device, steps=200, lr=2e-4, r=8, seed=0, max_len=512):
    """arXiv 2609.33898's fine-tunability probe: a LoRA SFT on IMDB reviews after RL."""
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    ds = load_dataset('stanfordnlp/imdb', split='train').shuffle(seed=seed).select(range(steps * 8))
    texts = [t for t in ds['text']]
    peft_model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, target_modules='all-linear',
                                                  task_type='CAUSAL_LM'))
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=lr)
    peft_model.train()
    bs = 8
    for i in range(steps):
        batch = texts[i * bs:(i + 1) * bs]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
        out = peft_model(**enc, labels=enc['input_ids'].masked_fill(enc['attention_mask'] == 0, -100))
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i % 50 == 0:
            print(f'[zero_rl] imdb sft step {i} loss {out.loss.item():.4f}', flush=True)
    merged = peft_model.merge_and_unload()
    merged.eval()
    return merged


def main():
    p = argparse.ArgumentParser(description='PRISM RL-anchor pilot (TRL GRPO on GSM8K)')
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--arm', default='beta0')
    p.add_argument('--regime', default='onpolicy')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--per_device_bs', type=int, default=8)
    p.add_argument('--grad_accum', type=int, default=4)
    p.add_argument('--num_generations', type=int, default=8)
    p.add_argument('--max_completion_length', type=int, default=256)
    p.add_argument('--eval_n', type=int, default=300)
    p.add_argument('--retention', default='hellaswag,arc_easy,winogrande', help="'none' to skip")
    p.add_argument('--retention_limit', type=int, default=300)
    p.add_argument('--kl_prompts', type=int, default=32)
    p.add_argument('--zero_rl', action='store_true')
    p.add_argument('--drift_k', type=int, default=256)
    p.add_argument('--drift_alpha', type=float, default=0.05)
    p.add_argument('--continuous_batching', type=int, default=1,
                   help='transformers continuous batching for generation (needs transformers>=5.8)')
    p.add_argument('--bf16', type=int, default=1)
    p.add_argument('--device', default=None)
    p.add_argument('--tag', default='')
    p.add_argument('--score', nargs='*', default=None, help='artifact paths to join into a table')
    a = p.parse_args()
    if a.score is not None:
        return score(a.score, a.tag)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    sys.path.insert(0, SRC_DIR)
    from prism_rl_anchor import AnchorCallback, StepTimeCallback, drift_metrics, kl_to_reference

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed)
    arm, regime = parse_arm(a.arm), parse_regime(a.regime)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name = f"rl_{arm['kind']}_{regime['regime'].replace(':', '')}_s{a.seed}" + (f'_{a.tag}' if a.tag else '')
    out_dir = f'out-rl-{name}'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f'[rl] arm={arm} regime={regime} steps={a.steps} seed={a.seed} device={device}', flush=True)

    dtype = torch.bfloat16 if (a.bf16 and device == 'cuda') else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(device)
    init_model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(device).eval()
    for q in init_model.parameters():
        q.requires_grad_(False)
    w0 = {n: q.detach().clone().cpu() for n, q in model.named_parameters() if q.dim() == 2}

    train, test = load_gsm8k(n_train=a.steps * a.per_device_bs * a.grad_accum * regime['steps_per_generation']
                             // a.num_generations + 64, n_eval=a.eval_n, seed=a.seed)
    t0 = time.time()
    acc_before = gsm8k_accuracy(model, tok, test, device, a.max_completion_length)
    print(f'[rl] gsm8k acc before: {acc_before:.3f} ({time.time() - t0:.0f}s)', flush=True)
    ret_tasks = [t for t in a.retention.split(',') if t and a.retention != 'none']
    ret_before = retention_suite(model, tok, ret_tasks, a.retention_limit, device) if ret_tasks else {}
    if ret_tasks:
        print(f'[rl] retention before: {ret_before}', flush=True)

    callbacks = [StepTimeCallback()]
    anchor_cb = None
    if arm['anchor']:
        anchor_cb = AnchorCallback(**arm['anchor'], device='cpu' if device == 'cpu' else device)
        callbacks.append(anchor_cb)
    cfg = GRPOConfig(
        output_dir=out_dir, seed=a.seed, max_steps=a.steps, learning_rate=a.lr,
        per_device_train_batch_size=a.per_device_bs, gradient_accumulation_steps=a.grad_accum,
        num_generations=a.num_generations, max_completion_length=a.max_completion_length,
        beta=arm['beta'],
        num_iterations=regime['num_iterations'], steps_per_generation=regime['steps_per_generation'],
        logging_steps=1, save_strategy='no', report_to='none', bf16=bool(a.bf16 and device == 'cuda'),
        gradient_checkpointing=(device == 'cuda'), use_cpu=(device == 'cpu'),
        use_transformers_continuous_batching=bool(a.continuous_batching and device == 'cuda'),
        temperature=1.0, loss_type='dapo',
    )
    trainer = GRPOTrainer(model=model, reward_funcs=gsm8k_reward, args=cfg, train_dataset=train,
                          processing_class=tok, callbacks=callbacks)
    t1 = time.time()
    trainer.train()
    train_wall = time.time() - t1
    log = trainer.state.log_history
    reward_curve = {str(r['step']): r.get('reward') for r in log if 'reward' in r}
    kl_curve = {str(r['step']): r.get('kl') for r in log if 'kl' in r}

    policy = trainer.model
    for attr in ('module', '_orig_mod'):
        while hasattr(policy, attr):
            policy = getattr(policy, attr)
    policy.eval()
    acc_after = gsm8k_accuracy(policy, tok, test, device, a.max_completion_length)
    ret_after = retention_suite(policy, tok, ret_tasks, a.retention_limit, device) if ret_tasks else {}
    # KL to init on held-out prompts, completions sampled from the INIT model (fixed across arms)
    kl_batches = []
    with torch.no_grad():
        held = test.select(range(min(a.kl_prompts, len(test))))
        prompts = [tok.apply_chat_template(pp, tokenize=False, add_generation_prompt=True) for pp in held['prompt']]
        for i in range(0, len(prompts), 8):
            enc = tok(prompts[i:i + 8], return_tensors='pt', padding=True, padding_side='left').to(device)
            gen = init_model.generate(**enc, max_new_tokens=128, do_sample=True, temperature=1.0,
                                      pad_token_id=tok.pad_token_id)
            kl_batches.append({'input_ids': gen, 'attention_mask': (gen != tok.pad_token_id).long()})
    kl_init = kl_to_reference(policy, init_model, kl_batches, device)
    drift = drift_metrics(policy, w0, k=a.drift_k, alpha=a.drift_alpha)
    step_ms = callbacks[0].mean_ms()
    anchor_ms = anchor_cb.timer.mean_ms() if anchor_cb else 0.0
    print(f'[rl] gsm8k acc after: {acc_after:.3f}  retention after: {ret_after}  KL(init) {kl_init:.4f}  '
          f'drift {drift["overall_rel_drift"]:.4f}  step {step_ms:.0f} ms (anchor {anchor_ms:.2f} ms)', flush=True)

    zero_rl = None
    if a.zero_rl:
        ft = imdb_lora_finetune(policy, tok, device, seed=a.seed)
        acc_post = gsm8k_accuracy(ft, tok, test, device, a.max_completion_length)
        zero_rl = {'protocol': 'LoRA r=8 SFT on IMDB, 200 steps × 8, lr 2e-4 (arXiv 2609.33898-style)',
                   'gsm8k_acc_after_imdb': acc_post, 'drop': acc_after - acc_post}
        print(f'[rl] zero-RL fine-tunability: gsm8k {acc_after:.3f} → {acc_post:.3f}', flush=True)

    art = {
        'schema': 'prism-rl/1', 'partial': False, 'name': name, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'probe': True, 'probe_note': 'single seed — a probe, not a result (3 seeds are Tier B)',
        'config': {'model': a.model, 'arm': arm, 'regime': regime, 'steps': a.steps, 'seed': a.seed,
                   'lr': a.lr, 'per_device_bs': a.per_device_bs, 'grad_accum': a.grad_accum,
                   'num_generations': a.num_generations, 'max_completion_length': a.max_completion_length,
                   'eval_n': a.eval_n, 'retention_tasks': ret_tasks, 'retention_limit': a.retention_limit,
                   'trl': __import__('trl').__version__, 'transformers': __import__('transformers').__version__,
                   'torch': torch.__version__, 'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
                   'loss_type': 'dapo', 'kl_beta': arm['beta'], 'reference_model_loaded': arm['beta'] > 0,
                   'argv': sys.argv[1:]},
        'gsm8k_acc': {'before': acc_before, 'after': acc_after},
        'retention': {'before': ret_before, 'after': ret_after},
        'kl_to_init': kl_init, 'drift': drift,
        'timing': {'train_wall_sec': round(train_wall, 1), 'step_ms_mean': step_ms, 'anchor_callback_ms': anchor_ms},
        'reward_curve': reward_curve, 'kl_curve': kl_curve, 'zero_rl': zero_rl,
    }
    path = os.path.join(RESULTS_DIR, f'{name}_{stamp}.json')
    json.dump(art, open(path, 'w'), indent=2)
    print(f'[rl] Artifact: results/{os.path.basename(path)} — COMMIT THIS FILE.', flush=True)


def score(paths, tag):
    rows = [json.load(open(p)) for p in paths]
    beta0 = {r['config']['regime']['regime']: r for r in rows if r['config']['arm']['kind'] == 'beta0'}
    print(f"{'arm':22} {'regime':10} {'acc→':>12} {'ret(avg)':>9} {'KL(init)':>9} {'drift':>7} {'overlap':>8} {'step ms':>8} {'ref share':>9}")
    for r in rows:
        c = r['config']
        ret = [v for v in r['retention']['after'].values() if v is not None]
        ret_avg = sum(ret) / len(ret) if ret else float('nan')
        ov = [g['principal_overlap'] for g in r['drift']['groups'].values() if g['principal_overlap'] is not None]
        ov_avg = sum(ov) / len(ov) if ov else float('nan')
        b0 = beta0.get(c['regime']['regime'])
        share = ''
        if b0 and c['arm']['kind'] == 'kl' and r['timing']['step_ms_mean'] > 0:
            share = f"{(r['timing']['step_ms_mean'] - b0['timing']['step_ms_mean']) / r['timing']['step_ms_mean']:.1%}"
        print(f"{c['arm']['arm']:22} {c['regime']['regime']:10} {r['gsm8k_acc']['before']:.3f}→{r['gsm8k_acc']['after']:.3f} "
              f"{ret_avg:9.3f} {r['kl_to_init']:9.4f} {r['drift']['overall_rel_drift']:7.4f} {ov_avg:8.3f} "
              f"{r['timing']['step_ms_mean']:8.0f} {share:>9}")
    out = {'schema': 'prism-rl-table/1', 'tag': tag, 'runs': [os.path.basename(p) for p in paths],
           'note': 'single-seed probes; the reference-forward share is (kl step − β0 step)/kl step on the same regime'}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    json.dump(out, open(os.path.join(RESULTS_DIR, f'rl_table_{stamp}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
