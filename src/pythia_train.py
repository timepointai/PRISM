"""pythia_train.py — the core-bet ladder on Pythia (W2b): same architecture, same
token stream, same schedule as the published run; only the initialization differs.

Arms (one run per arm; the harness is resumable and refuses to score a partial run):
    baseline   Pythia's published `step0` weights, untouched.
    prism      `step0` + EigenTransfer (and, unless --imprint False, the Spectral
               Imprint) from the SAME-SIZE published final checkpoint (`main`).
    control    `step0` with an externally built state dict (--control_init), the hook
               for the strongest W2a control from G3 on.

Data: EleutherAI/pile-standard-pythia-preshuffled, read in Pythia's exact order
(pile_mmap.py): training iteration k consumes sequences [k·1024, (k+1)·1024) of
2,048 tokens (targets left-shifted). Only the byte range a run needs is fetched.

Held-out slice: sequences [140000·1024, 140000·1024 + eval_seqs) — iteration
140,000 of the published run. No gate here trains anywhere near it, and every
published checkpoint up to step139000 had not seen it either, so the published
checkpoints can be scored on it to reproduce the published loss curve without
wandb (--published_curve). The final checkpoint (step143000, the teacher) HAS
trained on it; that is noted in the artifact and the teacher is never scored
as a comparator.

Schedule (models/<size>/pythia-<size>.yml): Adam β=(0.9, 0.95), ε=1e-8, weight
decay 0.1 (on ≥2-D params), grad-clip 1.0, cosine to min_lr over 143,000 iters
with 1% (1,430-iter) linear warmup, batch 1,024 × 2,048 tokens. 70M: lr 1e-3 →
1e-4; 160M: 6e-4 → 6e-5; 410M: 3e-4 → 3e-5. A gate stops at --iters but runs
THE SAME schedule the published run ran (no re-scaled cosine), so curves are
step-for-step comparable with the published checkpoints.

Deviations (all recorded in the artifact): bf16 autocast instead of fp16 with
dynamic loss scaling; gradient accumulation over micro-batches on one GPU instead
of 32-way data parallelism (same 1,024-sequence batch, same order inside it);
eval on a held-out slice of the training stream rather than the Pile validation
set; FlashAttention via torch SDPA where available.

Eval lines follow train.py's format (`step N: train loss X, val loss Y`) and the
`[timing]` line, so prism_eval.parse_curve / parse_extras bank them. Artifact
schema `prism-pythia/1`, written to results/pythia_<stamp>.json. `--score` turns
a baseline artifact + a method artifact into steps-to-target, the token-
equivalent multiplier (prism_metrics.py) and the donor-adjusted multiplier
(teacher compute 6·N·300B tokens, amortised).

Smoke (CPU, synthetic 2-layer model and stream, writes a conforming artifact):
    python3 pythia_train.py --dry --arm prism
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from prism_eval import provenance, parse_curve, parse_extras, RESULTS_DIR, log
import pile_mmap as pm
from prism_neox import (load_neox, extract_neox_fingerprint, save_fingerprint, load_fingerprint,
                        apply_prism_neox, synthetic_neox_config)

SIZES = {
    '70m':  dict(repo='EleutherAI/pythia-70m',  lr=1e-3, min_lr=1e-4, n_params=70_426_624,  n_nonemb=18_915_328),
    '160m': dict(repo='EleutherAI/pythia-160m', lr=6e-4, min_lr=6e-5, n_params=162_322_944, n_nonemb=85_056_000),
    '410m': dict(repo='EleutherAI/pythia-410m', lr=3e-4, min_lr=3e-5, n_params=405_334_016, n_nonemb=302_311_424),
}
TOTAL_ITERS = 143_000
WARMUP_ITERS = 1_430            # 1% of 143,000 (yml: warmup 0.01)
HOLDOUT_ITER = 140_000
PUBLISHED_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 143_001, 1000))
PYTHIA_EVAL_TASKS = ['lambada_openai', 'piqa', 'winogrande', 'arc_easy', 'arc_challenge', 'sciq', 'logiqa']
TEACHER_TOKENS = 299_892_736_000


def get_lr(it, lr, min_lr):
    if it < WARMUP_ITERS:
        return lr * (it + 1) / (WARMUP_ITERS + 1)
    if it > TOTAL_ITERS:
        return min_lr
    ratio = (it - WARMUP_ITERS) / (TOTAL_ITERS - WARMUP_ITERS)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


# ── data ──

def prepare_real_data(data_dir, iters, holdout_iter, eval_seqs, token=None):
    """Fetch exactly the byte ranges the run needs: the training prefix for
    iterations [0, iters) and the held-out window. Returns (train_stream, holdout_stream)."""
    os.makedirs(data_dir, exist_ok=True)
    n_train = iters * pm.SEQS_PER_ITER
    a, b = pm.byte_range_for_sequences(0, n_train)
    shards = {}
    for s, (lo, hi) in pm.shards_for_range(a, b).items():
        path = os.path.join(data_dir, pm.shard_name(s))
        need = hi - lo
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have < need:
            log(f'fetching {pm.shard_name(s)} bytes [{lo:,}, {hi:,}) ({(need - have) / 1e9:.2f} GB to go)…', 2)
            pm.fetch_range(pm.shard_name(s), lo, hi, path, token=token)
        shards[s] = (path, lo)
    idx_path = os.path.join(data_dir, f'train_prefix_{n_train}.idx')
    if not os.path.exists(idx_path):
        pm.fetch_idx_prefix(idx_path, n_train, token=token)
    train = pm.PileStream(idx_path, shards)
    h0 = holdout_iter * pm.SEQS_PER_ITER
    ha, hb = pm.byte_range_for_sequences(h0, h0 + eval_seqs)
    hshards = {}
    for s, (lo, hi) in pm.shards_for_range(ha, hb).items():
        path = os.path.join(data_dir, f'holdout_{s}_{lo}_{hi}.bin')
        if not os.path.exists(path) or os.path.getsize(path) < hi - lo:
            log(f'fetching held-out window from {pm.shard_name(s)} bytes [{lo:,}, {hi:,})…', 2)
            pm.fetch_range(pm.shard_name(s), lo, hi, path, token=token)
        hshards[s] = (path, lo)
    win_path = os.path.join(data_dir, f'holdout_{h0}_{eval_seqs}.npz')
    if os.path.exists(win_path):
        z = np.load(win_path)
        window = (int(z['first']), z['sizes'], z['pointers'], np.uint16)
    else:
        window = pm.fetch_idx_window(h0, eval_seqs, token=token)
        np.savez(win_path, first=window[0], sizes=window[1], pointers=window[2])
        window = (window[0], window[1], window[2], window[3])
    hold = pm.PileStream(None, hshards, window=window)
    return train, hold


def prepare_synthetic_data(data_dir, iters, seqs_per_iter, eval_seqs, seq_len=65, vocab=128):
    n = iters * seqs_per_iter + eval_seqs + 8
    idx, shards, _ = pm.write_synthetic_dataset(data_dir, n, seq_len=seq_len, vocab=vocab, seed=1)
    stream = pm.PileStream(idx, {0: shards[0]}, shard_bytes=10 ** 12)
    return stream, stream, iters * seqs_per_iter + 4     # holdout starts after the horizon


# ── model ──

def build_model(a, device):
    """The arm's starting weights. Returns (model, info)."""
    info = {}
    if a.dry:
        torch.manual_seed(1337)
        model = __import__('transformers').GPTNeoXForCausalLM(synthetic_neox_config())
        info['init'] = 'synthetic step0 (seed 1337)'
        if a.arm == 'prism':
            torch.manual_seed(7)
            teacher = __import__('transformers').GPTNeoXForCausalLM(synthetic_neox_config())
            fp = extract_neox_fingerprint(teacher)
            info['teacher'] = 'synthetic (seed 7)'
            apply_prism_neox(model, fp, align=a.align, imprint=a.imprint)
        return model.to(device), info
    size = SIZES[a.size]
    student_repo = a.student_repo or size['repo']
    model = load_neox(student_repo, revision='step0', dtype=torch.float32, device='cpu')
    info['init'] = f'{student_repo}@step0'
    if a.arm == 'prism':
        teacher_repo = a.teacher_repo or size['repo']
        fp_dir = os.path.join(a.data_dir, 'fingerprints', teacher_repo.replace('/', '_') + '_main')
        if not os.path.exists(os.path.join(fp_dir, 'directions.pt')):
            log(f'extracting fingerprint from {teacher_repo}@main …', 2)
            teacher = load_neox(teacher_repo, revision='main', dtype=torch.float32, device='cpu')
            save_fingerprint(extract_neox_fingerprint(teacher), fp_dir)
            del teacher
        apply_prism_neox(model, load_fingerprint(fp_dir), align=a.align, imprint=a.imprint)
        info['teacher'] = f'{teacher_repo}@main (step143000)'
        info['teacher_tokens'] = TEACHER_TOKENS
        info['teacher_params'] = size['n_params']
        info['donor_flops_6ND'] = 6.0 * size['n_params'] * TEACHER_TOKENS
    elif a.arm == 'control':
        assert a.control_init, '--arm control needs --control_init <state_dict.pt>'
        sd = torch.load(a.control_init, map_location='cpu', weights_only=False)
        res = model.load_state_dict(sd, strict=False)
        info['init'] = f'{student_repo}@step0 + control init {a.control_init} (missing {len(res.missing_keys)})'
    return model.to(device), info


def build_optimizer(model, lr, wd=0.1, betas=(0.9, 0.95), eps=1e-8):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW([{'params': decay, 'weight_decay': wd},
                              {'params': nodecay, 'weight_decay': 0.0}],
                             lr=lr, betas=betas, eps=eps, fused=torch.cuda.is_available())


@torch.no_grad()
def evaluate(model, stream, first, n_seqs, micro, device, ctx):
    model.eval()
    tot, cnt = 0.0, 0
    for s in range(first, first + n_seqs, micro):
        m = min(micro, first + n_seqs - s)
        x, y = stream.batch(s, m)
        x = torch.from_numpy(x).to(device); y = torch.from_numpy(y).to(device)
        with ctx:
            out = model(input_ids=x, labels=y)
        tot += out.loss.item() * m
        cnt += m
    model.train()
    return tot / cnt


def published_curve(a, hold, h0, device, ctx):
    """Score the published checkpoints on OUR held-out slice."""
    size = SIZES[a.size]
    repo = a.student_repo or size['repo']
    cache = os.path.join(a.data_dir, f'published_curve_{repo.replace("/", "_")}_{h0}_{a.eval_seqs}.json')
    curve = json.load(open(cache)) if os.path.exists(cache) else {}
    steps = [s for s in PUBLISHED_STEPS if s <= a.iters]
    for s in steps:
        if str(s) in curve:
            continue
        m = load_neox(repo, revision=f'step{s}', dtype=torch.float32, device=device)
        curve[str(s)] = evaluate(m, hold, h0, a.eval_seqs, a.eval_micro, device, ctx)
        log(f'published step{s}: val {curve[str(s)]:.4f}', 4)
        del m
        json.dump(curve, open(cache, 'w'), indent=2)
    return {k: curve[k] for k in sorted(curve, key=int) if int(k) <= a.iters}


def run_lm_eval(model, tokenizer_repo, tasks, limit=None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer_repo, batch_size=16)
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, limit=limit)
    out = {}
    for t, r in res['results'].items():
        out[t] = {k: v for k, v in r.items() if isinstance(v, (int, float))}
    return out


# ── main ──

def main():
    p = argparse.ArgumentParser(description='PRISM core bet on Pythia — one arm per run')
    p.add_argument('--size', default='70m', choices=list(SIZES))
    p.add_argument('--arm', default='baseline', choices=['baseline', 'prism', 'control'])
    p.add_argument('--student_repo', default=None,
                   help='HF repo whose step0 is the init (default EleutherAI/pythia-<size>; '
                        'seeds: EleutherAI/pythia-<size>-seed{1,2,3})')
    p.add_argument('--teacher_repo', default=None, help='HF repo whose main is the teacher (default same size, seed 0)')
    p.add_argument('--align', type=float, default=0.75)
    p.add_argument('--imprint', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=True)
    p.add_argument('--control_init', default=None)
    p.add_argument('--iters', type=int, default=8000, help='training iterations (the horizon); the schedule is the published 143k one')
    p.add_argument('--micro_bs', type=int, default=32, help='sequences per micro-batch (1024 per iteration)')
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--eval_seqs', type=int, default=512, help='held-out sequences scored per eval (×2048 tokens)')
    p.add_argument('--eval_micro', type=int, default=32)
    p.add_argument('--holdout_iter', type=int, default=HOLDOUT_ITER)
    p.add_argument('--data_dir', default='.prism_cache/pile')
    p.add_argument('--device', default=None)
    p.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float32'])
    p.add_argument('--published_curve', action='store_true', help='also score the published checkpoints on the held-out slice')
    p.add_argument('--lm_eval', action='store_true', help='run lm-eval-harness (Pythia suite, 0-shot) on the final model')
    p.add_argument('--lm_eval_limit', type=int, default=None)
    p.add_argument('--tag', default='')
    p.add_argument('--dry', action='store_true', help='synthetic 2-layer model + synthetic stream, CPU, 20 iterations')
    p.add_argument('--seqs_per_iter', type=int, default=None, help='(dry only) sequences per iteration')
    p.add_argument('--score', nargs=2, metavar=('BASELINE_JSON', 'METHOD_JSON'),
                   help='score a finished pair instead of training')
    p.add_argument('--compile', action='store_true')
    a = p.parse_args()

    if a.score:
        return score(a.score[0], a.score[1], a.tag)

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = torch.bfloat16 if (a.dtype == 'bfloat16' and device_type == 'cuda') else torch.float32
    ctx = torch.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else torch.autocast('cpu', enabled=False)
    seqs_per_iter = pm.SEQS_PER_ITER
    if a.dry:
        a.iters = min(a.iters, 20) if a.iters != 8000 else 20
        a.eval_every = min(a.eval_every, 5)
        seqs_per_iter = a.seqs_per_iter or 8
        a.eval_seqs = min(a.eval_seqs, 8)
        a.micro_bs = min(a.micro_bs, 4)
        a.eval_micro = min(a.eval_micro, 4)
        a.data_dir = os.path.join(a.data_dir, 'synthetic')
        train, hold, h0 = prepare_synthetic_data(a.data_dir, a.iters, seqs_per_iter, a.eval_seqs)
        lr, min_lr = 1e-3, 1e-4
        size_info = {'repo': 'synthetic', 'n_params': sum(p.numel() for p in []), 'n_nonemb': 0}
    else:
        train, hold = prepare_real_data(a.data_dir, a.iters, a.holdout_iter, a.eval_seqs)
        h0 = a.holdout_iter * pm.SEQS_PER_ITER
        lr, min_lr = SIZES[a.size]['lr'], SIZES[a.size]['min_lr']
        size_info = SIZES[a.size]
    assert a.iters * seqs_per_iter <= h0, 'the horizon must stay below the held-out slice'
    assert seqs_per_iter % a.micro_bs == 0
    n_micro = seqs_per_iter // a.micro_bs

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    key = f"pythia_{a.size}_{a.arm}_i{a.iters}_e{a.eval_every}" + (f"_{a.tag}" if a.tag else '') + ('_dry' if a.dry else '')
    run_dir = os.path.join(os.path.dirname(RESULTS_DIR), '.prism_runs', key)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    meta_p = os.path.join(run_dir, 'meta.json')
    if os.path.exists(meta_p):
        stamp = json.load(open(meta_p))['stamp']
    else:
        json.dump({'stamp': stamp}, open(meta_p, 'w'))

    print('=' * 64); log('PRISM × PYTHIA — core bet ladder'); print('=' * 64)
    log(f'run key {key} | device {device} | dtype {ptdtype} | arm {a.arm}', 2)
    model, info = build_model(a, device)
    n_params = sum(p.numel() for p in model.parameters())
    info['n_params'] = n_params
    if a.compile:
        model = torch.compile(model)
    optimizer = build_optimizer(model, lr)
    log(f'model {n_params:,} params | init {info["init"]} | teacher {info.get("teacher", "—")}', 2)
    log(f'stream: train seqs [0, {a.iters * seqs_per_iter:,}) in Pythia order; held-out seqs '
        f'[{h0:,}, {h0 + a.eval_seqs:,}) (iteration {h0 // seqs_per_iter:,} of the published run)', 2)

    # resume
    ckpt_p = os.path.join(run_dir, 'ckpt.pt')
    it0, curve, train_curve, step_times = 0, {}, {}, []
    if os.path.exists(ckpt_p):
        ck = torch.load(ckpt_p, map_location=device, weights_only=False)
        (model._orig_mod if hasattr(model, '_orig_mod') else model).load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        it0, curve, train_curve, step_times = ck['iter'], ck['curve'], ck['train_curve'], ck['step_times']
        log(f'[resume] from iteration {it0} ({len(curve)} evals banked)', 2)

    t_loop = time.time()
    it = it0
    last_train = float('nan')
    while True:
        if it % a.eval_every == 0 or it == a.iters:
            if str(it) not in curve:
                v = evaluate(model, hold, h0, a.eval_seqs, a.eval_micro, device, ctx)
                tl = train_curve.get(str(it), last_train if not math.isnan(last_train) else v)
                curve[str(it)] = v
                train_curve[str(it)] = tl
                print(f"step {it}: train loss {tl:.4f}, val loss {v:.4f}", flush=True)
                if it > 0 and not a.dry:
                    torch.save({'model': (model._orig_mod if hasattr(model, '_orig_mod') else model).state_dict(),
                                'optimizer': optimizer.state_dict(), 'iter': it, 'curve': curve,
                                'train_curve': train_curve, 'step_times': step_times}, ckpt_p)
        if it >= a.iters:
            break
        cur_lr = get_lr(it, lr, min_lr)
        for g in optimizer.param_groups:
            g['lr'] = cur_lr
        t0 = time.time()
        tot = 0.0
        for mi in range(n_micro):
            x, y = train.iteration(it, micro=a.micro_bs, which=mi) if seqs_per_iter == pm.SEQS_PER_ITER \
                else train.batch(it * seqs_per_iter + mi * a.micro_bs, a.micro_bs)
            x = torch.from_numpy(x).to(device); y = torch.from_numpy(y).to(device)
            with ctx:
                out = model(input_ids=x, labels=y)
                loss = out.loss / n_micro
            loss.backward()
            tot += out.loss.item() / n_micro
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device_type == 'cuda':
            torch.cuda.synchronize()
        step_times.append(time.time() - t0)
        last_train = tot
        if it % max(1, a.eval_every // 5) == 0:
            print(f"iter {it}: loss {tot:.4f}, time {step_times[-1] * 1000:.2f}ms, lr {cur_lr:.2e}", flush=True)
        it += 1
        if it % a.eval_every == 0 or it == a.iters:
            train_curve[str(it)] = last_train
    wall = time.time() - t_loop
    mean_ms = 1000.0 * float(np.mean(step_times)) if step_times else float('nan')
    print(f"[timing] steps {len(step_times)} mean_step_ms {mean_ms:.2f}", flush=True)

    pub = None
    if a.published_curve and not a.dry:
        log('scoring the published checkpoints on the held-out slice …', 2)
        pub = published_curve(a, hold, h0, device, ctx)
    lm = None
    if a.lm_eval and not a.dry:
        log('lm-eval-harness (Pythia suite, 0-shot) …', 2)
        lm = run_lm_eval(model._orig_mod if hasattr(model, '_orig_mod') else model,
                         a.student_repo or SIZES[a.size]['repo'], PYTHIA_EVAL_TASKS, a.lm_eval_limit)

    c = {int(k): v for k, v in curve.items()}
    best_step = min(c, key=c.get)
    art = {
        'schema': 'prism-pythia/1', 'partial': False, 'run_key': key, 'provenance': provenance(),
        'config': {'size': a.size, 'arm': a.arm, 'align': a.align, 'imprint': a.imprint,
                   'iters': a.iters, 'seqs_per_iter': seqs_per_iter, 'tokens_per_iter': seqs_per_iter * 2048,
                   'micro_bs': a.micro_bs, 'eval_every': a.eval_every, 'eval_seqs': a.eval_seqs,
                   'holdout_first_seq': h0, 'holdout_iteration': h0 // seqs_per_iter,
                   'schedule': {'lr': lr, 'min_lr': min_lr, 'warmup_iters': WARMUP_ITERS,
                                'decay_iters': TOTAL_ITERS, 'betas': [0.9, 0.95], 'eps': 1e-8,
                                'weight_decay': 0.1, 'grad_clip': 1.0, 'note': 'the published Pythia schedule; a gate stops early but does not rescale it'},
                   'optimizer': 'AdamW (torch, fused on CUDA)', 'dtype': str(ptdtype), 'device': device,
                   'student_repo': a.student_repo or size_info['repo'],
                   'teacher_repo': (a.teacher_repo or size_info['repo']) if a.arm == 'prism' else None,
                   'deviations': ['bf16 autocast instead of fp16 + dynamic loss scaling',
                                  f'gradient accumulation {n_micro}×{a.micro_bs} on one GPU instead of data parallelism',
                                  'held-out slice of the training stream (iteration 140000) instead of the Pile validation set',
                                  'the teacher (step143000) has trained on the held-out slice; it is never scored as a comparator'],
                   'dry': a.dry},
        'init': info,
        'runs': [{'arm': a.arm, 'curve': {str(k): v for k, v in sorted(c.items())},
                  'train_curve': {str(k): v for k, v in sorted((int(k), v) for k, v in train_curve.items())},
                  'best': c[best_step], 'best_step': best_step, 'at_end': c[max(c)],
                  'wall_sec': round(wall, 1), 'step_ms': mean_ms,
                  'overfits': c[max(c)] > c[best_step] * 1.05}],
        'published_curve_on_holdout': pub,
        'lm_eval': lm,
    }
    if pub:
        dev = {s: c[int(s)] - v for s, v in pub.items() if int(s) in c}
        art['reproduction'] = {'max_abs_dev_nats': max(abs(d) for d in dev.values()) if dev else None,
                               'per_step_dev': dev,
                               'tolerance_nats': 0.05,
                               'passes': (max(abs(d) for d in dev.values()) <= 0.05) if dev else None,
                               'note': 'baseline arm vs the published checkpoints scored on the same held-out slice; '
                                       'no PRISM ratio is reported until this passes (W2B-NOTES.md)'}
    path = os.path.join(RESULTS_DIR, f'pythia_{stamp}.json')
    json.dump(art, open(path, 'w'), indent=2)
    json.dump(art, open(os.path.join(RESULTS_DIR, 'pythia_latest.json'), 'w'), indent=2)
    log(f'Artifact: results/{os.path.basename(path)} — COMMIT THIS FILE.', 2)
    return 0


def score(baseline_json, method_json, tag=''):
    from prism_metrics import steps_to_target, token_equivalent_multiplier, donor_adjusted_multiplier
    b, m = json.load(open(baseline_json)), json.load(open(method_json))
    br, mr = b['runs'][0], m['runs'][0]
    tpi = b['config']['tokens_per_iter']
    stt = steps_to_target(mr['curve'], br['best'], b['config']['eval_every'], br['best_step'])
    teq = token_equivalent_multiplier(br['curve'], mr['curve'], rmse_tol=0.02, tokens_per_step=tpi)
    teq.pop('fit', None)
    if teq.get('multiplier') is not None and (teq['multiplier'] <= 0 or not math.isfinite(teq['multiplier'])):
        teq['reason'] = 'degenerate baseline fit (flat or non-decreasing curve); multiplier withheld'
        teq['multiplier'] = None
    donor = None
    if m['init'].get('teacher_params') and teq.get('multiplier'):
        donor = donor_adjusted_multiplier(teq['multiplier'], mr['curve'] and max(int(k) for k in mr['curve']),
                                          m['init']['teacher_params'], m['init']['teacher_tokens'],
                                          m['init']['n_params'], tpi, n_students_amortized=1)
    rep = b.get('reproduction')
    out = {'schema': 'prism-pythia-score/1', 'baseline': os.path.basename(baseline_json),
           'method': os.path.basename(method_json), 'steps_to_target': stt,
           'token_equivalent': teq, 'donor_adjusted': donor,
           'reproduction_passes': rep['passes'] if rep else None,
           'ratio_reportable': bool(rep and rep['passes']),
           'kill_rule_G2': 'ratio < 1.2x kills (docs/NEXT-EXPERIMENTS.md)',
           'provenance': provenance()}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = os.path.join(RESULTS_DIR, f'pythia_score_{stamp}{"_" + tag if tag else ""}.json')
    json.dump(out, open(path, 'w'), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != 'provenance'}, indent=1))
    log(f'Artifact: results/{os.path.basename(path)}', 2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
