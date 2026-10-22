"""
prism_prior_eval.py — Prior-Fused PRISM: does a T9-style fixed shared n-gram prior,
fused into the logits, stack with PRISM's spectral init for a much larger speedup?

The T9 idea: a tiny shared prior (~a dense n-gram table) already predicts held-out
same-modality text almost as well as the converged neural baseline (measured: a
context-3 char n-gram hits ~2.57 bits/char on Shakespeare val = the baseline's best).
Fuse it into the model (product of experts, final = model_logits + λ·log p_ngram) and
the model starts at baseline quality FOR FREE, then learns only the residual below the
n-gram floor — where PRISM's geometry helps.

Original four arms (Shakespeare, dense context-3 table), scored as steps to reach the
plain baseline's best val loss (the PRISM Score), plus the init loss (step 0) and best:
    baseline     plain
    prism        spectral init + mod wheel (needs a teacher)
    prior        plain + fused n-gram prior (the T9 dictionary)
    prism_prior  both — the hybrid

alpha-moves W5 (the COUNT control, the Act 2 composition gate; docs/ALPHA-MOVES-BRIEF.md):
    --corpus=enwik8 --sparse_context=6   a SPARSE interpolated prior at context 4+
                                         (prism_prior.py) with block-edge backoff,
                                         counted on the student's training slot
    dirs_only          init-only PRISM (no mod wheel — the enwik8 winner), matched schedule
    dirs_prior         dirs_only + the prior fused during training
    baseline_inffuse   baseline trained WITHOUT the prior, the same prior fused only at eval
    dirs_inffuse       dirs_only trained WITHOUT the prior, fused only at eval
    count              COUNT-style parameter-free n-gram input embedding (arXiv 2610.20456)
    count_dirs         count + dirs_only
Validation lands on every epoch boundary as well as the dense grid (train.py --eval_at),
and every arm gets the token-equivalent multiplier (prism_metrics.py) beside its score.

Every run writes a prism-prior/1 artifact (additive keys only). Resumable; partial runs raise.
"""
import argparse
import json
import math
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from prism_eval import (provenance, parse_curve, stream_train, run_training, compute_score,
                        acquire_lock, _stats, log, default_device, setup,
                        RESULTS_DIR, RUNS_DIR, SRC_DIR)

CONFIG = 'config/train_shakespeare_char.py'
# name: (prism mode, prior mode)   prism: None | 'recipe' (init + mod wheel, tuned LR)
#                                          | 'dirs' (init only, matched schedule)
#                                  prior: None | 'train' (fused in training and eval)
#                                          | 'eval' (fused ONLY at eval: inference-only control)
#                                          | 'count' (COUNT input embedding; no logit fusion)
ARMS = {
    'baseline':         (None,     None),
    'prism':            ('recipe', None),
    'prior':            (None,     'train'),
    'prism_prior':      ('recipe', 'train'),
    'dirs_only':        ('dirs',   None),
    'dirs_prior':       ('dirs',   'train'),
    'baseline_inffuse': (None,     'eval'),
    'dirs_inffuse':     ('dirs',   'eval'),
    'count':            (None,     'count'),
    'count_dirs':       ('dirs',   'count'),
}


def build_prior(context, lambdas, out_base, dataset='shakespeare_char'):
    """The original DENSE (V^C, V) table (build_ngram_prior.py)."""
    tag = 'default' if not lambdas else 'l' + lambdas.replace('.', '').replace(',', '_')
    out = f'{out_base}/{tag}' + ('' if dataset == 'shakespeare_char' else f'/{dataset}')
    tab = f'{out}/prior_c{context}.pt'
    if os.path.exists(tab):
        log(f'[resume] n-gram prior context {context} ({tag}) cached.', 2)
        return tab
    log(f'Building n-gram prior (context {context}, lambdas {lambdas or "default"})…', 2)
    cmd = [sys.executable, 'build_ngram_prior.py', f'--dataset={dataset}',
           f'--context={context}', f'--out={out}']
    if lambdas:
        cmd.append(f'--lambdas={lambdas}')
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f'build_ngram_prior failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}')
    for line in r.stdout.splitlines():
        if 'standalone' in line or 'saved' in line:
            log(line.strip(), 4)
    return tab


def build_sparse(a, corpus):
    """The SPARSE prior (prism_prior.py), counted on the student's training slot
    (data/shakespeare_eval/train.bin after setup) with the last 1M symbols held
    out; mus: Act 1's enwik8 weights for enwik8, tuned on the held-out slice
    otherwise. Cached by corpus/context/c_min/token count."""
    train = 'data/shakespeare_eval/train.bin'
    n_tok = os.path.getsize(train) // 2
    out = f'.prism_cache/sparse/{corpus}_c{a.sparse_context}_m{a.sparse_cmin}_n{n_tok}.pt'
    if os.path.exists(out) and os.path.exists(out + '.json'):
        log(f'[resume] sparse prior cached: {out}', 2)
        return out
    mus = a.sparse_mus or ('act1' if corpus == 'enwik8' else 'auto')
    log(f'Building SPARSE n-gram prior (orders 0..{a.sparse_context}, c_min {a.sparse_cmin}, '
        f'mus {mus}) on {n_tok:,} student-train symbols…', 2)
    cmd = [sys.executable, 'build_sparse_prior.py', f'--train={train}',
           f'--meta=data/shakespeare_eval/meta.pkl', f'--context={a.sparse_context}',
           f'--c_min={a.sparse_cmin}', f'--mus={mus}', f'--out={out}',
           '--val=data/shakespeare_eval/val.bin', '--val_max=5000000']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(f'build_sparse_prior failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    for line in r.stdout.splitlines():
        if 'VAL' in line or 'saved' in line or 'mus' in line:
            log(line.strip(), 4)
    return out


def train_teacher(seed, a, device, run_dir, corpus=None):
    out = f'out-prior-teacher-s{seed}' + (f'-{corpus}' if corpus else '')
    cache = f'.prism_cache/prior_teacher_s{seed}' + (f'_{corpus}' if corpus else '')
    if os.path.exists(f'{cache}/directions.pt'):
        log(f'[resume] teacher (seed {seed}) fingerprint cached.', 2)
        return cache
    # historical behaviour: Shakespeare teachers train on the full shakespeare_char set;
    # a --corpus run trains its teacher on the teacher slot setup() carved out.
    dataset = 'shakespeare_char' if corpus is None else 'shakespeare_teacher'
    log(f'Training teacher (seed {seed}, {a.teacher_steps} steps, {dataset})…', 2)
    rc, out_s = stream_train(
        [sys.executable, '-u', 'train.py', CONFIG, f'--dataset={dataset}',
         f'--seed={seed}', f'--device={device}', f'--max_iters={a.teacher_steps}',
         f'--eval_interval={a.teacher_steps}', f'--eval_iters={a.eval_iters}',
         '--log_interval=100', f'--out_dir={out}', '--always_save_checkpoint=True',
         '--compile=False', '--prism_init=False', '--wandb_log=False',
         f'--batch_size={a.batch_size}', f'--block_size={a.block_size}'],
        f's{seed} teacher', a.teacher_steps)
    if rc != 0:
        raise RuntimeError(f'teacher failed (seed {seed}):\n{out_s[-2000:]}')
    e = subprocess.run([sys.executable, 'prism_extract.py', '--ckpt', f'{out}/ckpt.pt',
                        '--out', cache], capture_output=True, text=True, timeout=300)
    if e.returncode != 0:
        raise RuntimeError(f'extract failed:\n{e.stderr[-2000:]}')
    return cache


def epoch_boundaries(n_train_tokens, batch_size, block_size, student_steps):
    """Steps at which the student has seen its training slot an integer number of
    times (tokens per step = batch × block). Empty in the zero-epoch regime."""
    per_epoch = n_train_tokens // (batch_size * block_size)
    if per_epoch <= 0:
        return []
    return [s for s in range(per_epoch, student_steps + 1, per_epoch)]


def run_arm(name, seed, a, device, run_dir, prior_tab, teacher_cache, eval_at):
    prism_mode, prior_mode = ARMS[name]
    meta_f = os.path.join(run_dir, f's{seed}_{name}.json')
    if os.path.exists(meta_f):
        r = json.load(open(meta_f))
        log(f'[resume] arm "{name}" (seed {seed}) done (best {r["best"]:.4f}).', 2)
        return r
    extra = []
    if prism_mode == 'recipe':
        extra += ['--prism_init=True', '--prism_align=0.75',
                  f'--prism_spectra={teacher_cache}/spectra.json',
                  f'--prism_directions={teacher_cache}/directions.pt',
                  '--prism_mod=0.01', '--prism_mod_decay=0.9999',
                  '--learning_rate=5e-4', '--warmup_iters=50']
    elif prism_mode == 'dirs':
        extra += ['--prism_init=True', '--prism_align=0.75',
                  f'--prism_spectra={teacher_cache}/spectra.json',
                  f'--prism_directions={teacher_cache}/directions.pt']
    else:
        extra += ['--prism_init=False']
    if prior_mode in ('train', 'eval'):
        if a.sparse_context > 0:
            extra += [f'--prior_sparse={prior_tab}']
        else:
            extra += [f'--prior_table={prior_tab}']
        extra += [f'--prior_strength={a.prior_strength}']
        if prior_mode == 'eval':
            extra += ['--prior_eval_only=True']
        if a.gate_warmup > 0 and prior_mode == 'train':
            extra += [f'--logit_gate_warmup={a.gate_warmup}']
    elif prior_mode == 'count':
        assert a.sparse_context > 0, 'count arms need --sparse_context'
        extra += [f'--prior_sparse={prior_tab}', '--prior_strength=0.0', '--count_embed=True']
        if a.count_cfg:
            extra += [f'--count_embed_cfg={a.count_cfg}']
    if eval_at:
        extra += ['--eval_at=' + ','.join(str(s) for s in eval_at)]
    if a.block_size != 256:
        extra += [f'--block_size={a.block_size}']

    r = run_training(name, extra, seed, a.student_steps, a.eval_every, a.eval_iters,
                     device, f's{seed}', batch_size=a.batch_size)
    # init loss = the step-0 eval (free head start)
    curve = {int(k): v for k, v in r['curve'].items()}
    r['init_val'] = curve.get(0)
    r['arm_spec'] = {'prism': prism_mode, 'prior': prior_mode, 'flags': extra,
                     'schedule_matched_to_baseline': prism_mode != 'recipe'}
    json.dump(r, open(meta_f, 'w'), indent=2)
    return r


def run_key(a):
    seeds = '-'.join(str(s) for s in a.seeds)
    key = (f"prior_c{a.context}_str{a.prior_strength}_g{a.gate_warmup}_s{a.student_steps}"
           f"_e{a.eval_every}_bs{a.batch_size}_seeds{seeds}")
    if a.corpus:
        key += f"_cor{a.corpus[:8]}"
    if a.sparse_context > 0:
        key += f"_sp{a.sparse_context}" + (f"m{a.sparse_cmin}" if a.sparse_cmin > 1 else '')
    if a.count_cfg:
        key += "_cc" + ''.join(ch for ch in a.count_cfg if ch.isalnum())
    return key + (f"_{a.tag}" if a.tag else '')


def main():
    p = argparse.ArgumentParser(description='Prior-Fused PRISM (T9 × PRISM)')
    p.add_argument('--context', type=int, default=3, help='DENSE prior context (build_ngram_prior)')
    p.add_argument('--prior_strength', type=str, default='1.0')
    p.add_argument('--prior_lambdas', default='',
                   help='interpolation weights orders 0..C (e.g. 0.0,0.03,0.12,0.85 = strong)')
    p.add_argument('--gate_warmup', type=int, default=0,
                   help='ramp the model logit contribution 0→1 over N steps (prior arms) so '
                        'the fused init = the pure prior; 0 = off')
    p.add_argument('--teacher_steps', type=int, default=2000)
    p.add_argument('--student_steps', type=int, default=1500)
    p.add_argument('--eval_every', type=int, default=25)
    p.add_argument('--eval_iters', type=int, default=100)
    p.add_argument('--seeds', default='1337')
    p.add_argument('--arms', default='baseline,prism,prior,prism_prior')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--device', default=None)
    p.add_argument('--tag', default='')
    # ── alpha-moves W5 ──
    p.add_argument('--corpus', default=None,
                   help='bench on data/<name>/ (e.g. enwik8) instead of shakespeare_char')
    p.add_argument('--sparse_context', type=int, default=0,
                   help='>0: use the SPARSE interpolated prior at this max order (prism_prior.py) '
                        'instead of the dense --context table')
    p.add_argument('--sparse_cmin', type=int, default=1, help='prune contexts seen < c_min times')
    p.add_argument('--sparse_mus', default='', help="'act1' | 'auto' | 'm0,m1,…' (default: act1 for enwik8, auto otherwise)")
    p.add_argument('--count_cfg', default='', help="count arms: 'K:8,amp:sqrt_d,orders:1-6'")
    p.add_argument('--epoch_evals', default=None, choices=['on', 'off'],
                   help='add epoch-boundary eval steps (default: on for --corpus/--sparse_context runs, '
                        'off for the legacy Shakespeare dense-prior protocol so its eval grid is unchanged)')
    a = p.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    arms = [x for x in a.arms.split(',') if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        sys.exit(f'unknown arm(s): {bad}. known: {", ".join(ARMS)}')
    device = a.device or default_device()
    os.chdir(SRC_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    key = run_key(a)
    run_dir = os.path.join(RUNS_DIR, key)
    os.makedirs(run_dir, exist_ok=True)
    meta_path = os.path.join(run_dir, 'meta.json')
    if os.path.exists(meta_path):
        stamp = json.load(open(meta_path))['stamp']
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        json.dump({'stamp': stamp}, open(meta_path, 'w'))

    print('=' * 64)
    log('PRIOR-FUSED PRISM (T9 × PRISM)')
    print('=' * 64)
    log(f'Run key: {key} | seeds {a.seeds} | arms {arms} | device {device}', 2)

    n_train, n_val, _ = setup(overlap=None, corpus=a.corpus)   # prepares the eval slots
    if a.sparse_context > 0:
        prior_tab = build_sparse(a, a.corpus or 'shakespeare_char')
        prior_info = json.load(open(prior_tab + '.json'))
    else:
        prior_tab = build_prior(a.context, a.prior_lambdas, '.prism_cache/ngram',
                                dataset=a.corpus or 'shakespeare_char')
        prior_info = None
    want_epochs = (a.epoch_evals == 'on') if a.epoch_evals else bool(a.corpus or a.sparse_context > 0)
    eval_at = epoch_boundaries(n_train, a.batch_size, a.block_size, a.student_steps) if want_epochs else []
    if eval_at:
        log(f'epoch-boundary evals at steps {eval_at[:6]}{"…" if len(eval_at) > 6 else ""} '
            f'({n_train:,} train tokens / {a.batch_size * a.block_size} per step)', 2)
    else:
        log(f'zero-epoch regime: {n_train:,} train tokens, no epoch boundary inside '
            f'{a.student_steps} steps', 2)
    lock = acquire_lock(run_dir)
    try:
        per_seed = []
        need_teacher = any(ARMS[x][0] for x in arms)
        for seed in a.seeds:
            print('-' * 64)
            log(f'SEED {seed}', 2)
            teacher_cache = (train_teacher(seed, a, device, run_dir, corpus=a.corpus)
                             if need_teacher else None)
            results = {name: run_arm(name, seed, a, device, run_dir, prior_tab, teacher_cache, eval_at)
                       for name in arms}
            base = results.get('baseline')
            entry = {'seed': seed, 'arms': {}}
            for name, r in results.items():
                sc = compute_score(base, r, a.eval_every) if base else None
                rec = {'init_val': r.get('init_val'), 'best': r['best'],
                       'best_step': r['best_step'], 'at_end': r['at_end'],
                       'score': sc}
                if base and name != 'baseline':
                    rec['token_equivalent'] = _tem(base, r, a.batch_size * a.block_size)
                for k in ('curve', 'step_ms', 'overlap', 'arm_spec', 'wall_sec'):   # curve: additive, for Act-2 charts
                    if k in r:
                        rec[k] = r[k]
                entry['arms'][name] = rec
            per_seed.append(entry)
            _persist(stamp, key, a, device, arms, per_seed, complete=False,
                     n_train=n_train, eval_at=eval_at, prior_info=prior_info)
        art = _persist(stamp, key, a, device, arms, per_seed, complete=True,
                       n_train=n_train, eval_at=eval_at, prior_info=prior_info)
        _report(art)
        log(f'Artifact: results/prior_{stamp}.json (also prior_latest.json)', 2)
        log('COMMIT THIS FILE — it is the evidence for any claim you publish.', 2)
    finally:
        try:
            os.remove(lock)
        except Exception:
            pass


def _tem(base, r, tokens_per_step):
    """Token-equivalent multiplier (prism_metrics.py) at the arm's end of training,
    against a monotone power-law fit of the baseline's own curve. Refusals are
    recorded as reasons, never as numbers."""
    try:
        from prism_metrics import token_equivalent_multiplier
        out = token_equivalent_multiplier(base['curve'], r['curve'], rmse_tol=0.01,
                                          tokens_per_step=tokens_per_step)
        fit = out.pop('fit')
        out['fit'] = {k: v for k, v in fit.items() if k != 'residuals'}
        return out
    except Exception as e:                      # a fit failure is a reason, not a crash
        return {'multiplier': None, 'reason': f'{type(e).__name__}: {e}'}


def _persist(stamp, key, a, device, arms, per_seed, complete, n_train=None, eval_at=None,
             prior_info=None):
    # baseline best (median) = the target; per-arm speedup medians
    base_step = _stats([e['arms']['baseline']['best_step'] for e in per_seed
                        if 'baseline' in e['arms']]) if per_seed else None
    summ = {}
    for name in arms:
        got = [e['arms'][name] for e in per_seed if name in e['arms']]
        scores = [g['score']['prism_score'] for g in got
                  if g['score'] and g['score']['prism_score'] is not None]
        summ[name] = {
            'init_val': _stats([g['init_val'] for g in got if g['init_val'] is not None]),
            'best': _stats([g['best'] for g in got]),
            'at_end': _stats([g['at_end'] for g in got]),
            'prism_score': _stats(scores) if scores else None,
            'any_reached_at_init': any(g['score'] and g['score'].get('reached_at_init')
                                       for g in got),
            'any_left_censored': any(g['score'] and g['score'].get('left_censored')
                                     for g in got),
        }
        tems = [g['token_equivalent']['multiplier'] for g in got
                if g.get('token_equivalent') and g['token_equivalent'].get('multiplier') is not None]
        if any(g.get('token_equivalent') for g in got):
            summ[name]['token_equivalent'] = _stats(tems) if tems else None
            summ[name]['token_equivalent_refusals'] = [g['token_equivalent'].get('reason') for g in got
                                                       if g.get('token_equivalent') and
                                                       g['token_equivalent'].get('multiplier') is None]
    config = {'context': a.context, 'prior_strength': a.prior_strength,
              'teacher_steps': a.teacher_steps, 'student_steps': a.student_steps,
              'eval_every': a.eval_every, 'eval_iters': a.eval_iters,
              'seeds': a.seeds, 'arms': arms, 'batch_size': a.batch_size,
              'block_size': a.block_size, 'device': device,
              'note': 'scored as steps-to-baseline-best val loss (nats). loss is '
                      'nanoGPT cross-entropy in NATS; bits/char = nats/ln(2).'}
    if a.corpus or a.sparse_context > 0:
        config.update({
            'corpus': a.corpus or 'shakespeare_char',
            'sparse_context': a.sparse_context, 'sparse_cmin': a.sparse_cmin,
            'count_cfg': a.count_cfg, 'n_train_tokens': n_train, 'eval_at': eval_at,
            'prior_info': prior_info,
            'arm_semantics': {k: {'prism': v[0], 'prior': v[1]} for k, v in ARMS.items() if k in arms},
            'schedule_note': "'dirs'-mode arms share the baseline's LR/warmup (matched); "
                             "'recipe' arms use the tuned 5e-4/50 schedule, as in the original study.",
            'framing': 'docs/NEXT-EXPERIMENTS.md two-act framing spec binds every prior arm.',
        })
    art = {
        'schema': 'prism-prior/1', 'partial': not complete, 'run_key': key,
        'provenance': provenance(),
        'config': config,
        'baseline_best_step': base_step,
        'runs': per_seed, 'summary': summ,
    }
    for f in (f'prior_{stamp}.json', 'prior_latest.json'):
        json.dump(art, open(os.path.join(RESULTS_DIR, f), 'w'), indent=2)
    return art


def _report(a):
    s, c = a['summary'], a['config']
    b2c = 1.0 / math.log(2)
    print()
    print('  ' + '-' * 78)
    label = (f'SPARSE context-{c["sparse_context"]} ({c.get("corpus")})'
             if c.get('sparse_context') else f'context-{c["context"]}')
    print(f'    PRIOR-FUSED PRISM — {label} n-gram × PRISM  (steps-to-baseline-best)')
    print('  ' + '-' * 78)
    bs = a.get('baseline_best_step')
    print(f'    seeds {c["seeds"]} · baseline best @ '
          f'{bs["median"] if bs else "?"} steps')
    print(f'    {"arm":16} | {"init loss":>9} {"(bits)":>7} | {"best":>7} | {"speedup":>14} | {"tok-equiv":>9}')
    print('  ' + '-' * 78)
    for name in c['arms']:
        g = s[name]
        iv = g['init_val']['median'] if g['init_val'] else float('nan')
        bb = g['best']['median'] if g['best'] else float('nan')
        if g['any_reached_at_init']:
            sp = 'reached at INIT'
        elif g['prism_score']:
            sp = f'{g["prism_score"]["median"]:.1f}×' + ('*' if g['any_left_censored'] else '')
        else:
            sp = 'never/—'
        te = g.get('token_equivalent')
        tes = f'{te["median"]:.2f}×' if te else ('—' if name == 'baseline' else 'refused')
        print(f'    {name:16} | {iv:>9.3f} {iv * b2c:>7.3f} | {bb:>7.3f} | {sp:>14} | {tes:>9}')
    print('  ' + '-' * 78)
    print('    init loss = val at step 0 (the free head start). loss in nats; bits = nats/ln2.')
    print('    "reached at INIT" is a flag, not a speedup — see the two-act framing spec before')
    print('    writing anything about it. tok-equiv = baseline steps the fitted baseline curve')
    print('    needs to reach the arm\'s final loss, over the arm\'s steps (refused if the fit is poor).')
    print()


if __name__ == '__main__':
    main()
