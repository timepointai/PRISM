"""prism_sweep.py — single-seed hyper-parameter sweeps for TUNING PARITY (W2a).

The rule (docs/NEXT-EXPERIMENTS.md, alpha-moves block): every arm, ours or a
control's, is tuned on seed 1336 only, with the same number of settings, by a
criterion written down before looking. This driver runs one arm over a small
grid on one seed and writes a `prism-sweep/1` artifact with every curve and the
chosen setting under the pre-registered criterion (default: lowest best val
loss inside the horizon). It never runs the result seeds.

    python prism_sweep.py --corpus=modernweb --seed=1336 --arm=baseline \
        --grid "lr=5e-4,1e-3,2e-3 init_std=0.01,0.02" --student_steps=1500 \
        --eval_every=10 --eval_iters=50 --tag=w2a

    python prism_sweep.py --corpus=modernweb --seed=1336 --arm=dirs_only \
        --grid "lr=5e-4,1e-3,2e-3 init_std=0.01,0.02" --teacher_steps=2000 ...

    python prism_sweep.py --corpus=enwik8 --seed=1336 --arm=baseline --optim=muon \
        --grid "muon_lr=0.01,0.02,0.04" ...

Grid keys → train.py flags: lr → --learning_rate, warmup → --warmup_iters,
init_std → --init_std, muon_lr → --muon_lr, qk → --init_mode=small_qk:<σ>.
Arms: `baseline` (no Prism), or any prism_eval method name (`dirs_only`,
`dirs_pure`, `spectral_only`, `recipe`, `small_qk`, `mimetic`, `statmatch`);
methods that need a teacher train one at --teacher_steps (cached per seed).
Stages bank under .prism_runs/<key>/ and resume. Partial runs raise.
"""
import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone

from prism_eval import (provenance, run_training, train_teacher, method_args_for, setup,
                        cached_stage, acquire_lock, log, default_device, _stats,
                        RESULTS_DIR, RUNS_DIR)

FLAG = {'lr': '--learning_rate', 'warmup': '--warmup_iters', 'init_std': '--init_std',
        'muon_lr': '--muon_lr'}


def parse_grid(spec):
    keys, vals = [], []
    for part in spec.split():
        k, v = part.split('=', 1)
        keys.append(k)
        vals.append([x for x in v.split(',') if x])
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def setting_flags(setting):
    out = []
    for k, v in setting.items():
        if k == 'qk':
            out.append(f'--init_mode=small_qk:{v}')
        else:
            out.append(f'{FLAG[k]}={v}')
    return out


def setting_tag(setting):
    return '_'.join(f'{k}{v}' for k, v in setting.items()).replace('-', 'm')


def main():
    p = argparse.ArgumentParser(description='PRISM tuning-parity sweep (one seed, one arm, a grid)')
    p.add_argument('--corpus', default=None)
    p.add_argument('--seed', type=int, default=1336)
    p.add_argument('--arm', default='baseline')
    p.add_argument('--grid', required=True, help="'lr=5e-4,1e-3 init_std=0.01,0.02' (space-separated keys)")
    p.add_argument('--optim', default='adamw', choices=['adamw', 'muon'])
    p.add_argument('--teacher_steps', type=int, default=2000)
    p.add_argument('--teacher_optim', default='adamw', choices=['adamw', 'muon'])
    p.add_argument('--student_steps', type=int, default=1500)
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--eval_iters', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--warmup', type=int, default=100, help='warmup when the grid does not set it')
    p.add_argument('--lr', default='1e-3', help='LR when the grid does not set it')
    p.add_argument('--criterion', default='best_val', choices=['best_val', 'at_end'],
                   help='pre-registered choice rule (default: lowest best val within the horizon)')
    p.add_argument('--device', default=None)
    p.add_argument('--tag', default='')
    a = p.parse_args()
    device = a.device or default_device()
    grid = parse_grid(a.grid)
    assert 1 <= len(grid) <= 9, f'tuning parity allows at most 9 settings; grid has {len(grid)}'

    key = (f"sweep_{a.arm}_{a.corpus or 'shakespeare_char'}_s{a.student_steps}_e{a.eval_every}"
           f"_seed{a.seed}_{a.optim}" + (f"_t{a.teacher_optim}" if a.teacher_optim != 'adamw' else '')
           + (f"_{a.tag}" if a.tag else ''))
    run_dir = os.path.join(RUNS_DIR, key)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    meta_path = os.path.join(run_dir, 'meta.json')
    if os.path.exists(meta_path):
        stamp = json.load(open(meta_path))['stamp']
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        json.dump({'stamp': stamp}, open(meta_path, 'w'))
    print('=' * 64)
    log('PRISM TUNING SWEEP (seed-1336 parity)')
    print('=' * 64)
    log(f'Run key: {key} | {len(grid)} settings | arm {a.arm} | optim {a.optim}', 2)

    setup(corpus=a.corpus)
    lock = acquire_lock(run_dir)
    try:
        cache = None
        if a.arm != 'baseline' and a.arm not in ('small_qk', 'mimetic', 'tuned'):
            tag = f'ev-{a.corpus[:8]}' if a.corpus else 'eval'
            cache = train_teacher(a.teacher_steps, a.seed, a.eval_iters, device, f's{a.seed}',
                                  cache_tag=tag, batch_size=a.batch_size,
                                  teacher_optim=a.teacher_optim)
        shared = ([f'--optim={a.optim}'] if a.optim != 'adamw' else [])
        results = []
        for i, setting in enumerate(grid):
            lr = setting.get('lr', a.lr)
            warmup = int(setting.get('warmup', a.warmup))
            if a.arm == 'baseline':
                flags = ['--prism_init=False', f'--learning_rate={lr}', f'--warmup_iters={warmup}']
            else:
                knobs = {'optim': a.optim if a.optim != 'adamw' else None}
                flags = method_args_for(a.arm, cache, lr, warmup, knobs={k: v for k, v in knobs.items() if v},
                                        seed=a.seed)
            flags += shared + [f for f in setting_flags(setting)
                               if not f.startswith('--learning_rate') and not f.startswith('--warmup_iters')]
            stg = setting_tag(setting)
            log(f'[setting {i + 1}/{len(grid)}] {setting}', 2)
            r = cached_stage(run_dir, f's{a.seed}_{stg}', lambda: run_training(
                f'{a.arm}-{stg}', flags, a.seed, a.student_steps, a.eval_every, a.eval_iters,
                device, f's{a.seed}', batch_size=a.batch_size))
            results.append({'setting': setting, 'flags': flags, **r})
            _persist(stamp, key, a, grid, results, device, complete=False)
        chosen = min(results, key=lambda r: r['best'] if a.criterion == 'best_val' else r['at_end'])
        art = _persist(stamp, key, a, grid, results, device, complete=True, chosen=chosen)
        print()
        print(f"    {'setting':40} | {'best':>7} | {'@step':>6} | {'at_end':>7}")
        for r in results:
            mark = ' <- chosen' if r is chosen else ''
            print(f"    {str(r['setting']):40} | {r['best']:7.4f} | {r['best_step']:6d} | {r['at_end']:7.4f}{mark}")
        log(f"Chosen by {a.criterion}: {chosen['setting']}", 2)
        log(f'Artifact: results/sweep_{stamp}.json', 2)
        log('COMMIT THIS FILE — it is the tuning record the parity rule requires.', 2)
    finally:
        try:
            os.remove(lock)
        except Exception:
            pass


def _persist(stamp, key, a, grid, results, device, complete, chosen=None):
    art = {
        'schema': 'prism-sweep/1', 'partial': not complete, 'run_key': key,
        'provenance': provenance(),
        'config': {'corpus': a.corpus, 'seed': a.seed, 'arm': a.arm, 'optim': a.optim,
                   'teacher_optim': a.teacher_optim, 'teacher_steps': a.teacher_steps,
                   'student_steps': a.student_steps, 'eval_every': a.eval_every,
                   'eval_iters': a.eval_iters, 'batch_size': a.batch_size, 'device': device,
                   'grid': grid, 'criterion': a.criterion,
                   'note': 'tuning-parity sweep on the tuning seed ONLY; the chosen setting is '
                           'applied to seeds 1337-1339 in the result run. Not a result.'},
        'settings': results,
        'chosen': None if chosen is None else {'setting': chosen['setting'], 'best': chosen['best'],
                                               'best_step': chosen['best_step'], 'at_end': chosen['at_end']},
        'summary': {'best_vals': _stats([r['best'] for r in results]) if results else None},
    }
    json.dump(art, open(os.path.join(RESULTS_DIR, f'sweep_{stamp}.json'), 'w'), indent=2)
    return art


if __name__ == '__main__':
    main()
