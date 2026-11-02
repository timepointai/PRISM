"""
w7_spectrum_fit.py — W7 "fingerprint redundancy", CPU-only half: how well does a
2-parameter spectrum family describe a trained teacher's group-averaged singular-value
curve, compared with the 8-coefficient DCT that PRISM's Spectral Imprint stores?

This measures reconstruction only. Whether a 2-parameter imprint trains as well as the
DCT-8 imprint is the GPU half (a 3-seed probe arm), not run here.

    python w7_spectrum_fit.py <label>=<directions.pt> [<label>=<directions.pt> ...] --out results/

Input: a fingerprint `directions.pt` written by `src/prism_extract.py` at a commit that
stores the singular values (`'S'`, added in 7a436db). Group averaging follows
`prism_extract.extract_from_checkpoint` exactly (per-matrix s / s.max(), resampled to the
longest length, averaged). Families, each fitted by least squares on the normalised curve:

  dct8      the stored imprint: coefficients fitted as prism_extract does, expanded with
            prism_init.dct_expand (softplus of 8 cosines, renormalised to max 1)
  powerlaw  s(i) = (1 + i/τ)^(−α)                    (2 parameters: τ, α)
  stretched s(i) = exp(−(i/λ)^β)                    (2 parameters: λ, β)

Metrics per group: relative L2 error ‖ŝ − s‖ / ‖s‖, max |ŝ − s|, and the error of the
top-16 values (the subspace PRISM's overlap tracker follows).
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
from scipy.optimize import least_squares

N_DCT = 8


def group_curves(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    groups = {}
    for name, rec in d.items():
        if 'S' not in rec:
            raise SystemExit(f'{path}: entry {name} has no singular values (extracted before 7a436db)')
        s = torch.as_tensor(rec['S']).float().numpy()
        groups.setdefault(rec['group'], []).append(s / s.max() if s.max() > 0 else s)
    out = {}
    for g, svs in groups.items():
        n = max(len(s) for s in svs)
        interp = [np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s)), s) for s in svs]
        out[g] = (np.mean(interp, axis=0), len(svs))
    return out


def fit_dct8(s):
    clipped = np.clip(s, 0.01, None)
    target = np.log(np.exp(clipped) - 1.0 + 1e-10)
    n = len(s)
    t = np.linspace(0, np.pi, n, endpoint=False)
    basis = np.stack([np.cos((k + 0.5) * t) for k in range(N_DCT)], axis=1)
    c, *_ = np.linalg.lstsq(basis, target, rcond=None)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
    from prism_init import dct_expand            # the exact curve a student is imprinted with
    return dct_expand(c, n), c.tolist()


def fit_2p(s, kind):
    i = np.arange(len(s), dtype=float)
    if kind == 'powerlaw':
        f = lambda p: (1 + i / np.exp(p[0])) ** (-np.exp(p[1]))
        p0 = [np.log(10.0), np.log(0.5)]
    else:
        f = lambda p: np.exp(-(i / np.exp(p[0])) ** np.exp(p[1]))
        p0 = [np.log(len(s) / 2), np.log(1.0)]
    r = least_squares(lambda p: f(p) - s, p0)
    names = ('tau', 'alpha') if kind == 'powerlaw' else ('lambda', 'beta')
    return f(r.x), dict(zip(names, np.exp(r.x).tolist()))


def metrics(fit, s):
    return {'rel_l2': float(np.linalg.norm(fit - s) / np.linalg.norm(s)),
            'max_abs': float(np.max(np.abs(fit - s))),
            'top16_rel_l2': float(np.linalg.norm(fit[:16] - s[:16]) / np.linalg.norm(s[:16]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='label=path/to/directions.pt')
    ap.add_argument('--origin', action='append', default=[],
                    help='label=where the fingerprint lives (e.g. a Modal volume path); recorded in the artifact')
    ap.add_argument('--out', default='results/')
    a = ap.parse_args()
    origins = dict(o.split('=', 1) for o in a.origin)
    report = {'schema': 'prism-w7-specfit/1', 'partial': False,
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'n_dct': N_DCT,
              'note': 'reconstruction of group-averaged normalised spectra only; no training run',
              'teachers': {}}
    for item in a.inputs:
        label, path = item.split('=', 1)
        rows = {}
        for g, (s, n_mat) in sorted(group_curves(path).items()):
            dct, coeffs = fit_dct8(s)
            pl, pl_p = fit_2p(s, 'powerlaw')
            se, se_p = fit_2p(s, 'stretched')
            rows[g] = {'n_matrices': n_mat, 'n_values': len(s),
                       'dct8': {**metrics(dct, s), 'params': coeffs},
                       'powerlaw': {**metrics(pl, s), 'params': pl_p},
                       'stretched': {**metrics(se, s), 'params': se_p}}
            print(f'{label:10s} {g:10s} rel-L2  dct8 {rows[g]["dct8"]["rel_l2"]:.4f}  '
                  f'powerlaw {rows[g]["powerlaw"]["rel_l2"]:.4f}  stretched {rows[g]["stretched"]["rel_l2"]:.4f}   '
                  f'top-16  dct8 {rows[g]["dct8"]["top16_rel_l2"]:.4f}  powerlaw {rows[g]["powerlaw"]["top16_rel_l2"]:.4f}  '
                  f'stretched {rows[g]["stretched"]["top16_rel_l2"]:.4f}')
        report['teachers'][label] = {'source': origins.get(label, os.path.basename(path)), 'groups': rows}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f'w7_specfit_{stamp}.json')
    json.dump(report, open(fn, 'w'), indent=1)
    print('wrote', fn)


if __name__ == '__main__':
    main()
