"""prism_metrics.py — the accounting the field now expects, as pure functions.

(a) steps_to_target     — `compute_score` semantics (prism_eval.py): the
                          baseline's steps-to-best divided by the method's
                          steps to reach that loss; left-censoring and
                          reached-at-init flagged, never divided by zero.
(b) token-equivalent    — fit a monotone power-law baseline curve
    multiplier            L(t) = a + b·(t + t0)^(−c) to the baseline's eval
                          points, report the fit residual, REFUSE (return the
                          reason) when RMSE exceeds a stated tolerance, then
                          invert the curve at the method's end-of-training loss:
                          t_equiv = L⁻¹(L_method_end); multiplier = t_equiv/t_method.
(c) donor compute       — teacher FLOPs ≈ 6·N·D, converted into student-step
                          units so any multiplier is quotable with and without
                          the amortized donor cost.

Everything is numpy; scipy is used for refinement when present, never required.
Unit tests on synthetic curves with known answers live in prism_selftest.py.
"""
import math

import numpy as np


# ── (a) steps-to-target ────────────────────────────────────────────────────

def steps_to_target(method_curve, target, eval_every, baseline_best_step=None):
    """method_curve: {step: loss}. Returns the compute_score-shaped dict."""
    curve = {int(k): float(v) for k, v in method_curve.items()}
    hit = next((s for s in sorted(curve) if curve[s] <= target), None)
    if hit is None:
        return {'prism_score': None, 'hit_step': None, 'reached_baseline_quality': False,
                'baseline_target': target, 'left_censored': False,
                'note': 'method never reached baseline best loss'}
    if hit == 0:
        return {'prism_score': None, 'hit_step': 0, 'reached_baseline_quality': True,
                'reached_at_init': True, 'baseline_target': target, 'left_censored': True,
                'note': 'baseline-best matched at the step-0 eval; steps-to-target is not '
                        'measurable on this run. Before publishing any at-init claim, see '
                        'the binding framing spec in docs/NEXT-EXPERIMENTS.md.'}
    out = {'prism_score': (baseline_best_step / hit) if baseline_best_step else None,
           'hit_step': hit, 'reached_baseline_quality': True, 'baseline_target': target,
           'left_censored': hit == eval_every,
           'note': ('score is a LOWER BOUND: target reached at first eval, true crossing '
                    f'unresolved below step {eval_every}') if hit == eval_every else ''}
    return out


def steps_to_target_ratio(curve_a, curve_b, target, eval_every):
    """steps(b reaches target)/steps(a reaches target): >1 means a is faster.
    Returns (ratio or None, hit_a, hit_b)."""
    ha = steps_to_target(curve_a, target, eval_every)['hit_step']
    hb = steps_to_target(curve_b, target, eval_every)['hit_step']
    if ha in (None, 0) or hb in (None, 0):
        return None, ha, hb
    return hb / ha, ha, hb


# ── (b) token-equivalent multiplier ───────────────────────────────────────

def _fit_ab(t, L, t0, c):
    """Closed-form a, b for fixed (t0, c): L ≈ a + b·x, x = (t+t0)^(-c)."""
    x = (t + t0) ** (-c)
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, L, rcond=None)
    a, b = coef
    pred = a + b * x
    rmse = float(np.sqrt(np.mean((pred - L) ** 2)))
    return a, b, rmse


def fit_power_law(curve, min_step=1, c_grid=None, t0_grid=None, refine=True):
    """Fit L(t) = a + b·(t+t0)^(−c), b>0, c>0, to {step: loss} (steps >= min_step).
    Grid search over (c, t0) with closed-form (a, b), optional scipy refinement.
    Returns dict(a, b, c, t0, rmse, n, residuals{step: resid})."""
    pts = sorted((int(k), float(v)) for k, v in curve.items() if int(k) >= min_step)
    if len(pts) < 4:
        raise ValueError('need at least 4 eval points to fit the baseline curve')
    t = np.array([p[0] for p in pts], dtype=float)
    L = np.array([p[1] for p in pts], dtype=float)
    c_grid = c_grid if c_grid is not None else np.geomspace(0.05, 4.0, 60)
    t0_grid = t0_grid if t0_grid is not None else np.concatenate(
        [[0.0], np.geomspace(1.0, max(10.0, t.max()), 40)])
    best = None
    for c in c_grid:
        for t0 in t0_grid:
            a, b, rmse = _fit_ab(t, L, t0, c)
            if b <= 0:
                continue
            if best is None or rmse < best[2]:
                best = (a, b, rmse, c, t0)
    if best is None:
        raise ValueError('no monotone-decreasing fit found (b>0 failed everywhere)')
    a, b, rmse, c, t0 = best
    if refine:
        try:
            from scipy.optimize import curve_fit

            def f(tt, a_, b_, c_, t0_):
                return a_ + b_ * (tt + t0_) ** (-c_)
            popt, _ = curve_fit(f, t, L, p0=[a, b, c, t0],
                                bounds=([-np.inf, 1e-12, 1e-4, 0.0], [np.inf, np.inf, 10.0, np.inf]),
                                maxfev=20000)
            pred = f(t, *popt)
            r2 = float(np.sqrt(np.mean((pred - L) ** 2)))
            if r2 <= rmse:
                a, b, c, t0, rmse = float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]), r2
        except Exception:
            pass
    pred = a + b * (t + t0) ** (-c)
    return {'a': float(a), 'b': float(b), 'c': float(c), 't0': float(t0), 'rmse': float(rmse),
            'n': int(len(t)), 'residuals': {str(int(s)): float(r) for s, r in zip(t, L - pred)}}


def invert_power_law(fit, loss):
    """t such that a + b·(t+t0)^(−c) == loss. None if loss <= a (below the
    asymptote — the baseline never gets there)."""
    a, b, c, t0 = fit['a'], fit['b'], fit['c'], fit['t0']
    if loss <= a:
        return None
    return float(((loss - a) / b) ** (-1.0 / c) - t0)


def token_equivalent_multiplier(baseline_curve, method_curve, rmse_tol=0.01,
                                min_step=1, tokens_per_step=None):
    """The multiplier at the method's end of training: t_equiv / t_method,
    where t_equiv is the baseline steps the fitted curve says are needed to
    reach the method's final loss. Refuses (multiplier=None, reason) when the
    baseline fit is worse than rmse_tol or the method's final loss lies below
    the fitted asymptote (then `lower_bound` gives t_max/t_method)."""
    fit = fit_power_law(baseline_curve, min_step=min_step)
    m_steps = sorted(int(k) for k in method_curve)
    t_method = m_steps[-1]
    L_end = float(method_curve[str(t_method)] if str(t_method) in method_curve
                  else method_curve[t_method])
    out = {'fit': fit, 'rmse_tol': rmse_tol, 't_method': t_method, 'method_end_loss': L_end,
           'multiplier': None, 'reason': ''}
    if fit['rmse'] > rmse_tol:
        out['reason'] = (f"baseline fit RMSE {fit['rmse']:.4f} > tolerance {rmse_tol}; "
                         'refusing to extrapolate')
        return out
    t_eq = invert_power_law(fit, L_end)
    b_max = max(int(k) for k in baseline_curve)
    if t_eq is None:
        out['reason'] = (f"method end loss {L_end:.4f} is below the fitted baseline asymptote "
                         f"a={fit['a']:.4f}; the baseline never reaches it")
        out['lower_bound'] = b_max / t_method
        return out
    out['t_equiv'] = t_eq
    out['multiplier'] = t_eq / t_method
    out['extrapolated'] = bool(t_eq > b_max)
    if tokens_per_step:
        out['tokens_equiv'] = t_eq * tokens_per_step
        out['tokens_method'] = t_method * tokens_per_step
    return out


# ── (c) donor compute ─────────────────────────────────────────────────────

def flops_6nd(n_params, n_tokens):
    return 6.0 * float(n_params) * float(n_tokens)


def donor_adjusted_multiplier(multiplier, t_method, teacher_params, teacher_tokens,
                              student_params, tokens_per_step, n_students_amortized=1):
    """Quote the multiplier with the donor (teacher) cost folded into the
    method's side, in student-step units: donor_steps = 6·N_t·D_t /
    (6·N_s·tokens_per_step) / n_students_amortized. Returns dict."""
    if multiplier is None:
        return {'with_donor': None, 'donor_steps': None}
    donor_steps = (flops_6nd(teacher_params, teacher_tokens)
                   / flops_6nd(student_params, tokens_per_step)) / max(1, n_students_amortized)
    t_eq = multiplier * t_method
    return {'without_donor': multiplier, 'donor_steps': donor_steps,
            'with_donor': t_eq / (t_method + donor_steps),
            'n_students_amortized': n_students_amortized}
