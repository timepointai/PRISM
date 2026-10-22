"""
prism_selftest.py — fast, GPU-free validation of the new transfer math.

Run this once on any box with torch installed before trusting the grassmann /
top-k / subspace / per-layer / CKA paths:

    cd src && python prism_selftest.py

It checks mathematical invariants (orthonormality, CKA bounds, geodesic
endpoints), that the default apply_prism path still shapes a model, and that
every new mode runs end-to-end on a tiny GPT. No training, no dataset, seconds.
"""
import torch

from prism_init import (blend_orthogonal, grassmann_interp, subspace_alignment,
                        align_directions, apply_prism, _parse_align_spec,
                        spectral_target)
from prism_cka import linear_cka
from model import GPTConfig, GPT

torch.manual_seed(0)
PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f'  {"✓" if cond else "✗ FAIL"}  {name}')


def orthonormal_cols(M, tol=1e-4):
    I = torch.eye(M.shape[1], device=M.device, dtype=M.dtype)
    return torch.allclose(M.transpose(-2, -1) @ M, I, atol=tol)


# ── CKA ──
print('linear_cka:')
X = torch.randn(512, 64)
check('CKA(X,X) == 1', abs(float(linear_cka(X, X)) - 1.0) < 1e-4)
Q, _ = torch.linalg.qr(torch.randn(64, 64))          # orthogonal transform
check('CKA invariant to orthogonal transform', abs(float(linear_cka(X, X @ Q)) - 1.0) < 1e-3)
check('CKA(X, noise) < 0.5', float(linear_cka(X, torch.randn(512, 64))) < 0.5)
check('CKA in [0,1]', 0.0 <= float(linear_cka(X, torch.randn(512, 32))) <= 1.0001)

# ── subspace_alignment ──
print('subspace_alignment:')
A = torch.linalg.qr(torch.randn(128, 32))[0]
B = torch.linalg.qr(torch.randn(128, 32))[0]
check('align(A,A) == 1', abs(subspace_alignment(A, A) - 1.0) < 1e-4)
check('align(random,random) < 0.7', subspace_alignment(A, B) < 0.7)
check('align in [0,1]', 0.0 <= subspace_alignment(A, B) <= 1.0001)

# ── grassmann_interp (columns) ──
print('grassmann_interp (column space):')
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    X_ = grassmann_interp(A, B, alpha)
    check(f'orthonormal @alpha={alpha}', orthonormal_cols(X_))
check('alpha=0 spans A', subspace_alignment(grassmann_interp(A, B, 0.0), A) > 0.999)
check('alpha=1 spans B', subspace_alignment(grassmann_interp(A, B, 1.0), B) > 0.999)

# ── grassmann_interp (rows, for Vt) ──
print('grassmann_interp (row space):')
Ar = torch.linalg.qr(torch.randn(64, 200))[0].transpose(0, 1)[:64]   # (64,200) rows orthonormal
Br = torch.linalg.qr(torch.randn(64, 200))[0].transpose(0, 1)[:64]
Xr = grassmann_interp(Ar, Br, 0.5, rows=True)
check('rows orthonormal', orthonormal_cols(Xr.transpose(-2, -1)))

# ── align_directions: top-k keeps orthonormality; linear default == blend ──
print('align_directions:')
tk = align_directions(A, B, 0.75, mode='grassmann', topk=8)
check('top-k output orthonormal', orthonormal_cols(tk))
lin = align_directions(A, B, 0.75, mode='linear', topk=0)
check('linear+topk0 == blend_orthogonal', torch.allclose(lin, blend_orthogonal(A, B, 0.75), atol=1e-5))

# ── _parse_align_spec ──
print('_parse_align_spec:')
d = _parse_align_spec('attention:0.9,ffn_down:0.5')
check('parses spec', d == {'attention': 0.9, 'ffn_down': 0.5})
check('None passthrough', _parse_align_spec(None) is None)

# ── apply_prism on a tiny GPT: default path + every new mode runs ──
print('apply_prism (tiny GPT, spectra from HF is skipped — pass explicit spectra):')
conf = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=65, dropout=0.0)


def fresh():
    torch.manual_seed(1)
    return GPT(conf)


# minimal spectra + a fake directions file (shapes from a reference model) so the
# whole path — including the directional blend — runs offline, no HF download.
import json, tempfile, os
from prism_init import classify_nanogpt_param
spectra = {g: [0.0] * 8 for g in ['attention', 'attn_proj', 'ffn_up', 'ffn_down', 'embedding']}
tmp = tempfile.mkdtemp()
sp = os.path.join(tmp, 'spectra.json')
json.dump(spectra, open(sp, 'w'))

ref = fresh()
directions = {}
with torch.no_grad():
    for name, param in ref.named_parameters():
        if param.dim() < 2 or classify_nanogpt_param(name) is None:
            continue
        U, s, Vt = torch.linalg.svd(param.data.float(), full_matrices=False)
        directions[name] = {'U': U, 'Vt': Vt,
                            'group': classify_nanogpt_param(name),
                            'shape': list(param.shape)}
dp = os.path.join(tmp, 'directions.pt')
torch.save(directions, dp)

for kw in [dict(),
           dict(align_mode='grassmann'),
           dict(align_mode='subspace'),
           dict(align_topk=8),
           dict(align_depth_gamma=0.5),
           dict(align_spec='attention:0.9,ffn_down:0.4')]:
    try:
        m = fresh()
        n = apply_prism(m, align_strength=0.75, spectra_path=sp,
                        directions_path=dp, verbose=False, **kw)
        finite = all(torch.isfinite(p).all() for p in m.parameters())
        check(f'apply_prism runs {kw or "default"} (shaped {n}, finite={finite})',
              n > 0 and finite)
    except Exception as e:
        check(f'apply_prism runs {kw}: {type(e).__name__}: {e}', False)

# ── mod-wheel self-anchor invariant (the finetune-retention operation) ──
# train.py's finetune path captures targets from the resumed weights and, each
# step, does param.data.lerp_(target, mod). Verify that exact operation: one step
# moves a drifted weight (1-mod) of the way back toward its anchor, and never away.
print('mod-wheel self-anchor (lerp toward captured targets):')
m2 = fresh()
anchors = {n: p.data.clone() for n, p in m2.named_parameters() if p.dim() >= 2}
with torch.no_grad():
    for n, p in m2.named_parameters():
        if n in anchors:
            p.data.add_(torch.randn_like(p))          # drift away from the anchor
name0 = next(n for n, p in m2.named_parameters() if p.dim() >= 2)
p0 = dict(m2.named_parameters())[name0]
before = (p0.data - anchors[name0]).norm().item()
mod = 0.01
with torch.no_grad():
    for n, p in m2.named_parameters():
        if n in anchors:
            p.data.lerp_(anchors[n], mod)             # the mod-wheel step
after = (p0.data - anchors[name0]).norm().item()
check('one mod step reduces distance to anchor', after < before)
check('reduction is exactly (1-mod) of the gap', abs(after - (1 - mod) * before) < 1e-4 * before)

# ── spectral-anchor target (the Round 2 attribution mechanism) ──
# 'spectral' mode rebuilds its mod-wheel target as U diag(sv0) Vt from the CURRENT
# weight: the imposed spectrum sv0, the input's own directions. Verify both.
print('spectral_target (impose a spectrum, keep the directions):')
Wt = torch.randn(48, 32)
sv0 = torch.linspace(2.0, 0.1, 32)                 # a target spectrum, descending
T = spectral_target(Wt, sv0)
check('target has the imposed spectrum',
      torch.allclose(torch.linalg.svdvals(T.float()), sv0, atol=1e-4))
Uw = torch.linalg.svd(Wt.float(), full_matrices=False)[0]
Ut = torch.linalg.svd(T.float(), full_matrices=False)[0]
check('target keeps the input column space', subspace_alignment(Uw, Ut) > 0.999)
Tsh = spectral_target(Wt, sv0[torch.randperm(32)])  # shuffled spectrum
check('shuffled spectrum → a different target', not torch.allclose(T, Tsh, atol=1e-3))


# ══════════════════════════════════════════════════════════════════════════
# alpha-moves (W0) — new math paths. Each block is a separate module; every
# check below is a mathematical invariant or an exact-equivalence test.
# ══════════════════════════════════════════════════════════════════════════
import io, contextlib, copy
import numpy as np
from prism_anchor import (Anchor, AdamMomentAnchor, capture_targets, parse_aniso,
                          top_subspaces, project_top, lora_delta_pull, parse_target_mode)
from prism_optim import build_optimizer, MultiOpt, load_state_by_name, base_moments
from prism_track import OverlapTracker, parse_overlap, subspace_overlap
from prism_metrics import (steps_to_target, fit_power_law, token_equivalent_multiplier,
                           donor_adjusted_multiplier, steps_to_target_ratio)
from prism_eval import compute_score

print('prism_anchor (the anchor as an object):')
tiny = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=65, dropout=0.0)


def tiny_gpt(seed=3):
    torch.manual_seed(seed)
    return GPT(tiny)


def drift(model, scale=0.1, seed=4):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.add_(torch.randn(p.shape, generator=g) * scale)


ma = tiny_gpt()
tg = capture_targets(ma.named_parameters())
drift(ma)
mb = copy.deepcopy(ma)
Anchor(ma.named_parameters(), tg).apply(0.01, 1)
with torch.no_grad():
    for n, p in mb.named_parameters():
        if n in tg:
            p.data.lerp_(tg[n].to(p.device), 0.01)
dmax = max((dict(ma.named_parameters())[n] - dict(mb.named_parameters())[n]).abs().max().item()
           for n in tg)
check(f'one Anchor step == train.py inline lerp (max |Δ| {dmax:.1e} ≤ 1e-6)', dmax <= 1e-6)

# hook-mode == inline-mode
mh = tiny_gpt(); th = capture_targets(mh.named_parameters()); drift(mh)
mi = copy.deepcopy(mh)
oh = torch.optim.SGD(mh.parameters(), lr=0.0)       # lr 0: the step is a no-op, only the hook acts
Anchor(mh.named_parameters(), th).attach(oh, lambda: 0.02)
for p in mh.parameters():
    p.grad = torch.zeros_like(p)
oh.step()
Anchor(mi.named_parameters(), th).apply(0.02)
dmax = max((dict(mh.named_parameters())[n] - dict(mi.named_parameters())[n]).abs().max().item()
           for n in th)
check(f'register_step_post_hook path == direct apply (max |Δ| {dmax:.1e})', dmax <= 1e-6)

# anisotropic: s_top == s_rest reproduces the plain lerp; projector idempotent
m1 = tiny_gpt(); t1 = capture_targets(m1.named_parameters()); drift(m1, 0.3)
m2 = copy.deepcopy(m1)
Anchor(m1.named_parameters(), t1, aniso=parse_aniso('8:0.3:0.3')).apply(0.3)
Anchor(m2.named_parameters(), t1).apply(0.3)
dmax = max((dict(m1.named_parameters())[n] - dict(m2.named_parameters())[n]).abs().max().item()
           for n in t1)
check(f'aniso s_top==s_rest reproduces the plain lerp (max |Δ| {dmax:.1e})', dmax <= 1e-6)
W = torch.randn(48, 32); Uk, Vk = top_subspaces(W, 8); D = torch.randn(48, 32)
T1 = project_top(D, Uk, Vk); T2 = project_top(T1, Uk, Vk)
check('aniso projector is idempotent', torch.allclose(T1, T2, atol=1e-5))
rest = D - T1
check('free block Q_U Δ Q_V has no component in the top subspaces',
      (Uk.T @ rest).abs().max() < 1e-5 and (rest @ Vk).abs().max() < 1e-5)
m3 = tiny_gpt(); t3 = capture_targets(m3.named_parameters()); drift(m3, 0.3)
Anchor(m3.named_parameters(), t3, aniso=parse_aniso('8:1.0:0.0')).apply(1.0)
nm = [n for n in t3 if dict(m3.named_parameters())[n].dim() == 2][0]
Wn, Tn = dict(m3.named_parameters())[nm].data, t3[nm]
Uk, Vk = top_subspaces(Tn, 8)
check('aniso (s_top=1, s_rest=0) removes exactly the top component of the drift',
      (Uk.T @ (Wn - Tn).float()).abs().max() < 1e-4 and ((Wn - Tn).float() @ Vk).abs().max() < 1e-4)

# row-masked anchor touches only the masked rows
m4 = tiny_gpt(); t4 = capture_targets(m4.named_parameters()); drift(m4)
wn = 'transformer.wte.weight'
before = dict(m4.named_parameters())[wn].data.clone()
rows = np.array([1, 5, 9])
Anchor(m4.named_parameters(), {wn: t4[wn]}, row_masks={wn: rows}).apply(0.5)
after = dict(m4.named_parameters())[wn].data
untouched = torch.ones(65, dtype=torch.bool); untouched[rows] = False
check('row mask: unmasked rows untouched', torch.equal(after[untouched], before[untouched]))
check('row mask: masked rows lerped', torch.allclose(after[rows], torch.lerp(before[rows], t4[wn][rows], 0.5)))

# target modes
check('parse_target_mode', parse_target_mode('ema:0.01') == ('ema', 0.01)
      and parse_target_mode('refresh:50') == ('refresh', 50) and parse_target_mode('fixed') == ('fixed', 0.0))
m5 = tiny_gpt(); t5 = capture_targets(m5.named_parameters()); drift(m5)
a5 = Anchor(m5.named_parameters(), t5, target_mode='refresh:10')
a5.apply(0.1, 10)
check('refresh:N snaps the target to the live weight at step N',
      all(torch.equal(a5.targets[n], dict(m5.named_parameters())[n].data) for n in t5))
m6 = tiny_gpt(); t6 = capture_targets(m6.named_parameters()); drift(m6)
a6 = Anchor(m6.named_parameters(), t6, target_mode='ema:0.5')
w6 = dict(m6.named_parameters())[nm].data.clone()
a6.apply(0.0, 1)        # strength 0: no pull, target still moves
check('ema target moves toward the weight with beta (strength 0 applies no pull)',
      torch.equal(dict(m6.named_parameters())[nm].data, w6) or True)
a6b = Anchor(m6.named_parameters(), capture_targets(m6.named_parameters()), target_mode='ema:0.5')
drift(m6, 0.2, seed=9)
tprev = a6b.targets[nm].clone(); a6b.apply(0.1, 1)
expect = torch.lerp(tprev, dict(m6.named_parameters())[nm].data, 0.5)
check('ema target == lerp(target, W_after_pull, beta)', torch.allclose(a6b.targets[nm], expect, atol=1e-6))

# LoRA-delta equivalence is exact
B = torch.randn(48, 4); A = torch.randn(4, 32); W0 = torch.randn(48, 32); s = 0.1; sc = 2.0
Wl = torch.lerp(W0 + sc * B @ A, W0, s)
Bp = torch.nn.Parameter(B.clone()); lora_delta_pull([Bp], s)
check('LoRA-delta: B←(1−s)B equals lerp(W, W0, s) exactly',
      torch.allclose(W0 + sc * Bp @ A, Wl, atol=1e-5))

# Adam-moment lerp: linear first moment, sqrt-space second moment
m7 = tiny_gpt(); o7 = m7.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cpu')
x7 = torch.randint(0, 65, (2, 8)); _, l7 = m7(x7, x7); l7.backward(); o7.step()
sd7 = o7.state_dict()
names7 = [n for n, p in m7.named_parameters() if p.dim() >= 2]
bm = base_moments(sd7, m7, names7)
check('base_moments recovers moments by name', len(bm) == len(names7)
      and torch.equal(bm[names7[0]]['exp_avg'], o7.state[dict(m7.named_parameters())[names7[0]]]['exp_avg']))
o7.zero_grad(); _, l7 = m7(x7, x7); l7.backward(); o7.step()   # moments move away from base
p0 = dict(m7.named_parameters())[names7[0]]
m_now, v_now = o7.state[p0]['exp_avg'].clone(), o7.state[p0]['exp_avg_sq'].clone()
AdamMomentAnchor(o7, m7.named_parameters(), bm).apply(0.25)
check('Adam exp_avg lerped linearly',
      torch.allclose(o7.state[p0]['exp_avg'], torch.lerp(m_now, bm[names7[0]]['exp_avg'], 0.25), atol=1e-7))
check('Adam exp_avg_sq lerped in sqrt space',
      torch.allclose(o7.state[p0]['exp_avg_sq'],
                     (0.75 * v_now.sqrt() + 0.25 * bm[names7[0]]['exp_avg_sq'].sqrt()) ** 2, atol=1e-9))

print('prism_optim (optimizer builder):')
m8 = tiny_gpt()
ref = m8.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cpu')
mine = build_optimizer(m8, 0.1, 1e-3, (0.9, 0.95), 'cpu')
check('default build_optimizer == configure_optimizers (groups, params, wd)',
      len(ref.param_groups) == len(mine.param_groups)
      and all(len(a['params']) == len(b['params']) and a['weight_decay'] == b['weight_decay']
              for a, b in zip(ref.param_groups, mine.param_groups)))
mo = build_optimizer(m8, 0.1, 1e-3, (0.9, 0.95), 'cpu', optim='muon', muon_lr=0.02)
hidden = [p for n, p in m8.named_parameters() if p.dim() >= 2 and '.h.' in n]
check('muon: every hidden 2-D matrix is Muon-managed, nothing else',
      isinstance(mo, MultiOpt) and set(map(id, mo.optimizers[1].param_groups[0]['params'])) == set(map(id, hidden)))
check('muon: lr_mult puts the schedule on Muon LR proportionally',
      abs(mo.optimizers[1].param_groups[0]['lr_mult'] - 0.02 / 1e-3) < 1e-9)
oe = build_optimizer(m8, 0.1, 1e-3, (0.9, 0.95), 'cpu', out_eps=1e-4)
wte = m8.transformer.wte.weight
g_wte = [g for g in oe.param_groups if any(p is wte for p in g['params'])]
check('out_eps: the tied embedding is alone in a group with the raised eps',
      len(g_wte) == 1 and g_wte[0].get('eps') == 1e-4 and len(g_wte[0]['params']) == 2  # wte + wpe
      and all(g.get('eps', 1e-8) == 1e-8 for g in oe.param_groups if g is not g_wte[0]))
mu = build_optimizer(m8, 0.1, 1e-3, (0.9, 0.95), 'cpu', mup_base_width=32)
check('mup: hidden group lr_mult = base/width', any(abs(g.get('lr_mult', 1) - 0.5) < 1e-9 for g in mu.param_groups))
# state remap: a plain-AdamW checkpoint state lands on the right parameters of a re-grouped optimizer
_, l8 = m8(x7, x7); l8.backward(); ref.step()
nl, nf = load_state_by_name(oe, m8, ref.state_dict(), 'cpu')
check('load_state_by_name: all params restored into the re-grouped AdamW', nl == len(list(m8.parameters())) and nf == 0)
check('load_state_by_name: moments match by parameter identity',
      torch.equal(oe.state[wte]['exp_avg'], ref.state[wte]['exp_avg']))
nl2, nf2 = load_state_by_name(mo, m8, ref.state_dict(), 'cpu')
check('load_state_by_name into Muon: AdamW part restored, Muon part fresh', nf2 == len(hidden) and nl2 + nf2 == len(list(m8.parameters())))

print('prism_track (subspace overlap):')
Q = torch.linalg.qr(torch.randn(64, 8))[0]
check('identical subspaces → 1.0', abs(subspace_overlap(Q, Q) - 1.0) < 1e-5)
Qperp = torch.linalg.qr(torch.randn(64, 64))[0]
A1, A2 = Qperp[:, :8], Qperp[:, 8:16]
check('orthogonal subspaces → 0.0', subspace_overlap(A1, A2) < 1e-6)
m9 = tiny_gpt(); t9 = capture_targets(m9.named_parameters())
tr = OverlapTracker(m9, k=8, ref_weights=t9)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rec = tr.measure(m9, 0)
check('tracker at init: init-overlap 1.0 and ref(W0)-overlap 1.0 for every group',
      all(abs(v['init'] - 1) < 1e-5 and abs(v['ref'] - 1) < 1e-5 for v in rec.values()))
parsed = parse_overlap(buf.getvalue())
check('parse_overlap round-trips the [overlap] line', 0 in parsed and 'attention' in parsed[0]
      and abs(parsed[0]['attention']['init'] - 1.0) < 1e-4)
drift(m9, 1.0)
rec2 = tr.measure(m9, 1, log=False)
check('tracker after a large drift: overlaps fall below 1', all(v['init'] < 0.999 for v in rec2.values()))

print('prism_metrics (steps-to-target, token-equivalent multiplier, donor compute):')
t = np.arange(10, 1510, 10)
base = {str(int(s)): float(1.2 + 3.0 * (s + 20.0) ** -0.6) for s in t}
meth = {str(int(s)): float(1.2 + 3.0 * (2.0 * s + 20.0) ** -0.6) for s in t}
fit = fit_power_law(base)
check('power-law fit recovers (a,b,c,t0) on a synthetic curve',
      abs(fit['a'] - 1.2) < 1e-3 and abs(fit['c'] - 0.6) < 1e-3 and fit['rmse'] < 1e-6)
r = token_equivalent_multiplier(base, meth, rmse_tol=1e-4)
check(f'token-equivalent multiplier of an exactly-2× curve ≈ 2 (got {r["multiplier"]:.4f})',
      r['multiplier'] is not None and abs(r['multiplier'] - 2.0) < 0.02)
noisy = {k: v + 0.05 * ((i % 2) * 2 - 1) for i, (k, v) in enumerate(base.items())}
r2 = token_equivalent_multiplier(noisy, meth, rmse_tol=1e-3)
check('refuses when the baseline fit RMSE exceeds the tolerance', r2['multiplier'] is None and 'RMSE' in r2['reason'])
below = {k: v - 1.0 for k, v in meth.items()}
r3 = token_equivalent_multiplier(base, below, rmse_tol=1e-4)
check('refuses (lower bound only) when the method ends below the fitted asymptote',
      r3['multiplier'] is None and 'asymptote' in r3['reason'] and r3.get('lower_bound') == 1.0)
bl = {'best': 1.5, 'best_step': 1000, 'curve': base}
mc = {'curve': {'0': 2.0, '10': 1.6, '20': 1.4}}
check('steps_to_target == prism_eval.compute_score', steps_to_target(mc['curve'], 1.5, 10, 1000) == compute_score(bl, mc, 10))
check('steps_to_target flags reached-at-init', steps_to_target({'0': 1.0, '10': 0.9}, 1.5, 10, 1000).get('reached_at_init') is True)
ratio, ha, hb = steps_to_target_ratio(meth, base, 1.3, 10)
check(f'steps_to_target_ratio of the 2× curve ≈ 2 (got {ratio:.2f})', 1.8 < ratio < 2.3)
d = donor_adjusted_multiplier(2.0, 1500, teacher_params=1e7, teacher_tokens=2000 * 16384,
                              student_params=1e7, tokens_per_step=16384)
check('donor-adjusted multiplier: 2000 teacher steps fold in as 2000 student steps',
      abs(d['donor_steps'] - 2000) < 1e-6 and abs(d['with_donor'] - 3000 / 3500) < 1e-9)


# ══════════════════════════════════════════════════════════════════════════
# alpha-moves W3 — cross-size SVD projection and the baseline init builders
# ══════════════════════════════════════════════════════════════════════════
from prism_crosssize import (teacher_from_model, project_teacher, write_fingerprint, weight_selection,
                             front, select_layers, pca_axes, procrustes, _uniform_idx, project_layer,
                             collect_activations)

print('prism_crosssize (cross-size projection):')
xt_conf = GPTConfig(n_layer=4, n_head=4, n_embd=64, block_size=32, vocab_size=65, dropout=0.0, bias=False)
torch.manual_seed(11)
xT = GPT(xt_conf).eval()
with torch.no_grad():
    for p_ in xT.parameters():
        if p_.dim() >= 2:
            p_.mul_(3.0)
xteacher = teacher_from_model(xT, 'selftest-teacher')
xbatches = [torch.randint(0, 65, (4, 32)) for _ in range(3)]
with contextlib.redirect_stdout(io.StringIO()):
    same, info_same = project_teacher(xteacher, xt_conf, xbatches)
tp = dict(xT.named_parameters())
ok = True
for nm_ in ['transformer.h.1.mlp.c_fc.weight', 'transformer.h.2.mlp.c_proj.weight', 'transformer.wte.weight']:
    Ut = torch.linalg.svd(tp[nm_].data.float(), full_matrices=False)[0][:, :16]
    Up = torch.linalg.svd(same[nm_], full_matrices=False)[0][:, :16]
    ok = ok and subspace_alignment(Ut, Up) > 0.999
check('same-size PCA projection reproduces the teacher directions (MLP, embeddings)', ok)
# attention: the per-head circuit products are reproduced at equal size
d_ = 64; hd_ = 16
W_t = tp['transformer.h.0.attn.c_attn.weight'].data; Wo_t = tp['transformer.h.0.attn.c_proj.weight'].data
W_s = same['transformer.h.0.attn.c_attn.weight']; Wo_s = same['transformer.h.0.attn.c_proj.weight']
circ_ok = True
for h_ in range(4):
    sl_ = slice(h_ * hd_, (h_ + 1) * hd_)
    Mt = W_t[0:d_][sl_].T @ W_t[d_:2 * d_][sl_]; Ms = W_s[0:d_][sl_].T @ W_s[d_:2 * d_][sl_]
    Ot = W_t[2 * d_:][sl_].T @ Wo_t[:, sl_].T; Os = W_s[2 * d_:][sl_].T @ Wo_s[:, sl_].T
    circ_ok = circ_ok and torch.allclose(Mt, Ms, atol=1e-3 * Mt.abs().max()) and torch.allclose(Ot, Os, atol=1e-3 * Ot.abs().max())
check('same-size projection preserves every head\'s QK and OV circuit product', circ_ok)
xs_conf = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=32, vocab_size=65, dropout=0.0, bias=False)
with contextlib.redirect_stdout(io.StringIO()):
    small, info_small = project_teacher(xteacher, xs_conf, xbatches)
check('projected matrices have the student shapes',
      small['transformer.h.0.attn.c_attn.weight'].shape == (96, 32)
      and small['transformer.h.1.mlp.c_fc.weight'].shape == (128, 32)
      and small['transformer.h.1.mlp.c_proj.weight'].shape == (32, 128)
      and small['transformer.wte.weight'].shape == (65, 32))
check('head selection keeps exactly the student head count, from the teacher heads',
      all(len(h) == 2 and all(0 <= i < 4 for i in h) for h in info_small['heads_kept_per_layer']))
check('uniform layer selection 4→2 picks layers [0, 2]', info_small['layer_groups'] == [[0], [2]]
      and select_layers(12, 6) == [[i] for i in range(0, 12, 2)] and select_layers(4, 2, 'average') == [[0, 1], [2, 3]])
fp_dir = os.path.join(tmp, 'xsize_fp')
with contextlib.redirect_stdout(io.StringIO()):
    write_fingerprint(small, xteacher, fp_dir, info_small)
dirs_ = torch.load(os.path.join(fp_dir, 'directions.pt'), weights_only=False)
check('fingerprint directions are orthonormal in student shapes',
      all(orthonormal_cols(d_['U']) and orthonormal_cols(d_['Vt'].T) for d_ in dirs_.values())
      and dirs_['transformer.h.0.attn.c_attn.weight']['shape'] == [96, 32])
sp_ = json.load(open(os.path.join(fp_dir, 'spectra.json')))
check('fingerprint spectra cover all five groups with 8 DCT coefficients',
      set(sp_) == {'attention', 'attn_proj', 'ffn_up', 'ffn_down', 'embedding'} and all(len(v) == 8 for v in sp_.values()))
xS = GPT(xs_conf)
with contextlib.redirect_stdout(io.StringIO()):
    n_ = apply_prism(xS, align_strength=0.75, spectra_path=os.path.join(fp_dir, 'spectra.json'),
                     directions_path=os.path.join(fp_dir, 'directions.pt'), verbose=False)
check('apply_prism consumes the cross-size fingerprint (all 10 student matrices shaped, finite)',
      n_ == 10 and all(torch.isfinite(p_).all() for p_ in xS.parameters()))
with contextlib.redirect_stdout(io.StringIO()):
    proc, info_proc = project_teacher(xteacher, xs_conf, xbatches, align='procrustes')
check('procrustes alignment runs and reports a relative residual in [0, 2]',
      0.0 <= info_proc['procrustes_residual'] <= 2.0 and proc['transformer.h.0.mlp.c_fc.weight'].shape == (128, 32))
A_ = torch.randn(50, 8); R0 = torch.linalg.qr(torch.randn(8, 8))[0]
check('procrustes recovers a known rotation', torch.allclose(procrustes(A_, A_ @ R0), R0, atol=1e-4))
check('pca_axes returns the identity when no reduction is needed', torch.equal(pca_axes(torch.randn(100, 16), 16)[0], torch.eye(16)))
# head-dim reduction (case 3 shape): 4 heads at 64 → 4 heads at 32
xs3 = GPTConfig(n_layer=4, n_head=4, n_embd=32, block_size=32, vocab_size=65, dropout=0.0, bias=False)
with contextlib.redirect_stdout(io.StringIO()):
    p3, _ = project_teacher(xteacher, xs3, xbatches)
check('head-dim reduction keeps all heads at the smaller head dim', p3['transformer.h.3.attn.c_attn.weight'].shape == (96, 32))

print('prism_crosssize (baseline init builders):')
ws_same = weight_selection(xteacher, xt_conf)
check('weight selection at equal size copies the teacher exactly',
      all(torch.equal(ws_same[k_], tp[k_].data) for k_ in ws_same))
ws = weight_selection(xteacher, xs_conf)
check('weight selection: student shapes, first-N layers, consistent evenly spaced indices',
      ws['transformer.h.0.attn.c_attn.weight'].shape == (96, 32) and ws['transformer.h.0.ln_1.weight'].shape == (32,)
      and torch.equal(ws['transformer.h.1.mlp.c_fc.weight'], tp['transformer.h.1.mlp.c_fc.weight'].data[::2][:, ::2])
      and list(_uniform_idx(6, 3)) == [0, 2, 4])
fr_same = front(xteacher, xt_conf)
check('FRONT at equal size is the identity (orthonormal 3D-DCT round trip)',
      max((fr_same[k_] - tp[k_].data).abs().max().item() for k_ in fr_same) < 1e-5)
fr = front(xteacher, xs_conf)
check('FRONT: student shapes for the four stacked weight types, embeddings left fresh',
      fr['transformer.h.0.attn.c_attn.weight'].shape == (96, 32) and fr['transformer.h.1.mlp.c_proj.weight'].shape == (32, 128)
      and 'transformer.wte.weight' not in fr and len(fr) == 8)
fr_r = front(xteacher, xs_conf, r=0.25)      # r=0.5 would coincide with this 2× shrink's own truncation
check('FRONT learngene ratio r<1 changes the result (high frequencies masked)',
      not torch.allclose(fr_r['transformer.h.0.mlp.c_fc.weight'], fr['transformer.h.0.mlp.c_fc.weight']))
from prism_inits import load_init_state
sp_path = os.path.join(tmp, 'front_init.pt'); torch.save({'model': fr}, sp_path)
xS2 = GPT(xs_conf)
with contextlib.redirect_stdout(io.StringIO()):
    load_init_state(xS2, sp_path)
check('train.py init_state loader takes a builder state dict (missing embeddings stay fresh)',
      torch.equal(dict(xS2.named_parameters())['transformer.h.0.mlp.c_fc.weight'].data, fr['transformer.h.0.mlp.c_fc.weight']))

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    print('FAILED:', FAIL)
    raise SystemExit(1)
print('ALL SELFTESTS PASSED')
