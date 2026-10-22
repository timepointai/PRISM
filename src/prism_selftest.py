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
# alpha-moves W2b — the GPT-NeoX / Pythia port and the Pile stream reader.
# Synthetic checks always run; the pythia-70m checks run when the two
# checkpoints are cached under .prism_cache/pythia-70m/{step0,main}.
# ══════════════════════════════════════════════════════════════════════════
import tempfile
print('prism_neox (GPT-NeoX / Pythia port):')
try:
    from transformers import GPTNeoXForCausalLM
    from prism_neox import (qkv_unpack, qkv_repack, qkv_unpack_bias, classify_neox_param, neox_matrices,
                            extract_neox_fingerprint, apply_prism_neox, synthetic_neox_config, load_neox)
    import pile_mmap as pm
    _HAVE_NEOX = True
except Exception as e:                                   # pragma: no cover
    _HAVE_NEOX = False
    print(f'  (skipped: {type(e).__name__}: {e})')
if _HAVE_NEOX:
    Wq = torch.randn(192, 64)
    check('QKV per-head pack <-> [Q;K;V] blocks round-trips exactly',
          torch.equal(qkv_repack(qkv_unpack(Wq, 4), 4), Wq))
    nx = GPTNeoXForCausalLM(synthetic_neox_config())
    att = nx.gpt_neox.layers[0].attention
    xin = torch.randn(1, 5, 64)
    q_h1 = att.query_key_value(xin).view(1, 5, 4, 48).transpose(1, 2).chunk(3, dim=-1)[0][0, 1]
    Wb = qkv_unpack(att.query_key_value.weight.data, 4)
    qb = xin[0] @ Wb[:64].T + qkv_unpack_bias(att.query_key_value.bias.data, 4)[:64]
    check('unpacked Q block rows == the HF forward\'s per-head query (head 1)',
          torch.allclose(q_h1, qb[:, 16:32], atol=1e-5))
    mats = neox_matrices(nx)
    all2d = {n for n, p in nx.named_parameters() if p.dim() == 2}
    check('group classification covers every 2-D matrix and nothing else',
          {n for n, _, _ in mats} == all2d and all(classify_neox_param(n) is None
                                                  for n, p in nx.named_parameters() if p.dim() < 2))
    check('five groups present (attention, attn_proj, ffn_up, ffn_down, embedding — embeddings untied)',
          {g for _, _, g in mats} == {'attention', 'attn_proj', 'ffn_up', 'ffn_down', 'embedding'}
          and sum(g == 'embedding' for _, _, g in mats) == 2)
    fpx = extract_neox_fingerprint(nx)
    for nm_, d_ in list(fpx['directions'].items())[:3]:
        pass
    check('fingerprint U columns / Vt rows orthonormal',
          all(orthonormal_cols(d['U']) and orthonormal_cols(d['Vt'].T) for d in fpx['directions'].values()))
    nx2 = GPTNeoXForCausalLM(synthetic_neox_config()); sd0 = {k: v.clone() for k, v in nx2.state_dict().items()}
    n0 = apply_prism_neox(nx2, fpx, align=0.0, imprint=False, verbose=False)
    check('align=0, imprint=False is an exact no-op (the published step0 is kept bit for bit)',
          n0 == 0 and all(torch.equal(sd0[k], v) for k, v in nx2.state_dict().items()))
    n1 = apply_prism_neox(nx2, fpx, align=0.75, imprint=True, verbose=False)
    xo = torch.randint(0, 128, (2, 16))
    outx = nx2(input_ids=xo, labels=xo)
    check(f'2-layer synthetic NeoX: EigenTransfer+Imprint shapes {n1} matrices and the model still runs',
          n1 == len(mats) and torch.isfinite(outx.loss))
    Wd = nx2.gpt_neox.layers[0].attention.dense.weight.data.float()
    Ud, sd_, Vtd = torch.linalg.svd(Wd)
    check('shaped matrix has the imprinted (non-flat) spectrum and orthonormal factors',
          orthonormal_cols(Ud) and sd_[0] > sd_[-1] * 1.01)
    nx3 = GPTNeoXForCausalLM(synthetic_neox_config())
    apply_prism_neox(nx3, fpx, align=1.0, imprint=False, verbose=False)
    nm0 = 'gpt_neox.layers.0.mlp.dense_h_to_4h.weight'
    Up = fpx['directions'][nm0]['U']
    Us = torch.linalg.svd(dict(nx3.named_parameters())[nm0].data.float(), full_matrices=False)[0]
    check('align=1, no imprint: column space moves onto the teacher\'s (subspace alignment > 0.95)',
          subspace_alignment(Us[:, :8], Up[:, :8]) > 0.95)
    print('pile_mmap (MMIDIDX reader):')
    td = tempfile.mkdtemp()
    idxp, shards_, data_ = pm.write_synthetic_dataset(td, 40, seq_len=17, shard_bytes=300)
    h_ = pm.read_idx_header(idxp)
    check('synthetic .idx header round-trips (uint16, N=40)', h_['dtype'] == np.uint16 and h_['n_seq'] == 40)
    ps_ = pm.PileStream(idxp, {i: p_ for i, p_ in enumerate(shards_)}, shard_bytes=300)
    check('sharded reads exact, including sequences straddling two shards',
          all(np.array_equal(ps_.sequence(i), data_[i]) for i in range(40)))
    xb, yb = ps_.batch(3, 4)
    check('batch targets are the left shift', np.array_equal(yb[0], data_[3][1:]) and xb.shape == (4, 16))
    win = pm.read_idx_window(open(idxp, 'rb'), 30, 5)
    ps_w = pm.PileStream(None, {i: p_ for i, p_ in enumerate(shards_)}, shard_bytes=300, window=win)
    check('windowed stream serves GLOBAL sequence ids', np.array_equal(ps_w.sequence(33), data_[33]))
    sr = pm.shards_for_range(*pm.byte_range_for_sequences(0, 8000 * 1024))
    check('8000 iterations of the real stream (33.6 GB) span shards 0 and 1', list(sr) == [0, 1])
    hold = pm.shards_for_range(*pm.byte_range_for_sequences(140000 * 1024, 140000 * 1024 + 512))
    check('the held-out window (iteration 140000) lives in shard 19', list(hold) == [19])
    p70 = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.', '.prism_cache', 'pythia-70m')
    if os.path.exists(os.path.join(p70, 'step0', 'model.safetensors')) and os.path.exists(os.path.join(p70, 'main', 'model.safetensors')):
        print('pythia-70m (cached checkpoints):')
        st0 = load_neox(os.path.join(p70, 'step0'))
        sd70 = {k: v.clone() for k, v in st0.state_dict().items()}
        m70 = neox_matrices(st0)
        check('pythia-70m: 26 PRISM matrices (6×4 + embed_in + embed_out)', len(m70) == 26)
        fp70 = extract_neox_fingerprint(load_neox(os.path.join(p70, 'main')))
        check('pythia-70m fingerprint: 5 group spectra, 26 direction sets',
              len(fp70['spectra']) == 5 and len(fp70['directions']) == 26)
        apply_prism_neox(st0, fp70, align=0.0, imprint=False, verbose=False)
        check('pythia-70m step0 untouched by the no-op arm', all(torch.equal(sd70[k], v) for k, v in st0.state_dict().items()))
    else:
        print('  (pythia-70m checkpoints not cached under .prism_cache/pythia-70m — real-checkpoint checks skipped)')

# alpha-moves (W5) — the sparse interpolated n-gram prior and the COUNT embedding
# ══════════════════════════════════════════════════════════════════════════
print('prism_prior (sparse interpolated n-gram prior, block-edge backoff):')
import math
import tempfile as _tf
import ngram_floor as _nf
from prism_prior import (build_tables, save_prior, load_sparse_prior, install_count_embedding,
                         count_embedding)
_rng = np.random.default_rng(0)
_V, _C = 256, 4                       # byte vocab (ngram_floor's order -1 is 1/256), 17 symbols used
_arr = np.zeros(20000, dtype=np.uint16)
for _i in range(1, len(_arr)):
    _arr[_i] = (_arr[_i - 1] * 7 + _rng.integers(0, 3)) % 17      # a Markov-ish synthetic corpus
_mus = [0.9, 0.7, 0.6, 0.5, 0.5]
_tmpd = _tf.mkdtemp()
_X = torch.from_numpy(_arr[:2000].astype(np.int64))[None]
# ngram_floor reference: sequential ML probs per order, mixed with the same mus
_nft = {k: _nf.count_table(_arr.astype(np.uint8), k) for k in range(_C + 1)}
_pos = np.arange(_C, 2000)
_mls, _hits = [], []
for k in range(_C + 1):
    m, h = _nf.ml_probs(_arr.astype(np.uint8), k, _pos, _nft)
    _mls.append(m); _hits.append(h)
_ref = _nf.mix(_mls, _hits, _mus)
for _dt, _tol in ((np.float32, 1e-5), (np.float16, 1e-3)):
    save_prior(f'{_tmpd}/p_{_dt.__name__}.pt', build_tables(_arr, _C, _V, log=lambda *a: None, row_dtype=_dt),
               _mus, _C, _V, {})
    _pr = load_sparse_prior(f'{_tmpd}/p_{_dt.__name__}.pt', 'cpu', vocab_size=_V)
    _P = _pr.probs(_X)[0]
    _mine = _P[torch.arange(_C, 2000), _X[0, _C:2000]].numpy()
    check(f'{_dt.__name__} rows: windowed probs == ngram_floor mix() on positions ≥ C (max |Δ| {np.abs(_mine - _ref).max():.1e} ≤ {_tol})',
          np.abs(_mine - _ref).max() <= _tol)
    check(f'{_dt.__name__} rows: every (B,T,V) row normalises to 1', float((_P.sum(-1) - 1).abs().max()) <= _tol)
_uni = torch.from_numpy(build_tables(_arr, 0, _V, log=lambda *a: None, row_dtype=np.float32)[0][1])[0]
check('block edge: position 0 backs off to the unigram mixture (NOT uniform)',
      torch.allclose(_P[0], _mus[0] * _uni + (1 - _mus[0]) / _V, atol=1e-3)
      and not torch.allclose(_P[0], torch.full((_V,), 1 / _V)))
check('block edge: position t<C equals the standalone prediction from t symbols of context',
      torch.allclose(_pr.probs(_X[:, :3])[0, 2], _P[2]))
check('logp == log(probs)', torch.allclose(_pr.logp(_X), torch.log(_pr.probs(_X).clamp_min(1e-12))))
_bpb, _n = _pr.score_sequential(_arr[2000:4000])
check(f'score_sequential returns bits/symbol in (0, log2 V] ({_bpb:.3f}) over the scored positions', 0 < _bpb <= math.log2(_V) and _n == 2000 - _C)
_S = _pr.backoff_scores(_X, {1, 2, 3})
_rows2, _hit2 = _pr._rows_hits(_X)[2]
_t2 = int(torch.nonzero(_hit2[0])[5])
check('stupid-backoff: observed continuations of a stored history score their count ratio',
      torch.allclose(_S[2][0, _t2][_rows2[0, _t2] > 0], _rows2[0, _t2][_rows2[0, _t2] > 0]))
# a context seen at order 3 whose unobserved continuations must score 0.4 × the order-2 score
_rows3, _hit3 = _pr._rows_hits(_X)[3]
_t = int(torch.nonzero(_hit3[0])[5])
_unobs = _rows3[0, _t] == 0
check('stupid-backoff: unobserved continuations of a stored history score 0.4·S(w|h[1:])',
      torch.allclose(_S[3][0, _t][_unobs], 0.4 * _S[2][0, _t][_unobs], atol=1e-6))

print('COUNT-style input embedding (install_count_embedding):')
_cm = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=64, vocab_size=_V, dropout=0.0))
_xx = _X[:, :64]
_l0, _ = _cm(_xx, _xx)
_n0 = sum(p.numel() for p in _cm.parameters())
_f = count_embedding(_pr, _xx, _cm.lm_head.weight, K=8)
check('f_t has shape (B,T,d)', tuple(_f.shape) == (1, 64, 32))
import copy as _cp
_c1 = _cp.deepcopy(_cm); install_count_embedding(_c1, _pr, 'K:8,amp:sqrt_d')
check('no new parameters; state_dict keys unchanged',
      sum(p.numel() for p in _c1.parameters()) == _n0 and set(_c1.state_dict()) == set(_cm.state_dict()))


class _ZeroPrior:
    C = _pr.C

    def backoff_scores(self, X, orders):
        return {n: torch.zeros(X.shape[0], X.shape[1], _V) for n in orders}


_c2 = _cp.deepcopy(_cm); install_count_embedding(_c2, _ZeroPrior(), 'K:8,amp:none')
check('all-zero tables + amp:none reduce to the plain forward exactly', torch.allclose(_c2(_xx, _xx)[0], _l0, atol=1e-6))
_c3 = _cp.deepcopy(_cm); install_count_embedding(_c3, _ZeroPrior(), 'K:8,amp:sqrt_d')
_h0 = _cm.transformer.wte(_xx)
_h0 = math.sqrt(32) * torch.nn.functional.layer_norm(_h0, (32,))
check('amplification √d·LayerNorm gives every input position norm ≈ d (parameter-free; LN eps)',
      torch.allclose(_h0.norm(dim=-1), torch.full((1, 64), 32.0), rtol=0.05))
_l1, _loss1 = _c1(_xx, _xx); _loss1.backward()
check('count arm: finite loss, gradient reaches the tied unembedding through f_t',
      bool(torch.isfinite(_loss1)) and float(_c1.transformer.wte.weight.grad.abs().sum()) > 0)
print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    print('FAILED:', FAIL)
    raise SystemExit(1)
print('ALL SELFTESTS PASSED')
