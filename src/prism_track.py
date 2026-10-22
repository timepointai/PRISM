"""prism_track.py — online geometry tracking (`train.py --prism_track_overlap=N`).

Every N steps, for each weight group, log the mean squared cosine of the
principal angles between the weight's CURRENT top-k singular subspaces and two
references:

  (a) `init` — the subspaces the weight had when training started (the init);
  (b) `ref`  — the transplanted teacher directions (scratch path, from the
               directions file) or W₀ (resume path).

Both the left (U) and right (V) sides are measured and averaged, then averaged
over the matrices in a group. For two k-dim subspaces of R^d the metric is
mean_i cos²θ_i = ‖U_kᵀ U'_k‖_F² / k ∈ [0, 1]; identical → 1, orthogonal → 0,
and two random k-subspaces sit near k/d (reported as `chance`).

Log line (a NEW format — the existing eval-line regexes are untouched):

    [overlap] step 120 attention init=0.8123 ref=0.4411 chance=0.0417

`parse_overlap(stdout)` banks it as {step: {group: {init, ref, chance}}}.
"""
import math
import re

import torch

from prism_init import classify_nanogpt_param

_LINE = re.compile(r'\[overlap\] step (\d+) (\w+) init=([\d.]+|nan) ref=([\d.]+|nan) '
                   r'chance=([\d.]+)')


def subspace_overlap(A, B):
    """Mean squared cosine of principal angles between col-spaces of A, B
    (both column-orthonormal, same k)."""
    c = torch.linalg.svdvals(A.transpose(0, 1).float() @ B.float())
    return float((c ** 2).mean())


def topk_uv(W, k):
    U, _, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    kk = min(k, U.shape[1])
    return U[:, :kk].contiguous(), Vt[:kk].transpose(0, 1).contiguous()


class OverlapTracker:
    def __init__(self, model, k=16, ref_dirs=None, ref_weights=None, min_dim=2):
        """ref_dirs: {name: {'U': (out,r), 'Vt': (r,in)}} (a directions.pt dict);
        ref_weights: {name: W₀ tensor}. Either may be None."""
        self.k = k
        self.names, self.groups, self.init, self.ref, self.chance = [], {}, {}, {}, {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() < min_dim:
                    continue
                g = classify_nanogpt_param(name)
                if g is None:
                    continue
                self.names.append(name)
                self.groups[name] = g
                U, V = topk_uv(p.data, k)
                self.init[name] = (U.cpu(), V.cpu())
                kk = U.shape[1]
                self.chance[name] = 0.5 * (kk / p.shape[0] + kk / p.shape[1])
                if ref_dirs and name in ref_dirs:
                    d = ref_dirs[name]
                    Ur, Vtr = d['U'], d['Vt']
                    if Ur.shape[0] == p.shape[0] and Vtr.shape[1] == p.shape[1]:
                        self.ref[name] = (Ur[:, :kk].contiguous().cpu(),
                                         Vtr[:kk].transpose(0, 1).contiguous().cpu())
                elif ref_weights and name in ref_weights:
                    Ur, Vr = topk_uv(ref_weights[name], k)
                    self.ref[name] = (Ur.cpu(), Vr.cpu())

    @torch.no_grad()
    def measure(self, model, step, log=True):
        params = dict(model.named_parameters())
        per_group = {}
        for name in self.names:
            W = params[name].data
            U, V = topk_uv(W, self.k)
            Ui, Vi = self.init[name]
            o_init = 0.5 * (subspace_overlap(U.cpu(), Ui) + subspace_overlap(V.cpu(), Vi))
            o_ref = float('nan')
            if name in self.ref:
                Ur, Vr = self.ref[name]
                o_ref = 0.5 * (subspace_overlap(U.cpu(), Ur) + subspace_overlap(V.cpu(), Vr))
            per_group.setdefault(self.groups[name], []).append((o_init, o_ref, self.chance[name]))
        out = {}
        for g, vals in per_group.items():
            inits = [v[0] for v in vals]
            refs = [v[1] for v in vals if not math.isnan(v[1])]
            ch = [v[2] for v in vals]
            rec = {'init': sum(inits) / len(inits),
                   'ref': (sum(refs) / len(refs)) if refs else float('nan'),
                   'chance': sum(ch) / len(ch)}
            out[g] = rec
            if log:
                print(f"[overlap] step {step} {g} init={rec['init']:.4f} "
                      f"ref={rec['ref']:.4f} chance={rec['chance']:.4f}", flush=True)
        return out


def parse_overlap(stdout):
    """{step: {group: {'init', 'ref', 'chance'}}} from [overlap] lines."""
    out = {}
    for line in stdout.split('\n'):
        m = _LINE.search(line)
        if m:
            step, g = int(m.group(1)), m.group(2)
            out.setdefault(step, {})[g] = {'init': float(m.group(3)),
                                           'ref': float(m.group(4)),
                                           'chance': float(m.group(5))}
    return out


def overlap_for_artifact(parsed):
    """JSON-safe form (string keys; NaN -> None)."""
    def clean(v):
        return None if (isinstance(v, float) and math.isnan(v)) else v
    return {str(s): {g: {k: clean(v) for k, v in rec.items()} for g, rec in gs.items()}
            for s, gs in sorted(parsed.items())}
