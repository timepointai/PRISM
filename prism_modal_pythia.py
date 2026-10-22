"""
prism_modal_pythia.py — the gated Pythia ladder (W2b), on the `exp/alpha-w2b`
branch, one Volume (`prism-alpha-pythia`) shared by every gate so the fetched
Pile prefix, the held-out window, fingerprints and the published-curve cache are
paid for once.

    modal run prism_modal_pythia.py --gate dry                 # Tier A: 20 real iterations of 70M on an L4 + artifact check
    modal run --detach prism_modal_pythia.py --gate g2         # Tier B (~$100): 70M, 1 seed, baseline + prism, 8,000 iters
    modal run --detach prism_modal_pythia.py --gate g3         # Tier B (~$500): 160M, 3 seeds (seed1..3 published inits)
    modal run --detach prism_modal_pythia.py --gate g4         # Tier C (~$1.2k): 410M, 2 seeds, + the control arm

Every launch prints the cost table and the exact flags before spawning. The
data for a gate is fetched by a CPU-only `prepare` call first (so concurrent
arm containers never race on the same shard file), then one detached GPU call
per (seed, arm). Gates G2+ launch ONLY with AUTHORIZE_TIER_B / Tier C per
docs/ALPHA-MOVES-BRIEF.md §6 — this file never decides that for you.

Cost model (estimates, stated so they can be audited):
  FLOPs/token = 6·N_eff + 12·L·d·T   (N_eff = non-embedding params + embed_out; T=2048)
  tokens/iter = 1024 × 2048 = 2,097,152
  H100 SXM bf16 dense 989 TFLOP/s at an assumed 30% MFU (small models) → 297 TFLOP/s
  Modal H100 $3.95/h (modal.com/pricing, 2026-10-21); A100-80 $2.50/h at 312 TFLOP/s
  is ~2× worse per FLOP, so H100 is the default GPU for G2–G4.

                 FLOPs/iter   s/iter   $/iter    gate plan                                      iters total   est. GPU $
  70M  (G2)      7.21e14      2.43     0.0027    1 seed × {baseline, prism} × 8,000             16,000        ~$43  (+ data, published-curve evals, lm-eval → ~$60–75)
  160M (G3)      2.03e15      6.84     0.0075    3 seeds × {baseline, prism} × 8,000            48,000        ~$360 (→ ~$420–500 with overheads)
  410M (G4)      5.72e15     19.3      0.0211    2 seeds × {baseline, prism, control} × 8,000   48,000        ~$1,010 (→ ~$1.2k)

  8,000 iterations = 16.8B tokens = 5.6% of Pythia's 300B run, well past the
  1,430-iteration warmup, at the published schedule (not rescaled). The same
  horizon at every size keeps the ladder's ratios comparable (G4's rule: the
  410M ratio must hold ≥80% of the 70M ratio).

Fetch + COMMIT artifacts:
    modal volume get prism-alpha-pythia nanogpt-prism/results/pythia_<stamp>.json ./results/
"""
import modal

app = modal.App("prism-alpha-pythia")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "scipy", "transformers", "datasets", "tiktoken",
                 "huggingface_hub", "safetensors", "accelerate", "lm_eval==0.4.13")
)

vol = modal.Volume.from_name("prism-alpha-pythia", create_if_missing=True)
WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/PRISM.git"
BRANCH = "exp/alpha-w2b"
DATA = f"{WORK}/pile"                      # shared Pile prefix / holdout / fingerprints / published-curve cache

GATES = {
    "dry": dict(size="70m", iters=20, eval_every=10, eval_seqs=64, seeds=[None], arms=["baseline", "prism"],
                gpu="L4", usd_per_h=0.80, published_curve=False, lm_eval=False, tier="A"),
    "g2":  dict(size="70m", iters=8000, eval_every=100, eval_seqs=512, seeds=[None], arms=["baseline", "prism"],
                gpu="H100", usd_per_h=3.95, published_curve=True, lm_eval=True, tier="B (~$100)"),
    "g3":  dict(size="160m", iters=8000, eval_every=100, eval_seqs=512,
                seeds=["EleutherAI/pythia-160m-seed1", "EleutherAI/pythia-160m-seed2", "EleutherAI/pythia-160m-seed3"],
                arms=["baseline", "prism"], gpu="H100", usd_per_h=3.95, published_curve=True, lm_eval=True, tier="B (~$500)"),
    "g4":  dict(size="410m", iters=8000, eval_every=100, eval_seqs=512,
                seeds=["EleutherAI/pythia-410m-seed1", "EleutherAI/pythia-410m-seed2"],
                arms=["baseline", "prism", "control"], gpu="H100", usd_per_h=3.95, published_curve=True, lm_eval=True, tier="C (~$1.2k)"),
}
# N_eff (non-embedding + embed_out), L, d
MODEL = {"70m": (44_671_000, 6, 512), "160m": (123_689_000, 12, 768), "410m": (353_822_000, 24, 1024)}
MFU, PEAK_H100, PEAK_L4 = 0.30, 989e12, 121e12


def cost_table(gate):
    g = GATES[gate]
    n_eff, L, d = MODEL[g["size"]]
    flops_tok = 6 * n_eff + 12 * L * d * 2048
    flops_iter = flops_tok * 1024 * 2048
    peak = PEAK_H100 if g["gpu"].startswith("H100") else PEAK_L4
    s_iter = flops_iter / (peak * MFU)
    runs = len(g["seeds"]) * len(g["arms"])
    iters = runs * g["iters"]
    usd = iters * s_iter / 3600 * g["usd_per_h"]
    return {"flops_per_iter": flops_iter, "s_per_iter": s_iter, "usd_per_iter": s_iter / 3600 * g["usd_per_h"],
            "runs": runs, "iters_total": iters, "gpu_usd": usd, "gpu": g["gpu"], "tier": g["tier"]}


def _sync_repo():
    import os
    import subprocess
    vol.reload()
    if not os.path.exists(REPO):
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "fetch", "origin", "--quiet"], check=False)
        subprocess.run(["git", "-C", REPO, "reset", "--hard", f"origin/{BRANCH}"], check=False)
    print("commit:", subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(), flush=True)
    vol.commit()


@app.function(image=image, volumes={WORK: vol}, timeout=6 * 3600, cpu=4, memory=16384)
def prepare(size: str, iters: int, eval_seqs: int):
    """CPU-only: fetch the exact Pile byte ranges and the published checkpoints this gate needs."""
    import os
    import subprocess
    import sys
    _sync_repo()
    sys.path.insert(0, f"{REPO}/src")
    os.chdir(f"{REPO}/src")
    import pythia_train as pt
    import pile_mmap as pm
    train, hold = pt.prepare_real_data(DATA, iters, pt.HOLDOUT_ITER, eval_seqs)
    print(f"train stream {train.seq_range}, holdout {hold.seq_range}", flush=True)
    vol.commit()


def _train(flags: list):
    import os
    import subprocess
    import sys
    import time
    _sync_repo()
    cmd = [sys.executable, "-u", "pythia_train.py", f"--data_dir={DATA}"] + flags
    proc = subprocess.Popen(cmd, cwd=f"{REPO}/src", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = time.time()
    for line in proc.stdout:
        print(line, end="", flush=True)
        if time.time() - last > 60:
            vol.commit()
            last = time.time()
    rc = proc.wait()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"pythia_train.py exited {rc}; resume state is on the Volume — re-run to continue.")


@app.function(image=image, gpu="L4", volumes={WORK: vol}, timeout=2 * 3600)
def train_l4(flags: list):
    _train(flags)


@app.function(image=image, gpu="H100", volumes={WORK: vol}, timeout=24 * 3600)
def train_h100(flags: list):
    _train(flags)


@app.local_entrypoint()
def main(gate: str = "dry", dry: bool = False, only_arm: str = "", only_seed: str = ""):
    assert gate in GATES, f"--gate must be one of {list(GATES)}"
    g, c = GATES[gate], cost_table(gate)
    print(f"gate {gate}: {g['size']} × {len(g['seeds'])} seed(s) × {g['arms']} × {g['iters']} iters on {c['gpu']} "
          f"[tier {c['tier']}]")
    print(f"  {c['flops_per_iter']:.2e} FLOPs/iter → {c['s_per_iter']:.2f} s/iter at {MFU:.0%} MFU → "
          f"${c['usd_per_iter']:.4f}/iter; {c['runs']} runs × {g['iters']} = {c['iters_total']:,} iters ≈ "
          f"${c['gpu_usd']:.0f} GPU (+ data fetch, published-curve evals, lm-eval)")
    if gate != "dry":
        print("  >>> Tier B/C: launch only with the matching authorization in docs/ALPHA-MOVES-BRIEF.md §0 <<<")
    if dry:
        return
    print("preparing data (CPU, blocking) …")
    prepare.remote(g["size"], g["iters"], g["eval_seqs"])
    fn = train_l4 if g["gpu"] == "L4" else train_h100
    for seed_repo in g["seeds"]:
        if only_seed and (seed_repo or "") != only_seed:
            continue
        for arm in g["arms"]:
            if only_arm and arm != only_arm:
                continue
            tag = gate + (f"_{seed_repo.rsplit('-', 1)[-1]}" if seed_repo else "")
            flags = [f"--size={g['size']}", f"--arm={arm}", f"--iters={g['iters']}",
                     f"--eval_every={g['eval_every']}", f"--eval_seqs={g['eval_seqs']}", f"--tag={tag}"]
            if g["gpu"] == "L4":
                # 22 GB: the fp32 logits of a 32×2048×50304 micro-batch alone are 12 GB → OOM (dry run, 2026-10-22)
                flags += ["--micro_bs=4", "--eval_micro=4"]
            if seed_repo:
                flags.append(f"--student_repo={seed_repo}")
            if g["published_curve"] and arm == "baseline":
                flags.append("--published_curve")
            if g["lm_eval"]:
                flags.append("--lm_eval")
            if arm == "control":
                flags.append(f"--control_init={DATA}/control_init_{g['size']}.pt")   # built by W2a's strongest control
            call = fn.spawn(flags)
            print(f"launched {arm} {seed_repo or ''}: call {call.object_id}  flags {' '.join(flags)}")
    print("watch: modal app list → modal app logs <ap-...>; artifacts land on prism-alpha-pythia under nanogpt-prism/results/")
