"""
prism_modal_alpha_w5.py — the W5 runner: prior fusion's COUNT control / the Act 2
composition gate on enwik8, on the `exp/alpha-w5` branch. One isolated Volume
(`prism-alpha-w5`), one detached L4 job per launch. Prints a cost estimate first.

    modal run --detach prism_modal_alpha_w5.py --extra "<prism_prior_eval.py flags>"

The pre-registered W5 launch (docs/alpha/W5-NOTES.md):
    modal run --detach prism_modal_alpha_w5.py --extra \\
     "--corpus=enwik8 --sparse_context=6 --teacher_steps=2000 --student_steps=1500 \\
      --eval_every=10 --eval_iters=50 --batch_size=32 --block_size=256 --seeds=1337,1338,1339 \\
      --arms=baseline,dirs_only,prior,dirs_prior,baseline_inffuse,dirs_inffuse,count,count_dirs --tag=w5"

Fetch + COMMIT the artifact:
    modal volume get prism-alpha-w5 nanogpt-prism/results/prior_<stamp>.json ./results/
"""
import modal

app = modal.App("prism-alpha-w5")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers", "tiktoken", "datasets", "scipy")
)

vol = modal.Volume.from_name("prism-alpha-w5", create_if_missing=True)
WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/PRISM.git"
BRANCH = "exp/alpha-w5"

# L4 timings: dirs_only enwik8 (results/dirs_only_20260728T054705Z.json) ≈ 0.27 s/step
# incl. dense eval every 10; the sparse-prior gather/fusion overhead was measured on CPU
# as a per-step ratio (see docs/alpha/W5-NOTES.md) and is applied to the prior/count arms.
SEC_PER_STEP = 0.27
PRIOR_OVERHEAD = {'baseline': 1.0, 'dirs_only': 1.0, 'prior': 1.25, 'dirs_prior': 1.25,
                  'baseline_inffuse': 1.05, 'dirs_inffuse': 1.05, 'count': 1.4, 'count_dirs': 1.4,
                  'prism': 1.0, 'prism_prior': 1.25}
L4_USD_PER_H = 0.80
# 24 GB L4; the 10.65M model + optimizer + activations take < 2 GB at batch 32 × 256.
GPU_BUDGET_GB = 24.0


@app.function(image=image, gpu="L4", volumes={WORK: vol}, timeout=24 * 3600, memory=32768)
def run_eval(extra: str = ""):
    import os
    import subprocess
    import sys
    import time

    vol.reload()
    if not os.path.exists(REPO):
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "remote", "set-url", "origin", REPO_URL], check=False)
        subprocess.run(["git", "-C", REPO, "fetch", "origin", "--quiet"], check=False)
        subprocess.run(["git", "-C", REPO, "reset", "--hard", f"origin/{BRANCH}"], check=False)
    print(f"on branch {BRANCH}:",
          subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip(), flush=True)
    vol.commit()
    # enwik8 is downloaded at prepare time (not redistributed); prism_eval.setup runs prepare.py
    cmd = [sys.executable, "-u", "prism_prior_eval.py"] + (extra.split() if extra else [])
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
        raise RuntimeError(f"prism_prior_eval.py exited {rc}. Resume state on the Volume "
                           f"— re-run to continue.")


def _estimate(extra: str):
    kv = {}
    for tok in extra.split():
        if tok.startswith('--') and '=' in tok:
            k, v = tok[2:].split('=', 1)
            kv[k] = v
    seeds = len([s for s in kv.get('seeds', '1337').split(',') if s])
    arms = [a for a in kv.get('arms', 'baseline,prism,prior,prism_prior').split(',') if a]
    teacher = int(kv.get('teacher_steps', 2000)) if any(a not in ('baseline', 'prior', 'baseline_inffuse', 'count') for a in arms) else 0
    student = int(kv.get('student_steps', 1500))
    steps = seeds * (teacher + sum(student * PRIOR_OVERHEAD.get(a, 1.2) for a in arms))
    hours = steps * SEC_PER_STEP / 3600 + 0.15   # + prior build (counting is CPU, minutes)
    return seeds, arms, hours


@app.local_entrypoint()
def main(gpu: str = "L4", extra: str = "", dry: bool = False):
    seeds, arms, hours = _estimate(extra)
    print(f"seeds={seeds} arms={arms}\n→ ~{hours:.1f} L4-h ≈ ${hours * L4_USD_PER_H:.2f} "
          f"(SEC_PER_STEP={SEC_PER_STEP}, prior/count overhead ratios {PRIOR_OVERHEAD})")
    if dry:
        return
    call = run_eval.with_options(gpu=gpu).spawn(extra=extra)
    print(f"\nLaunched W5 run (detached). call id: {call.object_id}")
    print("Watch: modal app list  →  modal app logs <ap-id>")
    print("Fetch when done: modal volume get prism-alpha-w5 "
          "nanogpt-prism/results/prior_latest.json ./results/")
