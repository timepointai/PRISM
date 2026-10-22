"""
prism_modal_alpha_eval.py — the alpha-moves runner for the from-scratch harnesses
(W2a controls, sweeps, W3 cases): runs ANY `src/` driver on an L4 with its own
Volume slot, on the `exp/alpha-w2a` branch (override with --branch).

    modal run --detach prism_modal_alpha_eval.py --slot a --script prism_sweep.py \
        --flags "--corpus=modernweb --seed=1336 --arm=baseline --grid 'lr=5e-4,1e-3,2e-3 init_std=0.01,0.02' ..."
    modal run --detach prism_modal_alpha_eval.py --slot b --script prism_eval.py \
        --flags "--corpus=enwik8 --method=dirs_only --seeds=1337,1338,1339 ..."

Slots share nothing (one Volume each: prism-alpha-ev-<slot>), so runs on the same
corpus in the same slot share the teacher cache (a cache hit per seed) while
runs in different slots never fight over a lock or a data slot. Fetch + COMMIT:

    modal volume get prism-alpha-ev-<slot> nanogpt-prism/results/<file>.json ./results/
"""
import shlex

import modal

app = modal.App("prism-alpha-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers", "tiktoken", "datasets", "scipy")
)

WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/PRISM.git"
SLOTS = ["a", "b", "c", "d", "e", "f"]
VOLS = {s: modal.Volume.from_name(f"prism-alpha-ev-{s}", create_if_missing=True) for s in SLOTS}
SEC_PER_STEP = 0.27          # measured: 10.65M model, batch 64×256, dense eval, L4
L4_USD_PER_H = 0.80


def _run(vol, branch: str, script: str, flags: str):
    import os
    import subprocess
    import sys
    import time

    vol.reload()
    if not os.path.exists(REPO):
        subprocess.run(["git", "clone", "-b", branch, REPO_URL, REPO], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "remote", "set-url", "origin", REPO_URL], check=False)
        subprocess.run(["git", "-C", REPO, "fetch", "origin", "--quiet"], check=False)
        subprocess.run(["git", "-C", REPO, "checkout", "-q", branch], check=False)
        subprocess.run(["git", "-C", REPO, "reset", "--hard", f"origin/{branch}"], check=False)
    print("commit:", subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(), flush=True)
    vol.commit()
    for part in flags.split("&&"):             # several sequential invocations per launch
        cmd = [sys.executable, "-u", script] + shlex.split(part.strip())
        print(">>>", " ".join(cmd[2:]), flush=True)
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
            raise RuntimeError(f"{script} exited {rc}. Resume state is on the Volume — re-run "
                               f"the same command to continue.")


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["a"]}, timeout=24 * 3600)
def run_slot_a(branch: str, script: str, flags: str):
    _run(VOLS["a"], branch, script, flags)


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["b"]}, timeout=24 * 3600)
def run_slot_b(branch: str, script: str, flags: str):
    _run(VOLS["b"], branch, script, flags)


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["c"]}, timeout=24 * 3600)
def run_slot_c(branch: str, script: str, flags: str):
    _run(VOLS["c"], branch, script, flags)


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["d"]}, timeout=24 * 3600)
def run_slot_d(branch: str, script: str, flags: str):
    _run(VOLS["d"], branch, script, flags)


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["e"]}, timeout=24 * 3600)
def run_slot_e(branch: str, script: str, flags: str):
    _run(VOLS["e"], branch, script, flags)


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["f"]}, timeout=24 * 3600)
def run_slot_f(branch: str, script: str, flags: str):
    _run(VOLS["f"], branch, script, flags)


FNS = {"a": run_slot_a, "b": run_slot_b, "c": run_slot_c, "d": run_slot_d, "e": run_slot_e, "f": run_slot_f}


@app.local_entrypoint()
def main(slot: str = "a", script: str = "prism_eval.py", flags: str = "", branch: str = "exp/alpha-w2a",
         gpu: str = "L4", est_steps: int = 0, dry: bool = False):
    assert slot in SLOTS, f"--slot must be one of {SLOTS}"
    assert flags, "--flags is required"
    if est_steps:
        h = est_steps * SEC_PER_STEP / 3600
        print(f"estimate: {est_steps:,} train steps ≈ {h:.1f} L4-h ≈ ${h * L4_USD_PER_H:.2f}")
    print(f"slot={slot} branch={branch} script={script}\nflags: {flags}")
    if dry:
        return
    call = FNS[slot].with_options(gpu=gpu).spawn(branch, script, flags)
    print(f"launched (detached). call id: {call.object_id}")
    print("watch: modal app list  →  modal app logs <ap-...>")
    print(f"artifacts land on Volume prism-alpha-ev-{slot} under nanogpt-prism/results/")
