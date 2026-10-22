"""
prism_modal_rl.py — the W4 RL-anchor pilot runner: TRL GRPO on GSM8K, one arm per
detached GPU call, on the `exp/alpha-w4` branch.

    modal run --detach prism_modal_rl.py --arm beta0 --regime onpolicy
    modal run --detach prism_modal_rl.py --arm kl:0.04 --regime recycle:4
    modal run --detach prism_modal_rl.py --arm anchor:0.0025 --regime stale:4 --zero-rl

Each call: A100-40GB ($2.10/h on Modal, 2026-10-21), ~300 GRPO steps of
Qwen2.5-0.5B-Instruct (full finetuning, 32 completions/step, 256-token
completions), GSM8K accuracy before/after on 300 test problems, lm-eval
retention (hellaswag / arc_easy / winogrande, 300 each), KL-to-init, drift
metrics. Rough cost per arm: ~1.3–2 GPU-h ≈ $3–4 (printed before spawn; the
first run's measured step time re-prices the rest). W4's total pilot cap is $100
(brief §5 W4): keep the arm list within it.

Artifacts land on Volume prism-alpha-rl under nanogpt-prism/results/:
    modal volume get prism-alpha-rl nanogpt-prism/results/rl_<name>_<stamp>.json ./results/
"""
import modal

app = modal.App("prism-alpha-rl")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers>=5.8", "trl==1.16.0", "datasets", "accelerate",
                 "peft", "lm_eval==0.4.13", "scipy")
)

vol = modal.Volume.from_name("prism-alpha-rl", create_if_missing=True)
WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/PRISM.git"
BRANCH = "exp/alpha-w4"
GPU = "A100-40GB"
USD_PER_H = 2.10
EST_SEC_PER_STEP = 18.0      # placeholder until the first run reports step_ms_mean


@app.function(image=image, gpu=GPU, volumes={WORK: vol}, timeout=12 * 3600,
              secrets=[modal.Secret.from_dict({"HF_HOME": f"{WORK}/hf"})])
def run_arm(flags: str):
    import os
    import subprocess
    import sys
    import time

    vol.reload()
    if not os.path.exists(REPO):
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "fetch", "origin", "--quiet"], check=False)
        subprocess.run(["git", "-C", REPO, "checkout", "-q", BRANCH], check=False)
        subprocess.run(["git", "-C", REPO, "reset", "--hard", f"origin/{BRANCH}"], check=False)
    print("commit:", subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(), flush=True)
    vol.commit()
    cmd = [sys.executable, "-u", "prism_rl_pilot.py"] + flags.split()
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
        raise RuntimeError(f"prism_rl_pilot.py exited {rc}")


@app.local_entrypoint()
def main(arm: str = "beta0", regime: str = "onpolicy", steps: int = 300, seed: int = 1337,
         tag: str = "w4", zero_rl: bool = False, extra: str = "", dry: bool = False):
    flags = f"--arm={arm} --regime={regime} --steps={steps} --seed={seed} --tag={tag}"
    if zero_rl:
        flags += " --zero_rl"
    if extra:
        flags += f" {extra}"
    h = (steps * EST_SEC_PER_STEP + 1500) / 3600          # + evals/retention/KL
    print(f"arm={arm} regime={regime} steps={steps} → ~{h:.1f} {GPU}-h ≈ ${h * USD_PER_H:.2f} "
          f"(EST_SEC_PER_STEP={EST_SEC_PER_STEP})\nflags: {flags}")
    if dry:
        return
    call = run_arm.spawn(flags)
    print(f"launched (detached). call id: {call.object_id}")
    print("artifact lands on Volume prism-alpha-rl under nanogpt-prism/results/")
