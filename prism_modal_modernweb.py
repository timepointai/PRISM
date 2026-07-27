"""
prism_modal_modernweb.py — the modern-web probe: PRISM on byte-level FineWeb-Edu
(2024 web text), on the `exp/modernweb` branch. Three arms, three ISOLATED
Volumes, launched as three detached L4 jobs in parallel:

    modal run --detach prism_modal_modernweb.py --arm recipe   # full PRISM, native teacher
    modal run --detach prism_modal_modernweb.py --arm spec     # spectrum-only, native teacher
    modal run --detach prism_modal_modernweb.py --arm gpt2     # spectrum-only, GPT-2's fingerprint

Identical probe rig (3 seeds, matched schedules, dense eval). What each arm
proves against its own in-run baseline:
  recipe — PRISM's full transfer works on modern web text (not just Shakespeare);
  spec   — how much the SPECTRUM alone carries (the never-committed ablation
           that gates the gpt2 arm's attribution);
  gpt2   — 40 numbers read off OpenAI's public GPT-2 weights, transplanted into
           a fresh byte-level model, no teacher trained at all. The universal-
           init claim, at probe cost (this arm trains no teacher — cheapest).

Fetch the artifacts when done, and COMMIT them:
    modal volume ls prism-eval-mw-<arm> nanogpt-prism/results
    modal volume get prism-eval-mw-<arm> nanogpt-prism/results/<*.json> ./results/
"""
import modal

app = modal.App("prism-eval-modernweb")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers", "tiktoken", "datasets")
)

vols = {
    "recipe": modal.Volume.from_name("prism-eval-mw-recipe", create_if_missing=True),
    "spec": modal.Volume.from_name("prism-eval-mw-spec", create_if_missing=True),
    "gpt2": modal.Volume.from_name("prism-eval-mw-gpt2", create_if_missing=True),
}
WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/PRISM.git"
BRANCH = "exp/modernweb"

# Shared probe rig — matched schedules, dense eval, converged native teachers.
PROBE = ("--corpus=modernweb --seeds=1337,1338,1339 --teacher_steps=2000 "
         "--student_steps=1500 --eval_every=10 --eval_iters=50 "
         "--method_lr=1e-3 --method_warmup=100")
ARM_FLAGS = {
    "recipe": "--method=recipe",
    "spec": "--method=spectral_only",
    "gpt2": "--method=spectral_only --fingerprint=../fingerprints/gpt2-124M",
}


def _run(vol, extra):
    import os
    import subprocess
    import sys
    import time

    vol.reload()
    if not os.path.exists(REPO):
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "remote", "set-url", "origin", REPO_URL],
                       check=False)
        subprocess.run(["git", "-C", REPO, "fetch", "origin", "--quiet"], check=False)
        subprocess.run(["git", "-C", REPO, "reset", "--hard", f"origin/{BRANCH}"],
                       check=False)
    print("commit:",
          subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip(), flush=True)
    vol.commit()

    cmd = [sys.executable, "-u", "prism_eval.py"] + extra.split()
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
        raise RuntimeError(f"prism_eval.py exited {rc}. Resume state is on the "
                           f"Volume — re-run the same arm to continue.")


@app.function(image=image, gpu="L4", volumes={WORK: vols["recipe"]}, timeout=24 * 3600)
def run_recipe(extra: str):
    _run(vols["recipe"], extra)


@app.function(image=image, gpu="L4", volumes={WORK: vols["spec"]}, timeout=24 * 3600)
def run_spec(extra: str):
    _run(vols["spec"], extra)


@app.function(image=image, gpu="L4", volumes={WORK: vols["gpt2"]}, timeout=24 * 3600)
def run_gpt2(extra: str):
    _run(vols["gpt2"], extra)


@app.local_entrypoint()
def main(arm: str = "recipe", gpu: str = "L4", extra: str = ""):
    if arm not in ARM_FLAGS:
        raise SystemExit(f"--arm must be one of {list(ARM_FLAGS)}")
    flags = f"{PROBE} {ARM_FLAGS[arm]}" + (f" {extra}" if extra else "")
    fn = {"recipe": run_recipe, "spec": run_spec, "gpt2": run_gpt2}[arm]
    call = fn.with_options(gpu=gpu).spawn(extra=flags)
    print(f"launched {arm} arm (detached). call id: {call.object_id}")
    print("watch: modal app list  →  modal app logs <ap-...>")
    print(f"artifact lands on Volume prism-eval-mw-{arm} under nanogpt-prism/results/")
