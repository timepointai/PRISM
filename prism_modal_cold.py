"""
prism_modal_cold.py — the cold-directions probe — never-trained DCT bases + GPT-2 spectrum, `exp/cold-directions`
branch. One isolated Volume, one detached L4 job:

    modal run --detach prism_modal_cold.py

The question (from the ReLoRA/fact-injection thread): when you inject NOVEL
facts into a trained modern-web base, what does the directional anchor buy vs
the low-LR alternative — on all three axes at once: fact recall (closed-book
exact match, seen AND held-out phrasing), retention (modern-web val climb),
and the facts-val loss curve. Arms share one base per seed and differ by one
flag each: plain (control), raw anchor 0.01 / 0.02, low-LR 1e-4.

Fetch + COMMIT the artifact:
    modal volume get prism-eval-cold nanogpt-prism/results/finetune_<stamp>.json ./results/
"""
import modal

app = modal.App("prism-eval-cold")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers", "tiktoken", "datasets")
)

vol = modal.Volume.from_name("prism-eval-cold", create_if_missing=True)
WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/PRISM.git"
BRANCH = "exp/cold-directions"

FLAGS = ("--corpus=modernweb --method=dirs_only "
         "--fingerprint=../fingerprints/cold-dct --seeds=1337,1338,1339 "
         "--student_steps=1500 --eval_every=10 --eval_iters=50 "
         "--method_lr=1e-3 --method_warmup=100")


@app.function(image=image, gpu="L4", volumes={WORK: vol}, timeout=24 * 3600)
def run_eval(extra: str = ""):
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

    subprocess.run([sys.executable, "gen_cold_directions.py",
                    "--out", "../fingerprints/cold-dct"],
                   cwd=f"{REPO}/src", check=True)
    cmd = [sys.executable, "-u", "prism_eval.py"] + FLAGS.split()
    if extra:
        cmd += extra.split()
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
        raise RuntimeError(f"prism_eval.py exited {rc}. Resume state is on "
                           f"the Volume — re-run to continue.")


@app.local_entrypoint()
def main(gpu: str = "L4", extra: str = ""):
    call = run_eval.with_options(gpu=gpu).spawn(extra=extra)
    print(f"launched (detached). call id: {call.object_id}")
    print("watch: modal app list  →  modal app logs <ap-...>")
    print("artifact lands on Volume prism-eval-cold under nanogpt-prism/results/")
