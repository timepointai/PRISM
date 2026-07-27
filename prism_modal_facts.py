"""
prism_modal_facts.py — the fact-injection probe, on the `exp/fact-injection`
branch. One isolated Volume, one detached L4 job:

    modal run --detach prism_modal_facts.py

The question (from the ReLoRA/fact-injection thread): when you inject NOVEL
facts into a trained modern-web base, what does the directional anchor buy vs
the low-LR alternative — on all three axes at once: fact recall (closed-book
exact match, seen AND held-out phrasing), retention (modern-web val climb),
and the facts-val loss curve. Arms share one base per seed and differ by one
flag each: plain (control), raw anchor 0.01 / 0.02, low-LR 1e-4.

Fetch + COMMIT the artifact:
    modal volume get prism-eval-facts nanogpt-prism/results/finetune_<stamp>.json ./results/
"""
import modal

app = modal.App("prism-eval-facts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers", "tiktoken", "datasets")
)

vol = modal.Volume.from_name("prism-eval-facts", create_if_missing=True)
WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/PRISM.git"
BRANCH = "exp/fact-injection"

FLAGS = ("--old_corpus=modernweb --far_corpus=data/facts/facts_train.txt "
         "--ft_val_corpus=data/facts/facts_val.txt "
         "--recall_prompts=data/facts/recall_prompts.tsv "
         "--base_steps=2000 --ft_steps=1000 --eval_every=25 --eval_iters=100 "
         "--seeds=1337,1338,1339 --batch_size=32 --block_size=256 "
         "--learning_rate=3e-4 --min_lr=3e-5 "
         "--arms=base,plain,raw_mid,raw_hi,lowlr_b --tag=facts1")


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

    cmd = [sys.executable, "-u", "prism_finetune_eval.py"] + FLAGS.split()
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
        raise RuntimeError(f"prism_finetune_eval.py exited {rc}. Resume state is on "
                           f"the Volume — re-run to continue.")


@app.local_entrypoint()
def main(gpu: str = "L4", extra: str = ""):
    call = run_eval.with_options(gpu=gpu).spawn(extra=extra)
    print(f"launched (detached). call id: {call.object_id}")
    print("watch: modal app list  →  modal app logs <ap-...>")
    print("artifact lands on Volume prism-eval-facts under nanogpt-prism/results/")
