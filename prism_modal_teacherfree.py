"""
prism_modal_teacherfree.py — the teacher-free init probe (experiment #1 in
docs/NEXT-EXPERIMENTS.md), on the `exp/teacher-free` branch.

Two arms, two ISOLATED Volumes, launched as two detached jobs in parallel
(they must never share a Volume — both rewrite data/shakespeare_eval and
out-eval-* dirs):

    modal run --detach prism_modal_teacherfree.py --arm control
    modal run --detach prism_modal_teacherfree.py --arm cross

Identical probe rig (3 seeds, matched schedules, dense eval). The ONLY
difference is `--cross_teacher=data/far.txt` on the cross arm: its teacher
trains exclusively on Sherlock (never sees Shakespeare), and its ~128-byte
fingerprint seeds a student trained and scored on the standard Shakespeare
benchmark. Win = cross score ≈ control score.

Fetch the artifacts when both finish, and COMMIT them:
    modal volume ls prism-eval-tf-cross nanogpt-prism/results
    modal volume get prism-eval-tf-cross   nanogpt-prism/results/<recipe_*.json> ./results/
    modal volume get prism-eval-tf-control nanogpt-prism/results/<recipe_*.json> ./results/
"""
import modal

app = modal.App("prism-eval-teacherfree")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers", "tiktoken", "datasets")
)

vol_control = modal.Volume.from_name("prism-eval-tf-control", create_if_missing=True)
vol_cross = modal.Volume.from_name("prism-eval-tf-cross", create_if_missing=True)
WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/PRISM.git"
BRANCH = "exp/teacher-free"

# The shared probe rig: 3 seeds, converged teacher (2k steps — the saturation
# point from Run F/I), dense eval for resolved (not left-censored) scores, and
# MATCHED schedules so baseline and method differ by nothing but the flags.
PROBE = ("--seeds=1337,1338,1339 --teacher_steps=2000 --student_steps=1500 "
         "--eval_every=10 --eval_iters=50 --method_lr=1e-3 --method_warmup=100")


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


@app.function(image=image, gpu="L4", volumes={WORK: vol_control}, timeout=24 * 3600)
def run_control(extra: str):
    _run(vol_control, extra)


@app.function(image=image, gpu="L4", volumes={WORK: vol_cross}, timeout=24 * 3600)
def run_cross(extra: str):
    _run(vol_cross, extra)


@app.local_entrypoint()
def main(arm: str = "cross", gpu: str = "L4", extra: str = ""):
    if arm not in ("cross", "control"):
        raise SystemExit("--arm must be 'cross' or 'control'")
    flags = PROBE
    if arm == "cross":
        flags += " --cross_teacher=data/far.txt"
    if extra:
        flags += f" {extra}"
    fn = (run_cross if arm == "cross" else run_control).with_options(gpu=gpu)
    call = fn.spawn(extra=flags)
    print(f"launched {arm} arm (detached). call id: {call.object_id}")
    print("watch: modal app list  →  modal app logs <ap-...>")
    print(f"artifact lands on Volume prism-eval-tf-{arm} under nanogpt-prism/results/")
