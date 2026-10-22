"""
prism_modal_alpha_w3.py — the cross-size projection study (alpha-moves W3), on the
`exp/alpha-w3` branch. The PROBE rig of prism_modal_modernweb.py verbatim (modern
web, 3 seeds, 2,000-step teachers where trained, 1,500-step students, eval every
10, matched method LR 1e-3 / warmup 100); one launch per (case, method), each on
its own Volume slot so concurrent launches never share a run lock or data slot.

Cases (pre-registered in docs/CROSS-SIZE.md):
  1  width 768→384, depth 12→6 : teacher 12L/768/12h (hd 64) → student 6L/384/6h (hd 64)
  2  heads 12→6 at width 384    : teacher 6L/384/12h (hd 32)  → student 6L/384/6h (hd 64)
  3  head-dim reduction         : teacher 6L/384/6h (hd 64)   → student 6L/192/6h (hd 32)
  4  GPT-2 124M → 10.65M        : HF gpt2 (12L/768/12h, BPE) → student 6L/384/6h (bytes), internal matrices only
Methods per case: xsize (dirs_only on the PCA-projected fingerprint), xsize_procrustes
(cases 1–3), weightsel, front, mimetic, scratch (the in-run baseline of every launch),
and same-size dirs_only as the upper reference (the committed Run R, modern web).

    modal run --detach prism_modal_alpha_w3.py --case 1 --method xsize --slot a
    modal run prism_modal_alpha_w3.py --case 1 --method xsize --dry      # cost estimate only

Fetch + COMMIT:  modal volume get prism-alpha-w3-<slot> nanogpt-prism/results/<method>_<stamp>.json ./results/
"""
import modal

app = modal.App("prism-alpha-w3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers", "tiktoken", "datasets", "scipy")
)

WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/PRISM.git"
BRANCH = "exp/alpha-w3"

PROBE = ("--corpus=modernweb --seeds=1337,1338,1339 --teacher_steps=2000 "
         "--student_steps=1500 --eval_every=10 --eval_iters=50 "
         "--method_lr=1e-3 --method_warmup=100")

CASES = {
    "1": dict(teacher="--teacher_n_layer=12 --teacher_n_embd=768 --teacher_n_head=12",
              student="", hf=None, label="width 768→384 (12L/768/12h → 6L/384/6h)"),
    "2": dict(teacher="--teacher_n_layer=6 --teacher_n_embd=384 --teacher_n_head=12",
              student="", hf=None, label="heads 12→6 at width 384"),
    "3": dict(teacher="", student="--student_n_layer=6 --student_n_embd=192 --student_n_head=6",
              hf=None, label="head-dim 64→32 (6L/384/6h → 6L/192/6h)"),
    "4": dict(teacher="", student="", hf="gpt2", label="GPT-2 124M → 10.65M (internal matrices)"),
}
METHODS = {
    "xsize":            "--method=dirs_only --crosssize=pca",
    "xsize_procrustes": "--method=dirs_only --crosssize=procrustes",
    "xsize_svdhidden":  "--method=dirs_only --crosssize=pca --xsize_hidden=svd",
    "weightsel":        "--method=init_state --init_builder=weightsel",
    "front":            "--method=init_state --init_builder=front",
    "mimetic":          "--method=mimetic",
}

# FLOP-count ratio of a 12L/768 teacher to the 10.65M student at T=256:
#   (6N + 12·L·d·T) per token = 538M vs 71M → ≈7.6×  (see docs/alpha/W3-NOTES.md for the CPU check)
TEACHER_STEP_RATIO = {"1": 7.6, "2": 1.0, "3": 1.0, "4": 0.0}
SEC_PER_STUDENT_STEP = 0.275       # measured: 1,500 steps ≈ 413 s on an L4 (Run R/X)
L4_USD_PER_H = 0.80

SLOTS = ["a", "b", "c", "d", "e", "f"]
VOLS = {s: modal.Volume.from_name(f"prism-alpha-w3-{s}", create_if_missing=True) for s in SLOTS}


def _run(vol, flags: str):
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
    print("commit:", subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(), flush=True)
    vol.commit()
    cmd = [sys.executable, "-u", "prism_eval.py"] + flags.split()
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
        raise RuntimeError(f"prism_eval.py exited {rc}. Resume state is on the Volume — "
                           f"re-run the same command to continue.")


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["a"]}, timeout=24 * 3600)
def run_slot_a(flags: str):
    _run(VOLS["a"], flags)


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["b"]}, timeout=24 * 3600)
def run_slot_b(flags: str):
    _run(VOLS["b"], flags)


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["c"]}, timeout=24 * 3600)
def run_slot_c(flags: str):
    _run(VOLS["c"], flags)


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["d"]}, timeout=24 * 3600)
def run_slot_d(flags: str):
    _run(VOLS["d"], flags)


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["e"]}, timeout=24 * 3600)
def run_slot_e(flags: str):
    _run(VOLS["e"], flags)


@app.function(image=image, gpu="L4", volumes={WORK: VOLS["f"]}, timeout=24 * 3600)
def run_slot_f(flags: str):
    _run(VOLS["f"], flags)


FNS = {"a": run_slot_a, "b": run_slot_b, "c": run_slot_c, "d": run_slot_d, "e": run_slot_e, "f": run_slot_f}


def estimate(case, method, n_seeds=3, teacher_steps=2000, student_steps=1500):
    c = CASES[case]
    teacher_h = (0.0 if c["hf"] else n_seeds * teacher_steps * SEC_PER_STUDENT_STEP * TEACHER_STEP_RATIO[case] / 3600)
    student_scale = {"3": 0.5}.get(case, 1.0)      # a 192-wide student is ~half the step time
    students_h = n_seeds * 2 * student_steps * SEC_PER_STUDENT_STEP * student_scale / 3600
    evals_h = 0.15 * n_seeds                        # dense eval every 10 steps, both arms
    hours = teacher_h + students_h + evals_h
    return hours, hours * L4_USD_PER_H, teacher_h


@app.local_entrypoint()
def main(case: str = "1", method: str = "xsize", slot: str = "a", gpu: str = "L4",
         extra: str = "", dry: bool = False):
    assert case in CASES and method in METHODS and slot in SLOTS
    c = CASES[case]
    if c["hf"]:
        assert method != "xsize_procrustes", "Procrustes needs the same tokenisation; case 4 uses pca"
        mflags = METHODS[method].replace("--crosssize=pca", f"--crosssize_hf={c['hf']}")
        if method in ("weightsel", "front"):
            mflags += f" --crosssize_hf={c['hf']}"
        if method == "mimetic":
            raise SystemExit("case 4 has no trained teacher stage; mimetic is covered by cases 1-3")
    else:
        mflags = METHODS[method]
    flags = " ".join(x for x in [PROBE, c["teacher"], c["student"], mflags, extra] if x)
    hours, usd, th = estimate(case, method)
    print(f"case {case} ({c['label']}) · method {method} · slot {slot}")
    print(f"estimate: ~{hours:.1f} L4-h ≈ ${usd:.2f} (teacher share {th:.1f} h; teachers cache per seed "
          f"on the slot's Volume, so later methods on the same slot skip it)")
    print("flags:", flags)
    if dry:
        return
    call = FNS[slot].with_options(gpu=gpu).spawn(flags)
    print(f"launched (detached). call id: {call.object_id}")
    print("watch: modal app list  →  modal app logs <ap-...>")
    print(f"artifact lands on Volume prism-alpha-w3-{slot} under nanogpt-prism/results/")
