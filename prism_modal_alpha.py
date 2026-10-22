"""
prism_modal_alpha.py — the alpha-moves runner for the finetune-retention rigs
(W1: the injection-dial head-to-head), on the `exp/alpha-w1` branch. Six
isolated Volumes ("slots" a–f) so concurrent launches never share a run lock, a
fork directory or a data slot — give every concurrent launch its own --slot:

    modal run --detach prism_modal_alpha.py --rig facts   --slot a --tag w1tune --seeds 1336 --arms base,plain,sst_t1,...
    modal run --detach prism_modal_alpha.py --rig catalog --slot b --tag w1a    --arms base,plain,...

Rigs reuse the committed probe FLAGS verbatim (prism_modal_facts.py /
prism_modal_catalog.py): modern-web base, 2,000 base steps, 1,000 finetune
steps, eval every 25, 3 seeds, LR 3e-4. Only --arms / --seeds / --tag change.
Every launch prints a cost estimate from the measured L4 step time before it
spawns.

Fetch + COMMIT the artifact:
    modal volume get prism-alpha-<slot> nanogpt-prism/results/finetune_<stamp>.json ./results/
"""
import modal

app = modal.App("prism-alpha")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers", "tiktoken", "datasets", "scipy")
)

WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/PRISM.git"
BRANCH = "exp/alpha-w1"

RIGS = {
    "facts": ("--old_corpus=modernweb --far_corpus=data/facts/facts_train.txt "
              "--ft_val_corpus=data/facts/facts_val.txt "
              "--recall_prompts=data/facts/recall_prompts.tsv,data/facts/recall_prompts_v2.tsv "
              "--base_steps=2000 --ft_steps=1000 --eval_every=25 --eval_iters=100 "
              "--batch_size=32 --block_size=256 --learning_rate=3e-4 --min_lr=3e-5"),
    "catalog": ("--old_corpus=modernweb --far_corpus=data/catalog/task_train.txt "
                "--ft_val_corpus=data/catalog/task_val.txt "
                "--recall_prompts=data/catalog/recall_prompts.tsv "
                "--base_steps=2000 --ft_steps=1000 --eval_every=25 --eval_iters=100 "
                "--batch_size=32 --block_size=256 --learning_rate=3e-4 --min_lr=3e-5"),
}

# measured on the committed facts2 run (L4): ~0.27 s/step incl. dense dual-val evals
SEC_PER_STEP = 0.30
L4_USD_PER_H = 0.80


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
    cmd = [sys.executable, "-u", "prism_finetune_eval.py"] + flags.split()
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
        raise RuntimeError(f"prism_finetune_eval.py exited {rc}. Resume state is on the "
                           f"Volume — re-run the same command to continue.")


# Modal functions must exist at import time (global scope), so concurrency comes
# from a fixed set of SLOTS, each with its own Volume. Pick a free slot per launch
# (--slot a..f); the --tag still names the run key / artifact.
SLOTS = ["a", "b", "c", "d", "e", "f"]
VOLS = {s: modal.Volume.from_name(f"prism-alpha-{s}", create_if_missing=True) for s in SLOTS}


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


@app.local_entrypoint()
def main(rig: str = "facts", tag: str = "", arms: str = "", seeds: str = "1337,1338,1339",
         slot: str = "a", gpu: str = "L4", extra: str = "", dry: bool = False):
    assert rig in RIGS, f"--rig must be one of {list(RIGS)}"
    assert slot in SLOTS, f"--slot must be one of {SLOTS}"
    assert tag, "--tag is required (it names the run key / artifact)"
    assert arms, "--arms is required"
    flags = f"{RIGS[rig]} --seeds={seeds} --arms={arms} --tag={tag}" + (f" {extra}" if extra else "")
    n_seeds = len([s for s in seeds.split(',') if s.strip()])
    arm_list = [a for a in arms.split(',') if a.strip() and a != 'base']
    steps = n_seeds * (2000 + 1000 * len(arm_list))
    hours = steps * SEC_PER_STEP / 3600 + 0.05 * n_seeds * len(arm_list)   # + recall decodes
    print(f"rig={rig} tag={tag} seeds={n_seeds} arms={len(arm_list)} → ~{hours:.1f} L4-h "
          f"≈ ${hours * L4_USD_PER_H:.2f} (SEC_PER_STEP={SEC_PER_STEP}); volume prism-alpha-{slot}")
    print("flags:", flags)
    if dry:
        return
    call = FNS[slot].with_options(gpu=gpu).spawn(flags)
    print(f"launched (detached). call id: {call.object_id}")
    print("watch: modal app list  →  modal app logs <ap-...>")
    print(f"artifact lands on Volume prism-alpha-{slot} under nanogpt-prism/results/")
