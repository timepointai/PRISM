"""TRL GRPO with the weight-space anchor in place of KL-to-reference.
beta=0.0 → TRL loads NO reference model; the callback is the only drift control.

    PYTHONPATH=src python examples/anchor_trl_grpo.py      # needs trl, datasets, a GPU
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from prism_rl_anchor import AnchorCallback

def reward_len(completions, **kwargs):          # a toy reward
    return [-abs(20 - len(c)) for c in completions]

dataset = load_dataset("trl-lib/tldr", split="train")
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_len,
    args=GRPOConfig(output_dir="out-grpo-anchor", beta=0.0, max_steps=50, logging_steps=5),
    train_dataset=dataset,
    callbacks=[AnchorCallback(strength=0.0025, target="fixed")],   # or target="ema:0.01", liti=0.7
)
trainer.train()
