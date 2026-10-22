"""The PRISM retention anchor in plain PyTorch: one post-step hook, no reference
model. Run from the repo's src/ on PYTHONPATH:

    PYTHONPATH=src python examples/anchor_pytorch.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from prism_anchor import Anchor, capture_targets

model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# W0 = the weights you want to stay near (here: the init; in a finetune: the base)
anchor = Anchor(model.named_parameters(), capture_targets(model.named_parameters()))
anchor.attach(optimizer, strength_fn=lambda: 0.0025)       # registers a step post-hook

for step in range(50):
    x = torch.randn(8, 16)
    loss = model(x).pow(2).mean()
    loss.backward()
    optimizer.step()            # the anchor fires here, after the update
    optimizer.zero_grad()
print('anchor applied', anchor.n_applied, 'times')
