"""fact_recall.py — closed-book exact-match recall on injected facts.

Loads a nanoGPT checkpoint, greedy-decodes the answer span for each prompt in
a recall TSV (kind<TAB>prompt<TAB>answer), and reports exact-match accuracy
per kind ('seen' = a trained phrasing, 'unseen' = the held-out phrasing).
Prints a single JSON object to stdout — the caller banks it in its artifact.

    python fact_recall.py --ckpt out-ft-plain-s1337/ckpt.pt \
        --meta data/facts_ft/meta.pkl --prompts data/facts/recall_prompts.tsv
"""
import argparse
import json
import pickle

import torch

from model import GPTConfig, GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--meta', required=True, help='meta.pkl with stoi/itos')
    p.add_argument('--prompts', required=True, help='TSV: kind, prompt, answer')
    p.add_argument('--device', default='cpu',
                   help='cpu is fine: 240 short greedy decodes')
    args = p.parse_args()

    meta = pickle.load(open(args.meta, 'rb'))
    stoi, itos = meta['stoi'], meta['itos']

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = GPT(GPTConfig(**ckpt['model_args']))
    sd = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    model.to(args.device).eval()

    rows = [line.rstrip('\n').split('\t') for line in open(args.prompts)
            if line.strip()]
    out = {'n': {}, 'correct': {}}
    with torch.no_grad():
        for kind, prompt, answer in rows:
            idx = torch.tensor([[stoi[c] for c in prompt]], device=args.device)
            got = []
            for _ in range(len(answer)):
                logits, _ = model(idx[:, -model.config.block_size:])
                nxt = int(torch.argmax(logits[0, -1]))
                got.append(itos[nxt])
                idx = torch.cat([idx, torch.tensor([[nxt]], device=args.device)],
                                dim=1)
            out['n'][kind] = out['n'].get(kind, 0) + 1
            out['correct'][kind] = (out['correct'].get(kind, 0)
                                    + (''.join(got) == answer))
    out['acc'] = {k: round(out['correct'][k] / out['n'][k], 4) for k in out['n']}
    print(json.dumps(out))


if __name__ == '__main__':
    main()
