"""Prepare the modern-web corpus (FineWeb-Edu slice) for byte-level training.

Reads the committed modern.txt VERBATIM as raw bytes — vocab is all 256 byte
values, so any modern text (and later: code, other languages) encodes without
dropping a single character. Emits train.bin / val.bin / meta.pkl in the same
format as data/shakespeare_char, so train.py and prism_eval.py work unchanged.

Split: first 90% train, last 10% val (held out, never trained on).
"""
import os
import pickle

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
data = open(os.path.join(here, 'modern.txt'), 'rb').read()
print(f'corpus: {len(data):,} bytes')

arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint16)
split = int(len(arr) * 0.9)
train, val = arr[:split], arr[split:]
train.tofile(os.path.join(here, 'train.bin'))
val.tofile(os.path.join(here, 'val.bin'))
print(f'train: {len(train):,} tokens  val: {len(val):,} tokens')

# Byte-level vocab: latin-1 gives a 1:1 char<->byte mapping for stoi/itos so
# sample.py and _encode_corpus stay usable; vocab_size is the full byte range.
stoi = {chr(i): i for i in range(256)}
itos = {i: chr(i) for i in range(256)}
meta = {'vocab_size': 256, 'stoi': stoi, 'itos': itos, 'encoding': 'latin-1'}
with open(os.path.join(here, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
print('vocab_size: 256 (raw bytes)')
