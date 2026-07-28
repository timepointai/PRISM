"""Prepare enwik8 — the canonical byte-level LM / compression benchmark
(first 10^8 bytes of English Wikipedia XML, the Hutter Prize corpus).

Downloads the fixed, world-known file (35MB zip) if absent, verifies its size,
and emits the STANDARD literature split: first 90M bytes train, next 5M val
(the remaining 5M is the untouched test tail, kept out of both). Byte-level
(vocab 256), same bin format as data/shakespeare_char, so train.py and
prism_eval.py (--corpus=enwik8) work unchanged.

Loss in nats/byte divides by ln(2) to give the literature's bits-per-byte.
The corpus is fixed for eternity — the sha256 of the unzipped file is printed
and recorded in this dir's README for provenance.
"""
import hashlib
import io
import os
import pickle
import urllib.request
import zipfile

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
raw = os.path.join(here, 'enwik8')

if not os.path.exists(raw):
    url = 'http://mattmahoney.net/dc/enwik8.zip'
    print(f'downloading {url} …')
    data = urllib.request.urlopen(url, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        content = z.read('enwik8')
    open(raw, 'wb').write(content)

content = open(raw, 'rb').read()
assert len(content) == 100_000_000, f'enwik8 must be 1e8 bytes, got {len(content):,}'
print('sha256:', hashlib.sha256(content).hexdigest())

arr = np.frombuffer(content, dtype=np.uint8).astype(np.uint16)
train, val = arr[:90_000_000], arr[90_000_000:95_000_000]
train.tofile(os.path.join(here, 'train.bin'))
val.tofile(os.path.join(here, 'val.bin'))
print(f'train: {len(train):,} tokens  val: {len(val):,} tokens  (standard 90/5 split; '
      f'final 5M test tail untouched)')

stoi = {chr(i): i for i in range(256)}
itos = {i: chr(i) for i in range(256)}
with open(os.path.join(here, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': 256, 'stoi': stoi, 'itos': itos,
                 'encoding': 'latin-1'}, f)
print('vocab_size: 256 (raw bytes)')
