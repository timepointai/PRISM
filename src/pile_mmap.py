"""pile_mmap.py — Pythia's exact token stream, read directly from the published
Megatron MMapIndexedDataset (EleutherAI/pile-standard-pythia-preshuffled).

Verified against the live repo on 2026-10-21 (header read by HTTP range request):
  document.idx : magic b'MMIDIDX\\x00\\x00', version 1, dtype code 8 (uint16),
                 N = 146,432,000 sequences = 143,000 iterations × 1,024,
                 D = 1 document; every sequence has 2,049 tokens (targets are the
                 left-shift), so pointer[i] = i · 4,098 bytes exactly.
  shards       : document-000{00..19}-of-00020.bin are 30,000,000,000 bytes each,
                 document-00020-of-00020.bin is 78,336,000 bytes; the official
                 utils/unshard_memmap.py simply concatenates them, so a global
                 byte offset o lives in shard o // 30e9 at local offset o % 30e9.
  batches      : training iteration k consumed sequences [k·1024, (k+1)·1024)
                 (EleutherAI/pythia utils/batch_viewer.py hard-codes 1024/step).

`.idx` layout (little-endian): 9 B magic | u64 version | u8 dtype | u64 N | u64 D |
int32 sizes[N] | int64 pointers[N] (byte offsets into the concatenated .bin) |
int64 doc_idx[D]. `write_synthetic_dataset` emits the same format for tests.

Range fetching: `fetch_range` streams an arbitrary byte window of a shard through
`huggingface_hub.HfFileSystem` (fsspec, seekable), so a gate that needs only the
first few GB of shard 0 plus an 8 MB held-out window from a far shard never
downloads 600 GB.
"""
import os
import struct

import numpy as np

MAGIC = b'MMIDIDX\x00\x00'
DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64,
          6: np.float32, 7: np.float64, 8: np.uint16}
CODE_OF = {v: k for k, v in DTYPES.items()}
REPO = 'EleutherAI/pile-standard-pythia-preshuffled'
SHARD_BYTES = 30_000_000_000
N_SHARDS = 21
SEQ_LEN = 2049
SEQS_PER_ITER = 1024


def shard_name(i):
    return f'document-{i:05d}-of-00020.bin'


def read_idx_header(path_or_bytes):
    hdr = (path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray))
           else open(path_or_bytes, 'rb').read(34))
    assert hdr[:9] == MAGIC, f'bad magic {hdr[:9]!r}'
    version = struct.unpack('<Q', hdr[9:17])[0]
    assert version == 1
    code = hdr[17]
    n = struct.unpack('<Q', hdr[18:26])[0]
    d = struct.unpack('<Q', hdr[26:34])[0]
    return {'dtype': DTYPES[code], 'code': code, 'n_seq': n, 'n_doc': d}


def read_idx(path, max_seq=None):
    """sizes (int32), pointers (int64) for the first `max_seq` sequences (all if None)."""
    h = read_idx_header(path)
    n = h['n_seq'] if max_seq is None else min(max_seq, h['n_seq'])
    sizes = np.fromfile(path, dtype='<i4', count=n, offset=34)
    pointers = np.fromfile(path, dtype='<i8', count=n, offset=34 + 4 * h['n_seq'])
    return h, sizes, pointers


def write_synthetic_dataset(out_dir, n_seq, seq_len=SEQ_LEN, vocab=128, seed=0,
                            shard_bytes=None, prefix='document'):
    """A tiny dataset in the exact MMIDIDX format (optionally sharded like the real
    one) for tests. Returns (idx_path, [shard paths])."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = rng.integers(0, vocab, size=(n_seq, seq_len), dtype=np.uint16)
    raw = data.tobytes()
    itemsize = 2
    sizes = np.full(n_seq, seq_len, dtype='<i4')
    pointers = (np.arange(n_seq, dtype='<i8') * seq_len * itemsize)
    doc_idx = np.array([0, n_seq], dtype='<i8')
    idx_path = os.path.join(out_dir, f'{prefix}.idx')
    with open(idx_path, 'wb') as f:
        f.write(MAGIC + struct.pack('<Q', 1) + bytes([CODE_OF[np.uint16]])
                + struct.pack('<Q', n_seq) + struct.pack('<Q', len(doc_idx)))
        f.write(sizes.tobytes()); f.write(pointers.tobytes()); f.write(doc_idx.tobytes())
    shards = []
    if shard_bytes is None:
        p = os.path.join(out_dir, f'{prefix}.bin')
        open(p, 'wb').write(raw)
        shards.append(p)
    else:
        n_sh = (len(raw) + shard_bytes - 1) // shard_bytes
        for i in range(n_sh):
            p = os.path.join(out_dir, f'{prefix}-{i:05d}-of-{n_sh - 1:05d}.bin')
            open(p, 'wb').write(raw[i * shard_bytes:(i + 1) * shard_bytes])
            shards.append(p)
    return idx_path, shards, data


class PileStream:
    """Sequence access over (possibly partial, possibly sharded) local shard files.

    shards: {shard_index: local_path} — a shard file may be a PREFIX of the real
    shard (range-fetched); reads beyond what is present raise. `shard_bytes` is the
    full-shard size the global offsets assume (30e9 for the real data)."""

    def __init__(self, idx_path, shards, shard_bytes=SHARD_BYTES, max_seq=None,
                 window=None):
        """window: (first_seq, sizes, pointers, dtype) for a non-prefix slice of the
        stream (see read_idx_window); then `sequence(i)` takes GLOBAL ids."""
        if window is None:
            self.header, self.sizes, self.pointers = read_idx(idx_path, max_seq=max_seq)
            self.first = 0
        else:
            self.first, self.sizes, self.pointers, dt = window
            self.header = {'dtype': dt}
        self.dtype = self.header['dtype']
        self.itemsize = np.dtype(self.dtype).itemsize
        self.shard_bytes = shard_bytes
        # value: path (a prefix of the shard) or (path, local_start) for a window
        self.shards, self.base = {}, {}
        for k, v in shards.items():
            path, base = (v if isinstance(v, (tuple, list)) else (v, 0))
            self.shards[int(k)], self.base[int(k)] = path, int(base)
        self.mm = {k: np.memmap(v, dtype=np.uint8, mode='r') for k, v in self.shards.items()}

    @property
    def n_seq(self):
        return len(self.sizes)

    @property
    def seq_range(self):
        return self.first, self.first + len(self.sizes)

    def sequence(self, i):
        j = i - self.first
        if j < 0 or j >= len(self.sizes):
            raise IndexError(f'sequence {i} outside this stream window {self.seq_range}')
        off, n = int(self.pointers[j]), int(self.sizes[j])
        nbytes = n * self.itemsize
        s, local = off // self.shard_bytes, off % self.shard_bytes
        if s not in self.mm:
            raise IndexError(f'sequence {i} lives in shard {s}, which is not present')
        if local + nbytes > self.shard_bytes:                  # a sequence straddling two shards
            a = self.mm[s][local - self.base[s]:]
            b = self.mm[s + 1][self.base.get(s + 1, 0) * 0:nbytes - len(a)]
            buf = np.concatenate([np.asarray(a), np.asarray(b)])
        else:
            lo = local - self.base[s]
            if lo < 0:
                raise IndexError(f'sequence {i}: shard {s} window starts at local byte {self.base[s]}')
            buf = self.mm[s][lo:lo + nbytes]
            if len(buf) < nbytes:
                raise IndexError(f'sequence {i}: shard {s} present only up to local byte '
                                 f'{self.base[s] + len(self.mm[s])}')
        return np.frombuffer(np.asarray(buf).tobytes(), dtype=self.dtype)

    def batch(self, first_seq, n):
        """(x, y) int64 arrays of shape (n, seq_len-1): sequences first_seq..first_seq+n."""
        seqs = np.stack([self.sequence(i) for i in range(first_seq, first_seq + n)]).astype(np.int64)
        return seqs[:, :-1], seqs[:, 1:]

    def iteration(self, k, micro=None, which=0):
        """Pythia training iteration k = sequences [k·1024, (k+1)·1024); with `micro`,
        micro-batch `which` of that iteration."""
        start = k * SEQS_PER_ITER
        if micro is None:
            return self.batch(start, SEQS_PER_ITER)
        return self.batch(start + which * micro, micro)


def byte_range_for_sequences(first, last_exclusive, seq_bytes=SEQ_LEN * 2):
    """Global byte window [a, b) covering sequences [first, last_exclusive) of the
    real stream (contiguous pointers verified)."""
    return first * seq_bytes, last_exclusive * seq_bytes


def shards_for_range(a, b, shard_bytes=SHARD_BYTES):
    """{shard_index: (local_start, local_end)} covering global bytes [a, b)."""
    out = {}
    s = a // shard_bytes
    while s * shard_bytes < b:
        lo = max(a, s * shard_bytes) - s * shard_bytes
        hi = min(b, (s + 1) * shard_bytes) - s * shard_bytes
        out[s] = (int(lo), int(hi))
        s += 1
    return out


def fetch_range(repo_file, start, end, out_path, repo=REPO, chunk=64 << 20, token=None):
    """Stream bytes [start, end) of a dataset file from the Hub into out_path
    (resumable: skips bytes already present when out_path exists and start==0)."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=token)
    have = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    mode = 'ab' if (have and start == 0) else 'wb'
    pos = start + (have if mode == 'ab' else 0)
    if pos >= end:
        return out_path
    with fs.open(f'datasets/{repo}/{repo_file}', 'rb', block_size=chunk) as src, open(out_path, mode) as dst:
        src.seek(pos)
        while pos < end:
            buf = src.read(min(chunk, end - pos))
            if not buf:
                raise IOError(f'short read at {pos} of {repo_file}')
            dst.write(buf)
            pos += len(buf)
    return out_path


def read_idx_window(fobj, first, n):
    """sizes/pointers for sequences [first, first+n) from an open .idx file object
    (local file or HfFileSystem handle). Returns (first, sizes, pointers, dtype)."""
    fobj.seek(0)
    h = read_idx_header(fobj.read(34))
    n = min(n, h['n_seq'] - first)
    fobj.seek(34 + 4 * first)
    sizes = np.frombuffer(fobj.read(4 * n), dtype='<i4')
    fobj.seek(34 + 4 * h['n_seq'] + 8 * first)
    pointers = np.frombuffer(fobj.read(8 * n), dtype='<i8')
    return first, sizes, pointers, h['dtype']


def fetch_idx_window(first, n, repo=REPO, token=None):
    """read_idx_window straight from the Hub (two small range reads)."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=token)
    with fs.open(f'datasets/{repo}/document.idx', 'rb') as f:
        return read_idx_window(f, first, n)


def fetch_idx_prefix(out_path, n_seq, repo=REPO, token=None):
    """The .idx header + sizes[:n_seq] + pointers[:n_seq] + doc_idx, written as a
    VALID .idx file describing only the first n_seq sequences (the full idx is 1.76 GB)."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=token)
    with fs.open(f'datasets/{repo}/document.idx', 'rb') as f:
        hdr = f.read(34)
        h = read_idx_header(hdr)
        n = min(n_seq, h['n_seq'])
        sizes = f.read(4 * n)
        f.seek(34 + 4 * h['n_seq'])
        pointers = f.read(8 * n)
    with open(out_path, 'wb') as g:
        g.write(MAGIC + struct.pack('<Q', 1) + bytes([h['code']]) + struct.pack('<Q', n)
                + struct.pack('<Q', 2))
        g.write(sizes); g.write(pointers)
        g.write(np.array([0, n], dtype='<i8').tobytes())
    return out_path, h
