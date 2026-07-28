# enwik8 — the canonical byte-level LM bench

The first 10^8 bytes of English Wikipedia XML (the Hutter Prize corpus) — the
byte-modeling / compression benchmark with two decades of published
bits-per-byte numbers around it. `prepare.py` downloads the fixed file
(`http://mattmahoney.net/dc/enwik8.zip`, not committed — 100MB) and emits the
**standard literature split**: first 90M bytes train, next 5M val, final 5M
test tail untouched by anything here.

- Unzipped sha256 (verified on prepare):
  `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`
- Byte-level, vocab 256, same bin format as the other benches
  (`prism_eval.py --corpus=enwik8`).
- Loss is nats/byte; divide by ln 2 for the literature's bits-per-byte.
