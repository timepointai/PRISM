# modernweb — the modern-web corpus

A ~5MB slice of **FineWeb-Edu** (educational-quality filtered 2024 CommonCrawl
web text), used as the modern-corpus bench for PRISM's byte-level experiments.

- Source: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
  config `sample-10BT`, file `sample/10BT/000_00000.parquet`, first 1,027
  documents in row-group order, joined with `\n\n`, truncated at the last
  newline before 5,000,000 bytes.
- `modern.txt`: 4,999,716 bytes,
  sha256 `a46ade308f026bbd7b15e27bcf068d200ed1384912dacebe35664d569df3aba5`.
- Fetched 2026-07-27.
- License: FineWeb-Edu is released under **ODC-By 1.0** — attribution above;
  subject to the CommonCrawl terms of use. This slice is redistributed here
  unmodified for reproducibility (the artifact ethos of this repo: the exact
  bytes every committed result trained on must remain available).

`prepare.py` encodes the text **byte-level** (vocab = 256, nothing dropped),
90/10 train/val split, same bin format as `data/shakespeare_char`.
