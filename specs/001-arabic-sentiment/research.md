# Research: Dataset (resolves T001, T002)

## Source file
`data/raw/train-00000-of-00001.parquet` — Hugging Face `datasets` export (columns `text`, `label`, `label_text`).
Its structure (99,999 rows, 3 perfectly balanced classes of 33,333, mixed book/hotel reviews) matches the public
"Arabic 100k Reviews" collection. **Verify the exact origin on the Hugging Face page you downloaded it from and
record the URL + license here before publishing results.**

## Measured facts (from the file itself)
| Item | Value |
|---|---|
| Rows | 99,999 (99,816 after cleaning: <2 words / no Arabic letters / duplicates removed) |
| Labels | 0 Negative, 1 Positive, 2 Mixed — 33,333 each |
| Median length | 26 words (p95 = 197, p99 = 455) |
| Dialects | MSA + Egyptian/Gulf dialect (e.g. "ايه دا دا واحد كان فاضى") |
| Domains seen | Mostly book (~36% of rows mention book words) and hotel (~20%) reviews; very few e-commerce product reviews |

## Decisions
1. **"Mixed" -> "neutral".** The project brief says Positive/Negative/Neutral; this dataset has Mixed (typically 3-star).
   Documented assumption, not an official equivalence.
2. **Domain gap.** The brief is about Noon/Amazon.eg product reviews; this data is books/hotels. Report this limitation
   in the final README. Do not claim product-review accuracy.
3. **Split:** stratified 80/10/10, seed 42, exact-text duplicates removed first (no train/test leakage).
4. **max_length = 128** tokens covers roughly the median-to-p90 review; longer texts are truncated.
