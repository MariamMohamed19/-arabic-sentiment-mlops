# Project Constitution

- Reproducibility over convenience.
- Measured results over assumed results.
- Never fabricate benchmark numbers.
- Use AraBERT or CAMeL-BERT.
- POST /predict returns label, confidence, model_version.
- GET /health returns {status: healthy}.
- Label mapping is fixed: 0=negative, 1=positive, 2=neutral (source column "Mixed" is treated as neutral).
- Splits are deterministic (seed 42, stratified 80/10/10); the test split is never used for tuning.
- Baseline (TF-IDF + LogReg) must exist before any transformer or optimization work.
