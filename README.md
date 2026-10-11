# Arabic Sentiment Analysis (MLOps project)

Classifies Arabic reviews as **positive / negative / neutral** with a fine-tuned AraBERT model,
served through a FastAPI endpoint. Built with Spec-Kit: read `.specify/memory/constitution.md`, then
`specs/001-arabic-sentiment/{spec,research,plan,tasks}.md`.

## Data

- File: `data/raw/train-00000-of-00001.parquet` (columns `text`, `label`, `label_text`), 99,999 rows,
  3 balanced classes. Its structure matches the public "Arabic 100k Reviews" collection; the exact
  origin and license still need to be confirmed (see `specs/001-arabic-sentiment/research.md`).
- Cleaning removed very short / non-Arabic / duplicate texts (99,816 rows kept). Deterministic
  stratified split (seed 42): train 79,852 / validation 9,982 / test 9,982.
- Label decision: the source class **"Mixed" is treated as "neutral"** (documented assumption).
- **Limitation:** the reviews are mostly books and hotels, not e-commerce products, so the results
  below say nothing about product-review accuracy.

## Results (all numbers measured)

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| TF-IDF char n-grams + Logistic Regression (baseline, CPU) | 0.6782 | 0.6777 |
| AraBERT, best of 5 runs (run3) | n/a | **0.7217** |

- Five MLflow runs (3 AraBERT configs and 2 CAMeL-BERT configs) were
  trained on **10,000 training samples each** (not the full 79,852), batch 32, max length 128, fp16, on a
  Colab T4 GPU. The best run was picked by **validation** macro-F1; the test split was used only for
  the final numbers.
- Run3 = AraBERT (`aubmindlab/bert-base-arabertv02`), lr 5e-5, 2 epochs.
- Per-run table (fill from `mlflow ui --backend-store-uri sqlite:///mlflow.db`):

| Run | Model | lr | Epochs | Val macro-F1 | Test macro-F1 |
|---|---|---:|---:|---:|---:|
| run1 | AraBERT | 2e-5 | 3 | 0.7235 | 0.7242 |
| run2 | AraBERT | 3e-5 | 3 | 0.7233 | 0.7210 |
| run3 | AraBERT | 5e-5 | 2 | 0.7251 | 0.7217 |
| run4 | CAMeL-BERT | 2e-5 | 3 | 0.7064 | 0.7064 |
| run5 | CAMeL-BERT | 3e-5 | 3 | 0.7093 | 0.7101 |

Run3 was selected because it has the highest **validation** macro-F1. Run1 happens to score slightly
higher on the test split (0.7242), but the test split was not used to choose the model. The three
AraBERT runs are within about half a point of each other, so the differences between them are small.

Training on the full training set was not tried, so 0.7217 is probably not the ceiling.

## Optimization benchmark

Measured on the **first 2,000 test samples**, on the Colab **CPU**, single-text latency (p95).
Original runs in PyTorch; the ONNX variants run in ONNX Runtime.

| Model | Accuracy | Macro-F1 | P95 latency (ms) | Size (MB) |
|---|---:|---:|---:|---:|
| Original (PyTorch) | 0.719 | 0.7192 | 371.4 | 542.6 |
| ONNX FP32 | 0.719 | 0.7192 | 360.5 | 541.0 |
| INT8 ONNX (default dynamic quantization) | 0.573 | 0.5640 | 236.0 | 136.0 |

INT8 cut size about 4x and latency by about a third versus ONNX FP32, but **accuracy dropped about
14.6 points**, which makes it worse than the baseline. The ONNX FP32 export matches the original, so
the loss comes from the quantization step. The cause has not been investigated further.

## API

```bash
pip install -e ".[dev]"
pip install torch transformers
# put the trained model (config.json, model.safetensors, tokenizer files) in artifacts/model
MODEL_PATH=artifacts/model MODEL_VERSION=v1 uvicorn src.api.main:app
```

- `POST /predict` body `{"text": "..."}` returns `label`, `confidence`, `model_version`.
  Empty or blank text returns HTTP 422.
- `GET /health` returns `{"status": "healthy"}`.
- Verified locally with the trained model: a positive sentence returned `positive` (0.925), a negative
  one returned `negative` (0.961), empty text returned 422, `/health` returned healthy.
- The model weights (about 516 MB) are **not in this repository**. TODO: add a download link here.

## Reproduce

```bash
python -m src.data.prepare      # splits -> data/processed
python -m src.models.baseline   # baseline metrics
pytest && ruff check .
```

Training, ONNX export and the benchmark were run in Colab (`arabic_sentiment_colab.ipynb`) using
`src/models/train.py`, `src/optimization/export_onnx.py` and `src/optimization/benchmark.py`.
## Docker

`Dockerfile` and `docker-compose.yml` are included (CPU torch, model mounted from `artifacts/model`).

- The GitHub Actions workflow (`.github/workflows/ci.yml`) runs `ruff` and `pytest`, then builds the
  image and pushes it to GitHub Container Registry on every push to `main`. Both the `test` and the
  `docker` jobs passed.
- A local build on my machine failed because of a slow, unstable network (pip hash error on a
  corrupted download), so the container was **not run end to end with the trained model**. The API
  itself was verified locally without Docker (see the API section).

## Not done (out of scope for this submission)

- 12-layer to 6-layer distillation (code in `src/optimization/distill.py`, never run), so there is no
  distilled row in the benchmark.
- TensorRT FP16 and Triton (script only, never run).
- BentoML service and Locust load test (code only, never run). Batch scoring script not run.
- DVC data versioning, Evidently drift report, Prometheus and Grafana dashboards, PSI alert, nginx 5/95
  canary. The PSI function and its tests exist (`src/monitoring/psi.py`) but are not connected to the API.

