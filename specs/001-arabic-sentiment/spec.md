# Feature Specification: Arabic Sentiment Analysis

## Goal
Classify Arabic product reviews into positive, negative, or neutral using AraBERT/CAMeL-BERT and expose the model as a production-oriented API.

## Requirements
- Labeled Arabic sentiment dataset; the handbook does not specify an official source. Chosen: `data/raw/train-00000-of-00001.parquet` (see research.md). Source label "Mixed" is mapped to "neutral".
- AraBERT or CAMeL-BERT.
- FastAPI /predict and /health.
- Empty text returns HTTP 422.
- MLflow with at least 5 runs.
- DVC data versioning.
- GitHub Actions lint, test, Docker build/push.
- BentoML batchable Runner and Locust benchmark.
- TensorRT FP16.
- 12-layer teacher distilled to 6-layer student.
- INT8 ONNX.
- Evidently PSI on text_length and confidence_score.
- Prometheus + Grafana; alert PSI > 0.25.
- Benchmark original vs distilled vs INT8 using accuracy, p95 latency, and size.
