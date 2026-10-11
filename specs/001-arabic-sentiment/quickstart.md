# Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m src.data.prepare        # splits -> data/processed
python -m src.models.baseline     # measured CPU baseline -> artifacts/baseline_metrics.json
pytest && ruff check .
uvicorn src.api.main:app --reload
# GPU machine:
pip install -e '.[ml,ops,opt]'
bash scripts/run_mlflow_sweep.sh                 # T008-T010
python -m src.optimization.distill               # T024
python -m src.optimization.export_onnx artifacts/model artifacts/onnx   # T025
python -m src.optimization.benchmark             # T026 -> benchmarks/results.json
```
Serve a trained model: `MODEL_PATH=artifacts/model MODEL_VERSION=v1 uvicorn src.api.main:app`.
