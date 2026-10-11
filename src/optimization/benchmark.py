"""Measure accuracy / macro-F1 / p95 latency / size for each model variant (T026, T035).
Writes benchmarks/results.json. Nothing is hard-coded: only measured values are written."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def size_mb(p: Path) -> float:
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


def bench(name, predict_fn, texts, labels, path, warmup=5, n_lat=200):
    for t in texts[:warmup]:
        predict_fn([t])
    lat = []
    for t in texts[:n_lat]:
        s = time.perf_counter(); predict_fn([t]); lat.append((time.perf_counter() - s) * 1000)
    pred = [x for i in range(0, len(texts), 64) for x in predict_fn(texts[i:i + 64])]
    return {"model": name, "accuracy": accuracy_score(labels, pred),
            "macro_f1": f1_score(labels, pred, average="macro"),
            "p95_latency_ms": float(np.percentile(lat, 95)), "size_mb": size_mb(Path(path)),
            "device": "see notes", "n_test": len(texts)}


def torch_fn(path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).eval()
    def f(batch):
        enc = tok(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            return m(**enc).logits.argmax(-1).tolist()
    return f


def onnx_fn(model_file, tok_dir):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    s = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])
    def f(batch):
        enc = tok(batch, truncation=True, max_length=128, padding="max_length", return_tensors="np")
        return s.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})[0].argmax(-1).tolist()
    return f


if __name__ == "__main__":
    te = pd.read_parquet("data/processed/test.parquet")
    texts, labels = te["text"].tolist(), te["label"].tolist()
    results = [
        bench("original", torch_fn("artifacts/model"), texts, labels, "artifacts/model"),
        bench("distilled-6L", torch_fn("artifacts/student"), texts, labels, "artifacts/student"),
        bench("int8-onnx", onnx_fn("artifacts/onnx/model.int8.onnx", "artifacts/onnx"), texts, labels,
              "artifacts/onnx/model.int8.onnx"),
    ]
    Path("benchmarks/results.json").write_text(json.dumps(results, indent=2))
    print("| Model | Accuracy | Macro-F1 | P95 Latency (ms) | Size (MB) |\n|---|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['p95_latency_ms']:.1f} | {r['size_mb']:.1f} |")
