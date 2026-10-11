"""Score >= 1000 samples in batches (T022). Usage: python scripts/batch_score.py [n]"""
import sys
import time

import pandas as pd

from src.models.predictor import Predictor

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
texts = pd.read_parquet("data/processed/test.parquet")["text"].head(n).tolist()
p, t0 = Predictor(), time.perf_counter()
out = [r for i in range(0, len(texts), 64) for r in p.predict(texts[i:i + 64])]
dt = time.perf_counter() - t0
print(f"scored {len(out)} samples in {dt:.2f}s ({len(out) / dt:.1f}/s)")
