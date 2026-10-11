"""TF-IDF + LogisticRegression baseline (measured, CPU-only). Run BEFORE any transformer work."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

D = Path("data/processed")


def main() -> dict:
    tr, te = pd.read_parquet(D / "train.parquet"), pd.read_parquet(D / "test.parquet")
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                        max_features=400_000),
        LogisticRegression(max_iter=300, C=3.0),
    )
    pipe.fit(tr["text"], tr["label"])
    pred = pipe.predict(te["text"])
    p, r, _f, _ = precision_recall_fscore_support(te["label"], pred, average="macro", zero_division=0)
    metrics = {"model": "tfidf-char-logreg", "split": "test", "n": len(te),
               "accuracy": accuracy_score(te["label"], pred), "macro_precision": p,
               "macro_recall": r, "macro_f1": f1_score(te["label"], pred, average="macro")}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
