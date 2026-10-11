"""Typed model wrapper (T007). Falls back to a clearly-labelled placeholder if no model is set."""
from __future__ import annotations

import os
from dataclasses import dataclass

LABELS = ["negative", "positive", "neutral"]


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    model_version: str


class Predictor:
    def __init__(self, model_path: str | None = None, version: str | None = None):
        self.model_path = model_path or os.getenv("MODEL_PATH") or None
        self.version = version or os.getenv("MODEL_VERSION", "dev")
        self._tok = self._model = None
        if self.model_path:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path).eval()

    def predict(self, texts: list[str]) -> list[Prediction]:
        if self._model is None:
            return [Prediction("neutral", 0.0, "dev-placeholder") for _ in texts]
        enc = self._tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
        with self._torch.no_grad():
            probs = self._model(**enc).logits.softmax(-1)
        conf, idx = probs.max(-1)
        return [Prediction(LABELS[i], float(c), self.version) for i, c in zip(idx.tolist(), conf.tolist())]
