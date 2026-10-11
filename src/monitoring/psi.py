"""Population Stability Index (T028). PSI > 0.25 => significant drift (T031)."""
from __future__ import annotations

import numpy as np

PSI_ALERT_THRESHOLD = 0.25


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_summary(ref: dict, cur: dict) -> dict:
    """ref/cur: {"text_length": [...], "confidence_score": [...]}"""
    out = {k: psi(ref[k], cur[k]) for k in ("text_length", "confidence_score")}
    out["alert"] = any(v > PSI_ALERT_THRESHOLD for v in out.values())
    return out
