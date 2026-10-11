import numpy as np

from src.monitoring.psi import drift_summary, psi


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.05


def test_psi_shifted_distribution_alerts():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(2, 1, size=5000)) > 0.25


def test_drift_summary_alert_flag():
    rng = np.random.default_rng(1)
    ref = {"text_length": rng.normal(50, 10, 3000), "confidence_score": rng.uniform(0.5, 1, 3000)}
    cur = {"text_length": rng.normal(120, 10, 3000), "confidence_score": rng.uniform(0.5, 1, 3000)}
    s = drift_summary(ref, cur)
    assert s["alert"] and s["text_length"] > 0.25
