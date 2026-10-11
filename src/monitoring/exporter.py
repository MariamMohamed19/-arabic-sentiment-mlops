"""Expose PSI gauges to Prometheus (T029). Run: python -m src.monitoring.exporter"""
import time

from prometheus_client import Gauge, start_http_server

from src.monitoring.psi import drift_summary

G = {k: Gauge(f"psi_{k}", f"PSI of {k}") for k in ("text_length", "confidence_score")}


def update(ref: dict, cur: dict) -> dict:
    s = drift_summary(ref, cur)
    for k, g in G.items():
        g.set(s[k])
    return s


if __name__ == "__main__":
    start_http_server(9100)
    while True:  # TODO: load reference (test split) + recent API logs, then call update()
        time.sleep(60)
