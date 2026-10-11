"""Evidently HTML drift report (T027). Needs `.[ops]`."""
import sys

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

ref, cur = pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2])
cols = ["text_length", "confidence_score"]
rep = Report(metrics=[DataDriftPreset(columns=cols, stattest="psi", stattest_threshold=0.25)])
rep.run(reference_data=ref[cols], current_data=cur[cols])
rep.save_html("artifacts/drift_report.html")
