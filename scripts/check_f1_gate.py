"""CI gate (T019): fail if macro-F1 of the candidate is below the measured baseline."""
import json
import sys
from pathlib import Path

base = json.loads(Path("artifacts/baseline_metrics.json").read_text())["macro_f1"]
cand_p = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/metrics.json")
cand = json.loads(cand_p.read_text()).get("test_macro_f1", json.loads(cand_p.read_text()).get("macro_f1"))
print(f"baseline={base:.4f} candidate={cand:.4f}")
sys.exit(0 if cand >= base else 1)
