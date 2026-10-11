import json
from pathlib import Path

import pytest

M = Path("data/processed/manifest.json")


@pytest.mark.skipif(not M.exists(), reason="run `python -m src.data.prepare` first")
def test_splits_are_disjoint_and_stratified():
    import pandas as pd
    d = Path("data/processed")
    s = {k: pd.read_parquet(d / f"{k}.parquet") for k in ("train", "validation", "test")}
    assert not set(s["train"].text) & set(s["test"].text)
    assert not set(s["train"].text) & set(s["validation"].text)
    assert not set(s["validation"].text) & set(s["test"].text)
    for df in s.values():
        shares = df.label.value_counts(normalize=True)
        assert shares.max() - shares.min() < 0.03
    assert json.loads(M.read_text())["seed"] == 42
