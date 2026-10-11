"""Deterministic data preparation (T002, T004, T005).

Source file columns: text, label (0/1/2), label_text (Negative/Positive/Mixed).
Decision (documented in specs/.../research.md): "Mixed" is treated as "neutral".
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
RAW = Path("data/raw/train-00000-of-00001.parquet")
OUT = Path("data/processed")
LABELS = ["negative", "positive", "neutral"]  # id order == source `label` column
SOURCE_TEXT_TO_NAME = {"Negative": "negative", "Positive": "positive", "Mixed": "neutral"}

_TASHKEEL = re.compile(r"[\u064B-\u0652\u0670]")
_TATWEEL = "\u0640"
_REPEAT = re.compile(r"(.)\1{2,}")
_SPACES = re.compile(r"\s+")
_ARABIC = re.compile(r"[\u0600-\u06FF]")


def clean_text(text: str) -> str:
    """Light normalization. Model-specific segmentation (AraBERT prep) happens at train time."""
    text = _TASHKEEL.sub("", text).replace(_TATWEEL, "")
    text = _REPEAT.sub(r"\1\1", text)  # "جميييييل" -> "جميل"
    return _SPACES.sub(" ", text).strip()


def verify_labels(df: pd.DataFrame) -> None:
    pairs = set(zip(df["label"], df["label_text"]))
    expected = {(0, "Negative"), (1, "Positive"), (2, "Mixed")}
    if pairs != expected:
        raise ValueError(f"Unexpected label mapping: {pairs}")


def prepare(raw: Path = RAW, out: Path = OUT) -> dict:
    df = pd.read_parquet(raw)
    verify_labels(df)
    df["text"] = df["text"].astype(str).map(clean_text)
    before = len(df)
    has_arabic = df["text"].map(lambda t: bool(_ARABIC.search(t)))
    df = df[has_arabic & (df["text"].str.split().str.len() >= 2)]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["label_name"] = df["label_text"].map(SOURCE_TEXT_TO_NAME)
    df["text_length"] = df["text"].str.split().str.len()

    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "rows_before": before, "rows_after": len(df), "labels": LABELS, "splits": {}}
    for name, part in {"train": train, "validation": val, "test": test}.items():
        part.reset_index(drop=True).to_parquet(out / f"{name}.parquet", index=False)
        manifest["splits"][name] = {"rows": len(part),
                                    "label_counts": part["label_name"].value_counts().to_dict()}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(*(Path(a) for a in sys.argv[1:])), ensure_ascii=False, indent=2))
