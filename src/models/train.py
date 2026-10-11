"""Fine-tune AraBERT / CAMeL-BERT with MLflow tracking (T006, T008-T010). Needs GPU + `.[ml,ops]`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

LABELS = ["negative", "positive", "neutral"]
MODELS = {"arabert": "aubmindlab/bert-base-arabertv02", "camelbert": "CAMeL-Lab/bert-base-arabic-camelbert-msa"}


def compute_metrics(eval_pred):
    logits, y = eval_pred
    pred = np.argmax(logits, -1)
    p, r, _f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "macro_precision": p, "macro_recall": r,
            "macro_f1": f1_score(y, pred, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="arabert")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--train-subset", type=int, default=0, help="0 = all rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/model")
    a = ap.parse_args()

    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    set_seed(a.seed)
    d = Path("data/processed")
    tr, va, te = (pd.read_parquet(d / f"{s}.parquet") for s in ("train", "validation", "test"))
    if a.train_subset:
        tr = tr.sample(a.train_subset, random_state=a.seed)
    name = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name)
    if a.model == "arabert":  # AraBERT v2 expects its own preprocessing
        from arabert.preprocess import ArabertPreprocessor
        prep = ArabertPreprocessor(model_name=name).preprocess
    else:
        prep = lambda s: s

    def to_ds(df):
        ds = Dataset.from_dict({"text": [prep(t) for t in df["text"]], "label": df["label"].tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_length), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    args = TrainingArguments(output_dir=a.out, learning_rate=a.lr, num_train_epochs=a.epochs,
                             per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=64,
                             eval_strategy="epoch", save_strategy="no", fp16=True, seed=a.seed,
                             report_to=[], logging_steps=100)
    trainer = Trainer(model=model, args=args, train_dataset=to_ds(tr), eval_dataset=to_ds(va),
                      processing_class=tok, compute_metrics=compute_metrics)
    mlflow.set_experiment("arabic-sentiment")
    with mlflow.start_run(run_name=f"{a.model}-lr{a.lr}-bs{a.batch_size}-ep{a.epochs}"):
        mlflow.log_params(vars(a))
        trainer.train()
        val = trainer.evaluate()
        test = trainer.evaluate(to_ds(te), metric_key_prefix="test")
        mlflow.log_metrics({k: v for k, v in {**val, **test}.items() if isinstance(v, float)})
        trainer.save_model(a.out)
        tok.save_pretrained(a.out)
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/metrics.json").write_text(json.dumps(test, indent=2))
        mlflow.log_artifacts(a.out, artifact_path="model")


if __name__ == "__main__":
    main()
