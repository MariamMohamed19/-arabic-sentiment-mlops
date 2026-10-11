"""12-layer teacher -> 6-layer student via knowledge distillation (T024). Needs GPU."""
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="artifacts/model")
ap.add_argument("--out", default="artifacts/student")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--alpha", type=float, default=0.5)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.teacher)
teacher = AutoModelForSequenceClassification.from_pretrained(a.teacher).to(dev).eval()
assert teacher.config.num_hidden_layers == 12, "teacher must have 12 layers"
student = AutoModelForSequenceClassification.from_pretrained(a.teacher, num_hidden_layers=6,
                                                             ignore_mismatched_sizes=True)
# init student from every 2nd teacher layer
for i, j in enumerate(range(1, 12, 2)):
    student.bert.encoder.layer[i].load_state_dict(teacher.bert.encoder.layer[j].state_dict())
student.to(dev).train()

df = pd.read_parquet("data/processed/train.parquet")
dl = DataLoader(list(zip(df["text"], df["label"])), batch_size=32, shuffle=True)
opt = torch.optim.AdamW(student.parameters(), lr=5e-5)
for ep in range(a.epochs):
    for texts, y in dl:
        enc = tok(list(texts), truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            t_logits = teacher(**enc).logits
        s_logits = student(**enc).logits
        kd = F.kl_div(F.log_softmax(s_logits / a.T, -1), F.softmax(t_logits / a.T, -1),
                      reduction="batchmean") * a.T**2
        loss = a.alpha * kd + (1 - a.alpha) * F.cross_entropy(s_logits, y.to(dev))
        loss.backward(); opt.step(); opt.zero_grad()
    print(f"epoch {ep} last loss {loss.item():.4f}")
student.save_pretrained(a.out); tok.save_pretrained(a.out)
