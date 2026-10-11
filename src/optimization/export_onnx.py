"""Export to ONNX and dynamic-quantize to INT8 (T025). Usage: python -m ... <model_dir> <out_dir>"""
import sys
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

src, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForSequenceClassification.from_pretrained(src).eval()
enc = tok(["مثال"], return_tensors="pt", padding="max_length", max_length=128, truncation=True)
args = (enc["input_ids"], enc["attention_mask"])
torch.onnx.export(model, args, out / "model.onnx", input_names=["input_ids", "attention_mask"],
                  output_names=["logits"], opset_version=17,
                  dynamic_axes={k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask", "logits")})
quantize_dynamic(out / "model.onnx", out / "model.int8.onnx", weight_type=QuantType.QInt8)
tok.save_pretrained(out)
print("wrote", out / "model.onnx", out / "model.int8.onnx")
