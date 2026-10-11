#!/usr/bin/env bash
# TensorRT FP16 engine (T023). Requires an NVIDIA GPU + TensorRT (trtexec).
set -euo pipefail
trtexec --onnx=artifacts/onnx/model.onnx --saveEngine=artifacts/model_fp16.engine --fp16 \
  --minShapes=input_ids:1x128,attention_mask:1x128 \
  --optShapes=input_ids:16x128,attention_mask:16x128 \
  --maxShapes=input_ids:64x128,attention_mask:64x128
