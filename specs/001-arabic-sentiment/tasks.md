# Task List

## Foundation
- [x] T001 Decide and document dataset source.
- [x] T002 Verify positive/negative/neutral labels.
- [ ] T003 Set up DVC data directories. (run `dvc init && dvc add data/raw/*.parquet`)
- [x] T004 Create deterministic train/validation/test split.
- [x] T005 Implement preprocessing.

## Model
- [ ] T006 Load AraBERT or CAMeL-BERT.
- [ ] T007 Implement typed model wrapper.
- [ ] T008 Fine-tune classifier.
- [ ] T009 Evaluate accuracy, precision, recall, macro-F1.
- [ ] T010 Track at least 5 MLflow runs and register best model.

## API
- [x] T011 Implement POST /predict.
- [x] T012 Return label, confidence, model_version.
- [x] T013 Reject empty text with 422.
- [x] T014 Implement GET /health.
- [x] T015 Add API tests.

## Deployment
- [ ] T016 Build Docker image and healthcheck.
- [x] T017 Add Docker Compose.
- [ ] T018 Add GitHub Actions lint/test/build/push workflow.
- [ ] T019 Add macro-F1 baseline gate.

## Serving and Optimization
- [ ] T020 Implement batchable BentoML Runner.
- [ ] T021 Run Locust baseline.
- [ ] T022 Support batch scoring for >=1000 samples.
- [ ] T023 Build TensorRT FP16 engine.
- [ ] T024 Distill 12-layer teacher to 6-layer student.
- [ ] T025 Export ONNX and quantize INT8.
- [ ] T026 Benchmark original vs distilled vs INT8.

## Monitoring
- [ ] T027 Generate Evidently drift report.
- [ ] T028 PSI for text_length and confidence_score.
- [ ] T029 Expose PSI to Prometheus.
- [ ] T030 Create Grafana p95 and PSI panels.
- [ ] T031 Add PSI > 0.25 alert.

## Final
- [ ] T032 Document nginx 5/95 canary.
- [ ] T033 Create architecture diagram.
- [ ] T034 Write reproducible README.
- [ ] T035 Add only measured benchmark values.
- [ ] T036 Peer review.

Complete tasks sequentially. Do not optimize before the baseline model and API work correctly.
