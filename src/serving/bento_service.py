"""BentoML batchable service (T020, T022). BentoML >=1.2: batchable API replaces the old Runner."""
import bentoml

from src.models.predictor import Predictor


@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 30})
class ArabicSentiment:
    def __init__(self) -> None:
        self.predictor = Predictor()

    @bentoml.api(batchable=True, max_batch_size=64, max_latency_ms=50)
    def classify(self, texts: list[str]) -> list[dict]:
        return [p.__dict__ for p in self.predictor.predict(texts)]
