from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src.models.predictor import Predictor

app = FastAPI(title="Arabic Sentiment Analysis API", version="0.1.0")
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    p = get_predictor().predict([request.text])[0]
    return PredictResponse(label=p.label, confidence=p.confidence, model_version=p.model_version)
