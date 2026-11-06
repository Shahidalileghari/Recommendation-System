"""Optional demo API. Replace the in-memory demo model with a persisted model in production."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from recsys_engineering.recommenders import PopularityRecommender

app = FastAPI(title="Recommendation & Ranking Demo API", version="0.1.0")

_demo = pd.DataFrame([
    {"user_id": "u1", "item_id": "item-a"},
    {"user_id": "u1", "item_id": "item-b"},
    {"user_id": "u2", "item_id": "item-a"},
    {"user_id": "u2", "item_id": "item-c"},
    {"user_id": "u3", "item_id": "item-b"},
    {"user_id": "u3", "item_id": "item-c"},
])
_model = PopularityRecommender().fit(_demo)


class RecommendationRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    k: int = Field(default=5, ge=1, le=50)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        items = _model.recommend(request.user_id, request.k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"user_id": request.user_id, "items": items, "model": "popularity-demo"}
