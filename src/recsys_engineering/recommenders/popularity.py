from __future__ import annotations
import pandas as pd


class PopularityRecommender:
    """Recommend globally popular items, excluding items already seen by user."""
    def __init__(self):
        self.item_scores = pd.Series(dtype=float)
        self.user_seen = {}

    def fit(self, interactions: pd.DataFrame):
        self.item_scores = interactions.groupby("item_id")["user_id"].nunique().sort_values(
            ascending=False, kind="stable"
        )
        self.user_seen = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
        return self

    def recommend(self, user_id, k: int = 10, candidates=None):
        if k < 1:
            raise ValueError("k must be >= 1.")
        pool = self.item_scores.index if candidates is None else candidates
        seen = self.user_seen.get(user_id, set())
        ranked = [item for item in pool if item not in seen and item in self.item_scores.index]
        ranked.sort(key=lambda item: (-float(self.item_scores[item]), str(item)))
        return ranked[:k]
