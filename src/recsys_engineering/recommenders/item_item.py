from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemRecommender:
    """Item-item cosine collaborative filtering over a user-item interaction matrix."""
    def __init__(self, min_similarity: float = 0.0):
        self.min_similarity = min_similarity
        self.similarity = None
        self.item_ids = []
        self.user_seen = {}
        self.item_popularity = {}

    def fit(self, interactions: pd.DataFrame):
        matrix = interactions.assign(value=1.0).pivot_table(
            index="user_id", columns="item_id", values="value", aggfunc="max", fill_value=0.0
        )
        self.item_ids = list(matrix.columns)
        self.similarity = cosine_similarity(matrix.T)
        self.user_seen = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
        self.item_popularity = interactions.groupby("item_id")["user_id"].nunique().to_dict()
        return self

    def recommend(self, user_id, k: int = 10):
        if self.similarity is None:
            raise RuntimeError("Call fit() before recommend().")
        if k < 1:
            raise ValueError("k must be >= 1.")
        seen = self.user_seen.get(user_id, set())
        score = defaultdict(float)
        index = {item: i for i, item in enumerate(self.item_ids)}
        for item in seen:
            if item not in index:
                continue
            row = self.similarity[index[item]]
            for j, candidate in enumerate(self.item_ids):
                if candidate not in seen and row[j] > self.min_similarity:
                    score[candidate] += float(row[j])
        ranked = sorted(score, key=lambda item: (-score[item], -self.item_popularity.get(item, 0),
                                                 str(item)))
        return ranked[:k]
