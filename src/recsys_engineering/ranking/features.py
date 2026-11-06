from __future__ import annotations
import pandas as pd


def build_user_item_features(interactions: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Build simple popularity/affinity features for supplied (user_id,item_id) pairs."""
    required = {"user_id", "item_id"}
    if not required.issubset(pairs.columns):
        raise ValueError("pairs must contain user_id and item_id.")
    user_counts = interactions.groupby("user_id")["item_id"].nunique()
    item_counts = interactions.groupby("item_id")["user_id"].nunique()
    user_items = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    rows = pairs[["user_id", "item_id"]].copy()
    rows["user_activity"] = rows.user_id.map(user_counts).fillna(0).astype(float)
    rows["item_popularity"] = rows.item_id.map(item_counts).fillna(0).astype(float)
    rows["already_seen"] = [
        float(item in user_items.get(user, set()))
        for user, item in zip(rows.user_id, rows.item_id)
    ]
    rows["user_activity_log"] = __import__("numpy").log1p(rows["user_activity"])
    rows["item_popularity_log"] = __import__("numpy").log1p(rows["item_popularity"])
    return rows
