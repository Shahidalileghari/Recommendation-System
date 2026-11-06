import pandas as pd
from recsys_engineering.recommenders import PopularityRecommender, ItemItemRecommender


def data():
    return pd.DataFrame([
        {"user_id": "u1", "item_id": "a"},
        {"user_id": "u1", "item_id": "b"},
        {"user_id": "u2", "item_id": "a"},
        {"user_id": "u2", "item_id": "c"},
        {"user_id": "u3", "item_id": "b"},
        {"user_id": "u3", "item_id": "c"},
    ])


def test_popularity_excludes_seen_items():
    model = PopularityRecommender().fit(data())
    assert "a" not in model.recommend("u1", 3)


def test_item_item_returns_unseen_items():
    model = ItemItemRecommender().fit(data())
    recs = model.recommend("u1", 3)
    assert "a" not in recs and "b" not in recs
    assert "c" in recs
