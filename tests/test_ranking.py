import numpy as np
import pandas as pd
import pytest
from recsys_engineering.ranking import build_user_item_features, pairwise_logistic_loss


def test_feature_builder_and_pairwise_loss():
    interactions = pd.DataFrame([
        {"user_id": "u1", "item_id": "a"},
        {"user_id": "u1", "item_id": "b"},
        {"user_id": "u2", "item_id": "a"},
    ])
    pairs = pd.DataFrame([{"user_id": "u1", "item_id": "a"},
                          {"user_id": "u1", "item_id": "c"}])
    features = build_user_item_features(interactions, pairs)
    assert features.loc[0, "already_seen"] == 1
    assert features.loc[1, "item_popularity"] == 0
    assert pairwise_logistic_loss([2, 3], [0, 1]) < pairwise_logistic_loss([0, 1], [2, 3])


def test_pairwise_loss_shape_validation():
    with pytest.raises(ValueError):
        pairwise_logistic_loss(np.array([1, 2]), np.array([1]))
