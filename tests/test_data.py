import pandas as pd
import pytest
from recsys_engineering.data import validate_interactions, temporal_split


def test_validation_sorts_and_deduplicates():
    frame = pd.DataFrame([
        {"user_id": "u", "item_id": "b", "timestamp": "2025-01-02"},
        {"user_id": "u", "item_id": "a", "timestamp": "2025-01-01"},
        {"user_id": "u", "item_id": "a", "timestamp": "2025-01-01"},
    ])
    result = validate_interactions(frame)
    assert len(result) == 2
    assert result.timestamp.is_monotonic_increasing


def test_temporal_split_is_disjoint():
    frame = pd.DataFrame([
        {"user_id": "u", "item_id": str(i), "timestamp": f"2025-01-{i+1:02d}"}
        for i in range(6)
    ])
    train, val, test = temporal_split(frame, .5, .25)
    assert len(train) and len(val) and len(test)
    assert train.timestamp.max() < val.timestamp.min() < test.timestamp.min()


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        validate_interactions(pd.DataFrame({"user_id": ["u"]}))
