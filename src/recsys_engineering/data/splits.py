from __future__ import annotations
import pandas as pd


def temporal_split(interactions: pd.DataFrame, train_ratio: float = 0.8,
                   validation_ratio: float = 0.1):
    """Global chronological split; keeps all events at a timestamp in one split."""
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("Ratios must be in valid ranges.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be < 1.")
    ordered = interactions.sort_values("timestamp").reset_index(drop=True)
    unique_times = ordered["timestamp"].drop_duplicates().sort_values().tolist()
    if len(unique_times) < 3:
        raise ValueError("At least three distinct timestamps are required.")
    train_end = max(1, min(len(unique_times) - 2, int(len(unique_times) * train_ratio)))
    val_end = max(train_end + 1, min(len(unique_times) - 1,
                                     int(len(unique_times) * (train_ratio + validation_ratio))))
    train_times = set(unique_times[:train_end])
    val_times = set(unique_times[train_end:val_end])
    test_times = set(unique_times[val_end:])
    return (
        ordered[ordered.timestamp.isin(train_times)].copy(),
        ordered[ordered.timestamp.isin(val_times)].copy(),
        ordered[ordered.timestamp.isin(test_times)].copy(),
    )
