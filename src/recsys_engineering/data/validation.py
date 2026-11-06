from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = {"user_id", "item_id", "timestamp"}


def validate_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an interaction table; return a sorted copy."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Interaction table is empty.")
    result = frame.copy()
    if result[["user_id", "item_id", "timestamp"]].isna().any().any():
        raise ValueError("user_id, item_id, and timestamp cannot contain null values.")
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    if "event_type" not in result:
        result["event_type"] = "interaction"
    if "rating" in result:
        result["rating"] = pd.to_numeric(result["rating"], errors="coerce")
    result = result.drop_duplicates(["user_id", "item_id", "timestamp"])
    return result.sort_values("timestamp").reset_index(drop=True)
