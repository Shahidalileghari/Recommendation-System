from __future__ import annotations
import argparse
import pandas as pd
from recsys_engineering.data import validate_interactions, temporal_split
from recsys_engineering.recommenders import PopularityRecommender, ItemItemRecommender
from recsys_engineering.metrics import precision_at_k, recall_at_k, ndcg_at_k


def synthetic_data():
    rows = []
    events = [
        ("u1", "A", "2025-01-01"), ("u1", "B", "2025-01-02"),
        ("u1", "C", "2025-01-08"), ("u2", "A", "2025-01-01"),
        ("u2", "C", "2025-01-03"), ("u2", "D", "2025-01-09"),
        ("u3", "B", "2025-01-02"), ("u3", "C", "2025-01-04"),
        ("u3", "E", "2025-01-10"), ("u4", "A", "2025-01-03"),
        ("u4", "D", "2025-01-05"), ("u4", "E", "2025-01-11"),
    ]
    for user, item, timestamp in events:
        rows.append({"user_id": user, "item_id": item, "timestamp": timestamp})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Recommendation & Ranking Systems CLI")
    parser.add_argument("command", choices=["demo"])
    args = parser.parse_args()
    if args.command == "demo":
        data = validate_interactions(synthetic_data())
        train, validation, test = temporal_split(data, 0.65, 0.15)
        model = PopularityRecommender().fit(train)
        print(f"events: total={len(data)}, train={len(train)}, validation={len(validation)}, test={len(test)}")
        users = sorted(test.user_id.unique())
        results = []
        for user in users:
            relevant = set(test.loc[test.user_id == user, "item_id"])
            recs = model.recommend(user, k=3)
            results.append((user, recs, precision_at_k(recs, relevant, 3),
                            recall_at_k(recs, relevant, 3), ndcg_at_k(recs, relevant, 3)))
        for row in results:
            print(f"user={row[0]} recommendations={row[1]} P@3={row[2]:.3f} "
                  f"R@3={row[3]:.3f} NDCG@3={row[4]:.3f}")
        print("Note: tiny synthetic data is for workflow demonstration, not model benchmarking.")


if __name__ == "__main__":
    main()
