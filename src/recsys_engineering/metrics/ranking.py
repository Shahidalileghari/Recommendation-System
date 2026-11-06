from __future__ import annotations
import numpy as np


def _relevance(recommended, relevant):
    relevant = set(relevant)
    return np.asarray([1.0 if item in relevant else 0.0 for item in recommended])


def precision_at_k(recommended, relevant, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1.")
    return float(_relevance(list(recommended)[:k], relevant).sum() / k)


def recall_at_k(recommended, relevant, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1.")
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return float(_relevance(list(recommended)[:k], relevant).sum() / len(relevant))


def average_precision_at_k(recommended, relevant, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1.")
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for rank, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return float(total / min(len(relevant), k))


def ndcg_at_k(recommended, relevant, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1.")
    rel = _relevance(list(recommended)[:k], relevant)
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float(np.sum(rel * discounts))
    ideal_count = min(len(set(relevant)), k)
    if ideal_count == 0:
        return 0.0
    ideal = float(np.sum(1.0 / np.log2(np.arange(2, ideal_count + 2))))
    return dcg / ideal


def mrr(recommended, relevant) -> float:
    relevant = set(relevant)
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
