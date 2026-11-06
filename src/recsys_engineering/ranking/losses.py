from __future__ import annotations
import numpy as np


def pairwise_logistic_loss(positive_scores, negative_scores) -> float:
    """Mean -log(sigmoid(pos - neg)), implemented stably with logaddexp."""
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    if pos.shape != neg.shape or pos.size == 0:
        raise ValueError("Positive and negative score arrays must have equal non-empty shapes.")
    return float(np.mean(np.logaddexp(0.0, -(pos - neg))))
