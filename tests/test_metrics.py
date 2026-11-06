import pytest
from recsys_engineering.metrics import (
    precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k, mrr,
    catalog_coverage,
)


def test_topk_metrics():
    recs, relevant = ["a", "x", "b"], {"a", "b"}
    assert precision_at_k(recs, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(recs, relevant, 3) == pytest.approx(1.0)
    assert average_precision_at_k(recs, relevant, 3) == pytest.approx((1 + 2/3) / 2)
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert mrr(["x", "b"], {"b"}) == pytest.approx(.5)


def test_catalog_coverage():
    assert catalog_coverage([["a", "b"], ["b"]], ["a", "b", "c", "d"]) == .5
