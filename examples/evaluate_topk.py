from recsys_engineering.metrics import (
    precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k, mrr
)

recommended = ["book-7", "book-2", "book-9", "book-4"]
relevant = {"book-2", "book-4"}
k = 3
print({
    "precision@k": precision_at_k(recommended, relevant, k),
    "recall@k": recall_at_k(recommended, relevant, k),
    "map@k": average_precision_at_k(recommended, relevant, k),
    "ndcg@k": ndcg_at_k(recommended, relevant, k),
    "mrr": mrr(recommended, relevant),
})
