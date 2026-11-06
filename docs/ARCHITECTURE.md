# Architecture: retrieval then ranking

A typical recommender is a multi-stage system:

1. **Event and catalog ingestion:** validate identifiers, timestamps, event semantics, and item availability.
2. **Candidate generation:** retrieve hundreds or thousands of plausible items from popularity, collaborative filtering, content similarity, or graph retrieval.
3. **Feature construction:** user history, item popularity, item metadata, recency, context, and candidate-source features.
4. **Ranking:** score candidates with a pointwise, pairwise, or listwise objective.
5. **Post-processing:** remove seen/unavailable items, apply business constraints, diversify, and cap repetition.
6. **Serving and feedback:** return ranked results and log impressions, positions, clicks, and downstream outcomes.

The included project implements popularity and item-item candidate generation, simple tabular ranking features, a pairwise logistic loss helper, top-K metrics, and a demo API. A trained production ranker and distributed retrieval service are intentionally out of scope.
