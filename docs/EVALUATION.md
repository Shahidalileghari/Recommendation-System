# Offline evaluation

For each user, compare a ranked list with held-out relevant items. Average metrics across eligible users and document whether users with no held-out positives are excluded.

| Metric | Main question |
|---|---|
| Precision@K | How many top-K results are relevant? |
| Recall@K | How much of the relevant set was retrieved? |
| MAP@K | Are relevant results consistently ranked early? |
| NDCG@K | Are high-ranked relevant results rewarded more? |
| MRR | How early is the first relevant result? |
| Catalog coverage | How much of the catalog is surfaced? |

Also consider novelty, diversity, calibration, latency, availability, and business outcomes. Offline metrics do not prove online impact. A/B testing or carefully designed online experiments are needed to assess causal product impact.
