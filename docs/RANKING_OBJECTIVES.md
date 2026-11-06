# Ranking objectives

## Pointwise

Predict a target for each user-item pair (e.g. click probability). Easy to train with standard classification/regression tooling, but pairwise ordering is indirect.

## Pairwise

Learn that a positive item should score above a sampled negative. The helper implements:

`loss = mean(log(1 + exp(-(score_positive - score_negative))))`

Negative sampling strategy strongly influences what the model learns.

## Listwise

Optimize a whole ranked list or a metric surrogate. Listwise methods can align training more closely with ranking metrics but require careful batching and objective design.

## Practical sequence

Start with popularity and item-item baselines, establish a temporal evaluation protocol, then add a pointwise model. Move to pairwise/listwise training only when the data and evaluation setup justify the extra complexity.
