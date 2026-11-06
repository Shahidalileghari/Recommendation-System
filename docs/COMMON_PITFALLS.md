# Common recommendation and ranking pitfalls

1. Random splits can leak future behavior into training.
2. Candidate recall can bottleneck the entire system even if ranking metrics look strong.
3. Popularity baselines are essential; complex models may not beat them.
4. Random negatives may be unobserved rather than truly disliked.
5. Repeat interactions and repeat consumption need explicit product rules.
6. Item cold-start requires content or metadata features.
7. User cold-start needs contextual, onboarding, or popularity fallbacks.
8. Catalog coverage and diversity can collapse under popularity-heavy ranking.
9. Offline metrics are proxies, not proof of user satisfaction or business lift.
10. Position and exposure bias distort click logs; clicks are not unbiased relevance labels.
