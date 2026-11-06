# Data modeling and splitting

## Explicit vs implicit feedback

- Explicit: ratings or direct preference labels.
- Implicit: clicks, views, saves, purchases, dwell events. An observed event is not automatically a positive preference; event semantics and weights matter.

## Temporal evaluation

When recommending future items, train on earlier events and evaluate on later events. The supplied splitter uses global timestamp boundaries so events sharing a timestamp remain in the same split. This is a simple baseline protocol; production systems may need per-user temporal holdouts, session boundaries, delayed labels, and rolling evaluation windows.

## Leakage checklist

- Do not fit popularity or item similarity on validation/test interactions.
- Do not use future item counts or future user activity as features.
- Keep held-out interactions out of training history.
- Define how repeat consumption and already-seen items are handled.
- Avoid using post-outcome fields (e.g. purchase completion) in pre-click ranking features.
