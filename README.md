# Recommendation & Ranking Systems Engineering

A practical engineering repository for building, evaluating, and serving recommendation and ranking systems. It moves from simple baselines to candidate generation, feature-based ranking, offline evaluation, and a lightweight API.

The project is designed to be CPU-first and reproducible. It uses transparent NumPy/Pandas/scikit-learn components; optional neural recommenders are an extension, not a mandatory dependency.

## What this repository covers

- Recommendation problem framing: explicit vs implicit feedback, user-item interactions, cold start.
- Popularity and item-item collaborative filtering baselines.
- Train/validation/test splits that respect interaction time.
- Candidate generation and ranking as separate system stages.
- Feature engineering for user-item ranking.
- Pairwise ranking objective and a scikit-learn pointwise baseline.
- Offline metrics: Precision@K, Recall@K, MAP@K, NDCG@K, MRR, coverage, and catalog diversity.
- A FastAPI recommendation endpoint and command-line demo.
- Experiment configuration, unit tests, Docker, and GitHub Actions CI.

## System architecture

```text
Interaction events + catalog metadata
                |
                v
       Validation / cleaning
                |
                v
       Temporal data splitting
                |
                v
       Candidate generation
       (popular / item-item)
                |
                v
       Feature construction
                |
                v
       Ranking model / baseline
                |
                v
       Top-K filtering and serving
                |
                v
       Offline evaluation + monitoring
```

**Key distinction:** candidate generation finds a manageable set of potentially relevant items; ranking orders those candidates for a user. Evaluating only the ranking stage can hide candidate-recall failures.

## Quick start

Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

Run a synthetic end-to-end baseline:

```bash
recsys-lab demo
```

Run the API (install the API extra first):

```bash
pip install -e ".[api]"
uvicorn recsys_engineering.api:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

## Data format

The interaction CSV expects:

| Column | Meaning |
|---|---|
| `user_id` | Stable user identifier |
| `item_id` | Stable item identifier |
| `timestamp` | Event timestamp parseable by pandas |
| `event_type` | Optional event name, e.g. view, click, purchase |
| `rating` | Optional explicit rating |

The catalog CSV is optional and should contain at least `item_id`; extra fields can include category, title, price band, or availability. Never include raw personal data in sample datasets.

## Evaluation

For a user \(u\), let \(R_u\) be relevant held-out items and \(L_u^K\) the recommended top-K list.

- Precision@K = relevant recommendations / K
- Recall@K = relevant recommendations / relevant held-out items
- NDCG@K rewards relevant items appearing near the top.
- MAP@K averages precision at each relevant rank.
- MRR uses the reciprocal rank of the first relevant result.
- Coverage measures the fraction of catalog items ever recommended.

Use a time-based split for event data whenever the product predicts future behavior. Avoid random interaction splits when they allow future events to leak into training.

## Repository structure

```text
src/recsys_engineering/
  data/          validation and temporal splitting
  recommenders/  popularity and item-item candidate generation
  ranking/       feature construction and pairwise loss
  metrics/       top-K relevance and catalog metrics
  api.py         optional FastAPI serving endpoint
  cli.py         synthetic demo
examples/        end-to-end and evaluation examples
configs/         experiment configuration
docs/            architecture, data, evaluation, deployment, pitfalls
tests/           unit tests
```

## GitHub repository description

**End-to-end recommendation and learning-to-rank engineering lab: candidate generation, collaborative filtering, ranking features, top-K metrics, temporal evaluation, and API serving.**

## Important limitations

This repository is a strong learning and baseline foundation, not a ready-made large-scale production recommender. It does not include a real user dataset, online experimentation, distributed retrieval, a feature store, or production-grade personalization/privacy controls. Item-item similarity is computed with a sparse interaction matrix and is intended for small/medium experiments. See `docs/ROADMAP.md`.

## License

MIT. See `LICENSE`.
