# Demo deployment

Install the API dependencies and run:

```bash
pip install -e ".[api]"
uvicorn recsys_engineering.api:app --host 127.0.0.1 --port 8000
```

`GET /health` checks process health. `POST /recommend` accepts a user ID and K. The included API deliberately uses a tiny in-memory popularity model to demonstrate request/response boundaries.

Before production use, add authentication/authorization where needed, request and response logging with privacy controls, model versioning, catalog availability checks, caching, timeouts, rate limiting, observability, and a persisted model-loading strategy. Do not expose internal user histories or sensitive features in API responses.
