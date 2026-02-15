# Movie Recommendation Engine — Team 05

A production-grade hybrid recommendation system that serves personalized movie recommendations via a REST API, backed by a real-time data pipeline consuming from Apache Kafka.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Kafka (128.2.220.241:9092)                        │
│                         Topic: movielog5                                  │
│  Events: ratings (1-10), watch minutes, new accounts, rec requests        │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  pipeline/ingestion/consumer.py                                         │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐                    │
│  │ parser.py  │→ │validators.py │→ │ postgres.py  │                    │
│  │ regex      │  │ range checks │  │ upsert       │                    │
│  │ extraction │  │ drift detect │  │ operations   │                    │
│  └────────────┘  └──────────────┘  └──────┬───────┘                    │
│                                           │                             │
│  On unknown movie/user:                   │  ┌──────────────┐          │
│  api_client.py ──────────────────────────►│  │ Course API   │          │
│  GET /movie/<slug>, GET /user/<id>        │  │ :8080        │          │
└───────────────────────────────────────────┼──└──────────────┘──────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PostgreSQL                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌────────┐  ┌───────┐                │
│  │ ratings  │  │ watch_events │  │ movies │  │ users │                │
│  │ (uid,mid,│  │ (uid,mid,    │  │ (mid,  │  │ (uid, │                │
│  │  rating, │  │  minutes)    │  │  title,│  │  age, │                │
│  │  ts)     │  │              │  │  genres│  │  likes│                │
│  │ PK(u,m)  │  │ PK(u,m)     │  │  cost, │  │  etc) │                │
│  └──────────┘  └──────────────┘  │  JSONB)│  └───────┘                │
│                                   └────────┘                            │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ features.py     │  │ svd.py               │  │ cold_start.py        │
│ TF-IDF on       │  │ scikit-surprise SVD  │  │ TF-IDF match user    │
│ genres+overview  │  │ 50 latent factors    │  │ self_description     │
│ +keywords       │  │ Rating scale: 1-10   │  │ against movie content│
│                 │  │                      │  │                      │
│ Cosine sim      │  │ Extracts:            │  │ Fallback: genre      │
│ top-50/movie    │  │ user_factors (N×50)  │  │ keyword matching     │
│                 │  │ item_factors (M×50)  │  │ from likes text      │
│ Saves:          │  │ user/item biases     │  │                      │
│ content_data.pkl│  │ global_mean          │  │ Writes to Redis      │
│ tfidf_matrix.npz│  │                      │  │ recs:user:<id>       │
│ tfidf_vec.pkl   │  │ Saves: model.pkl     │  │                      │
└────────┬────────┘  └──────────┬───────────┘  └──────────┬───────────┘
         │                      │                          │
         └──────────┬───────────┘                          │
                    ▼                                      │
         ┌──────────────────────┐                          │
         │ hybrid.py            │                          │
         │                      │                          │
         │ For each user:       │                          │
         │ score = α·SVD +      │                          │
         │        (1-α)·content │                          │
         │                      │                          │
         │ α = 0.7 (default)    │                          │
         │ Mask rated items     │                          │
         │ Top-20 via argpart   │                          │
         │                      │                          │
         │ Batch write to Redis │◄─────────────────────────┘
         │ Pipeline: 500/batch  │
         └──────────┬───────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Redis                                                                   │
│  recs:user:<id>    → "movie+slug+year,movie+slug+year,..." (TTL 24h)   │
│  recs:popular      → top 100 by avg rating (min 5 ratings)             │
│  model:version     → unix timestamp of last training run                │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  service/app/recommender.py                                              │
│                                                                          │
│  Prediction cascade (first hit wins):                                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Tier 1: Redis cache lookup        recs:user:<id>               │    │
│  │ Tier 2: Real-time SVD scoring     model.pkl in memory          │    │
│  │ Tier 3: Content-based fallback    content_data.pkl             │    │
│  │ Tier 4: Popularity from Redis     recs:popular                 │    │
│  │ Tier 5: Hardcoded defaults        20 classic films             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Hot-reload: checks model:version on each request, reloads if changed   │
│                                                                          │
│  GET /recommend/<int:userid>  →  text/plain CSV of up to 20 movie IDs  │
│  GET /health                  →  {"status": "healthy"}                  │
│  GET /                        →  {"message": "..."}                     │
│                                                                          │
│  Served via Gunicorn (4 workers, 2s timeout) on port 8082               │
└──────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
movie-rec-app/
├── docker-compose.yml              # Orchestrates all services
├── .env                            # Environment variables (gitignored)
├── .env.example                    # Template
├── .gitignore
├── .github/workflows/ci.yaml       # Lint → Test → Docker build
│
├── service/                         # Flask recommendation API
│   ├── Dockerfile                   # python:3.11-slim + gunicorn
│   ├── requirements.txt             # Flask, gunicorn, redis, numpy, scipy
│   ├── requirements-dev.txt         # + ruff, pytest, pytest-cov
│   ├── pyproject.toml               # Ruff + pytest config
│   ├── app/
│   │   ├── __init__.py              # Flask factory, create_app()
│   │   ├── __main__.py              # python -m app entry point
│   │   ├── config.py                # REDIS_HOST, REDIS_PORT, MODEL_DIR
│   │   ├── routes.py                # Blueprint: /recommend, /health, /
│   │   └── recommender.py           # 5-tier prediction cascade
│   └── tests/
│       ├── conftest.py              # Flask test client fixture
│       └── test_routes.py           # Endpoint tests
│
├── pipeline/                        # Data pipeline
│   ├── Dockerfile                   # python:3.11-slim + gcc (for surprise)
│   ├── requirements.txt             # surprise, sklearn, pandas, kafka, psycopg2
│   ├── pyproject.toml               # Ruff + pytest config
│   ├── config.py                    # Shared config (Kafka, Postgres, Redis, APIs)
│   │
│   ├── ingestion/                   # Real-time data ingestion
│   │   ├── consumer.py              # Kafka consumer loop → Postgres
│   │   ├── parser.py                # Regex-based log line parser
│   │   ├── api_client.py            # Course API client (user/movie metadata)
│   │   └── validators.py            # Rating/watch validation + drift detection
│   │
│   ├── storage/                     # Persistence layer
│   │   ├── postgres.py              # Schema DDL + upsert functions
│   │   └── cache.py                 # RedisCache class
│   │
│   ├── training/                    # Model training pipeline
│   │   ├── svd.py                   # Collaborative filtering (surprise SVD)
│   │   ├── content.py               # Content-based (TF-IDF cosine similarity)
│   │   ├── hybrid.py                # α-blended scoring + Redis precomputation
│   │   ├── cold_start.py            # Text-matching for users with no ratings
│   │   └── features.py              # TF-IDF vectorization, similarity matrices
│   │
│   ├── scripts/                     # Entry points
│   │   ├── run_consumer.py          # Start Kafka consumer
│   │   ├── run_training.py          # Train hybrid model + push to Redis
│   │   └── collect_data.py          # Batch fetch from course API
│   │
│   └── tests/
│       ├── conftest.py
│       ├── test_parser.py           # Parser against real Kafka format
│       └── test_validators.py       # Validation + drift detection tests
```

## Data Model

### Kafka Event Format (topic: `movielog5`)

Events arrive as comma-separated log lines at ~1000/sec:

```
# Watch event (98% of traffic)
2026-02-14T00:07:39,91903,GET /data/m/gladiator+2000/74.mpg

# Rating event (1-10 scale)
2026-02-14T00:09:21,130276,GET /rate/please+dont+eat+the+daisies+1960=7

# New account
2026-02-14T10:30:03,99999,GET /create_account

# Recommendation request (logged by evaluation harness)
2026-02-14T00:09:21.099,215124,recommendation request 17645-team05.isri.cmu.edu:8082, status 200, result: gladiator+2000,the+godfather+1972, 45ms
```

**Movie IDs are URL slugs** (e.g., `the+shawshank+redemption+1994`), not integers.

### PostgreSQL Schema

| Table | PK | Key columns | Notes |
|---|---|---|---|
| `ratings` | `(user_id, movie_id)` | `rating REAL`, `timestamp TEXT` | Scale 1-10 |
| `watch_events` | `(user_id, movie_id)` | `minutes_watched INTEGER` | Incremented per minute |
| `movies` | `movie_id TEXT` | `title`, `genres`, `cost`, `raw_data JSONB` | JSONB stores full API response |
| `users` | `user_id INTEGER` | `likes TEXT`, `dislikes TEXT`, `raw_data JSONB` | Self-descriptions from API |

Indices on `ratings(user_id)`, `ratings(movie_id)`, `watch_events(user_id)`.

### Course API Response Format

```json
// GET http://128.2.220.241:8080/movie/gladiator+2000
{
  "id": "gladiator+2000",
  "title": "Gladiator",
  "genres": [{"id": 28, "name": "Action"}, {"id": 18, "name": "Drama"}],
  "overview": "After the death of Emperor Marcus Aurelius...",
  "imdb_id": "tt0172495",
  "license_cost": 0,
  "runtime": 155
}

// GET http://128.2.220.241:8080/user/130276
{
  "user_id": 130276,
  "age": 33,
  "occupation": "K-12 student",
  "gender": "M",
  "self_description_likes": "I dig movies that are pretty well-known...",
  "self_description_dislikes": "I'm not really into musicals..."
}
```

## Recommendation Algorithms

### 1. Collaborative Filtering — SVD

Matrix factorization via `scikit-surprise`:

- **Input**: `ratings` table (user_id, movie_id, rating)
- **Rating scale**: 1-10
- **Hyperparameters**: 50 latent factors, 20 epochs, seed=42
- **Output**: `user_factors` (N×50), `item_factors` (M×50), biases, global mean
- **Prediction**: `score(u, i) = q_i · p_u + b_u + b_i + μ`
- **Serving**: `np.argpartition` for O(n) top-K selection, masking already-rated items

### 2. Content-Based Filtering — TF-IDF

Text-based similarity over movie metadata:

- **Features**: concatenation of `genres + overview + keywords` per movie
- **Vectorization**: `TfidfVectorizer(stop_words="english", max_features=5000)`
- **Similarity**: cosine similarity, top-50 neighbors stored per movie
- **Batched**: 500-movie batches to fit in memory
- **Cold-start use**: user's `self_description_likes` is transformed into TF-IDF space and matched against the movie matrix

### 3. Hybrid Blending

```
hybrid_score(u, i) = α · svd_score(u, i) + (1 - α) · content_score(u, i)
```

- `α = 0.7` by default (tunable via `--alpha`)
- Content scores normalized to [0, 10] to match SVD range
- Pre-computed for all users with ratings → written to Redis in pipelines of 500

### 4. Cold-Start Strategy

For users with self-descriptions but no ratings:

1. **TF-IDF matching**: transform `self_description_likes` text → find top-40 similar movies → filter out movies similar to `self_description_dislikes` → return top-20
2. **Genre fallback**: extract genre keywords from text (e.g., "action", "comedy") → SQL query for highest-rated movies in those genres, excluding disliked genres

### 5. Serving Cascade

The Flask `Recommender` class tries each tier in order:

| Tier | Source | Latency | When |
|---|---|---|---|
| 1 | Redis cache | <1ms | User has pre-computed recs |
| 2 | Real-time SVD | ~5ms | User in training set, cache miss |
| 3 | Content fallback | ~10ms | Unknown user, content data loaded |
| 4 | Redis popularity | <1ms | No models available |
| 5 | Hardcoded defaults | 0ms | Nothing else works |

Hot-reload: on every request, checks `model:version` in Redis. If changed, reloads `model.pkl` and `content_data.pkl` from disk.

## Data Quality

### Validation (`ingestion/validators.py`)

- Rating range: reject if `< 1` or `> 10`
- Movie ID: must be non-empty string
- User ID: must be positive integer
- Watch minute: must be non-negative

### Drift Detection

- **Rating drift**: compares rolling average of recent ratings against historical mean (5.5); alerts if deviation > 1.0
- **Genre drift**: compares genre distribution of recent watches against baseline; alerts if any genre shifts > 20%

## Infrastructure

### Docker Compose Services

| Service | Image | Port | Purpose |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | 5432 | Persistent storage |
| `redis` | `redis:7-alpine` | 6379 | Recommendation cache |
| `recommendation-service` | Custom (service/) | 8082 | Flask API |
| `kafka-consumer` | Custom (pipeline/) | — | Real-time ingestion |

Volumes: `pg-data` (Postgres persistence), `model-data` (shared model artifacts between consumer and service).

### Health Checks

- Postgres: `pg_isready -U movieapp -d moviedb` every 5s
- Redis: `redis-cli ping` every 5s
- Service: `curl http://localhost:8082/health` every 30s

## Local Development

### Prerequisites

- Docker + Docker Compose
- Python 3.11+ (3.11 recommended; 3.13 works except `scikit-surprise`)
- CMU VPN or campus network (for Kafka access)

### Quick Start

```bash
# 1. Clone and enter project
cd movie-rec-app

# 2. Copy env file
cp .env.example .env
# Edit .env with your TMDB_API_KEY if needed

# 3. Start infrastructure
docker compose up -d postgres redis

# 4. Set up SSH tunnel for Kafka (password: mlip-kafka)
sshpass -p 'mlip-kafka' ssh -o ServerAliveInterval=60 -o StrictHostKeyChecking=no \
  -L 9092:localhost:9092 tunnel@128.2.220.241 -NTf

# 5. Create venv and install deps
python3.11 -m venv venv && source venv/bin/activate
pip install -r service/requirements.txt -r pipeline/requirements.txt

# 6. Start Kafka consumer (background)
cd pipeline
POSTGRES_HOST=localhost REDIS_HOST=localhost KAFKA_BROKER=localhost:9092 \
  python scripts/run_consumer.py &

# 7. Wait for data (~60s for a few thousand ratings), then run features
POSTGRES_HOST=localhost MODEL_DIR=../models \
  python -c "import sys; sys.path.insert(0,'.'); from training.features import build_and_save; build_and_save()"

# 8. Train model (requires scikit-surprise / Python 3.11)
POSTGRES_HOST=localhost REDIS_HOST=localhost MODEL_DIR=../models \
  python scripts/run_training.py --alpha 0.7

# 9. Start the API
cd ../service
REDIS_HOST=localhost MODEL_DIR=../models FLASK_DEBUG=1 python -m app

# 10. Test
curl http://localhost:8082/health
curl http://localhost:8082/recommend/12345
```

### Running Tests

```bash
# Service tests (5 tests)
cd service && python -m pytest tests/ -v

# Pipeline tests (21 tests)
cd pipeline && python -m pytest tests/ -v
```

### Docker-Only Deploy

```bash
docker compose up -d --build
# All 4 services start, consumer begins ingesting, service serves on :8082
```

## API Reference

### `GET /recommend/<int:userid>`

Returns up to 20 movie recommendations as a plain-text, comma-separated list of movie slug IDs.

```
$ curl http://localhost:8082/recommend/12345
the+shawshank+redemption+1994,the+godfather+1972,pulp+fiction+1994,...
```

- **Response**: `text/plain`, status 200
- **Latency**: <5ms (cached), <50ms (real-time SVD)
- **Error**: empty body, status 500

### `GET /health`

```json
{"status": "healthy"}
```

### `GET /`

```json
{"message": "Movie Recommendation API is running"}
```

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yaml`):

1. **lint-and-test-service**: `ruff check` + `ruff format --check` + `pytest` on `service/`
2. **test-pipeline**: `pytest` on `pipeline/tests/`
3. **docker-build**: `docker compose build` (runs after tests pass)

Triggers on push to `main` and all pull requests.

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Used by |
|---|---|---|
| `KAFKA_BROKER` | `128.2.220.241:9092` | Consumer |
| `KAFKA_TOPIC` | `movielog5` | Consumer |
| `KAFKA_GROUP_ID` | `team05-consumer` | Consumer |
| `API_BASE_URL` | `http://128.2.220.241:8080` | Consumer, collect_data |
| `POSTGRES_HOST` | `postgres` | Pipeline |
| `POSTGRES_PORT` | `5432` | Pipeline |
| `POSTGRES_DB` | `moviedb` | Pipeline |
| `POSTGRES_USER` | `movieapp` | Pipeline |
| `POSTGRES_PASSWORD` | `movieapp123` | Pipeline |
| `REDIS_HOST` | `redis` | Service, Pipeline |
| `REDIS_PORT` | `6379` | Service, Pipeline |
| `MODEL_DIR` | `/app/models` | Service, Pipeline |
| `TMDB_API_KEY` | (empty) | Unused (course API has TMDB data) |
