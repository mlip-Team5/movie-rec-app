# Movie Recommendation Service — Team 05

## What Is This?

This is a **production movie recommendation system** built for the ML in Production course (17-445/645/745) at CMU. Imagine you're building an early Netflix: ~1 million users, ~20,000 movies, and your job is to suggest what each user should watch next.

The course server sends HTTP requests to our VM asking "what should user X watch?" and we have **600 milliseconds** to respond with up to 20 personalized movie recommendations. The server does this continuously, 24/7, and logs whether we answered correctly in a Kafka stream. Real users (simulated) then watch the movies we suggest and rate them — so our recommendations directly influence future behavior.

**In plain terms:** Kafka stream comes in with user activity (ratings, watches, new signups) → we ingest and store it → we train ML models on it → we serve personalized recommendations via a REST API → the course server evaluates us.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [How the Recommendation Works (The ML)](#how-the-recommendation-works-the-ml)
- [Data Model](#data-model)
- [Project Structure](#project-structure)
- [Setup and Run Guide](#setup-and-run-guide)
- [API Reference](#api-reference)
- [Operations Guide](#operations-guide)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## System Architecture

There are **4 Docker containers** running on our VM and **5 pipeline scripts** that feed them:

```
                    COURSE INFRASTRUCTURE
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Kafka (128.2.220.241:9092)          Course API (:8080)      │
│  Topic: movielog5                    /movie/<id>  /user/<id> │
│  ~1000 events/sec                    Rate-limited, JSON      │
│  - watch events (98%)                                        │
│  - ratings (GET /rate/movie=N)                               │
│  - new accounts                                              │
│  - recommendation logs (our results)                         │
│                                                              │
└──────────┬───────────────────────────────┬───────────────────┘
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────────────┐
│  OUR VM  (17645-team05.isri.cmu.edu / 128.2.220.246)        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  KAFKA CONSUMER (Docker container, always running)     │  │
│  │                                                        │  │
│  │  Kafka → parser.py → validators.py → postgres.py       │  │
│  │                                                        │  │
│  │  Every event:     → raw_events table (audit trail)     │  │
│  │  Ratings:         → ratings table (latest per user)    │  │
│  │  Watch events:    → watch_events table (aggregated)    │  │
│  │  New accounts:    → users table (fetches profile)      │  │
│  │  Rec requests:    → recommendation_logs table          │  │
│  └────────────────────────────┬───────────────────────────┘  │
│                               │                              │
│                               ▼                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  POSTGRES (Docker container)                           │  │
│  │                                                        │  │
│  │  Reference data:  movies, users                        │  │
│  │  Aggregated:      ratings, watch_events                │  │
│  │  Audit:           raw_events (append-only)             │  │
│  │  Observability:   recommendation_logs                  │  │
│  └────────────────────────────┬───────────────────────────┘  │
│                               │                              │
│          ┌────────────────────┼──────────────────┐           │
│          ▼                    ▼                   ▼           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ run_features  │  │ run_training │  │ collect_data     │   │
│  │              │  │              │  │                  │   │
│  │ TF-IDF +    │  │ SVD model + │  │ Batch fetch     │   │
│  │ similarity  │  │ hybrid recs │  │ movies + users  │   │
│  │ matrices    │  │ + cold-start│  │ from course API │   │
│  │              │  │              │  │                  │   │
│  │ Outputs:    │  │ Outputs:    │  │ Outputs:        │   │
│  │ .pkl files  │  │ model.pkl + │  │ rows in         │   │
│  │ to volume   │  │ Redis cache │  │ Postgres        │   │
│  └──────────────┘  └──────┬───────┘  └──────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  REDIS (Docker container)                              │  │
│  │                                                        │  │
│  │  recs:user:{id}  → precomputed recs (TTL 24h)         │  │
│  │  recs:popular    → top 100 movies by avg rating       │  │
│  │  model:version   → triggers hot-reload in service     │  │
│  └────────────────────────────┬───────────────────────────┘  │
│                               │                              │
│                               ▼                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  RECOMMENDATION SERVICE (Docker, port 8082)            │  │
│  │                                                        │  │
│  │  Flask + Gunicorn (4 workers, 2s timeout)              │  │
│  │                                                        │  │
│  │  GET /recommend/<userid>                               │  │
│  │  → 5-tier prediction cascade (see below)               │  │
│  │  → returns: "movie1,movie2,movie3,..." (plain text)   │  │
│  │                                                        │  │
│  │  Hot-reload: detects model:version change in Redis,    │  │
│  │  reloads .pkl files from shared Docker volume          │  │
│  └────────────────────────────────────────────────────────┘  │
│                               │                              │
└───────────────────────────────┼──────────────────────────────┘
                                │
                                ▼
                    Course server receives
                    comma-separated movie IDs
                    within 600ms and logs result
                    back to Kafka
```

### Docker Containers

| Container | Image | Exposed Port | Purpose |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | Internal only | Persistent data storage |
| `redis` | `redis:7-alpine` | Internal only | Recommendation cache + model versioning |
| `recommendation-service` | Custom (service/) | **8082** | Serves recommendations to course server |
| `kafka-consumer` | Custom (pipeline/) | None | Ingests Kafka stream into Postgres |

Postgres and Redis are **not exposed to the internet** — only accessible within the Docker network. Only port 8082 (the API) is publicly accessible.

### Shared Docker Volumes

| Volume | Mounted at | Shared between | Contents |
|---|---|---|---|
| `pg-data` | `/var/lib/postgresql/data` | postgres only | Database files |
| `model-data` | `/app/models` | kafka-consumer, recommendation-service | `.pkl` model files |

---

## How the Recommendation Works (The ML)

### The Problem

Given a `user_id`, return up to 20 movie IDs that the user is most likely to enjoy, within 600ms.

Challenges:
- **Cold-start**: New users have no ratings. We only have their self-description ("I like action movies, I don't like horror").
- **Scale**: ~1M users, ~20K movies. Must precompute where possible.
- **Latency**: 600ms budget including network. Inference must be <50ms.
- **Availability**: Must respond 24/7. Prefer a bad recommendation over no recommendation.

### Model 1: SVD Collaborative Filtering

**Idea:** Users who rated movies similarly will like similar movies.

We use Singular Value Decomposition (SVD) from `scikit-surprise` to factorize the user-item rating matrix into low-dimensional latent factors.

```
Rating matrix R (1M users × 20K movies, very sparse)
    ≈ P (1M × 50) × Q^T (50 × 20K) + biases

Predicted score for user u, item i:
    score(u, i) = q_i · p_u + b_u + b_i + μ

Where:
    p_u = 50-dimensional user embedding
    q_i = 50-dimensional item embedding
    b_u = user bias (does this user rate high/low generally?)
    b_i = item bias (is this movie generally rated high/low?)
    μ   = global mean rating
```

- **Training**: 50 latent factors, 20 epochs, rating scale 1-10
- **Serving**: For a given user, compute scores for all items via matrix multiply (`item_factors @ user_vec + biases`), mask already-rated items, pick top-20 via `np.argpartition` (O(n) instead of O(n log n) sort)
- **Output**: `model.pkl` containing numpy arrays for factors, biases, and ID mappings

### Model 2: Content-Based Filtering (TF-IDF)

**Idea:** Recommend movies similar to what the user already liked, based on movie metadata.

```
For each movie, build a text blob:
    "Action Drama" + "After the death of Emperor Marcus Aurelius..." + "gladiator roman empire"
     (genres)        (overview)                                        (keywords)

Vectorize all movies with TF-IDF (5000 features, English stopwords removed)
Compute cosine similarity between all pairs
Store top-50 most similar movies per movie
```

- **Training**: Reads movie metadata from Postgres, computes TF-IDF matrix, batches similarity computation (500 movies at a time to fit in memory)
- **Serving**: For a user, look at their top-rated movies, find similar movies via pre-computed similarity, aggregate scores weighted by rating
- **Output**: `content_data.pkl` (similarity maps, genre vectors), `tfidf_matrix.npz`, `tfidf_vectorizer.pkl`

### Hybrid Blending

We combine both models for users who have ratings:

```
final_score(u, i) = α × svd_score(u, i) + (1 - α) × content_score(u, i)
```

- Default `α = 0.7` (SVD gets more weight since it generally performs better with enough data)
- Content scores are normalized to [0, 10] to match SVD scale
- Already-rated items are masked to `-inf`
- Top-20 per user are pre-computed and written to Redis in batches of 500

### Cold-Start Strategy

For users who have no ratings but filled out their profile ("I like action movies and comedies"):

1. **TF-IDF text matching**: Transform the user's `self_description_likes` text into TF-IDF space → cosine similarity against all movies → get top-40 candidates → remove movies similar to `self_description_dislikes` → return top-20
2. **Genre keyword fallback**: If TF-IDF matching fails, extract genre keywords from the text (e.g., "action", "comedy") → SQL query for highest-rated movies in those genres, excluding disliked genres

### The 5-Tier Serving Cascade

The Flask service tries each tier in order. First hit wins. This guarantees we **always** return something:

| Tier | Source | Latency | When it's used |
|---|---|---|---|
| 1 | **Redis cache** | <1ms | User has pre-computed hybrid recs |
| 2 | **Real-time SVD** | ~5ms | User is in training set but cache miss/expired |
| 3 | **Content fallback** | ~10ms | Unknown user, content data is loaded |
| 4 | **Redis popularity** | <1ms | No ML models available |
| 5 | **Hardcoded defaults** | 0ms | Nothing else works (20 classic films) |

The service also **hot-reloads**: on every request, it checks `model:version` in Redis. If it changed (meaning training just ran), it reloads the `.pkl` files from disk. Zero downtime model updates.

---

## Data Model

### Kafka Events

The Kafka stream (`movielog5`) produces ~1000 events/sec in comma-separated format:

| Event | Format | Frequency |
|---|---|---|
| Watch | `<ts>,<uid>,GET /data/m/<movie>/<min>.mpg` | ~98% of traffic |
| Rating | `<ts>,<uid>,GET /rate/<movie>=<1-10>` | ~1% |
| New account | `<ts>,<uid>,GET /create_account` | Rare |
| Recommendation | `<ts>,<uid>,recommendation request <server>, status <code>, result: <recs>, <time>` | ~1% |

**Movie IDs are URL slugs**, not integers: `the+shawshank+redemption+1994`, `gladiator+2000`.

### PostgreSQL Tables

**Reference data** (from course API):

| Table | PK | Key Columns | Source |
|---|---|---|---|
| `movies` | `movie_id TEXT` | title, genres, vote_average, cost, raw_data JSONB | Course API `/movie/<id>` |
| `users` | `user_id INTEGER` | age, gender, likes, dislikes, raw_data JSONB | Course API `/user/<id>` |

**Derived from events** (from Kafka):

| Table | PK | Key Columns | Behavior |
|---|---|---|---|
| `ratings` | `(user_id, movie_id)` | rating REAL, timestamp | UPSERT — keeps latest rating |
| `watch_events` | `(user_id, movie_id)` | minutes_watched INTEGER | UPSERT — increments counter |

**Audit and observability:**

| Table | PK | Key Columns | Behavior |
|---|---|---|---|
| `raw_events` | `id SERIAL` | timestamp, user_id, event_type, raw_line | Append-only — every Kafka line stored verbatim |
| `recommendation_logs` | `id SERIAL` | timestamp, user_id, status, recommendations, response_time | Append-only — course server feedback |

### Redis Keys

| Key | Value | TTL | Purpose |
|---|---|---|---|
| `recs:user:{id}` | Comma-separated movie IDs | 24h | Pre-computed recommendations |
| `recs:popular` | Comma-separated movie IDs | 24h | Top 100 by avg rating (min 5 ratings) |
| `model:version` | Unix timestamp | None | Triggers hot-reload in service |

### Model Artifacts (Docker volume: model-data)

| File | Size | Contents |
|---|---|---|
| `model.pkl` | ~50-200 MB | SVD factors, biases, ID maps, user rated-item sets |
| `content_data.pkl` | ~10-50 MB | Similarity top-K maps, genre vectors, movie metadata |
| `tfidf_matrix.npz` | ~5-20 MB | Sparse TF-IDF matrix (movies × 5000 features) |
| `tfidf_vectorizer.pkl` | ~1 MB | Fitted TfidfVectorizer for cold-start text matching |

---

## Project Structure

```
movie-rec-app/
├── docker-compose.yml              # Orchestrates all 4 containers
├── .env.example                    # Environment variable template
├── .env                            # Actual config (gitignored)
├── .gitignore
├── .github/workflows/ci.yaml       # GitHub Actions: lint → test → build
├── README.md
│
├── service/                         # RECOMMENDATION API
│   ├── Dockerfile                   # python:3.11-slim + gunicorn
│   ├── requirements.txt             # Flask, gunicorn, redis, numpy, scipy
│   ├── requirements-dev.txt         # + ruff, pytest, pytest-cov
│   ├── pyproject.toml               # Ruff + pytest config
│   ├── app/
│   │   ├── __init__.py              # Flask app factory
│   │   ├── __main__.py              # Dev server entry point
│   │   ├── config.py                # REDIS_HOST, REDIS_PORT, MODEL_DIR
│   │   ├── routes.py                # /recommend/<userid>, /health, /
│   │   └── recommender.py           # 5-tier prediction cascade + hot-reload
│   └── tests/
│       ├── conftest.py              # Flask test client fixture
│       └── test_routes.py           # 5 endpoint tests
│
├── pipeline/                        # DATA PIPELINE + TRAINING
│   ├── Dockerfile                   # python:3.11-slim + gcc (scikit-surprise)
│   ├── requirements.txt             # surprise, sklearn, pandas, kafka, psycopg2
│   ├── pyproject.toml               # Ruff + pytest config
│   ├── config.py                    # All env-based config (Kafka, PG, Redis, API)
│   │
│   ├── ingestion/                   # DATA INGESTION
│   │   ├── consumer.py              # Kafka consumer loop → Postgres
│   │   ├── parser.py                # Regex parser for 4 event types
│   │   ├── api_client.py            # HTTP client for course API (bulk + retry)
│   │   └── validators.py            # Validation + drift detection
│   │
│   ├── storage/                     # PERSISTENCE
│   │   ├── postgres.py              # Schema DDL + all insert/upsert functions
│   │   └── cache.py                 # RedisCache class (recs, popularity, version)
│   │
│   ├── training/                    # ML MODELS
│   │   ├── svd.py                   # SVD collaborative filtering
│   │   ├── content.py               # TF-IDF content-based filtering
│   │   ├── hybrid.py                # Alpha-blended scoring + Redis precomputation
│   │   ├── cold_start.py            # Text-matching for new users
│   │   └── features.py              # TF-IDF vectorization + similarity matrices
│   │
│   ├── scripts/                     # ENTRY POINTS
│   │   ├── collect_data.py          # Batch fetch movies/users from API
│   │   ├── run_features.py          # Build TF-IDF + content similarity
│   │   ├── run_training.py          # Train SVD + hybrid + cold-start → Redis
│   │   └── run_consumer.py          # Start Kafka consumer (runs as container CMD)
│   │
│   └── tests/
│       ├── conftest.py
│       ├── test_parser.py           # 10 tests: all event types, edge cases
│       └── test_validators.py       # 16 tests: validation + drift + recommendations
```

---

## Setup and Run Guide

### Prerequisites

- SSH access to VM `17645-team05.isri.cmu.edu` (requires CMU VPN if off campus)
- Docker and docker-compose installed on the VM
- The repo cloned to `~/mlip-project/movie-rec-app`
- `.env` file created from `.env.example`

### Step-by-Step (VM)

Everything below uses `docker-compose` (with hyphen). VM has v1.29.2.

**Step 1 — Build all Docker images** (3-5 min first time):

```bash
cd ~/mlip-project/movie-rec-app
docker-compose build
```

**Step 2 — Start Postgres + Redis** (wait 15s for health checks):

```bash
docker-compose up -d postgres redis
sleep 15
docker-compose ps
```

Both should show `Up (healthy)`.

**Step 3 — Start the recommendation service** (immediately serves fallback defaults):

```bash
docker-compose up -d recommendation-service
sleep 10
curl http://localhost:8082/health
curl http://localhost:8082/recommend/12345
```

The course server can now reach you. You start accumulating 200-status responses right away.

**Step 4 — Collect movies from the API** (~20-30 min):

```bash
tmux new -s movies
docker-compose run --rm kafka-consumer python scripts/collect_data.py --movies
```

Detach: `Ctrl+B` then `D`. Reattach: `tmux attach -t movies`.

**Step 5 — Collect users from the API** (~60-90 min, can run in parallel):

```bash
tmux new -s users
docker-compose run --rm kafka-consumer python scripts/collect_data.py --users
```

**Step 6 — Start Kafka consumer** (runs forever in background):

```bash
docker-compose up -d kafka-consumer
docker-compose logs -f kafka-consumer   # Ctrl+C to stop watching
```

**Step 7 — Wait for enough ratings** (need 100+, check periodically):

```bash
docker-compose run --rm kafka-consumer python scripts/collect_data.py --stats
```

**Step 8 — Build content features** (needs movies from Step 4):

```bash
docker-compose run --rm kafka-consumer python scripts/run_features.py
```

**Step 9 — Train the model** (needs ratings from Step 7 + features from Step 8):

```bash
docker-compose run --rm kafka-consumer python scripts/run_training.py
```

**Step 10 — Verify personalized recommendations:**

```bash
curl http://localhost:8082/recommend/100
curl http://localhost:8082/recommend/5000
curl http://localhost:8082/recommend/99999
```

Different users should return different movie lists.

---

## API Reference

### `GET /recommend/<int:userid>`

Returns up to 20 movie recommendations as plain text, comma-separated movie slug IDs.

```
$ curl http://128.2.220.246:8082/recommend/12345
the+shawshank+redemption+1994,the+godfather+1972,pulp+fiction+1994,...
```

- **Content-Type**: `text/plain`
- **Status**: 200 on success, 500 on error
- **Latency**: <1ms (cached), <5ms (SVD), <50ms (content)
- **Constraint**: must respond within 600ms or course server marks it as failed

### `GET /health`

```json
{"status": "healthy"}
```

### `GET /`

```json
{"message": "Movie Recommendation API is running"}
```

---

## Operations Guide

### Retrain the model (zero downtime)

```bash
docker-compose run --rm kafka-consumer python scripts/run_features.py
docker-compose run --rm kafka-consumer python scripts/run_training.py
```

The service detects the new `model:version` in Redis and hot-reloads automatically.

### View logs

```bash
docker-compose logs recommendation-service --tail 30
docker-compose logs kafka-consumer --tail 30
```

### Check database stats

```bash
docker-compose run --rm kafka-consumer python scripts/collect_data.py --stats
```

### After code changes

```bash
cd ~/mlip-project/movie-rec-app
git pull
docker-compose build
docker-compose up -d
```

### Shutdown (preserves all data)

```bash
docker-compose down
```

Restart later: `docker-compose up -d` — everything comes back with data intact.

### Full reset (deletes all data and models)

```bash
docker-compose down -v
```

You would need to redo Steps 4-9.

### Rebuild a single service

```bash
docker-compose build recommendation-service
docker-compose up -d recommendation-service
```

---

## Testing

### Run all tests

```bash
# Service tests (5 tests)
cd service && python -m pytest tests/ -v

# Pipeline tests (26 tests)
cd pipeline && python -m pytest tests/ -v
```

### What's tested

**Parser tests** (`test_parser.py` — 10 tests):
- All 4 event types (rating, watch, new_account, recommendation)
- Error status parsing, edge cases, malformed input, empty input, rating scale boundaries

**Validator tests** (`test_validators.py` — 16 tests):
- Rating validation: valid, out-of-range (0, 11), max (10), negative user
- Watch validation: valid, negative minute
- Recommendation validation: valid, error status, bad user, wrong type, missing status
- Drift detection: rating drift (detected, none, too-few-samples), genre drift (detected, none)

**Route tests** (`test_routes.py` — 5 tests):
- Index endpoint, health endpoint, recommend returns CSV format, different users, invalid userid

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yaml`):

1. **lint-and-test-service**: `ruff check` + `ruff format --check` + `pytest` on `service/`
2. **test-pipeline**: `pytest` on `pipeline/tests/`
3. **docker-build**: `docker-compose build` (runs only after tests pass)

Triggers on push to `main` and all pull requests.

---

## Configuration Reference

All config is via environment variables. Defaults are set for Docker networking.

| Variable | Default | Used By | Purpose |
|---|---|---|---|
| `KAFKA_BROKER` | `128.2.220.241:9092` | Consumer | Kafka bootstrap server |
| `KAFKA_TOPIC` | `movielog5` | Consumer | Team-specific Kafka topic |
| `KAFKA_GROUP_ID` | `team05-consumer` | Consumer | Consumer group (tracks offset) |
| `API_BASE_URL` | `http://128.2.220.241:8080` | Consumer, collect_data | Course API for movie/user metadata |
| `POSTGRES_HOST` | `postgres` | Pipeline | Docker service name |
| `POSTGRES_PORT` | `5432` | Pipeline | Default Postgres port |
| `POSTGRES_DB` | `moviedb` | Pipeline | Database name |
| `POSTGRES_USER` | `movieapp` | Pipeline | Database user |
| `POSTGRES_PASSWORD` | `movieapp123` | Pipeline | Database password |
| `REDIS_HOST` | `redis` | Service + Pipeline | Docker service name |
| `REDIS_PORT` | `6379` | Service + Pipeline | Default Redis port |
| `MODEL_DIR` | `/app/models` | Service + Pipeline | Shared volume mount point |

---

## Troubleshooting

| Problem | Diagnosis | Fix |
|---|---|---|
| `curl: Connection refused` on 8082 | Service not running | `docker-compose up -d recommendation-service` and check `docker-compose logs recommendation-service` |
| Same recommendations for every user | Model not loaded in service | Retrain: `run_features.py` then `run_training.py`. Check logs for "SVD loaded" message |
| `Not enough ratings (<100)` during training | Kafka consumer hasn't ingested enough data yet | Wait 15-30 min, check `--stats`, retry |
| `No content_data.pkl — run features first` | Features step was skipped | Run `run_features.py` before `run_training.py` |
| Container keeps restarting | Crash loop | `docker-compose logs <name> --tail 50` to see the error |
| Kafka consumer can't connect | Network / VPN issue | From VM: `nc -zv 128.2.220.241 9092` — should say "succeeded" |
| Port 8082 not reachable externally | Firewall blocking it | `sudo ufw allow 8082/tcp` |
| Disk space running low | Docker images + Postgres data | `df -h` then `docker system prune -f` |
| Want to inspect the database directly | Need to query Postgres | `docker-compose exec postgres psql -U movieapp -d moviedb` |
