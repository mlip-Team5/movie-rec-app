# Movie Recommendation Service — Milestone 1 Report

**Team:** Cloudy with a Chance of Hallucinations (Team 05)
**Course:** ML in Production (17-445/645/745) — Spring 2026, Carnegie Mellon University
**Repository:** [group-project-s26-cloudy-with-a-chance-of-hallucinations](https://github.com/cmu-seai/group-project-s26-cloudy-with-a-chance-of-hallucinations)
**Branch:** `dev_shreyave_model`
**M1 Report:** [`m1_report.md`](m1_report.md)

---

## Table of Contents

1. [Data and Learning Technique](#1-data-and-learning-technique)
2. [Learning Implementation](#2-learning-implementation)
3. [Cold-Start Problem](#3-cold-start-problem)
4. [Quality Metrics (4 Qualities)](#4-quality-metrics-4-qualities)
5. [Prediction Service](#5-prediction-service)
6. [Request Volume and Response Time](#6-request-volume-and-response-time)
7. [Personalized Recommendations](#7-personalized-recommendations)
8. [System Architecture](#8-system-architecture)
9. [Data Model](#9-data-model)
10. [Project Structure](#10-project-structure)
11. [Setup and Run Guide](#11-setup-and-run-guide)
12. [How to Test Every Deliverable](#12-how-to-test-every-deliverable)
13. [Configuration Reference](#13-configuration-reference)
14. [Operations Guide](#14-operations-guide)
15. [Team Contract](#15-team-contract)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Data and Learning Technique

### Data Sources

| Source | Description | Volume |
|---|---|---|
| **Kafka stream** (`movielog5`) | Real-time user events: watch activity (~98%), ratings (~1%), new accounts, recommendation request logs | ~1000 events/sec, continuous |
| **Course REST API** (`128.2.220.241:8080`) | Movie metadata (title, genres, overview, keywords, vote_average, popularity, cost) and user profiles (age, gender, self-description likes/dislikes) | ~25K movies, ~100K users |

All data is ingested into PostgreSQL via our Kafka consumer and batch collection scripts. The Kafka consumer runs 24/7 and captures every event. The course API is queried on setup to seed the database with movie metadata and user profiles.

**Rating scale:** 1-10 (integer). Defined by the course spec.

### Why SVD + Content-Based Hybrid?

We chose a **hybrid model** combining SVD collaborative filtering with TF-IDF content-based filtering for several reasons:

1. **SVD collaborative filtering** captures latent user-item interactions. Users who rated movies similarly will get similar recommendations. It works well when sufficient ratings exist and is the standard baseline for recommendation systems.
2. **TF-IDF content-based filtering** computes cosine similarity between movie metadata (genres + overview + keywords). This covers cold-start cases where users have no ratings but do have profile descriptions.
3. **Hybrid blending** (`final = 0.7 * SVD + 0.3 * content`) gives SVD more weight (it's more accurate with enough data) while content-based filtering provides diversity and coverage for edge cases.

**Why not other approaches?**
- **Nearest-neighbor CF:** Slower at inference (requires computing user similarity at request time). SVD pre-factors the matrix so inference is a fast dot product.
- **Deep learning (NCF, autoencoders):** Requires much more data and training time for marginal gains at this scale. SVD is a well-understood, fast, and effective baseline.
- **Association rules / item-item CF:** Doesn't generalize as well to unseen users. SVD learns latent factors that transfer.

**Implementation pointers:**
- SVD training: [`pipeline/training/svd.py`](pipeline/training/svd.py)
- Content feature engineering: [`pipeline/training/features.py`](pipeline/training/features.py)
- Content-based scoring: [`pipeline/training/content.py`](pipeline/training/content.py)
- Hybrid blending and precomputation: [`pipeline/training/hybrid.py`](pipeline/training/hybrid.py)

---

## 2. Learning Implementation

### SVD Collaborative Filtering (Primary Model)

**Algorithm:** Singular Value Decomposition via `scikit-surprise`.

```
Rating matrix R (users x movies, very sparse)
    ~= P (users x 50) x Q^T (50 x movies) + biases

Predicted score(user u, item i) = q_i . p_u + b_u + b_i + mu

Where:
    p_u = 50-dim user latent factor vector
    q_i = 50-dim item latent factor vector
    b_u = user bias
    b_i = item bias
    mu  = global mean rating
```

- **Hyperparameters:** 50 latent factors, 20 epochs, rating scale 1-10 (all configurable via CLI args `--factors`, `--epochs` or env vars `SVD_FACTORS`, `SVD_EPOCHS`)
- **Training data:** All `(user_id, movie_id, rating)` tuples from the `ratings` table
- **Output:** `model.pkl` — numpy arrays for user/item factors, biases, ID mappings, and user's already-rated items

**Implementation:** [`pipeline/training/svd.py`](pipeline/training/svd.py)

### TF-IDF Content-Based Filtering (Secondary Model)

**Algorithm:** TF-IDF vectorization + cosine similarity.

```
For each movie, concatenate:
    genres + " " + overview + " " + keywords
    -> "Action Drama After the death of Emperor Marcus Aurelius... gladiator roman empire"

TfidfVectorizer(max_features=5000, stop_words="english")
    -> sparse matrix (movies x 5000)

cosine_similarity(movie_i, movie_j)
    -> store top-50 most similar movies per movie
```

- **Feature source:** `genres`, `overview`, `keywords` from the `movies` table
- **Output:** `content_data.pkl` (similarity maps, genre vectors, metadata), `tfidf_matrix.npz` (sparse matrix), `tfidf_vectorizer.pkl` (fitted vectorizer for cold-start text matching)

**Implementation:** [`pipeline/training/features.py`](pipeline/training/features.py), [`pipeline/training/content.py`](pipeline/training/content.py)

### Training Pipeline

```bash
# On the VM:
docker compose run --rm kafka-consumer python scripts/run_features.py   # Build TF-IDF features
docker compose run --rm kafka-consumer python scripts/run_training.py   # Train SVD + hybrid + cold-start
```

The training script: [`pipeline/scripts/run_training.py`](pipeline/scripts/run_training.py)
The feature engineering script: [`pipeline/scripts/run_features.py`](pipeline/scripts/run_features.py)

---

## 3. Cold-Start Problem

**Problem:** New users have no ratings, so SVD cannot generate recommendations for them.

**Our solution:** We use the user's **self-description** (the `likes` and `dislikes` fields from the user profile) to generate personalized recommendations even with zero ratings.

### Approach 1: TF-IDF Text Matching (Primary)

1. Load the pre-computed TF-IDF vectorizer and movie matrix
2. Transform the user's `likes` text into TF-IDF space using the same vectorizer
3. Compute cosine similarity between the user's text vector and all movie vectors
4. Take the top-N most similar movies
5. If the user also has `dislikes` text, repeat for dislikes and **exclude** those movies from the result

```
User profile: likes="I enjoy action movies, sci-fi, and thrillers"
              dislikes="I don't like horror or romance"

1. TF-IDF("I enjoy action movies, sci-fi, and thrillers") -> user_vec
2. cosine_similarity(user_vec, all_movie_vecs) -> ranked list
3. TF-IDF("I don't like horror or romance") -> dislike_vec
4. Remove movies similar to dislike_vec
5. Return top-20
```

### Approach 2: Genre Keyword Fallback

If TF-IDF matching fails (no content data loaded), we fall back to a simpler approach:

1. Extract genre keywords from the user's `likes` text by matching against all genres in the database
2. Query the `movies` table for highest-rated movies matching those genres
3. Exclude movies from genres mentioned in the user's `dislikes` text

**Implementation:** [`pipeline/training/cold_start.py`](pipeline/training/cold_start.py)

The cold-start function `process_all()` runs automatically as part of training (`run_training.py`) and pre-computes recommendations for all users who have self-descriptions but no ratings, storing them in Redis.

---

## 4. Quality Metrics (4 Qualities)

We evaluate **two models** — our primary SVD model and a NormalPredictor baseline — across four quality dimensions. Each quality follows the three-step framework: **metric**, **data**, and **operationalization**.

### Quality 1: Prediction Accuracy

| Step | Description |
|---|---|
| **Metric** | RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), Precision@K, Recall@K |
| **Data** | All ratings from the `ratings` table, split 80/20 train/test with `random_state=42` for reproducibility |
| **Operationalization** | Train each model on 80% of ratings. Predict on the held-out 20%. Compute RMSE and MAE over all predictions. For Precision@K and Recall@K, for each user: sort predicted items by score, take top-K, check how many are "relevant" (true rating >= 7.0, i.e., 70% of the max rating scale). Average across all users. |

**Implementation:** [`pipeline/scripts/run_eval.py`](pipeline/scripts/run_eval.py) — function `offline_eval()`, lines 46-187

**How to run:**
```bash
docker compose run --rm kafka-consumer python scripts/run_eval.py --offline
```

### Quality 2: Training Cost

| Step | Description |
|---|---|
| **Metric** | Wall-clock training time (seconds) |
| **Data** | Same 80% training split as accuracy evaluation |
| **Operationalization** | Measure `time.time()` before and after `model.fit(trainset)`. Report for both SVD and baseline. Lower is better — indicates computational efficiency. |

**Implementation:** [`pipeline/scripts/run_eval.py`](pipeline/scripts/run_eval.py) — `svd_train_time` and `bl_train_time` variables within `offline_eval()`

### Quality 3: Inference Cost

| Step | Description |
|---|---|
| **Metric** | Wall-clock inference time (seconds total + per-prediction milliseconds) |
| **Data** | The 20% test set predictions |
| **Operationalization** | Measure `time.time()` before and after `model.test(testset)`. Compute per-prediction time as `total_time / num_predictions * 1000` (ms). Also tracked at the serving layer: the Flask route logs `time=Xms` per request and flags slow requests (over 600ms). |

**Implementation:**
- Offline: [`pipeline/scripts/run_eval.py`](pipeline/scripts/run_eval.py) — `svd_infer_time` and `bl_infer_time` in `offline_eval()`
- Online: [`service/app/routes.py`](service/app/routes.py) — `elapsed_ms` tracking per request, slow-request counter

### Quality 4: Model Size

| Step | Description |
|---|---|
| **Metric** | Disk size of serialized model files (MB) |
| **Data** | The `.pkl` and `.npz` files in the models directory |
| **Operationalization** | After training, measure `os.path.getsize()` on `model.pkl`, `content_data.pkl`, `tfidf_matrix.npz`, `tfidf_vectorizer.pkl`. Report in MB. Smaller models load faster (important for hot-reload) and use less memory. The baseline (NormalPredictor) is in-memory only — no file. |

**Implementation:** [`pipeline/scripts/run_eval.py`](pipeline/scripts/run_eval.py) — `model_info()` function and model size logging in `offline_eval()`

### Comparison Table (Sample Output)

```
Quality                      SVD             Baseline
------------------------------------------------------------
RMSE (lower=better)          1.9926          2.0277
MAE (lower=better)           1.5248          1.5765
Precision@10 (higher)        0.1333          0.0800
Recall@10 (higher)           1.0000          0.6000
Training cost                0.05s           0.01s
Inference cost               0.0010s         0.0008s
Model size                   0.50 MB         ~0 MB
------------------------------------------------------------
SVD RMSE improvement over baseline: 1.7%
```

**How to run the full comparison:**
```bash
docker compose run --rm kafka-consumer python scripts/run_eval.py --offline
```

---

## 5. Prediction Service

### How a Ranking Is Computed

When the course server sends `GET /recommend/<userid>`, our Flask service computes a ranking through a **5-tier prediction cascade**. It tries each tier in order; the first one that returns results wins. This guarantees we **always** respond (availability > perfection).

| Tier | Source | Latency | When Used |
|---|---|---|---|
| 1 | **Redis cache** | <1ms | User has pre-computed hybrid recommendations from training |
| 2 | **Real-time SVD** | ~5ms | User is in the SVD training set but cache miss/expired |
| 3 | **Content fallback** | ~10ms | Unknown user, content similarity data is loaded |
| 4 | **Redis popularity** | <1ms | No ML models available, but popularity list exists |
| 5 | **Dynamic fallback** | <1ms | Last resort: top-voted movies from content data or first N items from SVD model |

**Tier 1 (Cache):** Check Redis for `recs:user:{userid}`. If found, return immediately. These are pre-computed during training by the hybrid blending step.

**Tier 2 (SVD):** Look up the user's inner ID in the SVD model. Compute scores for all items: `item_factors @ user_vec + item_biases + global_mean + user_bias`. Mask already-rated items. Use `np.argpartition` for O(n) top-K selection. Cache the result in Redis.

**Tier 3 (Content):** Use the content similarity data to find movies similar to top-voted movies. Aggregate similarity scores. Return the highest-scored candidates.

**Tier 4 (Popularity):** Return the pre-computed list of most popular movies (highest average rating with at least 5 ratings).

**Tier 5 (Fallback):** If nothing else is available, return top-voted movies from content data or the first N item IDs from the SVD model.

**Hot-reload:** On every request, the service checks `model:version` in Redis. If it changed (training just ran), it reloads the `.pkl` files from the shared Docker volume. Zero downtime model updates.

**Implementation:**
- Cascade logic: [`service/app/recommender.py`](service/app/recommender.py) — `predict_with_tier()` method
- Flask route: [`service/app/routes.py`](service/app/routes.py) — `recommend()` function
- Response format: plain text, comma-separated movie slug IDs (e.g., `the+shawshank+redemption+1994,gladiator+2000,...`)

---

## 6. Request Volume and Response Time

**Requirement:** Successfully answer at least 2000 recommendation requests in the 24 hours before or after submission. Successful = status 200, well-formed response, within the time limit (600ms).

**How we achieve this:**
- The service runs 24/7 via `docker compose up -d` with `restart: unless-stopped`
- Gunicorn runs 4 workers to handle concurrent requests
- The 5-tier cascade ensures we always return *something* — never an empty error
- Pre-computed recommendations in Redis respond in <1ms (Tier 1)
- Real-time SVD inference takes ~5ms (Tier 2)
- The service logs every request with its tier and response time

**How to verify:**
```bash
# Check recommendation_logs table for 200-status responses in the last 24h
docker compose run --rm kafka-consumer python scripts/run_eval.py --online

# Or query directly:
docker compose exec postgres psql -U movieapp -d moviedb -c \
  "SELECT COUNT(*) as total,
          SUM(CASE WHEN status = 200 THEN 1 ELSE 0 END) as successful
   FROM recommendation_logs
   WHERE timestamp::timestamp >= NOW() - INTERVAL '24 hours';"
```

---

## 7. Personalized Recommendations

**Requirement:** At least 10% of all recommendation requests must have *personalized* responses (not the same recommendation for every user).

**How we achieve personalization:**
- **SVD model:** Each user has a unique latent factor vector, producing unique item rankings
- **Hybrid precomputation:** Recs are pre-computed per-user and stored individually in Redis (`recs:user:{userid}`)
- **Cold-start:** Even users without ratings get personalized recs based on their unique self-descriptions
- **Content fallback (Tier 3):** Uses top-voted movies with similarity scoring — less personalized, but still varies

**How to verify:**
```bash
# Check personalization rate
docker compose run --rm kafka-consumer python scripts/run_eval.py --online

# Quick manual check — different users should get different results:
curl http://localhost:8082/recommend/100
curl http://localhost:8082/recommend/5000
curl http://localhost:8082/recommend/99999

# Check tier distribution (cache/svd = personalized, popularity/fallback = not):
curl http://localhost:8082/stats | python3 -m json.tool
```

---

## 8. System Architecture

Four Docker containers run on our VM, orchestrated by `docker-compose.yml`:

```
                    COURSE INFRASTRUCTURE
+--------------------------------------------------------------+
|                                                              |
|  Kafka (128.2.220.241:9092)          Course API (:8080)      |
|  Topic: movielog5                    /movie/<id>  /user/<id> |
|  ~1000 events/sec                                            |
|                                                              |
+----------+-------------------------------+-------------------+
           |                               |
           v (via SSH tunnel)              v
+--------------------------------------------------------------+
|  OUR VM  (17645-team05.isri.cmu.edu)                         |
|                                                              |
|  +--------------------------------------------------------+  |
|  |  KAFKA CONSUMER (Docker, host network)                 |  |
|  |  Kafka -> parser -> validators -> Postgres             |  |
|  |  Every event -> raw_events (audit trail)               |  |
|  |  Ratings -> ratings table (UPSERT latest)              |  |
|  |  Watches -> watch_events table (aggregated)            |  |
|  |  New accounts -> users table (fetch profile)           |  |
|  |  Rec logs -> recommendation_logs table                 |  |
|  +----------------------------+---------------------------+  |
|                               |                              |
|                               v                              |
|  +--------------------------------------------------------+  |
|  |  POSTGRES (Docker)            REDIS (Docker)           |  |
|  |  movies, users, ratings,      recs:user:{id} (cache)   |  |
|  |  watch_events, raw_events,    recs:popular (top list)   |  |
|  |  recommendation_logs          model:version (hot-reload)|  |
|  +----------------------------+---------------------------+  |
|                               |                              |
|  Pipeline scripts (run inside kafka-consumer container):     |
|  - run_features.py -> TF-IDF + similarity -> .pkl files     |
|  - run_training.py -> SVD + hybrid + cold-start -> Redis     |
|  - run_eval.py -> offline/online metrics                     |
|  - collect_data.py -> batch fetch movies/users from API      |
|                               |                              |
|                               v                              |
|  +--------------------------------------------------------+  |
|  |  RECOMMENDATION SERVICE (Docker, port 8082)            |  |
|  |  Flask + Gunicorn (4 workers)                          |  |
|  |  GET /recommend/<userid> -> 5-tier cascade -> CSV IDs  |  |
|  |  GET /health -> model status                           |  |
|  |  GET /stats -> live tier distribution + error rate     |  |
|  |  Hot-reload: detects model:version change in Redis     |  |
|  +--------------------------------------------------------+  |
|                               |                              |
+-------------------------------+------------------------------+
                                |
                                v
                    Course server receives
                    comma-separated movie IDs
                    within 600ms
```

### Docker Containers

| Container | Image | Port | Purpose |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | 127.0.0.1:5432 | Persistent data storage |
| `redis` | `redis:7-alpine` | 127.0.0.1:6379 | Recommendation cache + model versioning |
| `recommendation-service` | Custom (`service/`) | **8082** (public) | Serves recs to course server |
| `kafka-consumer` | Custom (`pipeline/`) | Host network | Ingests Kafka stream into Postgres |

### Shared Docker Volumes

| Volume | Mount path | Shared between | Contents |
|---|---|---|---|
| `pg-data` | `/var/lib/postgresql/data` | postgres | Database files (persisted across restarts) |
| `model-data` | `/app/models` | kafka-consumer + recommendation-service | `.pkl` model files |

---

## 9. Data Model

### Kafka Events (Input)

| Event | Format | Frequency |
|---|---|---|
| Watch | `<ts>,<uid>,GET /data/m/<movie>/<min>.mpg` | ~98% |
| Rating | `<ts>,<uid>,GET /rate/<movie>=<1-10>` | ~1% |
| New account | `<ts>,<uid>,GET /create_account` | Rare |
| Rec request | `<ts>,<uid>,recommendation request <server>, status <code>, result: <recs>, <time>` | ~1% |

Movie IDs are **URL slugs** (e.g., `the+shawshank+redemption+1994`), not integers.

### PostgreSQL Tables

**Reference data** (from course API):

| Table | PK | Key Columns | Source |
|---|---|---|---|
| `movies` | `movie_id TEXT` | title, genres, overview, keywords, vote_average, popularity, cost, raw_data JSONB | Course API `/movie/<id>` |
| `users` | `user_id INTEGER` | age, gender, likes, dislikes, raw_data JSONB | Course API `/user/<id>` |

**Event-derived** (from Kafka):

| Table | PK | Key Columns | Behavior |
|---|---|---|---|
| `ratings` | `(user_id, movie_id)` | rating REAL, timestamp | UPSERT — keeps latest rating per user+movie |
| `watch_events` | `(user_id, movie_id)` | minutes_watched INTEGER | UPSERT — increments counter |

**Audit and observability:**

| Table | PK | Key Columns | Behavior |
|---|---|---|---|
| `raw_events` | `id SERIAL` | timestamp, user_id, event_type, raw_line | Append-only — every Kafka event stored verbatim |
| `recommendation_logs` | `id SERIAL` | timestamp, user_id, status, recommendations, response_time | Append-only — course server feedback on our recs |

**Schema implementation:** [`pipeline/storage/postgres.py`](pipeline/storage/postgres.py)

**Key design decisions:**
- **UPSERT** for ratings and watches: prevents duplicates, keeps latest state
- **JSONB `raw_data`** on movies/users: stores full API response for future feature engineering
- **Separate `raw_events`** table: complete audit trail of every Kafka event, never modified
- **No foreign keys:** streaming data arrives out of order; we use `_ensure_movie()` to lazily fetch metadata
- **Indexes** on `ratings(user_id)`, `ratings(movie_id)`, `watch_events(user_id)`, `recommendation_logs(status, timestamp)` for query performance

### Redis Keys

| Key | Value | TTL | Purpose |
|---|---|---|---|
| `recs:user:{id}` | Comma-separated movie IDs | 24h | Pre-computed per-user recommendations |
| `recs:popular` | Comma-separated movie IDs | 24h | Top popular movies (fallback) |
| `model:version` | Unix timestamp | None | Triggers hot-reload in service |

### Model Artifacts (Docker volume: `model-data`)

| File | Typical Size | Contents |
|---|---|---|
| `model.pkl` | 0.5-200 MB | SVD factors, biases, ID maps, user rated-item sets |
| `content_data.pkl` | 1-50 MB | Similarity top-K maps, genre vectors, movie metadata |
| `tfidf_matrix.npz` | 0.2-20 MB | Sparse TF-IDF matrix (movies x 5000 features) |
| `tfidf_vectorizer.pkl` | ~0.2 MB | Fitted TfidfVectorizer for cold-start text matching |

---

## 10. Project Structure

```
movie-rec-app/
├── docker-compose.yml              # Orchestrates all 4 containers
├── .env                            # Environment variables (gitignored)
├── .github/workflows/ci.yaml       # GitHub Actions: lint + test + build
├── README.md                       # This file
│
├── service/                         # RECOMMENDATION API
│   ├── Dockerfile
│   ├── requirements.txt             # Flask, gunicorn, redis, numpy, scipy
│   ├── requirements-dev.txt         # + ruff, pytest
│   ├── pyproject.toml               # Ruff + pytest config
│   ├── app/
│   │   ├── __init__.py              # Flask app factory
│   │   ├── __main__.py              # Dev server entry point
│   │   ├── config.py                # Service config (Redis, models, response time limit)
│   │   ├── routes.py                # /recommend/<userid>, /health, /stats
│   │   └── recommender.py           # 5-tier prediction cascade + hot-reload
│   └── tests/
│       ├── conftest.py              # Flask test client fixture
│       └── test_routes.py           # Route tests
│
├── pipeline/                        # DATA PIPELINE + TRAINING
│   ├── Dockerfile
│   ├── requirements.txt             # surprise, sklearn, pandas, kafka, psycopg2
│   ├── pyproject.toml               # Ruff + pytest config
│   ├── config.py                    # Centralized config (all env vars, no hardcoding)
│   │
│   ├── ingestion/                   # DATA INGESTION
│   │   ├── consumer.py              # Kafka consumer loop -> Postgres
│   │   ├── parser.py                # Regex parser for 4 event types
│   │   ├── api_client.py            # HTTP client for course API (bulk + retry)
│   │   └── validators.py            # Data validation + drift detection
│   │
│   ├── storage/                     # PERSISTENCE
│   │   ├── postgres.py              # Schema DDL + insert/upsert functions
│   │   └── cache.py                 # RedisCache (recs, popularity, model version)
│   │
│   ├── training/                    # ML MODELS
│   │   ├── svd.py                   # SVD collaborative filtering
│   │   ├── content.py               # TF-IDF content-based filtering
│   │   ├── hybrid.py                # Alpha-blended scoring + Redis precomputation
│   │   ├── cold_start.py            # Text-matching for new users
│   │   └── features.py              # TF-IDF vectorization + similarity matrices
│   │
│   ├── scripts/                     # ENTRY POINTS
│   │   ├── collect_data.py          # Batch fetch movies/users from course API
│   │   ├── run_features.py          # Build TF-IDF + content similarity
│   │   ├── run_training.py          # Train SVD + hybrid + cold-start -> Redis
│   │   ├── run_eval.py              # Offline/online evaluation + model info
│   │   └── run_consumer.py          # Start Kafka consumer (container CMD)
│   │
│   └── tests/
│       ├── conftest.py
│       ├── test_parser.py           # 10 tests: all event types, edge cases
│       └── test_validators.py       # 16 tests: validation + drift detection
│
└── models/                          # Local model files (gitignored on VM)
```

---

## 11. Setup and Run Guide

### Prerequisites

- SSH access to VM `17645-team05.isri.cmu.edu` (requires CMU VPN)
- Docker and `docker compose` (v2) installed
- SSH tunnel for Kafka: `ssh -o ServerAliveInterval=60 -L 9092:localhost:9092 tunnel@128.2.220.241 -NTf`

### Step-by-Step Deployment

**Step 1 — Clone and configure:**
```bash
cd /home/resources/mlip-project
git clone <repo-url> movie-rec-app
cd movie-rec-app
git checkout dev_shreyave_model
```

**Step 2 — Establish Kafka SSH tunnel** (run in a tmux session):
```bash
tmux new -s tunnel
ssh -o ServerAliveInterval=60 -L 9092:localhost:9092 tunnel@128.2.220.241 -NTf
# Password: tunnel
# Verify: nc -zv localhost 9092 -w 5
```

**Step 3 — Build and start infrastructure:**
```bash
docker compose build
docker compose up -d postgres redis
sleep 15
docker compose ps  # Both should show "healthy"
```

**Step 4 — Start the recommendation service:**
```bash
docker compose up -d recommendation-service
sleep 10
curl http://localhost:8082/health
```

**Step 5 — Collect user profiles** (needed for cold-start):
```bash
docker compose run --rm kafka-consumer python scripts/collect_data.py --users
```

**Step 6 — Start the Kafka consumer** (runs forever, ingests events):
```bash
docker compose up -d kafka-consumer
docker compose logs -f kafka-consumer  # Watch for "Processed X events"
```

**Step 7 — Wait for ratings to accumulate** (need >= 100):
```bash
docker compose run --rm kafka-consumer python scripts/collect_data.py --stats
```

**Step 8 — Build content features:**
```bash
docker compose run --rm kafka-consumer python scripts/run_features.py
```

**Step 9 — Train the model:**
```bash
docker compose run --rm kafka-consumer python scripts/run_training.py
```

**Step 10 — Verify everything:**
```bash
curl http://localhost:8082/health              # svd_loaded: true
curl http://localhost:8082/recommend/100        # Personalized recs
curl http://localhost:8082/recommend/5000       # Different recs
curl http://localhost:8082/stats                # Tier distribution
```

---

## 12. How to Test Every Deliverable

### Deliverable 1: Data and Learning Technique (10pt)

**What to check:** This README sections 1 and 2 describe the data and learning technique.

**How to verify the implementation matches:**
```bash
# Verify SVD training runs and produces a model
docker compose run --rm kafka-consumer python scripts/run_training.py
# Should output: "SVD trained: X users, Y items", "Saved SVD model to /app/models"

# Verify content features exist
docker compose run --rm kafka-consumer python scripts/run_eval.py --info
# Should show model.pkl, content_data.pkl sizes and dimensions
```

### Deliverable 2: Learning Implementation (10pt)

**What to check:** Code links above point to the actual implementation.

**Key files to review:**
- [`pipeline/training/svd.py`](pipeline/training/svd.py) — SVD training and score computation
- [`pipeline/training/features.py`](pipeline/training/features.py) — TF-IDF feature engineering
- [`pipeline/training/content.py`](pipeline/training/content.py) — Content-based scoring
- [`pipeline/training/hybrid.py`](pipeline/training/hybrid.py) — Hybrid blending
- [`pipeline/scripts/run_training.py`](pipeline/scripts/run_training.py) — Training entry point

**How to verify:**
```bash
# Run training and check model dimensions match description
docker compose run --rm kafka-consumer python scripts/run_training.py --factors 50 --epochs 20
docker compose run --rm kafka-consumer python scripts/run_eval.py --info
# Should show: "Latent factors: 50"
```

### Deliverable 3: Cold-Start Problem (10pt)

**What to check:** [`pipeline/training/cold_start.py`](pipeline/training/cold_start.py) implements cold-start using user self-descriptions.

**How to verify:**
```bash
# Check that cold-start runs during training
docker compose run --rm kafka-consumer python scripts/run_training.py
# Look for log: "Cold-start recs: X/Y users"

# Test with a user who has no ratings but has a profile
# (new users from Kafka "create_account" events)
curl http://localhost:8082/recommend/99999
# Should return movies (not empty), even if this user has never rated

# Check user profiles exist in DB
docker compose exec postgres psql -U movieapp -d moviedb -c \
  "SELECT user_id, LEFT(likes, 50) as likes_preview FROM users WHERE likes IS NOT NULL LIMIT 5;"
```

### Deliverable 4: Four Quality Metrics (5pt each, up to 20pt)

**How to run ALL quality metrics at once:**
```bash
docker compose run --rm kafka-consumer python scripts/run_eval.py
```

This outputs the full comparison table with all four qualities for both models.

**Quality 1 — Prediction Accuracy:**
```bash
docker compose run --rm kafka-consumer python scripts/run_eval.py --offline
# Look for: RMSE, MAE, Precision@10, Recall@10 for both SVD and Baseline
```

**Quality 2 — Training Cost:**
```bash
# Same command. Look for: "Train time: X.XXs" for both models
docker compose run --rm kafka-consumer python scripts/run_eval.py --offline
```

**Quality 3 — Inference Cost:**
```bash
# Offline inference:
docker compose run --rm kafka-consumer python scripts/run_eval.py --offline
# Look for: "Inference time: X.XXXXs (N predictions)", "Per-prediction: X.XXXXms"

# Online inference (live service):
curl http://localhost:8082/stats | python3 -m json.tool
# Look for: "slow_requests_over_600ms" and tier distribution

# Also in service logs:
docker compose logs recommendation-service --tail 20
# Each line shows: "user=X recs=20 tier=cache time=0.3ms"
```

**Quality 4 — Model Size:**
```bash
docker compose run --rm kafka-consumer python scripts/run_eval.py --info
# Shows: model.pkl X.XX MB, content_data.pkl X.XX MB, etc.
```

### Deliverable 5: Prediction Service Description (10pt)

**What to check:** This README section 5 describes the prediction service and 5-tier cascade.

**How to verify:**
```bash
# Test the service is running
curl http://localhost:8082/health

# Test a recommendation request
curl http://localhost:8082/recommend/12345

# Check which tier served the request (in logs)
docker compose logs recommendation-service --tail 5
# Output: "user=12345 recs=20 tier=cache time=0.3ms"

# Check live stats
curl http://localhost:8082/stats | python3 -m json.tool
```

### Deliverable 6: Team Contract (10pt)

**What to check:** Team contract document covering: how to communicate, how to make decisions, how to handle conflict, how to handle someone not contributing, how to handle someone wanting to leave.

**Location:** Section 4 of [`m1_report.md`](m1_report.md) and Section 15 of this README.

The team contract must include a work division table with **who** did **what** by **when** for all team members, and a screenshot or link to meeting notes.

### Deliverable 7: 2000+ Successful Requests in 24h (10pt)

**How to verify:**
```bash
# Check total successful requests
docker compose run --rm kafka-consumer python scripts/run_eval.py --online
# Look for: "Total: X", "Successful: X (Y%)"

# Or query directly for the last 24 hours:
docker compose exec postgres psql -U movieapp -d moviedb -c \
  "SELECT COUNT(*) as total_requests,
          SUM(CASE WHEN status = 200 THEN 1 ELSE 0 END) as successful
   FROM recommendation_logs
   WHERE timestamp::timestamp >= NOW() - INTERVAL '24 hours';"

# Real-time monitoring:
docker compose logs -f recommendation-service
# Each line logs: "user=X recs=20 tier=Y time=Zms"
```

**If the count is low:** Make sure:
1. The service is running: `docker compose ps`
2. Port 8082 is reachable: `curl http://localhost:8082/health`
3. The Kafka consumer is ingesting recommendation logs: `docker compose logs kafka-consumer --tail 10`

### Deliverable 8: 10% Personalized Recommendations (10pt)

**How to verify:**
```bash
# Check personalization rate
docker compose run --rm kafka-consumer python scripts/run_eval.py --online
# Look for: "Unique recommendation lists: X / Y (Z%)"
# Z should be > 10%

# Quick manual check:
curl http://localhost:8082/recommend/100
curl http://localhost:8082/recommend/200
curl http://localhost:8082/recommend/300
# All three should return DIFFERENT movie lists

# Check tier distribution (cache/svd = personalized):
curl http://localhost:8082/stats | python3 -m json.tool
# "cache" and "svd" tiers are personalized
# "popularity" and "fallback" are NOT personalized
```

**If personalization is low:**
- Make sure SVD model is trained and loaded: `curl http://localhost:8082/health` (check `svd_loaded: true`)
- Retrain with more data: wait for more Kafka ratings, then re-run `run_training.py`
- Cold-start users need profiles: run `collect_data.py --users` first

### Deliverable 9: Debriefing Meeting (10pt, individual)

Schedule a meeting with your team mentor within one week after submission. Be prepared to:
- Explain the SVD + content hybrid approach and why it was chosen
- Discuss alternatives considered (nearest-neighbor CF, deep learning, item-item CF)
- Discuss cold-start alternatives (random popular, genre-only, collaborative approaches)
- Talk about teamwork experience and any issues

### Deliverable 10: Teamwork Survey (3pt, individual)

Fill out the milestone 1 teamwork survey after submission.

### Bonus: Social Activity (3pt)

> *[TODO: Document social activity]*

### Bonus: Beyond Comfort Zone (3pt)

> *[TODO: Document stretch goal]*

---

## 13. Configuration Reference

All configuration is via environment variables. No values are hardcoded in the application code. Defaults are set in [`pipeline/config.py`](pipeline/config.py) and [`service/app/config.py`](service/app/config.py).

### Data Sources

| Variable | Default | Used By | Purpose |
|---|---|---|---|
| `KAFKA_BROKER` | `128.2.220.241:9092` | Consumer | Kafka bootstrap server (use `127.0.0.1:9092` with SSH tunnel) |
| `KAFKA_TOPIC` | `movielog5` | Consumer | Team-specific Kafka topic |
| `KAFKA_GROUP_ID` | `team05-consumer` | Consumer | Consumer group (tracks offset) |
| `KAFKA_SESSION_TIMEOUT_MS` | `30000` | Consumer | Kafka session timeout |
| `API_BASE_URL` | `http://128.2.220.241:8080` | Pipeline | Course API for movie/user metadata |
| `API_BATCH_SIZE` | `200` | Pipeline | Batch size for bulk API requests |
| `API_TIMEOUT` | `10` | Pipeline | HTTP request timeout (seconds) |
| `API_RETRIES` | `2` | Pipeline | HTTP retry count |
| `API_SLEEP` | `0.3` | Pipeline | Sleep between API batches (rate limit) |

### Infrastructure

| Variable | Default | Used By | Purpose |
|---|---|---|---|
| `POSTGRES_HOST` | `postgres` | Pipeline | Postgres hostname |
| `POSTGRES_PORT` | `5432` | Pipeline | Postgres port |
| `POSTGRES_DB` | `moviedb` | Pipeline | Database name |
| `POSTGRES_USER` | `movieapp` | Pipeline | Database user |
| `POSTGRES_PASSWORD` | `movieapp123` | Pipeline | Database password |
| `REDIS_HOST` | `redis` | Service + Pipeline | Redis hostname |
| `REDIS_PORT` | `6379` | Service + Pipeline | Redis port |
| `MODEL_DIR` | `/app/models` | Service + Pipeline | Model file directory |

### Recommendation

| Variable | Default | Used By | Purpose |
|---|---|---|---|
| `MAX_RECS` | `20` | Service + Pipeline | Number of recommendations per user |
| `RECS_TTL` | `86400` (24h) | Pipeline | Redis TTL for pre-computed recs |
| `RECS_CACHE_TTL` | `3600` (1h) | Service | Redis TTL for on-the-fly cached recs |
| `MIN_RATINGS_TO_TRAIN` | `100` | Pipeline | Minimum ratings before training |
| `RESPONSE_TIME_LIMIT_MS` | `600` | Service + Eval | Course-spec response time budget |

### Feature Engineering

| Variable | Default | Used By | Purpose |
|---|---|---|---|
| `TFIDF_MAX_FEATURES` | `5000` | Pipeline | TF-IDF vocabulary size |
| `SIMILARITY_TOP_K` | `50` | Pipeline | Neighbors per movie in similarity map |

### Evaluation

| Variable | Default | Used By | Purpose |
|---|---|---|---|
| `SVD_FACTORS` | `50` | Eval | SVD latent factors for eval |
| `SVD_EPOCHS` | `20` | Eval | SVD training epochs for eval |
| `EVAL_K` | `10` | Eval | K for Precision@K and Recall@K |
| `EVAL_THRESHOLD` | `7.0` | Eval | Rating threshold for "relevant" (70% of max) |
| `EVAL_TEST_SIZE` | `0.2` | Eval | Train/test split ratio |

### Course Spec Constants

These are **fixed by the assignment** and are not configurable via environment variables:

| Constant | Value | Reason |
|---|---|---|
| `RATING_SCALE` | `(1, 10)` | Course defines ratings as 1-10 |
| `RESPONSE_TIME_LIMIT_MS` | `600` | Course requires response within 600ms |

---

## 14. Operations Guide

### Retrain the Model (Zero Downtime)

```bash
docker compose run --rm kafka-consumer python scripts/run_features.py
docker compose run --rm kafka-consumer python scripts/run_training.py
```

The service detects the new `model:version` in Redis and hot-reloads automatically.

### Run Full Evaluation

```bash
docker compose run --rm kafka-consumer python scripts/run_eval.py
```

### View Logs

```bash
docker compose logs recommendation-service --tail 30
docker compose logs kafka-consumer --tail 30
```

### Check Database Stats

```bash
docker compose run --rm kafka-consumer python scripts/collect_data.py --stats
```

### Query Database Directly

```bash
docker compose exec postgres psql -U movieapp -d moviedb
```

### After Code Changes

```bash
git pull
docker compose build
docker compose up -d
```

### Shutdown (Preserves Data)

```bash
docker compose down
# Restart later: docker compose up -d
```

### Full Reset (Deletes All Data)

```bash
docker compose down -v
# Then redo Steps 5-9 from Setup Guide
```

### Run Tests

```bash
# Service tests
cd service && python -m pytest tests/ -v

# Pipeline tests
cd pipeline && python -m pytest tests/ -v
```

---

## 15. Team Contract

### Communication

- **Primary channel:** Team Slack channel for day-to-day discussion, code reviews, and quick questions.
- **Meetings:** Weekly sync meeting (30-60 min) to review progress, blockers, and plan the next sprint. Ad-hoc calls for urgent issues.
- **Response time:** Team members respond to messages within 24 hours on weekdays.
- **Code updates:** All changes go through pull requests with at least one reviewer before merging.

### Decision Making

- **Technical decisions:** Discussed in team meetings or Slack. If there's disagreement, we present trade-offs and vote. Majority rules. For tie-breakers, we defer to whoever has the most context on that subsystem.
- **Architecture decisions:** Require consensus from all team members before implementation.
- **Deadlines:** Set collaboratively during weekly syncs. Each member commits to their deliverables.

### Conflict Resolution

- **Step 1:** Raise the issue directly with the person involved in a private, respectful conversation.
- **Step 2:** If unresolved, bring it to the full team during the next meeting for open discussion.
- **Step 3:** If still unresolved, escalate to the team mentor for mediation.
- We commit to assuming good intent and focusing on the problem, not the person.

### Handling Non-Contributing Members

- **First occurrence:** Bring it up respectfully in the team meeting. Understand if there are external factors (illness, other deadlines). Redistribute work temporarily if needed.
- **Repeated pattern:** Document the issue and have a direct conversation with the member. Set clear expectations and a timeline for improvement.
- **Escalation:** If no improvement after two weeks, raise with the team mentor and course staff.

### Handling a Member Wanting to Leave

- The member must communicate their intent at least one week before the next milestone deadline.
- The team will redistribute their remaining work and document the transition.
- The departing member must complete any in-progress work or provide a clear handoff.
- Escalate to course staff immediately to discuss grading implications.

### Work Division (Milestone 1)

| Team Member | Responsibilities | Deadline |
|---|---|---|
| Shreya Verma | ML pipeline (SVD + content-based + hybrid), Kafka consumer, PostgreSQL schema, cold-start implementation, Docker orchestration, evaluation metrics, deployment on VM | M1 submission |
| *(Add other team members and their tasks here)* | | |

> *[TODO: Add screenshot of teamwork notes or link to shared document showing detailed task tracking]*

---

## 16. Troubleshooting

| Problem | Diagnosis | Fix |
|---|---|---|
| `curl: Connection refused` on 8082 | Service not running | `docker compose up -d recommendation-service` |
| Same recs for every user | SVD model not loaded | Check `curl /health` for `svd_loaded: true`. Retrain if needed. |
| `Not enough ratings` during training | Need >= 100 ratings from Kafka | Wait, check `--stats`, retry |
| `content_data.pkl not found` | Features step was skipped | Run `run_features.py` before `run_training.py` |
| Kafka consumer can't connect | SSH tunnel not running | Re-establish: `ssh -L 9092:localhost:9092 tunnel@128.2.220.241 -NTf` |
| `bind 127.0.0.1:9092: Address in use` | Old tunnel still running | `sudo lsof -ti :9092 \| xargs sudo kill -9` then retry tunnel |
| Container keeps restarting | Crash loop | `docker compose logs <name> --tail 50` |
| Port 8082 not reachable externally | Firewall | `sudo ufw allow 8082/tcp` |
| Low personalization rate | Too few SVD users | Collect more data, retrain. Check tier distribution in `/stats`. |
| Response times over 600ms | Model too large or cache miss | Check `/stats` for slow request count. Redis should be handling most requests. |
| Want to inspect DB | Postgres shell | `docker compose exec postgres psql -U movieapp -d moviedb` |
