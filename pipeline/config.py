"""Shared configuration loaded from environment variables."""

import os

# --- Data sources ---
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "128.2.220.241:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "movielog5")
KAFKA_GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "team05-consumer")
KAFKA_SESSION_TIMEOUT_MS = int(os.environ.get("KAFKA_SESSION_TIMEOUT_MS", 30000))

API_BASE_URL = os.environ.get("API_BASE_URL", "http://128.2.220.241:8080")
API_BATCH_SIZE = int(os.environ.get("API_BATCH_SIZE", 200))
API_TIMEOUT = int(os.environ.get("API_TIMEOUT", 10))
API_RETRIES = int(os.environ.get("API_RETRIES", 2))
API_SLEEP = float(os.environ.get("API_SLEEP", 0.3))

# --- Infrastructure ---
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "moviedb")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "movieapp")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "movieapp123")

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")

# --- Recommendation ---
MAX_RECS = int(os.environ.get("MAX_RECS", 20))
RECS_TTL = int(os.environ.get("RECS_TTL", 86400))
MIN_RATINGS_TO_TRAIN = int(os.environ.get("MIN_RATINGS_TO_TRAIN", 100))

# --- Feature engineering ---
TFIDF_MAX_FEATURES = int(os.environ.get("TFIDF_MAX_FEATURES", 5000))
SIMILARITY_TOP_K = int(os.environ.get("SIMILARITY_TOP_K", 50))

# --- Course spec constants (fixed by assignment, not configurable) ---
RATING_SCALE = (1, 10)
RESPONSE_TIME_LIMIT_MS = 600
