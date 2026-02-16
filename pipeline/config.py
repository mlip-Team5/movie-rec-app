"""Shared configuration loaded from environment variables."""

import os

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "128.2.220.241:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "movielog5")
KAFKA_GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "team05-consumer")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://128.2.220.241:8080")

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "moviedb")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "movieapp")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "movieapp123")

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")

MAX_RECS = int(os.environ.get("MAX_RECS", 20))
RECS_TTL = int(os.environ.get("RECS_TTL", 86400))
MIN_RATINGS_TO_TRAIN = int(os.environ.get("MIN_RATINGS_TO_TRAIN", 100))
