"""Shared configuration — all values from environment variables with defaults."""

import os

# Kafka
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "128.2.220.241:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "movielog5")
KAFKA_GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "team05-consumer")

# Course API
API_BASE_URL = os.environ.get("API_BASE_URL", "http://128.2.220.241:8080")

# PostgreSQL
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "moviedb")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "movieapp")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "movieapp123")

# Redis
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

# Model artifacts
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
