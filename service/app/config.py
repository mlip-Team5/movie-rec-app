import os

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
MAX_RECS = int(os.environ.get("MAX_RECS", 20))
CACHE_TTL = int(os.environ.get("RECS_CACHE_TTL", 3600))

# Course spec: response time budget (ms)
RESPONSE_TIME_LIMIT_MS = int(os.environ.get("RESPONSE_TIME_LIMIT_MS", 600))
