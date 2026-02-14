import os

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
