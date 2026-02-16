#!/usr/bin/env python3
"""Entry point: start the Kafka consumer."""

import logging
import os
import sys
import time

# Ensure pipeline root is on the path (for local dev)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
  max_retries = int(os.environ.get("CONSUMER_MAX_RETRIES", 10))
  retry_delay = int(os.environ.get("CONSUMER_RETRY_DELAY", 5))
  for attempt in range(max_retries):
    try:
      from ingestion.consumer import run_consumer

      run_consumer()
      break
    except Exception as e:
      logging.error("Consumer failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
      time.sleep(retry_delay)
