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
  for attempt in range(10):
    try:
      from ingestion.consumer import run_consumer

      run_consumer()
      break
    except Exception as e:
      logging.error("Consumer failed (attempt %d): %s", attempt + 1, e)
      time.sleep(5)
