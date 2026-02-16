#!/usr/bin/env python3
"""Train the hybrid recommendation model and push to Redis.

Usage:
  python scripts/run_training.py
  python scripts/run_training.py --alpha 0.5
  python scripts/run_training.py --svd-only
"""

import argparse
import logging
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from config import MAX_RECS, MIN_RATINGS_TO_TRAIN, MODEL_DIR
from storage.cache import RedisCache
from storage.postgres import get_connection
from training import svd as svd_module
from training.cold_start import process_all as process_cold_start
from training.content import load_content_data
from training.hybrid import precompute

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--alpha", type=float, default=0.7, help="SVD weight (0=content, 1=SVD)")
  parser.add_argument("--svd-only", action="store_true", help="Skip content blending")
  parser.add_argument("--factors", type=int, default=50, help="SVD latent factors")
  parser.add_argument("--epochs", type=int, default=20, help="SVD training epochs")
  args = parser.parse_args()

  conn = get_connection()
  ratings_df = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", conn)
  conn.close()

  n_users = ratings_df["user_id"].nunique()
  n_movies = ratings_df["movie_id"].nunique()
  logger.info("Loaded %d ratings (%d users, %d movies)", len(ratings_df), n_users, n_movies)

  if len(ratings_df) < MIN_RATINGS_TO_TRAIN:
    logger.warning("Not enough ratings (<%d), skipping training", MIN_RATINGS_TO_TRAIN)
    return

  logger.info("Training SVD (factors=%d, epochs=%d)...", args.factors, args.epochs)
  t0 = time.time()
  model_data = svd_module.train(ratings_df, n_factors=args.factors, n_epochs=args.epochs)
  logger.info("SVD training: %.1fs", time.time() - t0)

  os.makedirs(MODEL_DIR, exist_ok=True)
  with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model_data, f)
  logger.info("Saved SVD model to %s", MODEL_DIR)

  alpha = 1.0 if args.svd_only else args.alpha
  content_data = None
  if not args.svd_only:
    content_data = load_content_data()
    if content_data is None:
      logger.warning("No content_data.pkl found. Using SVD only.")
      alpha = 1.0

  cache = RedisCache()
  logger.info("Pre-computing hybrid recs (alpha=%.2f)...", alpha)
  precompute(model_data, content_data, ratings_df, cache, alpha=alpha)

  popular_limit = MAX_RECS * 5
  conn = get_connection()
  popular_df = pd.read_sql(
    f"""SELECT movie_id, AVG(rating) as avg_r, COUNT(*) as cnt
       FROM ratings GROUP BY movie_id HAVING COUNT(*) >= 5
       ORDER BY avg_r DESC, cnt DESC LIMIT {popular_limit}""",
    conn,
  )
  conn.close()
  cache.set_popular(popular_df["movie_id"].tolist())
  logger.info("Cached %d popular movies", len(popular_df))

  logger.info("Processing cold-start users...")
  process_cold_start(cache)

  version = str(int(time.time()))
  cache.set_model_version(version)
  logger.info("Done — model version %s", version)


if __name__ == "__main__":
  main()
