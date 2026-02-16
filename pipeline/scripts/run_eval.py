#!/usr/bin/env python3
"""Evaluate everything: offline model metrics, online service stats, model info.

Usage:
  python scripts/run_eval.py              # full report
  python scripts/run_eval.py --offline    # only offline metrics
  python scripts/run_eval.py --online     # only online stats
  python scripts/run_eval.py --info       # only model file info
"""

import argparse
import logging
import math
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

from config import MODEL_DIR
from storage.postgres import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def offline_eval():
  """Offline metrics: RMSE, MAE, Precision@K, Recall@K on held-out test set."""
  logger.info("=" * 60)
  logger.info("OFFLINE EVALUATION")
  logger.info("=" * 60)

  conn = get_connection()
  ratings_df = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", conn)
  conn.close()

  n_ratings = len(ratings_df)
  n_users = ratings_df["user_id"].nunique()
  n_movies = ratings_df["movie_id"].nunique()
  logger.info("Dataset: %d ratings, %d users, %d movies", n_ratings, n_users, n_movies)
  logger.info("Rating distribution:")
  for stat, val in ratings_df["rating"].describe().items():
    logger.info("  %s: %.2f", stat, val)

  if n_ratings < 100:
    logger.warning("Not enough ratings for evaluation")
    return

  reader = Reader(rating_scale=(1, 10))
  data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)

  # 80/20 split, reproducible
  trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

  # --- SVD ---
  logger.info("\n--- SVD (50 factors, 20 epochs) ---")
  t0 = time.time()
  svd = SVD(n_factors=50, n_epochs=20, random_state=42)
  svd.fit(trainset)
  svd_train_time = time.time() - t0

  t0 = time.time()
  svd_preds = svd.test(testset)
  svd_inference_time = time.time() - t0

  svd_rmse = accuracy.rmse(svd_preds, verbose=False)
  svd_mae = accuracy.mae(svd_preds, verbose=False)
  svd_prec, svd_rec = _precision_recall_at_k(svd_preds, k=10, threshold=7.0)

  logger.info("  RMSE:           %.4f", svd_rmse)
  logger.info("  MAE:            %.4f", svd_mae)
  logger.info("  Precision@10:   %.4f", svd_prec)
  logger.info("  Recall@10:      %.4f", svd_rec)
  logger.info("  Train time:     %.1fs", svd_train_time)
  logger.info("  Inference time:  %.3fs (%d predictions)", svd_inference_time, len(svd_preds))

  # --- Content-based baseline (predict global mean) ---
  logger.info("\n--- Baseline (global mean) ---")
  global_mean = trainset.global_mean
  baseline_rmse = math.sqrt(sum((true_r - global_mean) ** 2 for _, _, true_r in testset) / len(testset))
  baseline_mae = sum(abs(true_r - global_mean) for _, _, true_r in testset) / len(testset)
  logger.info("  RMSE:           %.4f", baseline_rmse)
  logger.info("  MAE:            %.4f", baseline_mae)

  logger.info("\n--- Comparison ---")
  rmse_improvement = (1 - svd_rmse / baseline_rmse) * 100
  logger.info("  SVD vs Baseline RMSE improvement: %.1f%%", rmse_improvement)

  # --- Model sizes ---
  logger.info("\n--- Model sizes ---")
  _print_model_sizes()


def online_eval():
  """Online stats from recommendation_logs and Kafka feedback."""
  logger.info("=" * 60)
  logger.info("ONLINE EVALUATION")
  logger.info("=" * 60)

  conn = get_connection()

  # Recommendation log stats
  try:
    stats_df = pd.read_sql("""
      SELECT
        COUNT(*) as total_requests,
        SUM(CASE WHEN status = 200 THEN 1 ELSE 0 END) as successful,
        SUM(CASE WHEN status != 200 THEN 1 ELSE 0 END) as failed,
        MIN(timestamp) as first_request,
        MAX(timestamp) as last_request
      FROM recommendation_logs
    """, conn)

    row = stats_df.iloc[0]
    total = int(row["total_requests"])
    success = int(row["successful"])
    failed = int(row["failed"])
    logger.info("Recommendation requests:")
    logger.info("  Total:      %d", total)
    logger.info("  Successful: %d (%.1f%%)", success, (success / total * 100) if total else 0)
    logger.info("  Failed:     %d", failed)
    logger.info("  First:      %s", row["first_request"])
    logger.info("  Last:       %s", row["last_request"])
  except Exception as e:
    logger.warning("Could not read recommendation_logs: %s", e)

  # Personalization check: how many unique recommendation lists in recent 200 logs?
  try:
    recent_df = pd.read_sql("""
      SELECT recommendations FROM recommendation_logs
      WHERE status = 200 AND recommendations IS NOT NULL AND recommendations != ''
      ORDER BY id DESC LIMIT 200
    """, conn)

    if not recent_df.empty:
      unique_lists = recent_df["recommendations"].nunique()
      total_lists = len(recent_df)
      logger.info("\nPersonalization (last %d successful requests):", total_lists)
      logger.info("  Unique recommendation lists: %d / %d (%.1f%%)",
                   unique_lists, total_lists, unique_lists / total_lists * 100)
      if unique_lists / total_lists < 0.1:
        logger.warning("  LOW PERSONALIZATION: less than 10%% unique lists")
      else:
        logger.info("  Personalization looks good")
  except Exception as e:
    logger.warning("Could not check personalization: %s", e)

  # Response time stats
  try:
    rt_df = pd.read_sql("""
      SELECT response_time FROM recommendation_logs
      WHERE status = 200 AND response_time IS NOT NULL
      ORDER BY id DESC LIMIT 500
    """, conn)

    if not rt_df.empty:
      # Parse response times like "45ms" or "0.045s"
      times_ms = []
      for rt in rt_df["response_time"]:
        rt = str(rt).strip()
        if rt.endswith("ms"):
          try:
            times_ms.append(float(rt[:-2]))
          except ValueError:
            pass

      if times_ms:
        logger.info("\nResponse times (last %d requests):", len(times_ms))
        logger.info("  Mean:   %.0fms", np.mean(times_ms))
        logger.info("  Median: %.0fms", np.median(times_ms))
        logger.info("  P95:    %.0fms", np.percentile(times_ms, 95))
        logger.info("  P99:    %.0fms", np.percentile(times_ms, 99))
        logger.info("  Max:    %.0fms", np.max(times_ms))
        over_600 = sum(1 for t in times_ms if t > 600)
        if over_600:
          logger.warning("  %d requests over 600ms limit!", over_600)
  except Exception as e:
    logger.warning("Could not check response times: %s", e)

  # Rating distribution of recommended movies
  try:
    watch_df = pd.read_sql("""
      SELECT COUNT(*) as watches, AVG(r.rating) as avg_rating
      FROM watch_events w
      JOIN ratings r ON w.user_id = r.user_id AND w.movie_id = r.movie_id
    """, conn)
    if not watch_df.empty and watch_df.iloc[0]["watches"] > 0:
      logger.info("\nUser satisfaction proxy:")
      logger.info("  Watched movies that were also rated: %d", int(watch_df.iloc[0]["watches"]))
      logger.info("  Average rating of watched movies:    %.2f / 10", watch_df.iloc[0]["avg_rating"])
  except Exception as e:
    logger.warning("Could not compute satisfaction: %s", e)

  # DB stats
  logger.info("\nDatabase stats:")
  cur = conn.cursor()
  for table in ["movies", "users", "ratings", "watch_events", "raw_events", "recommendation_logs"]:
    try:
      cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
      logger.info("  %s: %d", table, cur.fetchone()[0])
    except Exception:
      conn.rollback()
  cur.close()
  conn.close()


def model_info():
  """Print model file info and key stats."""
  logger.info("=" * 60)
  logger.info("MODEL INFO")
  logger.info("=" * 60)

  _print_model_sizes()

  model_path = os.path.join(MODEL_DIR, "model.pkl")
  if os.path.exists(model_path):
    with open(model_path, "rb") as f:
      model_data = pickle.load(f)

    n_users = len(model_data.get("user_id_map", {}))
    n_items = len(model_data.get("raw_item_ids", []))
    n_factors = model_data["user_factors"].shape[1] if "user_factors" in model_data else 0

    logger.info("\nSVD model:")
    logger.info("  Users:          %d", n_users)
    logger.info("  Items:          %d", n_items)
    logger.info("  Latent factors: %d", n_factors)
    logger.info("  Global mean:    %.2f", model_data.get("global_mean", 0))
    logger.info("  User factors:   %s", model_data["user_factors"].shape)
    logger.info("  Item factors:   %s", model_data["item_factors"].shape)

    sample_ids = list(model_data["user_id_map"].keys())[:5]
    logger.info("  Sample user IDs: %s", sample_ids)
    sample_items = model_data["raw_item_ids"][:5]
    logger.info("  Sample item IDs: %s", sample_items)
  else:
    logger.warning("model.pkl not found at %s", model_path)

  content_path = os.path.join(MODEL_DIR, "content_data.pkl")
  if os.path.exists(content_path):
    with open(content_path, "rb") as f:
      content = pickle.load(f)
    logger.info("\nContent data:")
    logger.info("  Movies:     %d", len(content.get("movie_ids", [])))
    logger.info("  Genres:     %s", content.get("genre_names", []))
    logger.info("  Similarity: %d movies with neighbors", len(content.get("sim_top_k", {})))
  else:
    logger.warning("content_data.pkl not found at %s", content_path)


def _precision_recall_at_k(predictions, k=10, threshold=7.0):
  """Precision and Recall at K for all users."""
  user_est = {}
  user_true = {}
  for uid, iid, true_r, est, _ in predictions:
    user_est.setdefault(uid, []).append((iid, est))
    user_true.setdefault(uid, []).append((iid, true_r))

  precisions = []
  recalls = []
  for uid in user_est:
    top_k = sorted(user_est[uid], key=lambda x: x[1], reverse=True)[:k]
    relevant = {iid for iid, r in user_true[uid] if r >= threshold}
    if not relevant:
      continue
    recommended_relevant = sum(1 for iid, _ in top_k if iid in relevant)
    precisions.append(recommended_relevant / k)
    recalls.append(recommended_relevant / len(relevant))

  return np.mean(precisions) if precisions else 0.0, np.mean(recalls) if recalls else 0.0


def _print_model_sizes():
  files = ["model.pkl", "content_data.pkl", "tfidf_matrix.npz", "tfidf_vectorizer.pkl"]
  for f in files:
    path = os.path.join(MODEL_DIR, f)
    if os.path.exists(path):
      size_mb = os.path.getsize(path) / (1024 * 1024)
      logger.info("  %s: %.1f MB", f, size_mb)
    else:
      logger.info("  %s: not found", f)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--offline", action="store_true", help="Run offline evaluation only")
  parser.add_argument("--online", action="store_true", help="Run online evaluation only")
  parser.add_argument("--info", action="store_true", help="Show model info only")
  args = parser.parse_args()

  run_all = not (args.offline or args.online or args.info)

  if run_all or args.info:
    model_info()
  if run_all or args.offline:
    offline_eval()
  if run_all or args.online:
    online_eval()


if __name__ == "__main__":
  main()
