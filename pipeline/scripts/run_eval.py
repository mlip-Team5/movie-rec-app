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
from surprise import SVD, Dataset, NormalPredictor, Reader, accuracy
from surprise.model_selection import train_test_split

from config import MIN_RATINGS_TO_TRAIN, MODEL_DIR, RATING_SCALE, RESPONSE_TIME_LIMIT_MS
from storage.postgres import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_MB = 1024 * 1024


def _get_ratings_df():
  """Load ratings from Postgres using cursor to avoid SQLAlchemy warnings."""
  conn = get_connection()
  cur = conn.cursor()
  cur.execute("SELECT user_id, movie_id, rating FROM ratings")
  rows = cur.fetchall()
  cur.close()
  conn.close()
  return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])


def offline_eval():
  """Offline metrics: compare SVD vs Content-Based (NormalPredictor baseline).

  Reports the 4 M1 qualities for both models:
    1. Prediction accuracy (RMSE, MAE, Precision@10, Recall@10)
    2. Training cost (wall-clock seconds)
    3. Inference cost (wall-clock seconds for test set)
    4. Model size (bytes on disk)
  """
  logger.info("=" * 60)
  logger.info("OFFLINE EVALUATION — TWO-MODEL COMPARISON")
  logger.info("=" * 60)

  ratings_df = _get_ratings_df()

  n_ratings = len(ratings_df)
  n_users = ratings_df["user_id"].nunique()
  n_movies = ratings_df["movie_id"].nunique()
  logger.info("Dataset: %d ratings, %d users, %d movies", n_ratings, n_users, n_movies)
  logger.info("Rating distribution:")
  for stat, val in ratings_df["rating"].describe().items():
    logger.info("  %s: %.2f", stat, val)

  if n_ratings < MIN_RATINGS_TO_TRAIN:
    logger.warning("Not enough ratings for evaluation (need >= %d)", MIN_RATINGS_TO_TRAIN)
    return

  eval_k = int(os.environ.get("EVAL_K", 10))
  eval_threshold = float(os.environ.get("EVAL_THRESHOLD", RATING_SCALE[1] * 0.7))
  eval_test_size = float(os.environ.get("EVAL_TEST_SIZE", 0.2))
  svd_factors = int(os.environ.get("SVD_FACTORS", 50))
  svd_epochs = int(os.environ.get("SVD_EPOCHS", 20))

  reader = Reader(rating_scale=RATING_SCALE)
  data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)
  trainset, testset = train_test_split(data, test_size=eval_test_size, random_state=42)

  # ── Model 1: SVD Collaborative Filtering ──
  logger.info("\n" + "─" * 50)
  logger.info("MODEL 1: SVD (%d factors, %d epochs)", svd_factors, svd_epochs)
  logger.info("─" * 50)

  t0 = time.time()
  svd = SVD(n_factors=svd_factors, n_epochs=svd_epochs, random_state=42)
  svd.fit(trainset)
  svd_train_time = time.time() - t0

  t0 = time.time()
  svd_preds = svd.test(testset)
  svd_infer_time = time.time() - t0

  svd_rmse = accuracy.rmse(svd_preds, verbose=False)
  svd_mae = accuracy.mae(svd_preds, verbose=False)
  svd_prec, svd_rec = _precision_recall_at_k(svd_preds, k=eval_k, threshold=eval_threshold)

  svd_model_path = os.path.join(MODEL_DIR, "model.pkl")
  svd_size = os.path.getsize(svd_model_path) if os.path.exists(svd_model_path) else 0

  logger.info("  [Accuracy]")
  logger.info("    RMSE:           %.4f", svd_rmse)
  logger.info("    MAE:            %.4f", svd_mae)
  logger.info("    Precision@%d:   %.4f", eval_k, svd_prec)
  logger.info("    Recall@%d:      %.4f", eval_k, svd_rec)
  logger.info("  [Training cost]")
  logger.info("    Train time:     %.2fs", svd_train_time)
  logger.info("  [Inference cost]")
  logger.info("    Inference time: %.4fs (%d predictions)", svd_infer_time, len(svd_preds))
  logger.info("    Per-prediction: %.4fms", (svd_infer_time / max(len(svd_preds), 1) * 1000))
  logger.info("  [Model size]")
  svd_size_mb = svd_size / _MB
  logger.info("    model.pkl:      %.2f MB", svd_size_mb)

  # ── Model 2: Baseline (NormalPredictor — random from rating distribution) ──
  logger.info("\n" + "─" * 50)
  logger.info("MODEL 2: Baseline (NormalPredictor)")
  logger.info("─" * 50)

  t0 = time.time()
  baseline = NormalPredictor()
  baseline.fit(trainset)
  bl_train_time = time.time() - t0

  t0 = time.time()
  bl_preds = baseline.test(testset)
  bl_infer_time = time.time() - t0

  bl_rmse = accuracy.rmse(bl_preds, verbose=False)
  bl_mae = accuracy.mae(bl_preds, verbose=False)
  bl_prec, bl_rec = _precision_recall_at_k(bl_preds, k=eval_k, threshold=eval_threshold)

  logger.info("  [Accuracy]")
  logger.info("    RMSE:           %.4f", bl_rmse)
  logger.info("    MAE:            %.4f", bl_mae)
  logger.info("    Precision@%d:   %.4f", eval_k, bl_prec)
  logger.info("    Recall@%d:      %.4f", eval_k, bl_rec)
  logger.info("  [Training cost]")
  logger.info("    Train time:     %.2fs", bl_train_time)
  logger.info("  [Inference cost]")
  logger.info("    Inference time: %.4fs (%d predictions)", bl_infer_time, len(bl_preds))
  logger.info("    Per-prediction: %.4fms", (bl_infer_time / max(len(bl_preds), 1) * 1000))
  logger.info("  [Model size]")
  logger.info("    (in-memory only, no file)")

  # ── Comparison table ──
  logger.info("\n" + "=" * 60)
  logger.info("COMPARISON TABLE (for M1 report)")
  logger.info("=" * 60)
  logger.info("%-25s  %15s  %15s", "Quality", "SVD", "Baseline")
  logger.info("-" * 60)
  logger.info("%-25s  %15.4f  %15.4f", "RMSE (lower=better)", svd_rmse, bl_rmse)
  logger.info("%-25s  %15.4f  %15.4f", "MAE (lower=better)", svd_mae, bl_mae)
  logger.info("%-25s  %15.4f  %15.4f", f"Precision@{eval_k} (higher)", svd_prec, bl_prec)
  logger.info("%-25s  %15.4f  %15.4f", f"Recall@{eval_k} (higher)", svd_rec, bl_rec)
  logger.info("%-25s  %14.2fs  %14.2fs", "Training cost", svd_train_time, bl_train_time)
  logger.info("%-25s  %14.4fs  %14.4fs", "Inference cost", svd_infer_time, bl_infer_time)
  logger.info("%-25s  %13.2f MB  %15s", "Model size", svd_size_mb, "~0 MB")
  logger.info("-" * 60)
  rmse_improvement = (1 - svd_rmse / bl_rmse) * 100
  logger.info("SVD RMSE improvement over baseline: %.1f%%", rmse_improvement)
  if svd_rmse < bl_rmse:
    logger.info(">>> RECOMMENDATION: Use SVD in production (better accuracy)")
  else:
    logger.info(">>> NOTE: SVD not yet outperforming baseline — collect more ratings")

  # ── Content model info ──
  logger.info("\n" + "─" * 50)
  logger.info("CONTENT-BASED MODEL (used for cold-start & hybrid)")
  logger.info("─" * 50)
  content_path = os.path.join(MODEL_DIR, "content_data.pkl")
  tfidf_path = os.path.join(MODEL_DIR, "tfidf_matrix.npz")
  if os.path.exists(content_path):
    content_size = os.path.getsize(content_path)
    tfidf_size = os.path.getsize(tfidf_path) if os.path.exists(tfidf_path) else 0
    with open(content_path, "rb") as f:
      cd = pickle.load(f)
    logger.info("  Movies:     %d", len(cd.get("movie_ids", [])))
    logger.info("  Genres:     %d (%s)", len(cd.get("genre_names", [])), cd.get("genre_names", []))
    logger.info("  Similarity: %d movies with neighbors", len(cd.get("sim_top_k", {})))
    logger.info("  Size:       %.2f MB (content) + %.2f MB (tfidf)",
                 content_size / _MB, tfidf_size / _MB)
  else:
    logger.warning("  content_data.pkl not found")


def online_eval():
  """Online stats from recommendation_logs and Kafka feedback."""
  logger.info("=" * 60)
  logger.info("ONLINE EVALUATION")
  logger.info("=" * 60)

  conn = get_connection()
  cur = conn.cursor()

  # Recommendation log stats
  try:
    cur.execute("""
      SELECT
        COUNT(*) as total,
        COALESCE(SUM(CASE WHEN status = 200 THEN 1 ELSE 0 END), 0) as ok,
        COALESCE(SUM(CASE WHEN status != 200 THEN 1 ELSE 0 END), 0) as fail,
        MIN(timestamp), MAX(timestamp)
      FROM recommendation_logs
    """)
    total, success, failed, first_ts, last_ts = cur.fetchone()
    logger.info("Recommendation requests:")
    logger.info("  Total:      %d", total)
    logger.info("  Successful: %d (%.1f%%)", success, (success / total * 100) if total else 0)
    logger.info("  Failed:     %d", failed)
    logger.info("  First:      %s", first_ts)
    logger.info("  Last:       %s", last_ts)
    if total == 0:
      logger.info("  (No recommendation logs yet — course server may not have started hitting your API)")
  except Exception as e:
    logger.warning("Could not read recommendation_logs: %s", e)
    conn.rollback()

  # Personalization — use all available logs; no arbitrary LIMIT
  try:
    cur.execute("""
      SELECT recommendations FROM recommendation_logs
      WHERE status = 200 AND recommendations IS NOT NULL AND recommendations != ''
      ORDER BY id DESC
    """)
    rows = cur.fetchall()
    if rows:
      recs_list = [r[0] for r in rows]
      unique = len(set(recs_list))
      total_r = len(recs_list)
      logger.info("\nPersonalization (last %d successful requests):", total_r)
      logger.info("  Unique recommendation lists: %d / %d (%.1f%%)", unique, total_r, unique / total_r * 100)
      min_personalization = float(os.environ.get("MIN_PERSONALIZATION", 0.1))
      if unique / total_r < min_personalization:
        logger.warning("  LOW PERSONALIZATION: less than %.0f%% unique lists", min_personalization * 100)
      else:
        logger.info("  Personalization looks good")
  except Exception as e:
    logger.warning("Could not check personalization: %s", e)
    conn.rollback()

  # Response times — use all available logs; no arbitrary LIMIT
  try:
    cur.execute("""
      SELECT response_time FROM recommendation_logs
      WHERE status = 200 AND response_time IS NOT NULL
      ORDER BY id DESC
    """)
    rows = cur.fetchall()
    times_ms = []
    for (rt,) in rows:
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
      over_limit = sum(1 for t in times_ms if t > RESPONSE_TIME_LIMIT_MS)
      if over_limit:
        logger.warning("  %d requests OVER %dms limit!", over_limit, RESPONSE_TIME_LIMIT_MS)
      else:
        logger.info("  All within %dms limit", RESPONSE_TIME_LIMIT_MS)
  except Exception as e:
    logger.warning("Could not check response times: %s", e)
    conn.rollback()

  # User satisfaction
  try:
    cur.execute("""
      SELECT COUNT(*), AVG(r.rating)
      FROM watch_events w
      JOIN ratings r ON w.user_id = r.user_id AND w.movie_id = r.movie_id
    """)
    watches, avg_rating = cur.fetchone()
    if watches and watches > 0:
      logger.info("\nUser satisfaction proxy:")
      logger.info("  Watched movies that were also rated: %d", watches)
      logger.info("  Average rating of watched movies:    %.2f / %d", avg_rating, RATING_SCALE[1])
  except Exception as e:
    logger.warning("Could not compute satisfaction: %s", e)
    conn.rollback()

  # DB stats
  logger.info("\nDatabase stats:")
  for table in ["movies", "users", "ratings", "watch_events", "raw_events", "recommendation_logs"]:
    try:
      cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
      logger.info("  %-25s %d rows", table + ":", cur.fetchone()[0])
    except Exception:
      conn.rollback()
  cur.close()
  conn.close()


def model_info():
  """Print model file info and key stats."""
  logger.info("=" * 60)
  logger.info("MODEL INFO")
  logger.info("=" * 60)

  files = ["model.pkl", "content_data.pkl", "tfidf_matrix.npz", "tfidf_vectorizer.pkl"]
  for f in files:
    path = os.path.join(MODEL_DIR, f)
    if os.path.exists(path):
      size_mb = os.path.getsize(path) / _MB
      logger.info("  %-25s %.2f MB", f, size_mb)
    else:
      logger.info("  %-25s not found", f)

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


def main():
  parser = argparse.ArgumentParser(description="Evaluate recommendation system")
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
