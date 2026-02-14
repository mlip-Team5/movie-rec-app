"""Content-based recommendation using TF-IDF cosine similarity."""

import logging
import os
import pickle

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

from config import MODEL_DIR

logger = logging.getLogger(__name__)


def load_content_data():
  """Load pre-computed content similarity data."""
  return _load_pickle("content_data.pkl")


def load_tfidf():
  """Load TF-IDF matrix and vectorizer."""
  mat_path = os.path.join(MODEL_DIR, "tfidf_matrix.npz")
  vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
  if not os.path.exists(mat_path):
    return None, None
  matrix = load_npz(mat_path)
  with open(vec_path, "rb") as f:
    vectorizer = pickle.load(f)
  return vectorizer, matrix


def recommend_for_user(user_id, ratings_df, content_data, top_n=20):
  """Recommend movies similar to a user's top-rated items."""
  sim_top_k = content_data["sim_top_k"]
  mid_to_idx = content_data["movie_id_to_idx"]
  idx_to_mid = content_data["idx_to_movie_id"]

  user_ratings = ratings_df[ratings_df["user_id"] == user_id]
  if user_ratings.empty:
    return []

  top_rated = user_ratings.nlargest(5, "rating")
  rated_set = set(user_ratings["movie_id"].tolist())

  scores = {}
  for _, row in top_rated.iterrows():
    idx = mid_to_idx.get(row["movie_id"])
    if idx is None or idx not in sim_top_k:
      continue
    weight = row["rating"] / 10.0
    for sim_idx, sim_score in sim_top_k[idx]:
      sim_mid = idx_to_mid.get(sim_idx)
      if sim_mid and sim_mid not in rated_set:
        scores[sim_mid] = scores.get(sim_mid, 0) + sim_score * weight

  ranked = sorted(scores, key=scores.get, reverse=True)
  return ranked[:top_n]


def recommend_from_text(text, content_data, vectorizer, tfidf_matrix, top_n=20):
  """Recommend movies from free text (user self-description)."""
  if not text or vectorizer is None:
    return []

  idx_to_mid = content_data["idx_to_movie_id"]
  text_vec = vectorizer.transform([text])
  sim_scores = cosine_similarity(text_vec, tfidf_matrix).flatten()

  top_indices = np.argsort(sim_scores)[-top_n:][::-1]
  return [idx_to_mid[i] for i in top_indices if sim_scores[i] > 0]


def scores_for_user(user_id, ratings_df, content_data, n_items, raw_item_ids):
  """Compute content-based scores for all items (for hybrid blending)."""
  sim_top_k = content_data.get("sim_top_k", {})
  mid_to_idx = content_data.get("movie_id_to_idx", {})
  idx_to_mid = content_data.get("idx_to_movie_id", {})

  item_id_to_pos = {mid: i for i, mid in enumerate(raw_item_ids)}
  scores = np.zeros(n_items, dtype=np.float32)

  user_ratings = ratings_df[ratings_df["user_id"] == user_id]
  if user_ratings.empty:
    return scores

  for _, row in user_ratings.nlargest(10, "rating").iterrows():
    content_idx = mid_to_idx.get(row["movie_id"])
    if content_idx is None or content_idx not in sim_top_k:
      continue
    weight = row["rating"] / 10.0
    for sim_content_idx, sim_score in sim_top_k[content_idx]:
      sim_mid = idx_to_mid.get(sim_content_idx)
      if sim_mid is not None:
        pos = item_id_to_pos.get(sim_mid)
        if pos is not None:
          scores[pos] += sim_score * weight

  # Normalize to match SVD score range
  if scores.max() > 0:
    scores = scores / scores.max() * 10.0

  return scores


def _load_pickle(filename):
  path = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(path):
    return None
  with open(path, "rb") as f:
    return pickle.load(f)
