"""Feature engineering: TF-IDF vectors, content similarity, genre vectors."""

import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

from config import MODEL_DIR
from storage.postgres import get_connection

logger = logging.getLogger(__name__)

ALL_GENRES = [
  "action", "adventure", "animation", "comedy", "crime", "documentary",
  "drama", "family", "fantasy", "history", "horror", "music", "mystery",
  "romance", "science fiction", "thriller", "tv movie", "war", "western",
]


def build_and_save():
  """Run the full feature engineering pipeline."""
  os.makedirs(MODEL_DIR, exist_ok=True)

  movies_df = _load_movies()
  if movies_df.empty:
    logger.warning("No movies in DB -- run data collection first")
    return

  # TF-IDF on movie content
  logger.info("Computing TF-IDF on %d movies...", len(movies_df))
  content_strings = _build_content_strings(movies_df)
  tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
  tfidf_matrix = tfidf.fit_transform(content_strings)
  logger.info("TF-IDF matrix: %s", tfidf_matrix.shape)

  # Content similarity (top-50 per movie)
  logger.info("Computing content similarity...")
  sim_top_k = _compute_similarity(tfidf_matrix, top_k=50)

  # Genre vectors
  genre_matrix, genre_names = _compute_genre_vectors(movies_df)

  # Build index maps
  movie_id_to_idx = {mid: i for i, mid in enumerate(movies_df["movie_id"])}
  idx_to_movie_id = {i: mid for mid, i in movie_id_to_idx.items()}

  # Save content data
  content_data = {
    "movie_id_to_idx": movie_id_to_idx,
    "idx_to_movie_id": idx_to_movie_id,
    "sim_top_k": sim_top_k,
    "genre_matrix": genre_matrix,
    "genre_names": list(genre_names),
    "movie_ids": movies_df["movie_id"].tolist(),
    "vote_averages": movies_df["vote_average"].values,
    "popularities": movies_df["popularity"].values,
    "costs": movies_df["cost"].values,
  }
  _save_pickle(content_data, "content_data.pkl")

  # Save TF-IDF artifacts
  save_npz(os.path.join(MODEL_DIR, "tfidf_matrix.npz"), tfidf_matrix)
  _save_pickle(tfidf, "tfidf_vectorizer.pkl")

  logger.info("Feature engineering complete")


# ── Internal helpers ──────────────────────────────────────────────


def _load_movies():
  conn = get_connection()
  df = pd.read_sql(
    "SELECT movie_id, title, genres, vote_average, popularity, adult, cost, raw_data FROM movies",
    conn,
  )
  conn.close()
  return df


def _build_content_strings(movies_df):
  """Build a text blob per movie for TF-IDF: genres + overview + keywords."""
  contents = []
  for _, row in movies_df.iterrows():
    parts = []
    if row.get("genres"):
      parts.append(str(row["genres"]).replace(",", " "))
    raw = row.get("raw_data")
    if isinstance(raw, dict):
      overview = raw.get("overview", "")
      if overview:
        parts.append(overview)
      keywords = raw.get("keywords", "")
      if isinstance(keywords, str) and keywords:
        parts.append(keywords.replace(",", " "))
    contents.append(" ".join(parts) if parts else "unknown")
  return contents


def _compute_similarity(tfidf_matrix, top_k=50, batch_size=500):
  """Cosine similarity, keeping only top-K per movie."""
  n = tfidf_matrix.shape[0]
  sim_top_k = {}
  for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    sim_batch = cosine_similarity(tfidf_matrix[start:end], tfidf_matrix)
    for i in range(end - start):
      row = sim_batch[i]
      row[start + i] = -1  # exclude self
      top_idx = np.argpartition(row, -top_k)[-top_k:]
      top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
      sim_top_k[start + i] = [(int(j), float(row[j])) for j in top_idx if row[j] > 0]
    if start % 2000 == 0:
      logger.info("  similarity: %d/%d", start, n)
  return sim_top_k


def _compute_genre_vectors(movies_df):
  def parse(g):
    if not g or pd.isna(g):
      return []
    return [x.strip().lower() for x in str(g).split(",")]

  genre_lists = movies_df["genres"].apply(parse)
  mlb = MultiLabelBinarizer(classes=ALL_GENRES)
  return mlb.fit_transform(genre_lists), mlb.classes_


def _save_pickle(obj, filename):
  path = os.path.join(MODEL_DIR, filename)
  with open(path, "wb") as f:
    pickle.dump(obj, f)
  logger.info("Saved %s", path)
