"""Cold-start: recommend movies for users with no ratings using their self-descriptions."""

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import MAX_RECS
from storage.postgres import get_connection
from training.content import load_content_data, load_tfidf

logger = logging.getLogger(__name__)


def get_genre_list():
  """Load distinct genres from the movies table instead of hardcoding."""
  conn = get_connection()
  cur = conn.cursor()
  cur.execute("SELECT DISTINCT genres FROM movies WHERE genres IS NOT NULL AND genres != ''")
  rows = cur.fetchall()
  cur.close()
  conn.close()

  genres = set()
  for (raw,) in rows:
    for g in raw.split(","):
      g = g.strip().lower()
      if g:
        genres.add(g)
  return sorted(genres)


def process_all(cache):
  """Generate recs for all users who have self-descriptions but no ratings."""
  content_data = load_content_data()
  vectorizer, tfidf_matrix = load_tfidf()
  all_genres = get_genre_list()

  conn = get_connection()
  cur = conn.cursor()
  cur.execute("""
    SELECT u.user_id, u.likes, u.dislikes
    FROM users u
    LEFT JOIN ratings r ON u.user_id = r.user_id
    WHERE r.user_id IS NULL
      AND u.likes IS NOT NULL AND u.likes != ''
  """)
  users = cur.fetchall()
  cur.close()

  count = 0
  for user_id, likes, dislikes in users:
    if cache.user_has_recs(user_id):
      continue

    recs = _from_content(likes, dislikes or "", content_data, vectorizer, tfidf_matrix)
    if not recs:
      recs = _from_genres(conn, likes, dislikes or "", all_genres)

    if recs:
      cache.set_user_recs(user_id, recs)
      count += 1

  conn.close()
  logger.info("Cold-start recs: %d/%d users", count, len(users))


def _from_content(likes_text, dislikes_text, content_data, vectorizer, tfidf_matrix):
  """Match user description against movie content via TF-IDF cosine similarity."""
  if content_data is None or vectorizer is None or tfidf_matrix is None:
    return None

  try:
    idx_to_mid = content_data["idx_to_movie_id"]
    oversampled = MAX_RECS * 2

    likes_vec = vectorizer.transform([likes_text])
    sim_scores = cosine_similarity(likes_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(sim_scores)[-oversampled:][::-1]
    recs = [idx_to_mid[i] for i in top_indices if sim_scores[i] > 0]

    if not recs:
      return None

    if dislikes_text:
      dis_vec = vectorizer.transform([dislikes_text])
      dis_scores = cosine_similarity(dis_vec, tfidf_matrix).flatten()
      dis_top = np.argsort(dis_scores)[-MAX_RECS:][::-1]
      bad = {idx_to_mid[i] for i in dis_top if dis_scores[i] > 0}
      recs = [m for m in recs if m not in bad]

    return recs[:MAX_RECS]
  except Exception as e:
    logger.debug("Content cold-start failed: %s", e)
    return None


def _from_genres(conn, likes_text, dislikes_text, all_genres):
  """Fallback: extract genre keywords from text, query DB for top-rated matches."""
  liked = [g for g in all_genres if g in likes_text.lower()]
  if not liked:
    return None

  like_pats = [f"%{g}%" for g in liked]
  conditions = " OR ".join(["LOWER(genres) LIKE %s"] * len(like_pats))
  query = f"SELECT movie_id FROM movies WHERE ({conditions})"
  params = list(like_pats)

  disliked = [g for g in all_genres if g in (dislikes_text or "").lower()]
  if disliked:
    dis_pats = [f"%{g}%" for g in disliked]
    exclude = " AND NOT (" + " OR ".join(["LOWER(genres) LIKE %s"] * len(dis_pats)) + ")"
    query += exclude
    params.extend(dis_pats)

  query += f" ORDER BY vote_average DESC, popularity DESC LIMIT {MAX_RECS}"

  cur = conn.cursor()
  cur.execute(query, params)
  recs = [row[0] for row in cur.fetchall()]
  cur.close()
  return recs
