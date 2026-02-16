"""Cold-start: recommend movies for users with no ratings using their self-descriptions."""

import logging

from storage.postgres import get_connection
from training.content import load_content_data, load_tfidf, recommend_from_text

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

    recs = _from_content(likes, dislikes or "")
    if not recs:
      recs = _from_genres(conn, likes, dislikes or "")

    if recs:
      cache.set_user_recs(user_id, recs)
      count += 1

  conn.close()
  logger.info("Cold-start recs: %d/%d users", count, len(users))


def _from_content(likes_text, dislikes_text=""):
  """Match user description against movie content via TF-IDF cosine similarity."""
  try:
    content_data = load_content_data()
    vectorizer, tfidf_matrix = load_tfidf()
    if content_data is None or vectorizer is None:
      return None

    recs = recommend_from_text(likes_text, content_data, vectorizer, tfidf_matrix, top_n=40)
    if not recs:
      return None

    if dislikes_text:
      bad = set(recommend_from_text(dislikes_text, content_data, vectorizer, tfidf_matrix, top_n=20))
      recs = [m for m in recs if m not in bad]

    return recs[:20]
  except Exception as e:
    logger.debug("Content cold-start failed: %s", e)
    return None


def _from_genres(conn, likes_text, dislikes_text=""):
  """Fallback: extract genre keywords from text, query DB for top-rated matches."""
  all_genres = get_genre_list()
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

  query += " ORDER BY vote_average DESC, popularity DESC LIMIT 20"

  cur = conn.cursor()
  cur.execute(query, params)
  recs = [row[0] for row in cur.fetchall()]
  cur.close()
  return recs
