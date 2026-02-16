"""PostgreSQL schema and upsert operations."""

import logging

import psycopg2
from psycopg2.extras import Json

from config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
  -- ── Reference data (from course API) ─────────────────────────────

  CREATE TABLE IF NOT EXISTS movies (
    movie_id     TEXT PRIMARY KEY,
    title        TEXT,
    genres       TEXT,
    vote_average REAL DEFAULT 0,
    popularity   REAL DEFAULT 0,
    adult        BOOLEAN DEFAULT FALSE,
    cost         REAL DEFAULT 0,
    raw_data     JSONB
  );

  CREATE TABLE IF NOT EXISTS users (
    user_id    INTEGER PRIMARY KEY,
    age        INTEGER,
    gender     TEXT,
    occupation TEXT,
    likes      TEXT,
    dislikes   TEXT,
    raw_data   JSONB
  );

  -- ── Derived / aggregated event tables ────────────────────────────

  CREATE TABLE IF NOT EXISTS ratings (
    user_id   INTEGER NOT NULL,
    movie_id  TEXT    NOT NULL,
    rating    REAL    NOT NULL,
    timestamp TEXT,
    PRIMARY KEY (user_id, movie_id)
  );

  CREATE TABLE IF NOT EXISTS watch_events (
    user_id         INTEGER NOT NULL,
    movie_id        TEXT    NOT NULL,
    minutes_watched INTEGER DEFAULT 1,
    PRIMARY KEY (user_id, movie_id)
  );

  -- ── Append-only event log (audit trail, never modified) ──────────

  CREATE TABLE IF NOT EXISTS raw_events (
    id         SERIAL PRIMARY KEY,
    timestamp  TEXT        NOT NULL,
    user_id    INTEGER     NOT NULL,
    event_type TEXT        NOT NULL,
    raw_line   TEXT        NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
  );

  -- ── Recommendation logs (course server feedback) ─────────────────

  CREATE TABLE IF NOT EXISTS recommendation_logs (
    id              SERIAL PRIMARY KEY,
    timestamp       TEXT        NOT NULL,
    user_id         INTEGER     NOT NULL,
    server          TEXT,
    status          INTEGER     NOT NULL,
    recommendations TEXT,
    response_time   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
  );

  -- ── Indexes ──────────────────────────────────────────────────────

  CREATE INDEX IF NOT EXISTS idx_ratings_user    ON ratings(user_id);
  CREATE INDEX IF NOT EXISTS idx_ratings_movie   ON ratings(movie_id);
  CREATE INDEX IF NOT EXISTS idx_watch_user      ON watch_events(user_id);
  CREATE INDEX IF NOT EXISTS idx_raw_events_type ON raw_events(event_type);
  CREATE INDEX IF NOT EXISTS idx_raw_events_user ON raw_events(user_id);
  CREATE INDEX IF NOT EXISTS idx_rec_logs_user   ON recommendation_logs(user_id);
  CREATE INDEX IF NOT EXISTS idx_rec_logs_status ON recommendation_logs(status);
"""


def get_connection():
  return psycopg2.connect(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
  )


def init_db():
  conn = get_connection()
  cur = conn.cursor()
  cur.execute(_SCHEMA_SQL)
  conn.commit()
  cur.close()
  conn.close()
  logger.info("Database tables initialized")


def upsert_rating(conn, user_id, movie_id, rating, timestamp=None):
  cur = conn.cursor()
  cur.execute(
    """INSERT INTO ratings (user_id, movie_id, rating, timestamp)
       VALUES (%s, %s, %s, %s)
       ON CONFLICT (user_id, movie_id)
       DO UPDATE SET rating = EXCLUDED.rating, timestamp = EXCLUDED.timestamp""",
    (user_id, movie_id, rating, timestamp),
  )
  conn.commit()
  cur.close()


def upsert_watch(conn, user_id, movie_id):
  cur = conn.cursor()
  cur.execute(
    """INSERT INTO watch_events (user_id, movie_id, minutes_watched)
       VALUES (%s, %s, 1)
       ON CONFLICT (user_id, movie_id)
       DO UPDATE SET minutes_watched = watch_events.minutes_watched + 1""",
    (user_id, movie_id),
  )
  conn.commit()
  cur.close()


def upsert_movie(conn, movie_id, data):
  genres = data.get("genres", "")
  if isinstance(genres, list):
    genres = ",".join(
      g["name"] if isinstance(g, dict) and "name" in g else str(g) for g in genres
    )
  cost = data.get("license_cost") or data.get("cost") or 0
  cur = conn.cursor()
  cur.execute(
    """INSERT INTO movies (movie_id, title, genres, vote_average, popularity, adult, cost, raw_data)
       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
       ON CONFLICT (movie_id)
       DO UPDATE SET title = EXCLUDED.title, genres = EXCLUDED.genres,
                     vote_average = EXCLUDED.vote_average, cost = EXCLUDED.cost,
                     raw_data = EXCLUDED.raw_data""",
    (
      movie_id,
      data.get("title") or data.get("original_title") or "",
      genres,
      data.get("vote_average") or 0,
      data.get("popularity") or 0,
      data.get("adult", False),
      cost,
      Json(data),
    ),
  )
  conn.commit()
  cur.close()


def upsert_user(conn, user_id, data):
  likes = (
    data.get("self_description_likes")
    or data.get("likes")
    or ""
  )
  dislikes = (
    data.get("self_description_dislikes")
    or data.get("dislikes")
    or ""
  )
  cur = conn.cursor()
  cur.execute(
    """INSERT INTO users (user_id, age, gender, occupation, likes, dislikes, raw_data)
       VALUES (%s, %s, %s, %s, %s, %s, %s)
       ON CONFLICT (user_id)
       DO UPDATE SET likes = EXCLUDED.likes, dislikes = EXCLUDED.dislikes,
                     raw_data = EXCLUDED.raw_data""",
    (
      user_id,
      data.get("age"),
      data.get("gender"),
      data.get("occupation"),
      likes,
      dislikes,
      Json(data),
    ),
  )
  conn.commit()
  cur.close()


# ── Append-only event storage ─────────────────────────────────────────


def insert_raw_event(conn, timestamp, user_id, event_type, raw_line):
  """Store a raw Kafka line. Append-only, never modified."""
  cur = conn.cursor()
  cur.execute(
    """INSERT INTO raw_events (timestamp, user_id, event_type, raw_line)
       VALUES (%s, %s, %s, %s)""",
    (timestamp, user_id, event_type, raw_line),
  )
  conn.commit()
  cur.close()


def insert_recommendation_log(conn, timestamp, user_id, server, status, recommendations, response_time):
  """Store a recommendation request result from the course server."""
  recs_str = ",".join(str(r) for r in recommendations) if recommendations else ""
  cur = conn.cursor()
  cur.execute(
    """INSERT INTO recommendation_logs
       (timestamp, user_id, server, status, recommendations, response_time)
       VALUES (%s, %s, %s, %s, %s, %s)""",
    (timestamp, user_id, server, status, recs_str, response_time),
  )
  conn.commit()
  cur.close()
