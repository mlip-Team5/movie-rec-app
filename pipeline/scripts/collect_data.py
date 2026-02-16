#!/usr/bin/env python3
"""Batch data collection from the course API.

Usage:
  python scripts/collect_data.py --movies
  python scripts/collect_data.py --users
  python scripts/collect_data.py --all
  python scripts/collect_data.py --stats
  python scripts/collect_data.py --movies --max-id 5000
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.api_client import fetch_movies_bulk, fetch_users_bulk
from storage.postgres import get_connection, init_db, upsert_movie, upsert_user

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def collect_movies(conn, max_id):
  logger.info("Fetching movies 1-%d from course API...", max_id)
  all_ids = list(range(1, max_id + 1))
  fetched = 0
  for i in range(0, len(all_ids), 200):
    batch = all_ids[i : i + 200]
    movies = fetch_movies_bulk(batch)
    for m in movies:
      mid = m.get("id", m.get("movie_id"))
      if mid:
        upsert_movie(conn, mid, m)
        fetched += 1
    if (i // 200) % 10 == 0:
      logger.info("  fetched %d movies so far...", fetched)
    time.sleep(0.3)
  logger.info("Collected %d movies", fetched)


def collect_users(conn, max_id):
  logger.info("Fetching users 1-%d from course API...", max_id)
  all_ids = list(range(1, max_id + 1))
  fetched = 0
  for i in range(0, len(all_ids), 200):
    batch = all_ids[i : i + 200]
    users = fetch_users_bulk(batch)
    for u in users:
      uid = u.get("user_id", u.get("id"))
      if uid:
        upsert_user(conn, int(uid), u)
        fetched += 1
    if (i // 200) % 50 == 0:
      logger.info("  fetched %d users so far...", fetched)
    time.sleep(0.3)
  logger.info("Collected %d users", fetched)


def show_stats(conn):
  cur = conn.cursor()
  for table in ["movies", "users", "ratings", "watch_events", "raw_events", "recommendation_logs"]:
    try:
      cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
      logger.info("  %s: %d rows", table, cur.fetchone()[0])
    except Exception:
      conn.rollback()
      logger.info("  %s: (table not yet created)", table)
  cur.close()


def main():
  parser = argparse.ArgumentParser(description="Collect data from course API")
  parser.add_argument("--movies", action="store_true")
  parser.add_argument("--users", action="store_true")
  parser.add_argument("--all", action="store_true")
  parser.add_argument("--stats", action="store_true")
  parser.add_argument("--max-movie-id", type=int, default=25000, help="Highest movie ID to fetch")
  parser.add_argument("--max-user-id", type=int, default=100000, help="Highest user ID to fetch")
  args = parser.parse_args()

  init_db()
  conn = get_connection()

  if args.all or args.movies:
    collect_movies(conn, args.max_movie_id)
  if args.all or args.users:
    collect_users(conn, args.max_user_id)

  show_stats(conn)
  conn.close()


if __name__ == "__main__":
  main()
