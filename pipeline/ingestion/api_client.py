"""HTTP client for the course User/Movie API."""

import logging
import time

import requests

from config import API_BASE_URL, API_BATCH_SIZE, API_RETRIES, API_SLEEP, API_TIMEOUT

logger = logging.getLogger(__name__)
_session = requests.Session()


def fetch_user(user_id):
  return _fetch(f"{API_BASE_URL}/user/{user_id}")


def fetch_movie(movie_id):
  return _fetch(f"{API_BASE_URL}/movie/{movie_id}")


def fetch_users_bulk(user_ids, batch_size=API_BATCH_SIZE):
  return _fetch_bulk("user", user_ids, batch_size)


def fetch_movies_bulk(movie_ids, batch_size=API_BATCH_SIZE):
  return _fetch_bulk("movie", movie_ids, batch_size)


def _fetch_bulk(entity, ids, batch_size):
  results = []
  for i in range(0, len(ids), batch_size):
    batch = ids[i : i + batch_size]
    ids_str = ",".join(str(x) for x in batch)
    data = _fetch(f"{API_BASE_URL}/{entity}/{ids_str}")
    if isinstance(data, list):
      results.extend(data)
    elif isinstance(data, dict):
      results.append(data)
    time.sleep(API_SLEEP)
  return results


def _fetch(url, retries=API_RETRIES):
  for attempt in range(retries + 1):
    try:
      resp = _session.get(url, timeout=API_TIMEOUT)
      if resp.status_code == 200:
        return resp.json()
      logger.warning("API %d for %s", resp.status_code, url)
      return None
    except Exception as e:
      if attempt < retries:
        time.sleep(1)
      else:
        logger.error("Failed to fetch %s: %s", url, e)
        return None
  return None
