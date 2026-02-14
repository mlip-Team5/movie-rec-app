"""Data quality validation for incoming events."""

import logging

logger = logging.getLogger(__name__)


def validate_rating(event):
  if event.get("type") != "rating":
    return False, "not a rating event"
  if not isinstance(event.get("user_id"), int) or event["user_id"] <= 0:
    return False, "invalid user_id"
  mid = event.get("movie_id")
  if not mid or not isinstance(mid, str):
    return False, "invalid movie_id"
  r = event.get("rating")
  if not isinstance(r, (int, float)) or r < 1 or r > 10:
    return False, f"rating out of range: {r}"
  return True, None


def validate_watch(event):
  if event.get("type") != "watch":
    return False, "not a watch event"
  if not isinstance(event.get("user_id"), int) or event["user_id"] <= 0:
    return False, "invalid user_id"
  mid = event.get("movie_id")
  if not mid or not isinstance(mid, str):
    return False, "invalid movie_id"
  if not isinstance(event.get("minute"), int) or event["minute"] < 0:
    return False, "invalid minute"
  return True, None


def check_rating_drift(recent_ratings, historical_avg=5.5, threshold=1.0):
  """Returns True if the recent average deviates from historical by more than threshold."""
  if not recent_ratings or len(recent_ratings) < 10:
    return False
  avg = sum(recent_ratings) / len(recent_ratings)
  drift = abs(avg - historical_avg)
  if drift > threshold:
    logger.warning("Rating drift: avg=%.2f historical=%.1f drift=%.2f", avg, historical_avg, drift)
    return True
  return False


def check_genre_drift(recent_genres, baseline_genres, threshold=0.2):
  """Returns True if genre distribution has shifted beyond threshold."""
  if not recent_genres or not baseline_genres:
    return False
  all_genres = set(recent_genres.keys()) | set(baseline_genres.keys())
  total_recent = sum(recent_genres.values()) or 1
  total_baseline = sum(baseline_genres.values()) or 1
  max_shift = max(
    abs(recent_genres.get(g, 0) / total_recent - baseline_genres.get(g, 0) / total_baseline)
    for g in all_genres
  )
  if max_shift > threshold:
    logger.warning("Genre drift: max_shift=%.3f", max_shift)
    return True
  return False
