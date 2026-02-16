from ingestion.validators import (
  check_genre_drift,
  check_rating_drift,
  validate_rating,
  validate_recommendation,
  validate_watch,
)


def test_validate_rating_valid():
  event = {"type": "rating", "user_id": 1, "movie_id": "gladiator+2000", "rating": 7}
  valid, _ = validate_rating(event)
  assert valid is True


def test_validate_rating_out_of_range():
  event = {"type": "rating", "user_id": 1, "movie_id": "foo+2000", "rating": 11}
  valid, _ = validate_rating(event)
  assert valid is False


def test_validate_rating_zero():
  event = {"type": "rating", "user_id": 1, "movie_id": "foo+2000", "rating": 0}
  valid, _ = validate_rating(event)
  assert valid is False


def test_validate_rating_max():
  event = {"type": "rating", "user_id": 1, "movie_id": "foo+2000", "rating": 10}
  valid, _ = validate_rating(event)
  assert valid is True


def test_validate_rating_negative_user():
  event = {"type": "rating", "user_id": -1, "movie_id": "foo+2000", "rating": 3}
  valid, _ = validate_rating(event)
  assert valid is False


def test_validate_watch_valid():
  event = {"type": "watch", "user_id": 1, "movie_id": "the+matrix+1999", "minute": 5}
  valid, _ = validate_watch(event)
  assert valid is True


def test_validate_watch_negative_minute():
  event = {"type": "watch", "user_id": 1, "movie_id": "foo+2000", "minute": -1}
  valid, _ = validate_watch(event)
  assert valid is False


def test_rating_drift_detected():
  assert check_rating_drift([9.0] * 20, historical_avg=5.5, threshold=1.0) is True


def test_rating_drift_none():
  assert check_rating_drift([5.5] * 20, historical_avg=5.5, threshold=1.0) is False


def test_rating_drift_too_few():
  assert check_rating_drift([9.0, 9.0], historical_avg=5.5) is False


def test_genre_drift():
  baseline = {"action": 50, "comedy": 50}
  recent = {"action": 90, "comedy": 10}
  assert check_genre_drift(recent, baseline, threshold=0.2) is True


def test_genre_no_drift():
  baseline = {"action": 50, "comedy": 50}
  recent = {"action": 48, "comedy": 52}
  assert check_genre_drift(recent, baseline, threshold=0.2) is False


# ── Recommendation validation ────────────────────────────────────────


def test_validate_recommendation_valid():
  event = {
    "type": "recommendation",
    "user_id": 215124,
    "server": "17645-team05.isri.cmu.edu:8082",
    "status": 200,
    "recommendations": ["gladiator+2000", "inception+2010"],
    "response_time": "45ms",
  }
  valid, _ = validate_recommendation(event)
  assert valid is True


def test_validate_recommendation_error_status():
  event = {
    "type": "recommendation",
    "user_id": 1,
    "status": 0,
    "recommendations": [],
  }
  valid, _ = validate_recommendation(event)
  assert valid is True  # status 0 is still valid to log — it records a failure


def test_validate_recommendation_bad_user():
  event = {"type": "recommendation", "user_id": -5, "status": 200}
  valid, _ = validate_recommendation(event)
  assert valid is False


def test_validate_recommendation_wrong_type():
  event = {"type": "rating", "user_id": 1, "status": 200}
  valid, _ = validate_recommendation(event)
  assert valid is False


def test_validate_recommendation_missing_status():
  event = {"type": "recommendation", "user_id": 1}
  valid, _ = validate_recommendation(event)
  assert valid is False
