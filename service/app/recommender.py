"""Multi-tier recommendation engine.

Prediction cascade:
  1. Redis cache (pre-computed hybrid recs)
  2. Real-time SVD scoring
  3. Content-based fallback
  4. Popularity list from Redis
  5. Hardcoded defaults
"""

import logging
import os
import pickle

import numpy as np

from .config import MODEL_DIR, REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)

DEFAULT_RECS = [
  "the+shawshank+redemption+1994",
  "the+godfather+1972",
  "the+dark+knight+2008",
  "pulp+fiction+1994",
  "forrest+gump+1994",
  "inception+2010",
  "the+matrix+1999",
  "goodfellas+1990",
  "interstellar+2014",
  "the+lord+of+the+rings+the+return+of+the+king+2003",
  "fight+club+1999",
  "the+lord+of+the+rings+the+fellowship+of+the+ring+2001",
  "gladiator+2000",
  "the+silence+of+the+lambs+1991",
  "schindlers+list+1993",
  "saving+private+ryan+1998",
  "spirited+away+2001",
  "the+green+mile+1999",
  "se7en+1995",
  "casablanca+1942",
]


class Recommender:
  def __init__(self):
    self._redis = self._connect_redis()
    self._svd = None
    self._content = None
    self._model_version = None
    self._load_models()

  # ── Connections ──────────────────────────────────────────────────

  def _connect_redis(self):
    try:
      import redis

      r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
      r.ping()
      logger.info("Connected to Redis")
      return r
    except Exception as e:
      logger.warning("Redis unavailable: %s", e)
      return None

  # ── Model loading ───────────────────────────────────────────────

  def _load_models(self):
    self._svd = self._load_pickle("model.pkl")
    self._content = self._load_pickle("content_data.pkl")

    if self._svd:
      n_u = len(self._svd.get("user_id_map", {}))
      n_i = len(self._svd.get("raw_item_ids", []))
      logger.info("SVD loaded: %d users, %d items", n_u, n_i)
    if self._content:
      logger.info("Content data loaded: %d movies", len(self._content.get("movie_ids", [])))

  def _load_pickle(self, filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
      return None
    try:
      with open(path, "rb") as f:
        return pickle.load(f)
    except Exception as e:
      logger.warning("Failed to load %s: %s", filename, e)
      return None

  def _check_model_update(self):
    if not self._redis:
      return
    try:
      version = self._redis.get("model:version")
      if version and version != self._model_version:
        self._load_models()
        self._model_version = version
        logger.info("Models reloaded, version=%s", version)
    except Exception:
      pass

  # ── Prediction cascade ──────────────────────────────────────────

  def predict(self, userid, limit=20):
    self._check_model_update()

    # Tier 1: Redis cache
    recs = self._from_cache(userid)
    if recs:
      return recs

    # Tier 2: Real-time SVD
    if self._svd:
      recs = self._svd_predict(userid, limit)
      if recs:
        self._to_cache(userid, recs)
        return recs

    # Tier 3: Content-based fallback
    if self._content:
      recs = self._content_fallback(limit)
      if recs:
        self._to_cache(userid, recs)
        return recs

    # Tier 4: Popularity from Redis
    recs = self._from_popularity(limit)
    if recs:
      return recs

    # Tier 5: Hardcoded defaults
    return DEFAULT_RECS[:limit]

  # ── Tier implementations ────────────────────────────────────────

  def _from_cache(self, userid):
    if not self._redis:
      return None
    try:
      cached = self._redis.get(f"recs:user:{userid}")
      if cached:
        return [x.strip() for x in cached.split(",") if x.strip()]
    except Exception:
      pass
    return None

  def _svd_predict(self, userid, limit):
    uid_map = self._svd.get("user_id_map", {})
    if userid not in uid_map:
      return None

    inner_uid = uid_map[userid]
    user_vec = self._svd["user_factors"][inner_uid]
    user_bias = self._svd["user_biases"][inner_uid]

    scores = (
      self._svd["item_factors"] @ user_vec
      + self._svd["item_biases"]
      + self._svd["global_mean"]
      + user_bias
    )

    # Zero out already-rated items
    for idx in self._svd.get("user_rated_items", {}).get(inner_uid, set()):
      scores[idx] = -np.inf

    raw_ids = self._svd["raw_item_ids"]
    top_idx = np.argpartition(scores, -limit)[-limit:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    return [str(raw_ids[i]) for i in top_idx[:limit]]

  def _content_fallback(self, limit):
    sim = self._content.get("sim_top_k", {})
    idx_to_mid = self._content.get("idx_to_movie_id", {})
    votes = self._content.get("vote_averages")
    if votes is None or len(votes) == 0:
      return None

    top_voted = np.argsort(votes)[-5:][::-1]
    candidates = {}
    for seed in top_voted:
      for sim_idx, score in sim.get(seed, [])[:10]:
        mid = idx_to_mid.get(sim_idx)
        if mid:
          candidates[mid] = candidates.get(mid, 0) + score

    ranked = sorted(candidates, key=candidates.get, reverse=True)
    return ranked[:limit] if ranked else None

  def _from_popularity(self, limit):
    if not self._redis:
      return None
    try:
      popular = self._redis.get("recs:popular")
      if popular:
        return [x.strip() for x in popular.split(",") if x.strip()][:limit]
    except Exception:
      pass
    return None

  def _to_cache(self, userid, recs, ttl=3600):
    if not self._redis:
      return
    try:
      self._redis.setex(f"recs:user:{userid}", ttl, ",".join(str(x) for x in recs))
    except Exception:
      pass
