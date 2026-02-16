"""Multi-tier recommendation engine with dynamic fallbacks."""

import logging
import os
import pickle

import numpy as np

from .config import CACHE_TTL, MAX_RECS, MODEL_DIR, REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)


class Recommender:
  def __init__(self):
    self._redis = self._connect_redis()
    self._svd = None
    self._content = None
    self._model_version = None
    self._load_models()

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

  def predict(self, userid, limit=MAX_RECS):
    recs, _ = self.predict_with_tier(userid, limit)
    return recs

  def predict_with_tier(self, userid, limit=MAX_RECS):
    """Return (recs, tier_name) for observability."""
    self._check_model_update()

    recs = self._from_cache(userid)
    if recs:
      return recs[:limit], "cache"

    if self._svd:
      recs = self._svd_predict(userid, limit)
      if recs:
        self._to_cache(userid, recs)
        return recs, "svd"

    if self._content:
      recs = self._content_fallback(limit)
      if recs:
        self._to_cache(userid, recs)
        return recs, "content"

    recs = self._from_popularity(limit)
    if recs:
      return recs, "popularity"

    return self._dynamic_fallback(limit), "fallback"

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

    seed_count = max(1, limit // 4)
    top_voted = np.argsort(votes)[-seed_count:][::-1]
    candidates = {}
    neighbors_per_seed = max(1, limit // seed_count)
    for seed in top_voted:
      for sim_idx, score in sim.get(seed, [])[:neighbors_per_seed]:
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

  def _dynamic_fallback(self, limit):
    """Last resort: top-rated movies from content data, or first N item IDs from model."""
    if self._content:
      votes = self._content.get("vote_averages")
      idx_to_mid = self._content.get("idx_to_movie_id", {})
      if votes is not None and len(votes) > 0:
        top_idx = np.argsort(votes)[-limit:][::-1]
        return [idx_to_mid[i] for i in top_idx if i in idx_to_mid]

    if self._svd:
      return [str(mid) for mid in self._svd["raw_item_ids"][:limit]]

    return []

  def _to_cache(self, userid, recs):
    if not self._redis:
      return
    try:
      self._redis.setex(f"recs:user:{userid}", CACHE_TTL, ",".join(str(x) for x in recs))
    except Exception:
      pass
