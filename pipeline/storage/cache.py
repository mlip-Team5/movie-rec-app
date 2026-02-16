"""Redis cache for pre-computed recommendations."""

import logging

import redis as redis_lib

from config import RECS_TTL, REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)


class RedisCache:
  def __init__(self):
    self._r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    self._r.ping()
    logger.info("Connected to Redis at %s:%d", REDIS_HOST, REDIS_PORT)

  @property
  def client(self):
    return self._r

  def set_user_recs(self, user_id, recs, ttl=RECS_TTL):
    self._r.setex(f"recs:user:{user_id}", ttl, ",".join(str(x) for x in recs))

  def set_popular(self, movie_ids, ttl=RECS_TTL):
    self._r.setex("recs:popular", ttl, ",".join(str(x) for x in movie_ids))

  def set_model_version(self, version):
    self._r.set("model:version", version)

  def user_has_recs(self, user_id):
    return self._r.exists(f"recs:user:{user_id}")

  def pipeline(self):
    return self._r.pipeline()
