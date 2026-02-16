import logging
import time
from collections import defaultdict

from flask import Blueprint, Response, jsonify

from .config import RESPONSE_TIME_LIMIT_MS
from .recommender import Recommender

logger = logging.getLogger(__name__)

main = Blueprint("main", __name__)
recommender = Recommender()

_stats = defaultdict(int)


@main.route("/recommend/<int:userid>", methods=["GET"])
def recommend(userid):
  start = time.time()
  try:
    recs, tier = recommender.predict_with_tier(userid)
    result = ",".join(str(mid) for mid in recs)
    elapsed_ms = (time.time() - start) * 1000

    _stats["total"] += 1
    _stats[f"tier:{tier}"] += 1
    if elapsed_ms > RESPONSE_TIME_LIMIT_MS:
      _stats["slow"] += 1

    logger.info(
      "user=%d recs=%d tier=%s time=%.1fms",
      userid,
      len(recs),
      tier,
      elapsed_ms,
    )
    return Response(result, mimetype="text/plain")
  except Exception as e:
    elapsed_ms = (time.time() - start) * 1000
    _stats["total"] += 1
    _stats["errors"] += 1
    logger.error("user=%d error=%s time=%.1fms", userid, e, elapsed_ms)
    return Response("", mimetype="text/plain"), 500


@main.route("/", methods=["GET"])
def index():
  return jsonify({"message": "Movie Recommendation API is running"})


@main.route("/health", methods=["GET"])
def health():
  svd_loaded = recommender._svd is not None
  content_loaded = recommender._content is not None
  return jsonify(
    {
      "status": "healthy",
      "svd_loaded": svd_loaded,
      "content_loaded": content_loaded,
      "svd_users": len(recommender._svd["user_id_map"]) if svd_loaded else 0,
      "svd_items": len(recommender._svd["raw_item_ids"]) if svd_loaded else 0,
    }
  )


@main.route("/stats", methods=["GET"])
def stats():
  """Live serving stats: tier distribution, error rate, model status."""
  total = _stats.get("total", 0)
  data = {
    "total_requests": total,
    "errors": _stats.get("errors", 0),
    f"slow_requests_over_{RESPONSE_TIME_LIMIT_MS}ms": _stats.get("slow", 0),
    "tier_distribution": {},
    "model": {
      "svd_loaded": recommender._svd is not None,
      "content_loaded": recommender._content is not None,
    },
  }
  for key, val in _stats.items():
    if key.startswith("tier:"):
      tier_name = key.split(":", 1)[1]
      data["tier_distribution"][tier_name] = {
        "count": val,
        "pct": f"{val / total * 100:.1f}%" if total else "0%",
      }
  return jsonify(data)
