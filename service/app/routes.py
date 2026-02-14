import logging
import time

from flask import Blueprint, Response, jsonify

from .recommender import Recommender

logger = logging.getLogger(__name__)

main = Blueprint("main", __name__)
recommender = Recommender()


@main.route("/recommend/<int:userid>", methods=["GET"])
def recommend(userid):
  start = time.time()
  try:
    recs = recommender.predict(userid)
    result = ",".join(str(mid) for mid in recs)
    elapsed_ms = (time.time() - start) * 1000
    logger.info("user=%d recs=%d time=%.1fms", userid, len(recs), elapsed_ms)
    return Response(result, mimetype="text/plain")
  except Exception as e:
    elapsed_ms = (time.time() - start) * 1000
    logger.error("user=%d error=%s time=%.1fms", userid, e, elapsed_ms)
    return Response("", mimetype="text/plain"), 500


@main.route("/", methods=["GET"])
def index():
  return jsonify({"message": "Movie Recommendation API is running"})


@main.route("/health", methods=["GET"])
def health():
  return jsonify({"status": "healthy"})
