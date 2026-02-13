import logging
import time

from flask import Blueprint, Response, jsonify

from models import RecommenderModel

logger = logging.getLogger(__name__)

main = Blueprint("main", __name__)
model = RecommenderModel()


@main.route("/recommend/<int:userid>", methods=["GET"])
def recommend(userid):
  start = time.time()
  try:
    predictions = model.predict(userid)
    result = ",".join(str(mid) for mid in predictions)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(f"user={userid} recs={len(predictions)} time={elapsed_ms:.1f}ms")
    return Response(result, mimetype="text/plain")
  except Exception as e:
    elapsed_ms = (time.time() - start) * 1000
    logger.error(f"user={userid} error={e} time={elapsed_ms:.1f}ms")
    return Response("", mimetype="text/plain"), 500


@main.route("/", methods=["GET"])
def index():
  return jsonify({"message": "Movie Recommendation API is running"})


@main.route("/health", methods=["GET"])
def health():
  return jsonify({"status": "healthy"})
