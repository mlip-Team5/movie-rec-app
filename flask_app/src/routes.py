from flask import Blueprint, jsonify

from models import RecommenderModel

main = Blueprint("main", __name__)
model = RecommenderModel()


@main.route("/recommend/<userid>", methods=["GET"])
def recommend(userid):
  try:
    predictions = model.predict(userid)
    movie_ids = ",".join(map(str, predictions))
    return movie_ids
  except Exception as e:
    return jsonify({"status": "error", "message": str(e)}), 500


@main.route("/", methods=["GET"])
def index():
  return jsonify({"message": "Movie Recommendation API is running"})


@main.route("/health", methods=["GET"])
def health():
  return jsonify({"status": "healthy"})
