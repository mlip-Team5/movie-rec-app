from flask import Blueprint, jsonify, request

from models import RecommenderModel

main = Blueprint("main", __name__)
model = RecommenderModel()


@main.route("/recommend", methods=["POST"])
def recommend():
  try:
    # force=True tells Flask to skip content-type check for the time being before model integration.
    data = request.get_json(silent=True)

    # In the future, extract features from data
    user_input = data.get("input") if data else None

    predictions = model.predict(user_input)

    return jsonify({"status": "success", "recommended_movie_ids": predictions})
  except Exception as e:
    return jsonify({"status": "error", "message": str(e)}), 500


@main.route("/", methods=["GET"])
def index():
  return jsonify({"message": "Movie Recommendation API is running"})


@main.route("/health", methods=["GET"])
def health():
  return jsonify({"status": "healthy"})
