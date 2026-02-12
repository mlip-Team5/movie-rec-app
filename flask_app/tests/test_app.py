import json

import pytest

from app import app


@pytest.fixture
def client():
  app.config["TESTING"] = True
  with app.test_client() as client:
    yield client


def test_index_route(client):
  """Test the index route returns a 200 and correct message."""
  response = client.get("/")
  assert response.status_code == 200
  data = json.loads(response.data)
  assert data["message"] == "Movie Recommendation API is running"


def test_recommend_route(client):
  """Test the recommend route returns hardcoded IDs."""
  # Send a sample payload
  payload = {"input": "some user data"}
  response = client.post("/recommend", data=json.dumps(payload), content_type="application/json")

  assert response.status_code == 200
  data = json.loads(response.data)

  assert data["status"] == "success"
  assert "recommended_movie_ids" in data
  # Check for the hardcoded IDs we defined in models.py
  assert data["recommended_movie_ids"] == [101, 204, 550, 892, 12]


def test_recommend_route_no_data(client):
  """Test that the route handles empty requests gracefully."""
  response = client.post("/recommend", content_type="application/json")

  # Depending on how request.get_json() behaves with no data,
  # it might return None or raise an error internally if force=True.
  # Our current implementation accepts None but standard flask behavior
  # for get_json() without data allows it.

  assert response.status_code == 200
  data = json.loads(response.data)
  assert data["status"] == "success"
  assert isinstance(data["recommended_movie_ids"], list)
