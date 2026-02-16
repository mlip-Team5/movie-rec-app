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
  """Test the recommend route returns hardcoded IDs as CSV."""
  userid = "123"
  response = client.get(f"/recommend/{userid}")

  assert response.status_code == 200
  # The response is now a CSV string
  data = response.data.decode("utf-8")
  expected_ids = "101,204,550,892,12"
  assert data == expected_ids


def test_recommend_route_missing_userid(client):
  """Test that the route returns 404 if userid is missing."""
  response = client.get("/recommend")
  assert response.status_code == 404
