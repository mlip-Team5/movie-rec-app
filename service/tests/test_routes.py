def test_index(client):
  resp = client.get("/")
  assert resp.status_code == 200
  assert b"Movie Recommendation API" in resp.data


def test_health(client):
  resp = client.get("/health")
  assert resp.status_code == 200
  data = resp.get_json()
  assert data["status"] == "healthy"
  assert "svd_loaded" in data
  assert "content_loaded" in data


def test_recommend_returns_csv(client):
  resp = client.get("/recommend/12345")
  assert resp.status_code == 200
  assert resp.content_type.startswith("text/plain")
  body = resp.data.decode()
  if body:
    ids = body.split(",")
    assert len(ids) <= 20
    assert all(len(mid.strip()) > 0 for mid in ids)


def test_recommend_different_users(client):
  for uid in [1, 999, 500000]:
    resp = client.get(f"/recommend/{uid}")
    assert resp.status_code == 200


def test_recommend_invalid_userid(client):
  resp = client.get("/recommend/notanumber")
  assert resp.status_code == 404


def test_stats_endpoint(client):
  client.get("/recommend/1")
  resp = client.get("/stats")
  assert resp.status_code == 200
  data = resp.get_json()
  assert "total_requests" in data
  assert "tier_distribution" in data
  assert "model" in data
  assert data["total_requests"] >= 1
