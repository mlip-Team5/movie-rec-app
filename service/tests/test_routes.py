def test_index(client):
  resp = client.get("/")
  assert resp.status_code == 200
  assert b"Movie Recommendation API" in resp.data


def test_health(client):
  resp = client.get("/health")
  assert resp.status_code == 200
  assert resp.get_json()["status"] == "healthy"


def test_recommend_returns_csv(client):
  resp = client.get("/recommend/12345")
  assert resp.status_code == 200
  assert resp.content_type.startswith("text/plain")
  ids = resp.data.decode().split(",")
  assert 1 <= len(ids) <= 20
  assert all(len(mid.strip()) > 0 for mid in ids)


def test_recommend_different_users(client):
  for uid in [1, 999, 500000]:
    resp = client.get(f"/recommend/{uid}")
    assert resp.status_code == 200
    assert len(resp.data.decode()) > 0


def test_recommend_invalid_userid(client):
  resp = client.get("/recommend/notanumber")
  assert resp.status_code == 404
