from ingestion.parser import parse_log_line


def test_parse_rating():
  line = "2026-02-14T00:09:21,130276,GET /rate/please+dont+eat+the+daisies+1960=7"
  event = parse_log_line(line)
  assert event is not None
  assert event["type"] == "rating"
  assert event["user_id"] == 130276
  assert event["movie_id"] == "please+dont+eat+the+daisies+1960"
  assert event["rating"] == 7


def test_parse_watch():
  line = "2026-02-14T00:09:20.426,219411,GET /data/m/escape+to+athena+1979/73.mpg"
  event = parse_log_line(line)
  assert event is not None
  assert event["type"] == "watch"
  assert event["user_id"] == 219411
  assert event["movie_id"] == "escape+to+athena+1979"
  assert event["minute"] == 73


def test_parse_watch_classic():
  line = "2026-02-14T00:07:39,166882,GET /data/m/the+godfather+1972/148.mpg"
  event = parse_log_line(line)
  assert event["movie_id"] == "the+godfather+1972"
  assert event["minute"] == 148


def test_parse_new_account():
  line = "2026-02-14T10:30:03,99999,GET /create_account"
  event = parse_log_line(line)
  assert event is not None
  assert event["type"] == "new_account"
  assert event["user_id"] == 99999


def test_parse_recommendation():
  line = (
    "2026-02-14T00:09:21.099,215124,"
    "recommendation request 17645-team05.isri.cmu.edu:8082, status 200, "
    "result: gladiator+2000,the+godfather+1972,inception+2010, 45ms"
  )
  event = parse_log_line(line)
  assert event is not None
  assert event["type"] == "recommendation"
  assert event["user_id"] == 215124
  assert event["status"] == 200
  assert "gladiator+2000" in event["recommendations"]


def test_parse_recommendation_error():
  line = (
    "2026-02-14T00:09:21.099,215124,"
    "recommendation request 17645-team05.isri.cmu.edu:8082, status 0, "
    "result: java.net.ConnectException: Connection refused, 1ms"
  )
  event = parse_log_line(line)
  assert event is not None
  assert event["status"] == 0


def test_parse_empty():
  assert parse_log_line("") is None
  assert parse_log_line("   ") is None


def test_parse_malformed():
  assert parse_log_line("no commas here") is None
  assert parse_log_line("2024,notanumber,GET /rate/foo=5") is None


def test_parse_rating_scale():
  line = "2026-02-14T10:30:00,1,GET /rate/some+movie+2020=10"
  event = parse_log_line(line)
  assert event["rating"] == 10

  line = "2026-02-14T10:30:00,1,GET /rate/another+movie+1999=1"
  event = parse_log_line(line)
  assert event["rating"] == 1
