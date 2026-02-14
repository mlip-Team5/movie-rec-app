"""Parse Kafka log lines into structured event dicts.

Event types:
  - rating:         GET /rate/<movie_slug>=<1-10>
  - watch:          GET /data/m/<movie_slug>/<minute>.mpg
  - new_account:    GET /create_account
  - recommendation: recommendation request <server>, status <code>, result: ..., <time>
"""

import re

_RATING_RE = re.compile(r"GET /rate/(.+?)=(\d+)")
_WATCH_RE = re.compile(r"GET /data/m/(.+?)/(\d+)\.mpg")
_REC_RE = re.compile(r"recommendation request (.+?), status (\d+), result: (.+), (\S+)$")


def parse_log_line(line):
  """Parse a single Kafka log line. Returns dict or None."""
  line = line.strip()
  if not line:
    return None

  parts = line.split(",", 2)
  if len(parts) < 3:
    return None

  timestamp = parts[0].strip()
  try:
    user_id = int(parts[1].strip())
  except (ValueError, IndexError):
    return None

  action = parts[2].strip()

  m = _RATING_RE.match(action)
  if m:
    return {
      "type": "rating",
      "timestamp": timestamp,
      "user_id": user_id,
      "movie_id": m.group(1),
      "rating": int(m.group(2)),
    }

  m = _WATCH_RE.match(action)
  if m:
    return {
      "type": "watch",
      "timestamp": timestamp,
      "user_id": user_id,
      "movie_id": m.group(1),
      "minute": int(m.group(2)),
    }

  if "GET /create_account" in action:
    return {"type": "new_account", "timestamp": timestamp, "user_id": user_id}

  m = _REC_RE.match(action)
  if m:
    raw = m.group(3).strip()
    recs = [r.strip() for r in raw.split(",") if r.strip()] if raw else []
    return {
      "type": "recommendation",
      "timestamp": timestamp,
      "user_id": user_id,
      "server": m.group(1),
      "status": int(m.group(2)),
      "recommendations": recs,
      "response_time": m.group(4),
    }

  return None
