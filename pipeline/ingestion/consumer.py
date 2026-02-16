"""Kafka consumer: ingests events from the movielog stream into Postgres."""

import logging
import os

from confluent_kafka import Consumer, KafkaError

from config import KAFKA_BROKER, KAFKA_GROUP_ID, KAFKA_SESSION_TIMEOUT_MS, KAFKA_TOPIC
from ingestion.api_client import fetch_movie, fetch_user
from ingestion.parser import parse_log_line
from ingestion.validators import validate_rating, validate_recommendation, validate_watch
from storage.postgres import (
  get_connection,
  init_db,
  insert_raw_event,
  insert_recommendation_log,
  upsert_movie,
  upsert_rating,
  upsert_user,
  upsert_watch,
)

logger = logging.getLogger(__name__)

LOG_INTERVAL = int(os.environ.get("CONSUMER_LOG_INTERVAL", 10000))


def run_consumer():
  init_db()
  conn = get_connection()

  consumer = Consumer({
    "bootstrap.servers": KAFKA_BROKER,
    "group.id": KAFKA_GROUP_ID,
    "auto.offset.reset": "earliest",
    "enable.auto.commit": True,
    "session.timeout.ms": KAFKA_SESSION_TIMEOUT_MS,
  })
  consumer.subscribe([KAFKA_TOPIC])
  logger.info("Consuming from %s at %s", KAFKA_TOPIC, KAFKA_BROKER)

  known_movies = set()
  known_users = set()
  count = 0
  errors = 0

  try:
    while True:
      msg = consumer.poll(1.0)
      if msg is None:
        continue
      if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
          continue
        logger.error("Kafka error: %s", msg.error())
        continue

      raw_line = msg.value().decode("utf-8", errors="replace")
      event = parse_log_line(raw_line)
      if not event:
        continue

      try:
        etype = event["type"]

        insert_raw_event(conn, event.get("timestamp", ""), event["user_id"], etype, raw_line)

        if etype == "rating":
          ok, reason = validate_rating(event)
          if not ok:
            errors += 1
            logger.debug("Invalid rating: %s", reason)
            continue
          upsert_rating(conn, event["user_id"], event["movie_id"], event["rating"], event.get("timestamp"))
          _ensure_movie(conn, event["movie_id"], known_movies)

        elif etype == "watch":
          ok, reason = validate_watch(event)
          if not ok:
            errors += 1
            logger.debug("Invalid watch: %s", reason)
            continue
          upsert_watch(conn, event["user_id"], event["movie_id"])
          _ensure_movie(conn, event["movie_id"], known_movies)

        elif etype == "new_account":
          _ensure_user(conn, event["user_id"], known_users)

        elif etype == "recommendation":
          ok, reason = validate_recommendation(event)
          if not ok:
            errors += 1
            logger.debug("Invalid recommendation log: %s", reason)
            continue
          insert_recommendation_log(
            conn,
            event.get("timestamp", ""),
            event["user_id"],
            event.get("server", ""),
            event["status"],
            event.get("recommendations", []),
            event.get("response_time", ""),
          )

        count += 1
        if LOG_INTERVAL > 0 and count % LOG_INTERVAL == 0:
          logger.info("Processed %d events (%d validation errors)", count, errors)

      except Exception as e:
        logger.error("Error processing event: %s", e)
        try:
          conn.close()
        except Exception:
          pass
        conn = get_connection()

  except KeyboardInterrupt:
    logger.info("Shutting down (processed %d events, %d errors)", count, errors)
  finally:
    consumer.close()
    conn.close()


def _ensure_movie(conn, movie_id, known):
  if movie_id in known:
    return
  known.add(movie_id)
  cur = conn.cursor()
  cur.execute("SELECT 1 FROM movies WHERE movie_id = %s", (movie_id,))
  exists = cur.fetchone()
  cur.close()
  if not exists:
    data = fetch_movie(movie_id)
    if data:
      upsert_movie(conn, data.get("id", movie_id), data)


def _ensure_user(conn, user_id, known):
  if user_id in known:
    return
  known.add(user_id)
  data = fetch_user(user_id)
  if data:
    upsert_user(conn, user_id, data)
    logger.debug("Stored new user %d", user_id)
