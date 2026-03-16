import logging
import os

from flask import Flask

from routes import main

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app():
  application = Flask(__name__)
  application.register_blueprint(main)
  logger.info("App initialized")
  return application


app = create_app()

if __name__ == "__main__":
  debug = os.environ.get("FLASK_DEBUG", "0") == "1"
  port = int(os.environ.get("PORT", 8082))
  app.run(host="0.0.0.0", port=port, debug=debug)
