import logging
import os

from flask import Flask

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_app():
  application = Flask(__name__)
  from .routes import main

  application.register_blueprint(main)
  return application


app = create_app()

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8082))
  app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
