import os

from . import app

if __name__ == "__main__":
  debug = os.environ.get("FLASK_DEBUG", "0") == "1"
  port = int(os.environ.get("PORT", 8082))
  app.run(host="0.0.0.0", port=port, debug=debug)
