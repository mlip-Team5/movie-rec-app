from flask import Flask

from routes import main

app = Flask(__name__)

# Register the blueprint (controller)
app.register_blueprint(main)

if __name__ == "__main__":
  app.run(debug=True)
