# Movie Recommendation Flask App

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note that it is recommended to set up a virtual python environment prior to running the app to prevent dependency conflicts

## Running the app
You can either run `python app.py` or do the following in your terminal
```
export FLASK_APP=src/app.py
export FLASK_ENV=development
flask run
```

## Project Structure

*   **src/**: Source code for the application.
    *   `app.py`: Main entry point that initializes the Flask app and registers blueprints.
    *   `models.py`: Contains the `RecommenderModel` class responsible for prediction logic.
    *   `routes.py`: Defines the API endpoints (Controller) and handles request processing.
*   **tests/**: Contains unit tests.
    *   `test_app.py`: Tests for the API endpoints using `pytest`.
*   **pyproject.toml**: Configuration file for development tools like `ruff` (linting/formatting) and `pytest`.