# Movie Recommendation Service

## Quick Start (Docker)

```bash
# from movie-rec-app/
docker compose up --build
```

Service runs at `http://localhost:8082/recommend/<userid>`

## Local Development

```bash
cd flask_app
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
python src/app.py
```

## Testing

```bash
cd flask_app
pytest
```

## API

| Endpoint | Method | Response |
|---|---|---|
| `/recommend/<userid>` | GET | Comma-separated movie IDs (plain text) |
| `/health` | GET | `{"status": "healthy"}` |
| `/` | GET | Status message |

## Project Structure

```
flask_app/
  src/
    app.py       - Flask app factory + entrypoint
    routes.py    - API endpoints
    models.py    - Recommendation model interface
  tests/
    test_app.py  - Endpoint tests
  Dockerfile     - Production container
docker-compose.yml - Service orchestration
```
