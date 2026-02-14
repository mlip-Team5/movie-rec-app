FROM python:3.11-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8082

CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8082", "--timeout", "2", "app:app"]
