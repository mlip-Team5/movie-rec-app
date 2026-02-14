FROM python:3.11-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends gcc g++ \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY ingestion/ ./ingestion/
COPY storage/ ./storage/
COPY training/ ./training/
COPY scripts/ ./scripts/

CMD ["python", "scripts/run_consumer.py"]
