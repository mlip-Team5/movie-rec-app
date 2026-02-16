#!/usr/bin/env python3
"""Entry point: build content features (TF-IDF + similarity).

Reads movies from Postgres, computes TF-IDF vectors and cosine similarity,
and saves artifacts to MODEL_DIR (content_data.pkl, tfidf_matrix.npz,
tfidf_vectorizer.pkl).

Requires: movies collected in Postgres (run collect_data.py --movies first).

Usage:
  python scripts/run_features.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
  from training.features import build_and_save

  build_and_save()
