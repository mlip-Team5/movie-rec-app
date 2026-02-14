"""Hybrid recommendation: blend SVD + content scores, precompute to Redis."""

import logging

import numpy as np

from training import svd as svd_module
from training import content as content_module

logger = logging.getLogger(__name__)


def precompute(model_data, content_data, ratings_df, cache, alpha=0.7, batch_size=500):
  """Compute hybrid recs for all users and store in Redis.

  hybrid_score = alpha * svd + (1 - alpha) * content
  """
  uid_map = model_data["user_id_map"]
  raw_ids = model_data["raw_item_ids"]
  rated_map = model_data["user_rated_items"]
  n_items = len(raw_ids)
  use_content = content_data is not None and alpha < 1.0

  pipe = cache.pipeline()
  count = 0

  for raw_uid, inner_uid in uid_map.items():
    svd_scores = svd_module.scores_for_user(inner_uid, model_data)

    if use_content:
      cont_scores = content_module.scores_for_user(
        raw_uid, ratings_df, content_data, n_items, raw_ids
      )
      final = alpha * svd_scores + (1 - alpha) * cont_scores
    else:
      final = svd_scores

    # Mask already-rated items
    for idx in rated_map.get(inner_uid, set()):
      final[idx] = -np.inf

    # Top-20
    top_idx = np.argpartition(final, -20)[-20:]
    top_idx = top_idx[np.argsort(final[top_idx])[::-1]]
    top_ids = [str(raw_ids[i]) for i in top_idx[:20]]

    pipe.setex(f"recs:user:{raw_uid}", 86400, ",".join(top_ids))
    count += 1

    if count % batch_size == 0:
      pipe.execute()
      pipe = cache.pipeline()
      logger.info("Pre-computed recs: %d/%d users", count, len(uid_map))

  pipe.execute()
  logger.info("Pre-computed hybrid recs for %d users (alpha=%.2f)", count, alpha)
