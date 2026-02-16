"""SVD collaborative filtering using scikit-surprise."""

import logging

import numpy as np
from surprise import SVD, Dataset, Reader

from config import RATING_SCALE

logger = logging.getLogger(__name__)


def train(ratings_df, n_factors=50, n_epochs=20):
  """Train SVD model and return (model_data dict, trainset)."""
  reader = Reader(rating_scale=RATING_SCALE)
  data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)
  trainset = data.build_full_trainset()

  model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
  model.fit(trainset)

  model_data = _extract(model, trainset)
  logger.info("SVD trained: %d users, %d items", len(model_data["user_id_map"]), len(model_data["raw_item_ids"]))
  return model_data


def _extract(model, trainset):
  """Extract numpy arrays and mappings from trained SVD model."""
  user_id_map = {trainset.to_raw_uid(i): i for i in trainset.all_users()}
  raw_item_ids = [trainset.to_raw_iid(i) for i in trainset.all_items()]
  user_rated_items = {i: {iid for iid, _ in trainset.ur[i]} for i in trainset.all_users()}

  return {
    "user_factors": np.array(model.pu),
    "item_factors": np.array(model.qi),
    "user_biases": np.array(model.bu),
    "item_biases": np.array(model.bi),
    "global_mean": trainset.global_mean,
    "user_id_map": user_id_map,
    "raw_item_ids": raw_item_ids,
    "user_rated_items": user_rated_items,
  }


def scores_for_user(inner_uid, model_data):
  """Compute raw SVD predicted scores for all items."""
  return (
    model_data["item_factors"] @ model_data["user_factors"][inner_uid]
    + model_data["item_biases"]
    + model_data["global_mean"]
    + model_data["user_biases"][inner_uid]
  )
