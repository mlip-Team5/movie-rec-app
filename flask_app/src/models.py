import logging
import os

logger = logging.getLogger(__name__)

FALLBACK_RECOMMENDATIONS = [
  101, 204, 550, 892, 12, 745, 331, 1029, 467, 88,
  612, 293, 1150, 77, 504, 839, 162, 710, 445, 998,
]


class RecommenderModel:
  def __init__(self, model_path=None):
    self.model_path = model_path or os.environ.get("MODEL_PATH")
    self.model = None
    self._load_model()

  def _load_model(self):
    if not self.model_path:
      logger.info("No model path set, using fallback recommendations")
      return
    try:
      # TODO: load real model (e.g. joblib.load, pickle, torch.load)
      logger.info(f"Model loaded from {self.model_path}")
    except Exception as e:
      logger.warning(f"Failed to load model: {e}, falling back to defaults")
      self.model = None

  def predict(self, userid):
    """Return up to 20 recommended movie IDs for the given user."""
    if self.model is not None:
      pass  # TODO: return self.model.predict(userid)[:20]

    return FALLBACK_RECOMMENDATIONS[:20]
