import time

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

from .PSGEModel import PSGEModel

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

class PSGE(RecMixin, BaseRecommenderModel):
    r"""
    PSGE: Pure Spectral Graph Embeddings: Reinterpreting Graph Convolution for Top-N Recommendation

    For further details, please refer to the `paper <http://arxiv.org/abs/2305.18374>`_

    Args:
        factors: Number of latent factors
        alpha: Regularization factor of the users' degree in the graph
        beta:Regularization factor of the items' degree in the graph

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        external.PSGE:
          meta:
            save_recs: True
          factors: 1500
          alpha: 0.3
          beta: 0.3
          seed: 2026
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "factors", 1500, int, None),
            ("_alpha", "alpha", "alpha", 0.3, float, None),
            ("_beta", "beta", "beta", 0.3, float, None)
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._model = PSGEModel(
            num_users=self._num_users,
            num_items=self._num_items,
            factors=self._factors,
            alpha=self._alpha,
            beta=self._beta,
            data=self._data,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "PSGE" \
            + f"_{self.get_base_params_shortcut()}" \
            + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        logger.info(f"The similarity computation has taken: {end - start}")

        logger.info(f"Transactions: {self._data.transactions}")

        logger.info(f"Start evaluation")
        start = time.time()
        self.evaluate()
        end = time.time()
        logger.info(f"Evaluation has taken: {end - start}")

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._ratings.keys()}

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test