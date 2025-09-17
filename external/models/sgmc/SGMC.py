import time

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

from .SGMCModel import SGMCModel

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

class SGMC(RecMixin, BaseRecommenderModel):
    r"""
    SGMC: Scalable and Explainable 1-Bit Matrix Completion via Graph Signal Learning

    For further details, please refer to the `paper <https://cdn.aaai.org/ojs/16863/16863-13-20357-1-2-20210518.pdf>`_

    Args:
        factors: Number of latent factors

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        external.PSGE:
          meta:
            save_recs: True
          factors: 1500
          seed: 2026
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "factors", 1500, int, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._model = SGMCModel(
            num_users=self._num_users,
            num_items=self._num_items,
            factors=self._factors,
            data=self._data,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "SGMC" \
            + f"_{self.get_params_shortcut()}"

    def train(self):
        start = time.time()
        self._model.initialize()
        end = time.time()
        logger.info(f"The similarity computation has taken: {end - start}")

        logger.info(f"Transactions: {self._data.transactions}")

        self.evaluate()

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._ratings.keys()}

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test
