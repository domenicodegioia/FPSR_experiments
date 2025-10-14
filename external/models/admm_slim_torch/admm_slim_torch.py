import gc
import torch
import pickle
import time
import numpy as np
from tqdm import tqdm

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from .admm_slim_model import ADMMSlimModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class ADMMSlimTorch(RecMixin, BaseRecommenderModel):
    r"""
    ADMM SLIM: Sparse Recommendations for Many Users

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3336191.3371774>`_

    Args:
        eigen_dim: Number of eigenvectors extracted

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        external.ADMMSlim:
          meta:
            verbose: True
          eigen_dim: 256

    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_l1", "l1", "l1", 0.001, float, None),
            ("_l2", "l2", "l2", 0.001, float, None),
            ("_alpha", "alpha", "alpha", 0.001, float, None),
            ("_rho", "rho", "rho", 100, int, None),
            ("_iterations", "iterations", "iterations", 50, int, None),
            ("_batch_eval", "batch_eval", "batch_eval", 1024, int, None),
        ]

        self.autoset_params()

        self._ratings = self._data.train_dict
        # self._sp_i_train = self._data.sp_i_train
        # self._i_items_set = list(range(self._num_items))

        self._model = ADMMSlimModel(self._data,
                                    self._num_users,
                                    self._num_items,
                                    self._l1,
                                    self._l2,
                                    self._alpha,
                                    self._iterations,
                                    self._rho)


    @property
    def name(self):
        return "ADMMSlimTorch" \
               + f"_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
#        return {u: self._model.get_user_recs(u, mask, k) for u in self._ratings.keys()}
        recs = {}
        for i in tqdm(range(0, len(self._ratings.keys()), 1024), desc="Processing batches", total=len(self._ratings.keys()) // 1024 + (1 if len(self._ratings.keys()) % 1024 != 0 else 0)):
            batch = list(self._ratings.keys())[i:i+1024]
            mat = self._model.get_user_recs(batch, mask, k)
            proc_batch = dict(zip(batch, mat))
            recs.update(proc_batch)
        return recs

    def train(self):
        start = time.time()
        self._model.train()
        self.logger.info(f"The similarity computation has taken:\t{time.time() - start}")

        torch.cuda.empty_cache()
        gc.collect()

        self.evaluate()