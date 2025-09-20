import os
import time
import sys

import torch
import numpy as np
from scipy.sparse import coo_matrix, dok_matrix
from tqdm import tqdm

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .BISMModel import BISMModel
from elliot.utils import logging as logging_project

logger = logging_project.get_logger("__main__")

class BISM(RecMixin, BaseRecommenderModel):
    r"""
    Block-Aware Item Similarity Models for Top-N Recommendation

    For further details, please refer to the `paper <https://doi.org/10.1145/3411754>`_

    Args:


    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        external.BISM:
          meta:
            save_recs: True

    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        if self._batch_size < 1:
            self._batch_size = self._num_users

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._params_list = [
            ("_alpha", "alpha", "alpha", 0.1, float, None),
            ("_beta", "beta", "beta", 0.1, float, None),
            ("_lamb", "lamb", "lamb", 0.1, float, None),
            ("_c", "c", "c", 1, int, None),
        ]
        self.autoset_params()

        row, col = data.sp_i_train.nonzero()
        self._inter = coo_matrix((np.ones_like(row, dtype=np.float64), (row, col)),
                                 shape=(self._num_users, self._num_items))

        self._model = BISMModel(
            num_users=self._num_users,
            num_items=self._num_items,
            inter=self._inter,
            alpha=self._alpha,
            beta=self._beta,
            lamb=self._lamb,
            c=self._c,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "BISM" \
            + f"_{self.get_base_params_shortcut()}" \
            + f"_{self.get_params_shortcut()}"

    def train(self):
        start = time.time()
        for it in range(self._epochs):
            self.logger.info(f"Epoch: {it+1}")
            loss = 0
            loss += self._model.train_step()
            self.evaluate(it, loss / (it + 1))
        end = time.time()
        logger.info(f"The similarity computation has taken: {end - start}")


    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        with torch.no_grad():
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(offset, offset_stop)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))


    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), torch.tensor(preds).to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)