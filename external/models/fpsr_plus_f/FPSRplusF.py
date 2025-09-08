import os
import time
import sys

import torch
import numpy as np
from scipy.sparse import coo_matrix

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .FPSRplusFModel import FPSRplusFModel
from elliot.utils import logging as logging_project

logger = logging_project.get_logger("__main__")

class FPSRplusF(RecMixin, BaseRecommenderModel):
    r"""
    FPSR+_F: Toward Robust, Efficient, and Scalable Collaborative Filtering With Partition-Aware Item Similarity Modeling

    For further details, please refer to the `paper <https://doi.org/10.1109/TKDE.2024.3418080>`_

    Args:
        eigen_dim: Number of eigenvectors extracted
        l_w: Regularization coefficient
        rho: Hyperparameter introduced by ADMM
        w_1: l_1 regularization term
        w_2: l_2 regularization term
        eta: regularizes the similarity between items within the partition
        eps: tolerance
        tau: Size ratio
        pop_ratio: percentage of items to be selected as “hubs”

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        external.FPSRplusF:
          meta:
            save_recs: True
          eigen_dim: 256
          l_w: 0.2
          rho: 500
          w_1: 0.8
          w_2: 0.1
          eta: 1.0
          eps: 5e-3
          tau: 0.5
          pop_ratio: 0.1
          seed: 2026
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._sampler = cs.Sampler(self._data.i_train_dict)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._params_list = [
            ("_eigen_dim", "eigen_dim", "eigen_dim", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.5, float, None),
            ("_rho", "rho", "rho", 5000, int, None),
            ("_w_1", "w_1", "w_1", 0.1, float, None),
            ("_w_2", "w_2", "w_2", 1.0, float, None),
            ("_eta", "eta", "eta", 1.0, float, None),
            ("_tau", "tau", "tau", 0.2, float, None),
            ("_eps", "eps", "eps", 5e-3, float, None),
            ("_pop_ratio", "pop_ratio", "pop_ratio", 0.1, float, None),
        ]
        self.autoset_params()

        row, col = data.sp_i_train.nonzero()
        # col = np.array([c + self._num_users for c in col])
        # self._inter = np.array([row, col])
        # self._inter = coo_matrix(np.ones_like(row), (row, col), dtype=np.float64)  # coo_matrix

        # # Most/Least voted item
        # item_votes = np.array(data.sp_i_train.sum(axis=0)).flatten()
        # most_voted_item = item_votes.argmax()
        # least_voted_item = item_votes.argmin()
        # most_votes = item_votes[most_voted_item]
        # least_votes = item_votes[least_voted_item]
        # logger.info(f"Most voted item: {most_voted_item}, votes: {most_votes}")
        # logger.info(f"Most voted item: {least_voted_item}, votes: {least_votes}")
        # most_voted_vector = data.sp_i_train[:, most_voted_item].toarray().flatten()
        # least_voted_vector = data.sp_i_train[:, least_voted_item].toarray().flatten()
        # pearson_corr, p_value = pearsonr(most_voted_vector, least_voted_vector)
        # logger.info(f"Pearson correlation between item {most_voted_item} and item {least_voted_item}: {pearson_corr}, p-value: {p_value}")
        # sys.exit()

        self._inter = coo_matrix((np.ones_like(row, dtype=np.float64), (row, col)),
                                 shape=(self._num_users, self._num_items))

        self._model = FPSRplusFModel(
            num_users=self._num_users,
            num_items=self._num_items,
            eigen_dim=self._eigen_dim,
            l_w=self._l_w,
            tau=self._tau,
            eta=self._eta,
            eps=self._eps,
            w_1=self._w_1,
            w_2=self._w_2,
            rho=self._rho,
            pop_ratio=self._pop_ratio,
            inter=self._inter,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "FPSRplusF" \
            + f"_{self.get_params_shortcut()}"

    def train(self):
        start = time.time()

        # Recursive Spectral Graph Partitioning + Fine-tuning Intra-partition Item Similarities
        self._model.initialize()

        end = time.time()
        logger.info(f"The similarity computation has taken: {end - start}")

        logger.info(f"Transactions: {self._data.transactions}")
        logger.info("Start evaluation")
        self.evaluate()

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