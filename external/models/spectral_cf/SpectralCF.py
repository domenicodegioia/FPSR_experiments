import time
import sys
import numpy as np
import os
from tqdm import tqdm
import math

import torch
from scipy.sparse import coo_matrix, dok_matrix

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from .SpectralCFModel import SpectralCFModel
from elliot.recommender.base_recommender_model import init_charger


class SpectralCF(RecMixin, BaseRecommenderModel):
    r"""
    Spectral Collaborative Filtering

    For further details, please refer to the `paper <https://arxiv.org/abs/1808.10523>`_

    Args:
        neighbors: Number of item neighbors
        similarity: Similarity function

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        SpectralCF:
          meta:
            save_recs: True
          lr: 0.001
          epochs: 200
          batch_size: 1024
          factors: 16
          n_layers: 3
          n_filters: 16
          l_reg: 0.001
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._sampler = cs.Sampler(self._data.i_train_dict)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.001, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_reg", "l_reg", "l_reg", 0.001, float, None),
            ("_n_layers", "n_layers", "n_layers", 3, int, None),
            ("_n_filters", "n_filters", "n_filters", 16, int, None)
        ]
        self.autoset_params()

        row, col = self._data.sp_i_train.nonzero()
        self.inter = coo_matrix((np.ones_like(row, dtype=np.float64), (row, col)),
                                 shape=(self._num_users, self._num_items))
        self._inter = torch.sparse_coo_tensor(
            indices=torch.LongTensor(np.array([self.inter.row, self.inter.col])),
            values=torch.FloatTensor(self.inter.data),
            size=self.inter.shape, dtype=torch.float
        ).coalesce().to(self.device)


        self._adj = self.get_adj_matrix()

        self._model = SpectralCFModel(learning_rate=self._learning_rate, factors=self._factors,
                                      n_layers=self._n_layers, n_filters=self._n_filters, inter=self._inter,
                                      adj=self._adj, num_users=self._num_users, num_items=self._num_items,
                                      seed=self._seed, l_reg=self._l_reg)
        del self.inter

    def get_adj_matrix(self):
        num_users, num_items = self._num_users, self._num_items
        row, col, data = self.inter.row, self.inter.col, self.inter.data

        rows = np.concatenate([row, col + num_users])
        cols = np.concatenate([col + num_users, row])
        values = np.concatenate([data, data])

        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        shape = (num_users + num_items, num_users + num_items)

        return torch.sparse_coo_tensor(indices, values, size=shape, device=self.device).coalesce()

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


    @property
    def name(self):
        return f"SpectralCF_{self.get_params_shortcut()}"

    def train(self):
        # if self._restore:
        #     return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

            # if it > 50 and it % 5 == 0:
            #     self.evaluate(it, loss / (it + 1))

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), torch.tensor(preds).to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)