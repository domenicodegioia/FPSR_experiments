from operator import itemgetter

import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.benchmark.utils.fuzzer import dtype_size
from tqdm import tqdm
import time
import gc

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class BiRank(RecMixin, BaseRecommenderModel):
    r"""


    For further details, please refer to the `paper <htt>`_

    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 1024, int, None),
            ("_alpha", "alpha", "alpha", 0.85, float, None),
            ("_beta", "beta", "beta", 0.85, float, None)
        ]

        self.autoset_params()

        self._ratings = self._data.train_dict

        self._R = self._data.sp_i_train#.astype(np.float64)

        self._item_propagator = None

    def train(self):
        start = time.time()

        user_degrees = np.array(self._R.sum(axis=1)).flatten()
        item_degrees = np.array(self._R.sum(axis=0)).flatten()

        user_degrees[user_degrees == 0] = 1e-8
        item_degrees[item_degrees == 0] = 1e-8

        Du_inv_sqrt = sp.diags(1.0 / np.sqrt(user_degrees))
        Dp_inv_sqrt = sp.diags(1.0 / np.sqrt(item_degrees))

        S = Du_inv_sqrt @ self._R @ Dp_inv_sqrt

        STS = (S.T @ S).toarray()

        del S
        gc.collect()

        I = np.identity(self._num_items, dtype=np.float32)
        self._item_item_matrix = I - self._alpha * self._beta * STS

        del I, STS
        gc.collect()

        if torch.cuda.is_available():
            self._item_item_matrix = torch.from_numpy(self._item_item_matrix).cuda()
            temp = np.zeros((self._data.num_items, self._data.num_items))
            I = torch.eye(self._data.num_items).cuda()
            batch_size = 1024
            for start in tqdm(range(0, self._data.num_items, batch_size), disable=False):
                end = min(start + batch_size, self._data.num_items)
                block = I[:, start:end]
                X = torch.linalg.solve(self._item_item_matrix, block)
                temp[:, start:end] = X.cpu().numpy()
            self._item_item_matrix = temp
        else:
            temp = np.zeros((self._data.num_items, self._data.num_items))
            I = np.eye(self._data.num_items)
            batch_size = 1024
            for start in tqdm(range(0, self._data.num_items, batch_size), disable=False):
                end = min(start + batch_size, self._data.num_items)
                block = I[:, start:end]
                temp[:, start:end] = np.linalg.solve(self._item_item_matrix, block)
            self._item_item_matrix = temp
        end = time.time()
        self.logger.info(f"Training has taken: {end-start} seconds")

        gc.collect()
        torch.cuda.empty_cache()

        self.evaluate()

    def get_recommendations(self, k: int = 100):
        predictions_top_k_val = {}
        predictions_top_k_test = {}
        recs_val, recs_test = self.process_protocol(k)
        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
        recs = {}
        for i in tqdm(range(0, len(self._ratings.keys()), 1024), desc="Processing batches",
                      total=len(self._ratings.keys()) // 1024 + (1 if len(self._ratings.keys()) % 1024 != 0 else 0)):
            batch = list(self._ratings.keys())[i:i + 1024]
            mat = self.get_user_recs(batch, mask, k)
            proc_batch = dict(zip(batch, mat))
            recs.update(proc_batch)
        return recs

    def get_user_recs(self, u, mask, k):
        u_index = itemgetter(*u)(self._data.public_users)
        preds = self.predict(u_index)
        users_recs = np.where(mask[u_index, :], preds, -np.inf)
        index_ordered = np.argpartition(users_recs, -k, axis=1)[:, -k:]
        value_ordered = np.take_along_axis(users_recs, index_ordered, axis=1)
        local_top_k = np.take_along_axis(index_ordered, value_ordered.argsort(axis=1)[:, ::-1], axis=1)
        value_sorted = np.take_along_axis(users_recs, local_top_k, axis=1)
        mapper = np.vectorize(self._data.private_items.get)
        return [[*zip(item, val)] for item, val in zip(mapper(local_top_k), value_sorted)]

    def predict(self, user_indices):
        user_inter = self._R[user_indices,:]
        return user_inter @ self._item_item_matrix

    @property
    def name(self):
        return "BiRank" \
               + f"_{self.get_params_shortcut()}"