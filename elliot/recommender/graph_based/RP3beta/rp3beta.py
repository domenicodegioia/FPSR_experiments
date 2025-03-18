"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import time

import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

import warnings
warnings.filterwarnings("ignore")


class RP3beta(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_neighborhood", "neighborhood", "neighborhood", 10, int, None),
            ("_alpha", "alpha", "alpha", 1., float, None),
            ("_beta", "beta", "beta", 0.6, float, None),
            ("_normalize_similarity", "normalize_similarity", "normalize_similarity", False, bool, None)
        ]

        self.autoset_params()
        if self._neighborhood == -1:
            self._neighborhood = self._data.num_items


    @property
    def name(self):
        return f"RP3beta_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
        return {u: self.get_user_recs(u, mask, k) for u in self._data.train_dict.keys()}

    def get_user_recs(self, u, mask, k):  # NEW
        user_id = self._data.public_users.get(u)
        user_recs = self._preds[user_id].toarray()[0]
        masked_recs = np.where(mask[user_id], user_recs, -np.inf)
        valid_items = int(np.sum(mask[user_id]))
        local_k = min(k, valid_items)
        top_k_indices = np.argpartition(masked_recs, -local_k)[-local_k:]
        top_k_values = masked_recs[top_k_indices]
        sorted_top_k_indices = top_k_indices[np.argsort(-top_k_values)]
        return [(self._data.private_items[idx], masked_recs[idx]) for idx in sorted_top_k_indices]

    def train(self):
        if self._restore:
            return self.restore_weights()

        self._train = self._data.sp_i_train_ratings.copy()
        self.Pui = normalize(self._train, norm='l1', axis=1)

        X_bool = self._train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        self.degree = np.zeros(self._train.shape[1])

        nonZeroMask = X_bool_sum != 0.0

        self.degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self._beta)

        self.Piu = normalize(X_bool, norm='l1', axis=1)
        del (X_bool)

        if self._alpha != 1.:
            self.Pui = self.Pui.power(self._alpha)
            self.Piu = self.Piu.power(self._alpha)

        block_dim = 200
        d_t = self.Piu

        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start = time.time()

        for current_block_start_row in range(0, self.Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > self.Pui.shape[1]:
                block_dim = self.Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * self.Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], self.degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self._neighborhood]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

        self._similarity_matrix = sparse.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                                    shape=(self.Pui.shape[1], self.Pui.shape[1]))

        if self._normalize_similarity:
            self._similarity_matrix = normalize(self._similarity_matrix, norm='l1', axis=1)

        self._similarity_matrix = self._similarity_matrix.tocsc()

        data, rows_indices, cols_indptr = [], [], []

        for item_idx in range(len(self._data.items)):
            cols_indptr.append(len(data))

            start_position = self._similarity_matrix.indptr[item_idx]
            end_position = self._similarity_matrix.indptr[item_idx + 1]

            column_data = self._similarity_matrix.data[start_position:end_position]
            column_row_index = self._similarity_matrix.indices[start_position:end_position]


            non_zero_data = column_data != 0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-self._neighborhood:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

        cols_indptr.append(len(data))

        W_sparse = sparse.csc_matrix((data, rows_indices, cols_indptr),
                                     shape=(len(self._data.items), len(self._data.items)), dtype=np.float32).tocsr()

        self._preds = self._train.dot(W_sparse)

        end = time.time()
        logger.info(f"The similarity computation has taken: {end - start}")

        self.evaluate()
