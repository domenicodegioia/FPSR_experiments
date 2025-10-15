"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import time
from operator import itemgetter

from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
import scipy.sparse as sp


from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class MRF(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_l2_norm", "l2_norm", "l2_norm", 1e3, float, None)
        ]

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train

        self.autoset_params()

        self.B = None


    @property
    def name(self):
        return f"MRF_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
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


    def train(self):
        start = time.time()

        self.logger.info("Step 1: Computing X^T * X...")
        xtx = self._sp_i_train.T @ self._sp_i_train

        # Step 2: Compute the regularized empirical covariance matrix S_x
        # S_x = (1/n) * (X^T*X + lambda*I)
        self.logger.info("Step 2: Computing regularized covariance matrix S_x...")
        lambda_identity = self._l2_norm * sp.identity(self._num_items, dtype=np.float32)
        s_x = (1 / self._num_users) * (xtx + lambda_identity)

        # Step 3: Compute the precision matrix C_hat by inverting S_x
        # This is the most computationally expensive step.
        self.logger.info("Step 3: Inverting S_x to get the precision matrix C_hat...")
        s_x_dense = s_x.toarray()
        c_hat = np.linalg.inv(s_x_dense)

        # Step 4: Compute the final B matrix from C_hat
        # B_ij = -C_ij / C_jj for i != j, and B_ii = 0
        self.logger.info("Step 4: Computing the final parameter matrix B...")

        # SOLUZIONE: Creare una copia esplicita della diagonale per renderla modificabile
        diag_c_hat = np.diag(c_hat).copy()

        # Ora la modifica funzionerà senza errori
        # Add a small epsilon to the diagonal to prevent division by zero
        diag_c_hat[diag_c_hat == 0] = 1e-8

        # Use broadcasting to divide each element C_ij by C_jj
        b_matrix = -c_hat / diag_c_hat

        # Enforce the zero-diagonal constraint
        np.fill_diagonal(b_matrix, 0)

        self.B = b_matrix

        end = time.time()
        self.logger.info(f"The similarity computation has taken: {end - start}")


        self.evaluate()

    def predict(self, user_indices):
        # Get the interaction profiles of the users
        user_profiles = self._sp_i_train[user_indices,:]
        # Compute the predictions: P = X_batch * B
        # safe_sparse_dot is efficient for sparse * dense matrix multiplication
        return safe_sparse_dot(user_profiles, self.B)
