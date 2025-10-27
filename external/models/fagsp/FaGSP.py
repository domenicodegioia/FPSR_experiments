import torch
import numpy as np
from scipy.sparse import csr_matrix
import scipy
from tqdm import tqdm
import scipy.sparse as sp
import time

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class FaGSP(RecMixin, BaseRecommenderModel):
    r"""
    Frequency-aware Graph Signal Processing for Collaborative Filtering

    For further details, please refer to the `paper <https://arxiv.org/abs/2402.08426>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 1024, int, None),
            ("_pri_factor1", "pri_factor1", "pri_factor1", 256, int, None),
            ("_pri_factor2", "pri_factor2", "pri_factor2", 128, int, None),
            ("_alpha_1", "alpha_1", "alpha_1", 0.3, float, None),
            ("_alpha_2", "alpha_2", "alpha_2", 0.5, float, None),
            ("_order1", "order1", "order1", 12, int, None),
            ("_order2", "order2", "order2", 14, int, None),
            ("_q", "q", "q", 0.7, float, None)
        ]

        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        row, col = data.sp_i_train.nonzero()
        self.rat_mat = csr_matrix((np.ones(len(row)), (row, col)), shape=(self._num_users, self._num_items))

        self.adj_mat = data.sp_i_train.tolil()
        self.d_mat_i, self.d_mat_i_inv, self.vt, self.norm_adj = None, None, None, None

    def matpow(self, mat, order):
        R = mat
        if order == 1:
            return R
        for ord in range(2, order+1):
            R = R.T @ mat
        return R

    def train(self):
        start = time.time()

        adj_mat = self.adj_mat
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = scipy.sparse.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = scipy.sparse.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = scipy.sparse.diags(1 / d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        _, _, self.vt2 = np.linalg.svd(self.norm_adj.A, full_matrices=False)

        RTR1 = self.norm_adj.T @ self.norm_adj
        self.RTR1_pow = self.matpow(np.eye(RTR1.shape[0]) - RTR1, self._order1)
        RTR2 = self.norm_adj @ self.norm_adj.T
        self.RTR2_pow = self.matpow(np.eye(RTR2.shape[0]) - RTR2, self._order2)

        rat_mat = self.rat_mat
        batch_test = np.array(rat_mat.todense())

        self.preds = 0

        P11 = batch_test @ (np.eye(self.RTR1_pow.shape[0]) - self.RTR1_pow)
        self.preds += P11
        P12 = (np.eye(self.RTR2_pow.shape[0]) - self.RTR2_pow) @ batch_test
        self.preds += P12

        vt2 = self.vt2[-self._pri_factor2:]
        P30 = batch_test @ self.d_mat_i @ vt2.T @ vt2 @ self.d_mat_i_inv
        quan = np.quantile(P30, q=self._q, axis=0, keepdims=True)
        P30[P30 > quan] = 1.0
        P30[P30 <= quan] = 0.0
        P30[batch_test < 1] = 0.0
        P3 = batch_test + self._alpha_2 * P30
        P3 = sp.csr_matrix(P3)

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_P3 = d_mat.dot(adj_mat)
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i_P3 = d_mat
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_i_inv_P3 = sp.diags(d_inv)
        norm_P3 = norm_P3.dot(d_mat)
        norm_P3 = norm_P3.tocsc()
        _, _, vt1 = np.linalg.svd(norm_P3.A, full_matrices=False)
        vt1 = vt1[:self._pri_factor1]
        P2 = P3 @ d_mat_i_P3 @ vt1.T @ vt1 @ d_mat_i_inv_P3
        self.preds += self._alpha_1 * P2

        end = time.time()
        self.logger.info(f"Training has taken: {end - start}")
        self.evaluate()

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(tqdm(range(0, self._num_users, self._batch_eval))):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = self.get_users_rating(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_users_rating(self, batch_start, batch_stop):
        return self.preds[batch_start:batch_stop]

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), torch.tensor(preds).to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    @property
    def name(self):
        return "FaGSP" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"