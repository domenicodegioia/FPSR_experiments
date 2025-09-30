import torch
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import time

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class LinkProp(RecMixin, BaseRecommenderModel):
    r"""
    Revisiting Neighborhood-based Link Prediction for Collaborative Filtering

    For further details, please refer to the `paper <https://arxiv.org/abs/2203.15789>`_

    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 1024, int, None),
            # ("_svd_factors", "svd_factors", "svd_factors", 256, int, None),
            ("_alpha", "alpha", "alpha", 0.5, float, None),
            ("_beta", "beta", "beta", 0.5, float, None),
            ("_gamma", "gamma", "gamma", 0.5, float, None),
            ("_delta", "delta", "delta", 0.5, float, None)
        ]

        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._inter = self._data.sp_i_train.astype(np.float64)

        self._user_degrees = np.array(self._inter.sum(axis=1)).flatten() + 1e-9
        self._item_degrees = np.array(self._inter.sum(axis=0)).flatten() + 1e-9

        self._preds = None

        # row, col = data.sp_i_train.nonzero()
        # self.inter = coo_matrix((np.ones_like(row, dtype=np.float64), (row, col)),
        #                          shape=(self._num_users, self._num_items))
        #
        # self.adj_mat = data.sp_i_train.tolil()

    def train(self):
        start = time.time()

        M = self._inter.tocsr()
        M_T = self._inter.transpose().tocsc()

        user_deg_alpha = np.power(self._user_degrees, -self._alpha)
        item_deg_beta = np.power(self._item_degrees, -self._beta)
        user_deg_gamma = np.power(self._user_degrees, -self._gamma)
        item_deg_delta = np.power(self._item_degrees, -self._delta)

        D_u_alpha = sp.diags(user_deg_alpha)
        D_i_beta = sp.diags(item_deg_beta)
        D_u_gamma = sp.diags(user_deg_gamma)
        D_i_delta = sp.diags(item_deg_delta)

        A = (D_u_alpha @ M) @ D_i_beta
        B = (D_u_gamma @ M) @ D_i_delta

        self._preds = (A @ M_T) @ B

        end = time.time()
        self.logger.info(f"Training has taken: {end-start} seconds")

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
        return self._preds[batch_start:batch_stop].toarray()

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
        return "LinkProp" \
               + f"_{self.get_params_shortcut()}"