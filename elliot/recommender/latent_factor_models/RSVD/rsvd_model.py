"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np
from scipy import sparse as sp
from sklearn.utils.extmath import randomized_svd


class RSVDModel(object):

    def __init__(self, factors, reg, data, random_seed):

        self._data = data
        self._private_users = data.private_users
        self._public_users = data.public_users
        self._private_items = data.private_items
        self._public_items = data.public_items
        self.factors = factors
        self.reg = reg
        self.random_seed = random_seed
        self.train_dict = self._data.train_dict
        self.user_num, self.item_num = self._data.num_users, self._data.num_items

        self.user_vec, self.item_vec = None, None

    def train(self):
        U, sigma, Vt = randomized_svd(self._data.sp_i_train,
                                      n_components=self.factors,
                                      random_state=self.random_seed)
        self.user_vec = U
        sigma_squared_minus_reg = np.maximum(0, sigma ** 2 - self.reg)
        omega_diag = np.sqrt(sigma_squared_minus_reg)
        self.item_vec = Vt.T * omega_diag

    def predict(self, user, item):
        return self.user_vec[self._data.public_users[user], :].dot(self.item_vec[self._data.public_items[item], :])

    def get_user_recs(self, user_id, mask, top_k=100):
        user_id = self._public_users.get(user_id)
        b = self.user_vec[user_id] @ self.item_vec.T
        a = mask[user_id]
        b[~a] = -np.inf
        indices, values = zip(*[(self._private_items.get(u_list[0]), u_list[1])
                                for u_list in enumerate(b.data)])
        indices = np.array(indices)
        values = np.array(values)
        local_k = min(top_k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]
