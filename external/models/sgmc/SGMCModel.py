import numpy as np

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import svds

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")


class SGMCModel:
    def __init__(self,
                 num_users,
                 num_items,
                 factors,
                 data,
                 random_seed,
                 name="SGMC",
                 **kwargs
                 ):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.num_users = num_users
        self.num_items = num_items
        self.factors = factors
        self.data = data

        logger.info(f"Computing rate_matrix...")
        row, col = self.data.sp_i_train.nonzero()
        values = np.ones(len(row), dtype=np.int8)
        self.rate_matrix = csr_matrix((values, (row, col)), shape=(self.num_users, self.num_items))


    def initialize(self):
        user_degree = np.array(self.rate_matrix.sum(axis=1))
        d_user_inv = np.power(user_degree, -0.5).flatten()
        d_user_inv[np.isinf(d_user_inv)] = 0.0
        d_user_inv_diag = diags(d_user_inv)

        item_degree = np.array(self.rate_matrix.sum(axis=0))
        d_item_inv = np.power(item_degree, -0.5).flatten()
        d_item_inv[np.isinf(d_item_inv)] = 0.0
        d_item_inv_diag = diags(d_item_inv)

        d_item_alpha = np.power(item_degree, -0.5).flatten()
        d_item_alpha[np.isinf(d_item_alpha)] = 0.0
        d_item_alpha = diags(d_item_alpha)

        d_item_inv_alpha = np.power(item_degree, 0.5).flatten()
        d_item_inv_alpha[np.isinf(d_item_inv_alpha)] = 0.0
        d_item_inv_alpha_diag = diags(d_item_inv_alpha)

        int_norm = d_user_inv_diag.dot(self.rate_matrix).dot(d_item_inv_diag)
        logger.info("Computing svd...")
        _, _, vt = svds(int_norm.tocsc(), self.factors)

        logger.info("Computing similarity matrix...")
        self.similarity_matrix = (d_item_alpha @ vt.T) @ (d_item_inv_alpha_diag @ vt.T).T

        logger.info("Computing preds...")
        self._preds = self.rate_matrix.dot(self.similarity_matrix)


    def get_user_recs(self, u, mask, k):
        user_id = self.data.public_users.get(u)
        user_recs = self._preds[user_id]
        masked_recs = np.where(mask[user_id], user_recs, -np.inf)
        valid_items = np.sum(mask[user_id])
        local_k = min(k, valid_items)
        top_k_indices = np.argpartition(masked_recs, -local_k)[-local_k:]
        top_k_values = masked_recs[top_k_indices]
        sorted_top_k_indices = top_k_indices[np.argsort(-top_k_values)]
        return [(self.data.private_items[idx], masked_recs[idx]) for idx in sorted_top_k_indices]

    def get_model_state(self):
        saving_dict = {}
        saving_dict['_preds'] = self._preds
        saving_dict['_alpha'] = self._alpha
        saving_dict['_beta'] = self._beta
        saving_dict['_factors'] = self._factors
        return saving_dict

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._alpha = saving_dict['_alpha']
        self._beta = saving_dict['_beta']
        self._factors = saving_dict['_factors']


