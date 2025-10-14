import pickle
import time
from operator import itemgetter

import numpy as np
import sys
import scipy.sparse as sp
from tqdm import tqdm
import torch
import random
import gc

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

def soft_threshold(x, threshold):
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

class ADMMSlimModel:
    def __init__(self,
                 data,
                 num_users,
                 num_items,
                 l1,
                 l2,
                 alpha,
                 iterations,
                 rho,
                 random_seed=42):
        # set seed
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._data = data
        self.num_users = num_users
        self.num_items = num_items
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.iterations = iterations
        self.rho = rho

        self.X = self._data.sp_i_train
        self.item_means = self.X.mean(axis=0).getA1()
        self.X = self.X.toarray() - self.item_means

        self.X = torch.from_numpy(self.X).to(self.device, dtype=torch.float32)
        self.item_means = torch.from_numpy(self.item_means).to(self.device, dtype=torch.float32)

        self._w_sparse = None
        self.pred_mat = None

    def train(self):
        # pre-compute
        # self.X = self._data.sp_i_train
        G = (self.X.T @ self.X)
        # diag = self.l2 * np.diag(np.power(self.item_means, self.alpha)) + self.rho * np.identity(self.num_items)
        diag_l2 = self.l2 * torch.diag(torch.pow(self.item_means, self.alpha))
        diag_rho = self.rho * torch.eye(self.num_items, device=self.device)
        diag = diag_l2 + diag_rho

        logger.info("Computing P...")
        P = torch.linalg.inv(G + diag)

        logger.info("Computing B_aux...")
        B_aux = (P @ G)

        del diag_l2, diag_rho, diag, G
        torch.cuda.empty_cache()
        gc.collect()


        # initialize
        Gamma = torch.zeros((self.num_items, self.num_items), device=self.device, dtype=torch.float32)
        C = torch.zeros((self.num_items, self.num_items), device=self.device, dtype=torch.float32)


        # iterate until convergence
        for _ in tqdm(range(self.iterations), disable=False):
            # start = time.time()
            B_tilde = B_aux + P @ (self.rho * C - Gamma)
            gamma = torch.diag(B_tilde) / (torch.diag(P) + 1e-7)
            B = B_tilde - P * gamma
            C = soft_threshold(B + Gamma / self.rho, self.l1 / self.rho)
            Gamma += self.rho * (B - C)
            # logger.info(f"Iteration {i} has taken:\t{time.time() - start}")

        self._w_sparse = C

        self.X = torch.from_numpy(self._data.sp_i_train.toarray()).to(self.device, dtype=torch.float32)

    # def prepare_predictions(self):
    #     self.pred_mat = self.X.dot(self._w_sparse) #.toarray()

    def predict(self, u):
        u = torch.tensor(u, device=self.device)
        return self.X[u] @ self._w_sparse

    # def get_user_recs(self, user, mask, k=100):
    #     ui = self._data.public_users[user]
    #     user_mask = mask[ui]
    #     predictions = self.predict(user) #self.pred_mat[ui].copy()
    #     predictions[~user_mask] = -np.inf
    #     valid_items = user_mask.sum()
    #     local_k = min(k, valid_items)
    #     top_k_indices = np.argpartition(predictions, -local_k)[-local_k:]
    #     top_k_values = predictions[top_k_indices]
    #     sorted_top_k_indices = top_k_indices[np.argsort(-top_k_values)]
    #     return [(self._data.private_items[idx], predictions[idx]) for idx in sorted_top_k_indices]

    def get_user_recs(self, u, mask, k):
        u_index = itemgetter(*u)(self._data.public_users)
        preds = self.predict(u_index)
        users_recs = np.where(mask[u_index, :], preds.cpu().numpy(), -np.inf)
        index_ordered = np.argpartition(users_recs, -k, axis=1)[:, -k:]
        value_ordered = np.take_along_axis(users_recs, index_ordered, axis=1)
        local_top_k = np.take_along_axis(index_ordered, value_ordered.argsort(axis=1)[:, ::-1], axis=1)
        value_sorted = np.take_along_axis(users_recs, local_top_k, axis=1)
        mapper = np.vectorize(self._data.private_items.get)
        return [[*zip(item, val)] for item, val in zip(mapper(local_top_k), value_sorted)]