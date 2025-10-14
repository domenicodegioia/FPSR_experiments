import pickle
import time
from operator import itemgetter

import numpy as np
import sys
import scipy.sparse as sp
from tqdm import tqdm
import torch
import random

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

def soft_threshold(x, threshold):
    return (np.abs(x) > threshold) * (np.abs(x) - threshold) * np.sign(x)

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

        self._w_sparse = None
        self.pred_mat = None

    def train(self):
        # pre-compute
        G = (self.X.T @ self.X)
        diag = self.l2 * np.diag(np.power(self.item_means, self.alpha)) + self.rho * np.identity(self.num_items)
        logger.info("Computing P...")
        P = np.linalg.inv(G + diag).astype(np.float32)
        logger.info("Computing B_aux...")
        B_aux = (P @ G).astype(np.float32)

        # initialize
        Gamma = np.zeros(G.shape, dtype=float)
        C = np.zeros(G.shape, dtype=float)
        del diag, G

        # iterate until convergence
        for i in tqdm(range(self.iterations), disable=True):
            start = time.time()
            B_tilde = B_aux + P @ (self.rho * C - Gamma)
            gamma = np.diag(B_tilde) / (np.diag(P) + 1e-7)
            B = B_tilde - P * gamma
            C = soft_threshold(B + Gamma / self.rho, self.l1 / self.rho)
            Gamma += self.rho * (B - C)
            logger.info(f"Iteration {i} has taken:\t{round(time.time() - start,2)}")

        self._w_sparse = C

        self.X = self._data.sp_i_train

    def predict(self, u):
        return self.X[u,:].dot(self._w_sparse)

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
