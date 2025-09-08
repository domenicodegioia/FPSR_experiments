import os
import random
import numpy as np
import torch
import sys
from tqdm import tqdm

from elliot.utils import logging as logging_project

logger = logging_project.get_logger("__main__")

import warnings

warnings.filterwarnings("ignore")


class FPSRplusFModel:
    def __init__(self,
                 num_users,
                 num_items,
                 eigen_dim,
                 l_w,
                 tau,
                 eta,
                 eps,
                 w_1,
                 w_2,
                 rho,
                 pop_ratio,
                 inter,
                 random_seed,
                 name="FPSR",
                 **kwargs
                 ):

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

        # load parameters info
        self.num_users = num_users
        self.num_items = num_items
        self.eigen_dim = eigen_dim
        self.l_w = l_w
        self.tau = tau
        self.eta = eta
        self.eps = eps
        self.w_1 = w_1
        self.w_2 = w_2
        self.rho = rho
        self.pop_ratio = pop_ratio
        self.p_list = []
        self.inter = inter

        # compute user-item interaction matrix
        self.inter = torch.sparse_coo_tensor(
            indices=torch.LongTensor(np.array([self.inter.row, self.inter.col])),
            values=torch.FloatTensor(self.inter.data),
            size=self.inter.shape, dtype=torch.float
        ).coalesce().to(self.device)

        # storage variables for item similarity matrix S
        self.S_indices = []
        self.S_values = []

        self.item_list = []

        # compute these matrices: d_i , d_i_inv , V
        # W = D_I^(1/2) * V * V^T * D_I^(-1/2)
        self.update_W()

    def update_W(self) -> None:
        self.d_i = self._degree(dim=0, exp=-0.5).reshape(-1, 1)  # D_I^(1/2)
        self.d_i_inv = self._degree(dim=0, exp=0.5).reshape(1, -1)  # D_I^(-1/2)
        self.V = self._svd(self._norm_adj(), self.eigen_dim)  # V

    def _degree(self, inter_mat=None, dim=0, exp=-0.5) -> torch.Tensor:
        # degree of users and items (row/column sum)
        if inter_mat is None:
            inter_mat = self.inter
        return torch.nan_to_num(torch.sparse.sum(inter_mat, dim=dim).to_dense().pow(exp), nan=0, posinf=0, neginf=0)

    def _svd(self, mat, k) -> torch.Tensor:
        _, _, V = torch.svd_lowrank(mat, q=max(4 * k, 32), niter=10)
        return V[:, :k]

    def _norm_adj(self, item_list=None) -> torch.Tensor:
        if item_list is None:
            vals = self.inter.values() * self.d_i[self.inter.indices()[1]].squeeze()
            return torch.sparse_coo_tensor(
                self.inter.indices(),
                self._degree(dim=1)[self.inter.indices()[0]] * vals,
                size=self.inter.shape, dtype=torch.float
            ).coalesce()
        else:
            inter = self.inter.index_select(dim=1, index=item_list).coalesce()
            vals = inter.values() * self.d_i[item_list][inter.indices()[1]].squeeze()
            return torch.sparse_coo_tensor(
                inter.indices(),
                self._degree(inter, dim=1)[inter.indices()[0]] * vals,
                size=inter.shape, dtype=torch.float
            ).coalesce()

    def partitioning(self, V, pop=False) -> torch.Tensor:
        if pop:
            pop_thres = torch.quantile(self.d_i_inv, 1 - self.pop_ratio)
            split = self.d_i_inv.squeeze() >= pop_thres
        else:
            split = V[:, 1] >= 0
            if split.sum() == split.shape[0] or split.sum() == 0:
                split = V[:, 1] >= torch.median(V[:, 1])
        return split

    def initialize(self):
        abs_fielder_vector = torch.abs(self.V[:, 1])
        _, sorted_indices = torch.sort(abs_fielder_vector, descending=False)

        num_hubs = int(self.num_items * self.pop_ratio)
        logger.info(f"Total number of hubs created: {num_hubs}")

        self.pop_list = sorted_indices[:num_hubs]

        first_split = self.partitioning(self.V)
        self.update_S(torch.arange(self.num_items, device=self.device)[torch.where(first_split)[0]])
        self.update_S(torch.arange(self.num_items, device=self.device)[torch.where(~first_split)[0]])

        for ilist in self.item_list:
            self.item_similarity(ilist)

        self.S = torch.sparse_coo_tensor(indices=torch.cat(self.S_indices, dim=1),
                                         values=torch.cat(
                                             self.S_values, dim=0),
                                         size=(self.num_items, self.num_items)).coalesce().T.to_sparse_csr()

        logger.info(f"Total number of partitions created: {len(self.S_indices)}")
        del self.S_indices, self.S_values

    def update_S(self, item_list) -> None:

        if item_list.shape[0] <= self.tau * self.num_items:
            # If the partition size is samller than size limit, model item similarity for this partition.
            self.item_list.append(item_list)
        else:
            # If the partition size is larger than size limit, perform graph partitioning on this partition.
            split = self.partitioning(self._svd(self._norm_adj(item_list), 2))
            self.update_S(item_list[torch.where(split)[0]])
            self.update_S(item_list[torch.where(~split)[0]])

    def item_similarity(self, item_list) -> torch.Tensor:
        r"""Update partition-aware item similarity matrix S in a specific partition.

        Returns:
            Partition-aware item similarity matrix of a partition.
        """
        # Initialize
        print("Part {:d}: ".format(item_list.shape[0]), end="")
        combined = torch.cat([item_list, self.pop_list, self.pop_list], dim=0)
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 2]
        aug_list = torch.cat([item_list, difference], dim=0)
        deaug_list = item_list

        full_mat = self.inter.index_select(dim=1, index=aug_list).to_dense()

        W = self.l_w * self.d_i[aug_list, :] * self.V[aug_list, :] @ \
            self.V[deaug_list, :].T * self.d_i_inv[:, deaug_list]

        Q_hat = torch.mm(full_mat.T, full_mat) + self.w_2 * \
                torch.diag(
                    torch.pow(self.d_i_inv[:, aug_list].squeeze(), 2)) + self.eta
        del full_mat

        Z_aux = Q_hat[:, :deaug_list.shape[0]] - Q_hat @ W
        del W

        Q_inv = torch.inverse(
            Q_hat + self.rho * torch.eye(Q_hat.shape[0], device=self.device))
        diag_inv = torch.diag(Q_inv)[:deaug_list.shape[0]] + 1e-10
        del Q_hat

        Z_aux = Q_inv @ Z_aux

        Phi = torch.zeros_like(Z_aux, device=self.device)
        S = torch.zeros_like(Z_aux, device=self.device)
        for _ in tqdm(range(50)):
            # Iteration
            Z_tilde = Z_aux + Q_inv @ (self.rho * (S - Phi))
            gamma = torch.diag(Z_tilde) / diag_inv
            Z = Z_tilde - Q_inv[:, :deaug_list.shape[0]] * gamma  # Update Z
            S = torch.clip(Z + Phi - self.w_1 /
                           self.rho, min=0)  # Update S
            Phi += Z - S  # Update Phi
        S = torch.where(S >= self.eps, S, 0).to_sparse_coo()
        self.S_indices.append(aug_list[S.indices()])
        self.S_values.append(S.values())

    def predict(self, start, stop):
        # C = S + lambda * W (C -> item-similarity matrix, W -> global info, S-> local info)
        batch = torch.arange(start, stop).to(self.device)
        user = self.inter.index_select(dim=0, index=batch).to_dense()
        r = torch.sparse.mm(self.S, user.T).T
        r += self.l_w * user * self.d_i.T @ self.V @ self.V.T * self.d_i_inv
        return r