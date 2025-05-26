import os
import random
import numpy as np
import torch
import sys
from tqdm import tqdm

from sklearn.preprocessing import normalize
import similaripy as sim

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

import warnings
warnings.filterwarnings("ignore")


class FPSR_RP3beta_model:
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
                 inter,
                 data,
                 # for rp3beta
                 neighborhood,
                 alpha,
                 beta,
                 normalize_similarity,
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
        self.inter = inter

        # compute user-item interaction matrix
        self.inter = torch.sparse_coo_tensor(
            indices=torch.LongTensor(np.array([self.inter.row, self.inter.col])),
            # indices=torch.LongTensor(self.inter),
            values=torch.FloatTensor(self.inter.data),
            # values=torch.ones((self.inter.shape[1], 1)).flatten(),
            size=self.inter.shape, dtype=torch.float
        ).coalesce().to(self.device)

        # storage variables for item similarity matrix S
        self.S_indices = []
        self.S_values = []

        # compute these matrices: d_i , d_i_inv , V
        # W = D_I^(1/2) * V * V^T * D_I^(-1/2)
        self.update_W()
        # these are structures needed also for partitioning, so I can't remove it


        # new global item-similarity matrix
        self._data = data
        self._neighborhood = neighborhood
        self._alpha = alpha
        self._beta = beta
        self._normalize_similarity = normalize_similarity
        self.W_sparse = sim.rp3beta(self._data.sp_i_train_ratings.T, k=self._neighborhood, alpha=self._alpha,
                               beta=self._beta, format_output='csr', verbose=False)
        if self._normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)
        self.W_sparse = self.csr2coo_tensor(self.W_sparse)

    def csr2coo_tensor(self, matrix) -> torch.Tensor:
        coo = matrix.tocoo()
        indices = torch.tensor([coo.row, coo.col], dtype=torch.int64).to(self.device)
        values = torch.tensor(coo.data, dtype=torch.float32).to(self.device)
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
        return sparse_tensor

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

    def partitioning(self, V) -> torch.Tensor:
        split = V[:, 1] >= 0
        if split.sum() == split.shape[0] or split.sum() == 0:
            split = V[:, 1] >= torch.median(V[:, 1])
        return split

    def initialize(self):
        # first split
        self.first_split = self.partitioning(self.V)
        # recursive partitioning_item #1
        self.update_S(torch.arange(self.num_items, device=self.device)[torch.where(self.first_split)[0]])
        # recursive partitioning_item #2
        self.update_S(torch.arange(self.num_items, device=self.device)[torch.where(~self.first_split)[0]])
        # concatenation of similarity matrix in all partitions
        self.S = torch.sparse_coo_tensor(indices=torch.cat(self.S_indices, dim=1),
                                         values=torch.cat(self.S_values, dim=0),
                                         size=(self.num_items, self.num_items)).coalesce().T.to_sparse_csr()
        logger.info(f"Total number of partitions created: {len(self.S_indices)}")
        del self.S_indices, self.S_values

        ###### COMPUTING HEATMAP
        # # to save S in numpy dense version
        # logger.info(f"Creating heatmap")
        # S_dense = self.S.clone().to_dense().cpu().numpy()
        # # S_mean = np.mean(S_dense)
        # # S_std = np.std(S_dense)
        # # S_dense = (S_dense - S_mean) / S_std
        # S_min = np.min(S_dense)
        # S_max = np.max(S_dense)
        # S_dense = (S_dense - S_min) / (S_max - S_min)
        # # np.save(f'{os.getcwd()}/heatmap/yelp2018/fpsr/similarity_matrix.npy', S_dense)
        # np.save('/home/ironman/projects/fpsr-sir_elliot/heatmap/yelp2018/fpsr/similarity_matrix.npy', S_dense)
        # # np.savetxt('self_S.tsv', S_dense, delimiter='\t')
        # del S_dense
        # logger.info(f"Created")
        # sys.exit()

    def update_S(self, item_list) -> None:
        if item_list.shape[0] <= self.tau * self.num_items:  # |I_n| <= tau * |I|
            # If the partition size is smaller than size limit, model item similarity for this partition.
            comm_inter = self.inter.index_select(dim=1, index=item_list).to_dense()
            comm_inter = torch.mm(comm_inter.T, comm_inter)
            comm_ae = self.item_similarity(
                comm_inter,
                self.V[item_list, :],
                self.d_i[item_list, :],
                self.d_i_inv[:, item_list]
            )
            comm_ae = torch.where(comm_ae >= self.eps, comm_ae, 0).to_sparse_coo()
            # comm_ae = comm_ae.to_sparse_coo()  # no tollerance
            self.S_indices.append(item_list[comm_ae.indices()])
            self.S_values.append(comm_ae.values())
        else:
            # If the partition size is larger than size limit, perform graph partitioning_item on this partition.
            split = self.partitioning(self._svd(self._norm_adj(item_list), 2))
            self.update_S(item_list[torch.where(split)[0]])
            self.update_S(item_list[torch.where(~split)[0]])

    def item_similarity(self, inter_mat, V, d_i, d_i_inv) -> torch.Tensor:
        Q_hat = inter_mat + self.w_2 * torch.diag(torch.pow(d_i_inv, 2).flatten()) + self.eta
        Q_inv = torch.inverse(Q_hat + self.rho * torch.eye(inter_mat.shape[0], device=self.device))
        Z_aux = (Q_inv @ Q_hat @ (torch.eye(inter_mat.shape[0], device=self.device) - self.l_w * d_i * V @ V.T * d_i_inv))
        del Q_hat
        Phi = torch.zeros_like(Q_inv, device=self.device)
        S = torch.zeros_like(Q_inv, device=self.device)
        for _ in tqdm(range(50)):
            Z_tilde = Z_aux + Q_inv @ (self.rho * (S - Phi))
            gamma = torch.diag(Z_tilde) / (torch.diag(Q_inv) + 1e-10)
            Z = Z_tilde - Q_inv * gamma  # Update Z
            S = torch.clip(Z + Phi - self.w_1 / self.rho, min=0)
            Phi += Z - S
        return S

    def predict(self, start, stop):
        # C = S + lambda * W (C -> item-similarity matrix, W -> global info, S-> local info)
        batch = torch.arange(start, stop).to(self.device)
        user = self.inter.index_select(dim=0, index=batch).to_dense()
        r = torch.sparse.mm(self.S, user.T).T
        r += self.l_w * torch.sparse.mm(self.W_sparse, user.T).T
        return r
