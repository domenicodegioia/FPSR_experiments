import os
import random
import numpy as np
import torch
import sys
from tqdm import tqdm

from .utils import dist, trace

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")


class BISMModel(torch.nn.Module):
    def __init__(self,
                 num_users,
                 num_items,
                 inter,
                 alpha,
                 beta,
                 lamb,
                 c,
                 random_seed,
                 name="FPSR",
                 **kwargs
                 ):
        super(BISMModel, self).__init__()

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
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        self.c = c
        self.iterations = 10
        self.inter = inter

        # compute user-item interaction matrix
        self.inter = torch.sparse_coo_tensor(
            indices=torch.LongTensor(np.array([self.inter.row, self.inter.col])),
            values=torch.FloatTensor(self.inter.data),
            size=self.inter.shape, dtype=torch.float
        ).coalesce().to(self.device)

        self.RR = torch.sparse.mm(self.inter.transpose(0, 1), self.inter).to(self.device)

        # local item similarity matrix
        self.Sl = torch.nn.Parameter(torch.empty(self.num_items, self.num_items).to(self.device))
        torch.nn.init.uniform_(self.Sl, 0, 1)
        self.Sl.data.fill_diagonal_(0)

        # global item similarity matrix
        self.Sg = torch.nn.Parameter(torch.empty(self.num_items, self.num_items).to(self.device))
        torch.nn.init.uniform_(self.Sg, 0, 1)
        self.Sg.data.fill_diagonal_(0)

        self.F = torch.nn.Parameter(torch.rand(self.num_items, self.c).to(self.device))

        self.D = 1e-5

    def train_step(self):
        with torch.no_grad():
            # update F
            logger.info(f"Updating F...")
            S = self.Sl.data
            S0 = S.clone()
            S0.add_(S.t())
            S0.div_(2)
            L = torch.diag(torch.sum(S0, 0)) - S0
            L = (L + L.t()) / 2
            del S, S0
            _, F = torch.lobpcg(L, k=self.c, largest=False, tol=1e-4)
            del L
            self.F.data = F
            self.D = self.lamb * dist(F) + 1e-5

            # update S
            logger.info(f"Updating S...")
            Sl = self.Sl.data
            Sg = self.Sg.data

            S_sum = Sl + Sg
            RR_S_sum = torch.matmul(self.RR, S_sum)
            del S_sum

            denominator = RR_S_sum + self.D
            denominator.add_(Sl, alpha=self.beta)
            Sl *= self.RR
            Sl /= denominator

            denominator = RR_S_sum + self.alpha
            denominator.add_(Sg, alpha=self.beta)
            Sg *= self.RR
            Sg /= denominator

            # model.object()
            logger.info(f"Computing Loss...")
            # Sl = self.Sl.data
            # Sg = self.Sg.data
            # S = Sl + Sg
            # loss = torch.sparse.sum(self.inter.pow(2)) / 2
            # loss -= trace(self.RR, S)
            # loss += trace(S.t() @ self.RR, S) / 2
            # loss += self.alpha * torch.sum(Sg)
            # loss += trace(self.D, Sl) * self.lamb
            # loss += (trace(Sl) + trace(Sg)) * self.beta / 2
            # obj += trace(torch.sum(S, 0) - 1) * self.gamma / 2
        return 0

    def predict(self, start, stop):
        batch = torch.arange(start, stop).to(self.device)
        user = self.inter.index_select(dim=0, index=batch).to_dense()
        return torch.mm(user, self.Sl + self.Sg)