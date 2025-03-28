from abc import ABC

from .loss import BPRLoss, EmbLoss

import torch
import numpy as np
import random
import scipy

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")


class SpectralCFModel(torch.nn.Module, ABC):
    def __init__(self,
                 learning_rate,
                 factors,
                 l_reg,
                 n_layers,
                 n_filters,
                 inter,
                 adj,
                 num_users,
                 num_items,
                 seed,
                 name="SpectralCF",
                 **kwargs
                 ):
        super().__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = factors
        self.learning_rate = learning_rate
        self.l_reg = l_reg
        self.n_layers = n_layers
        self.n_filters = n_filters

        self.inter = inter  # interaction matrix in 'coo' format
        self.A = adj

        A_tilde = self._get_norm_adj()
        I = torch.eye(self.num_items + self.num_users)
        L = I - A_tilde
        self.A_hat = (I + L).to(self.device)

        self.user_embeddings = None
        self.item_embeddings = None

        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k).to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k).to(self.device)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        torch.nn.init.xavier_uniform_(self.Gi.weight)

        self.filters = torch.nn.ParameterList([
            torch.nn.Parameter(torch.normal(mean=0.01, std=0.02, size=(self.embed_k, self.embed_k)), requires_grad=True)
            for _ in range(self.n_filters)
            ]).to(self.device)

        self.sigmoid = torch.nn.Sigmoid()

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)

    def _get_norm_adj(self):
        sumArr = np.array(self.A.sum(axis=1)).flatten() + 1e-7
        diag = 1.0 / sumArr
        D = scipy.sparse.diags(diag)
        A_tilde = D @ self.A
        A_tilde = A_tilde.tocoo()

        indices = torch.tensor([A_tilde.row, A_tilde.col], dtype=torch.long)
        values = torch.tensor(A_tilde.data, dtype=torch.float32)
        A_tilde = torch.sparse_coo_tensor(indices, values, size=A_tilde.shape)

        return A_tilde

    def get_ego_embeddings(self):
        user_embeddings = self.Gu.weight
        item_embeddings = self.Gi.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0).to(self.device)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for k in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.A_hat, all_embeddings)
            all_embeddings = self.sigmoid(torch.mm(all_embeddings, self.filters[k]))
            embeddings_list.append(all_embeddings)

        new_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            new_embeddings, [self.num_users, self.num_items]
        )
        self.user_embeddings = user_all_embeddings
        self.item_embeddings = item_all_embeddings
        return user_all_embeddings, item_all_embeddings


    def train_step(self, batch):
        user, pos, neg = batch
        user_all_embeddings, item_all_embeddings = self.forward()
        user = user.squeeze()
        pos = pos.squeeze()
        neg = neg.squeeze()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos]
        neg_embeddings = item_all_embeddings[neg]
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        loss = mf_loss + self.l_reg * reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.user_embeddings[start: stop].to(self.device),
                            torch.transpose(self.item_embeddings.to(self.device), 0, 1))
