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

        indices = torch.arange(self.num_users + self.num_items, device=self.device)
        I = torch.sparse_coo_tensor(indices=torch.stack([indices, indices]),
                                    values=torch.ones(self.num_users + self.num_items, device=self.device),
                                    size=(self.num_users + self.num_items, self.num_users + self.num_items))

        # L = torch.sparse.add(I, -1.0, A_tilde)  # L = I - A_tilde
        L = I - A_tilde

        # self.A_hat = torch.sparse.add(I, 1.0, L)  # A_hat = I + L = 2I - A_tilde
        self.A_hat = I + L

        del I, L, A_tilde


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
        A = self.A.coalesce()
        indices = A.indices()
        values = A.values()
        row = indices[0]

        row_sum = torch.zeros(A.shape[0], device=self.device).index_add_(0, row, values)
        inv_deg = 1.0 / (row_sum + 1e-7)

        norm_values = values * inv_deg[row]
        A_tilde = torch.sparse_coo_tensor(indices, norm_values, A.shape, device=self.device)

        return A_tilde.coalesce()

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
