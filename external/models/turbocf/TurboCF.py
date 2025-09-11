import time
import random

import torch
import numpy as np
from tqdm import tqdm

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

from .sparse_matmul import batch_sparse_matmul_sparse_output

class TurboCF(RecMixin, BaseRecommenderModel):
    r"""
    Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation

    For further details, please refer to the `paper <https://arxiv.org/abs/2404.14243>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml


    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 512, int, None),
            ("_alpha", "alpha", "alpha", 0.3, float, None),
            ("_power", "power", "power", 1.0, float, None),
            ("_filter", "filter", "filter", 1, int, None),
            ("_seed", "seed", "seed", 42, int, None),
        ]

        self.autoset_params()

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self._filter < 1 or self._filter > 3:
            raise ValueError(f"chosen between: (1) linear, (2) 2nd order, (3) polynomial !!!!!!")

        # compute user-item interaction matrix
        row, col = data.sp_i_train.nonzero()
        self.R = torch.sparse_coo_tensor(
            indices=torch.LongTensor(np.array([row, col])),
            values=torch.FloatTensor(np.ones_like(row, dtype=np.float64)),
            size=(self._num_users, self._num_items),
            dtype=torch.float
        ).coalesce().to(self.device)

        self.LPF = None


    def normalized_sparse_rating_matrix(self, m, alpha):
        values = torch.ones((m.shape[1], 1)).to(self.device)
        rowsum = torch.sparse.mm(m, values).to(self.device)
        rowsum = torch.pow(rowsum, -alpha).squeeze()

        values = torch.ones((m.shape[0], 1)).to(self.device)
        colsum = torch.sparse.mm(m.t(), values).to(self.device)
        colsum = torch.pow(colsum, alpha - 1).squeeze()

        diag_indices = torch.arange(rowsum.shape[0], device=self.device)
        indices = torch.stack([diag_indices, diag_indices], dim=0)
        d_mat_rows = torch.sparse_coo_tensor(
            indices, rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])
        ).to(self.device)

        diag_indices = torch.arange(colsum.shape[0], device=self.device)
        indices = torch.stack([diag_indices, diag_indices], dim=0)
        d_mat_cols = torch.sparse_coo_tensor(
            indices, colsum, torch.Size([colsum.size(0), colsum.size(0)])
        ).to(self.device)

        R_tilde = d_mat_rows.mm(m).mm(d_mat_cols)

        return R_tilde


    def train(self):
        start = time.time()

        R_tilde = self.normalized_sparse_rating_matrix(self.R, self._alpha).to(self.device)
        P = R_tilde.T @ R_tilde
        P.float().coalesce().to(self.device)
        P._values().pow_(self._power)

        del R_tilde

        if self._filter == 1:
            self.LPF = (P)

        elif self._filter == 2:
            PP = batch_sparse_matmul_sparse_output(A=P, B=P, device=self.device, batch_size=1000)
            P_dense = P.to_dense()
            PP_dense = PP.to_dense()
            # self.LPF = (2 * P - P @ P)
            LPF_dense = 2 * P_dense - PP_dense
            self.LPF = LPF_dense.to_sparse()

            del P_dense, PP_dense, LPF_dense
            torch.cuda.empty_cache()

        elif self._filter == 3:
            # P @ P
            PP = batch_sparse_matmul_sparse_output(A=P, B=P, device=self.device, batch_size=1000)
            # P @ P @ P
            PPP = batch_sparse_matmul_sparse_output(A=PP, B=P, device=self.device, batch_size=1000)

            P_dense = P.to_dense()
            PP_dense = PP.to_dense()
            PPP_dense = PPP.to_dense()

            # self.LPF = (P + 0.01 * (-P @ P @ P + 10 * P @ P - 29 * P))
            LPF_dense = P_dense + 0.01 * (-PPP_dense + 10 * PP_dense - 29 * P_dense)
            self.LPF = LPF_dense.to_sparse()

            del P_dense, PP_dense, PPP_dense, PP, PPP, LPF_dense
            torch.cuda.empty_cache()

        end = time.time()
        logger.info(f"The similarity computation has taken: {end - start}")

        self.evaluate()

    # def batch_sparse_matmul_sparse_output(self, P, batch_size=1000):
    #     """
    #     Calcola P @ P a blocchi e restituisce una matrice sparse COO come output.
    #     """
    #     P = P.coalesce()
    #     indices = P.indices()
    #     values = P.values()
    #     n = P.size(0)
    #
    #     P_dense = P.to_dense()
    #     result_rows = []
    #
    #     for start in tqdm(range(0, n, batch_size)):
    #         end = min(start + batch_size, n)
    #
    #         # Seleziona solo le righe del batch
    #         mask = (indices[0] >= start) & (indices[0] < end)
    #         batch_indices = indices[:, mask].clone()
    #         batch_indices[0] -= start
    #         batch_values = values[mask]
    #
    #         # Costruisci blocco sparso
    #         P_block = torch.sparse_coo_tensor(
    #             batch_indices,
    #             batch_values,
    #             size=(end - start, n),
    #             device=P.device,
    #             dtype=P.dtype
    #         ).coalesce()
    #
    #         # Sparse @ Dense (eseguito su GPU)
    #         block_result = torch.matmul(P_block, P_dense)  # (batch_size, n), denso
    #
    #         result_rows.append(block_result.cpu())
    #
    #     # Ricostruisci tutta la matrice densa su CPU
    #     full_dense_cpu = torch.cat(result_rows, dim=0)  # shape: (n, n)
    #
    #     # Applica soglia per costruire sparse
    #     mask = full_dense_cpu != 0
    #
    #     indices = mask.nonzero(as_tuple=False).T  # shape: (2, nnz)
    #     values = full_dense_cpu[mask]
    #
    #     # Costruisci tensore sparso sul device finale
    #     return torch.sparse_coo_tensor(
    #             indices.to(self.device),
    #             values.to(self.device),
    #             size=(n, n),
    #             device=self.device,
    #             dtype=P.dtype
    #     ).coalesce()

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(tqdm(range(0, self._num_users, self._batch_eval))):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = self.predict(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def predict(self, start, stop):
        batch = torch.arange(start, stop).to(self.device)
        user = self.R.index_select(dim=0, index=batch).to_dense()
        return torch.sparse.mm(self.LPF, user.T).T

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
        return "TurboCF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"
