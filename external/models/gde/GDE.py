import scipy as sp
from tqdm import tqdm
import numpy as np
import torch
import os
import random
import scipy.sparse as sp
import gc

from elliot.utils.write import store_recommendation
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .GDEModel import GDEModel


class GDE(RecMixin, BaseRecommenderModel):
    r"""
    Less is More: Reweighting Important Spectral Graph Features for Recommendation

    For further details, please refer to the `paper <http://arxiv.org/abs/2204.11346>`_

    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        if self._batch_size < 1:
            self._batch_size = self._num_users

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.03, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_beta", "beta", "beta", 5.0, float, None),
            ("_loss_type", "loss_type", "loss_type", "adaptive", str, None),
            ("_feature_type", "feature_type", "feature_type", "both", str, None),
            ("_smooth_ratio", "smooth_ratio", "smooth_ratio", 0.1, float, None),
            ("_rough_ratio", "rough_ratio", "rough_ratio", 0.0, float, None),
            ("_dropout", "dropout", "dropout", 0.1, float, None)
        ]
        self.autoset_params()

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rate_matrix = torch.from_numpy(data.sp_i_train.todense()).to(self.device)

        smooth_uu_value, smooth_uu_vector, rough_uu_value, rough_uu_vector, smooth_ii_value, smooth_ii_vector, rough_ii_value, rough_ii_vector = None, None, None, None, None, None, None, None

        if self._config.data_config.strategy == 'fixed':
            self.logger.info(f"Preprocessing {self._config.data_config.train_path}")
            smooth_uu_value, smooth_uu_vector, rough_uu_value, rough_uu_vector, smooth_ii_value, smooth_ii_vector, rough_ii_value, rough_ii_vector = self.preprocess()
            self.logger.info(f"Preprocessing finished!")
            # path = os.path.split(self._config.data_config.train_path)[0]
            # if not (os.path.exists(path + f'/svd_u_{self._alpha}.npy')
            #         or os.path.exists(path + f'/svd_i_{self._alpha}.npy') or os.path.exists(
            #         path + f'/svd_value_{self._alpha}.npy')):
            #     self.logger.info(
            #         f"Processing singular values as they haven't been calculated before on this dataset...")
            #     U, value, V = self.preprocess(path, data.num_users, data.num_items)
            #     self.logger.info(f"Processing end!")
            # else:
            #     self.logger.info(f"Singular values have already been processed for this dataset!")
            #     value = torch.Tensor(np.load(path + f'/svd_value_{self._alpha}.npy'))
            #     U = torch.Tensor(np.load(path + f'/svd_u_{self._alpha}.npy'))
            #     V = torch.Tensor(np.load(path + f'/svd_v_{self._alpha}.npy'))
        else:
            raise NotImplementedError('The check when strategy is different from fixed has not been implemented yet!')

        self._model = GDEModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            factors=self._factors,
            l_w=self._l_w,
            beta=self._beta,
            dropout=self._dropout,
            loss_type=self._loss_type,
            feature_type=self._feature_type,
            smooth_uu_value=smooth_uu_value,
            smooth_uu_vector=smooth_uu_vector,
            rough_uu_value=rough_uu_value,
            rough_uu_vector=rough_uu_vector,
            smooth_ii_value=smooth_ii_value,
            smooth_ii_vector=smooth_ii_vector,
            rough_ii_value=rough_ii_value,
            rough_ii_vector=rough_ii_vector,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "GDE" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def cal_spectral_feature(self, Adj, size, largest=True, niter=5):
        # k: the number of required features
        # largest: Ture (default) for k-largest (smoothed)  and Flase for k-smallest (rough) eigenvalues
        # niter: maximum number of iterations
        # for more information, see https://pytorch.org/docs/stable/generated/torch.lobpcg.html
        value, vector = torch.lobpcg(Adj, k=size, largest=largest, niter=niter)
        return value, vector

    def preprocess(self):
        D_u = self.rate_matrix.sum(1)
        D_i = self.rate_matrix.sum(0)
        for i in range(self._num_users):
            if D_u[i] != 0:
                D_u[i] = 1 / D_u[i].sqrt()
        for i in range(self._num_items):
            if D_i[i] != 0:
                D_i[i] = 1 / D_i[i].sqrt()
        rate_matrix = D_u.unsqueeze(1) * self.rate_matrix * D_i

        del D_u, D_i
        gc.collect()
        torch.cuda.empty_cache()

        # user-user relations
        L_u = rate_matrix.mm(rate_matrix.t())
        smooth_uu_value, smooth_uu_vector = self.cal_spectral_feature(L_u,
                                                                      int(self._smooth_ratio * self._num_users),
                                                                      largest=True)
        if self._rough_ratio != 0:
            rough_uu_value, rough_uu_vector = self.cal_spectral_feature(L_u,
                                                                        int(self._rough_ratio * self._num_users),
                                                                        largest=False)
        else:
            rough_uu_value, rough_uu_vector = None, None

        del L_u
        gc.collect()
        torch.cuda.empty_cache()

        # item-item relations
        L_i = rate_matrix.t().mm(rate_matrix)
        smooth_ii_value, smooth_ii_vector = self.cal_spectral_feature(L_i,
                                                                      int(self._smooth_ratio * self._num_items),
                                                                      largest=True)
        if self._rough_ratio != 0:
            rough_ii_value, rough_ii_vector = self.cal_spectral_feature(L_i,
                                                                        int(self._rough_ratio * self._num_items),
                                                                        largest=False)
        else:
            rough_ii_value, rough_ii_vector = None, None

        del L_i
        gc.collect()
        torch.cuda.empty_cache()

        return smooth_uu_value, smooth_uu_vector, rough_uu_value, rough_uu_vector, smooth_ii_value, smooth_ii_vector, rough_ii_value, rough_ii_vector

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for _, _ in enumerate(range(0, self._data.transactions, self._batch_size)):
                    steps += 1
                    u = torch.LongTensor(np.random.randint(0, self._num_users, self._batch_size)).to(self.device)
                    p = torch.multinomial(self.rate_matrix[u], 1, True).squeeze(1)
                    nega = torch.multinomial(1 - self.rate_matrix[u], 1, True).squeeze(1)
                    batch = u, p, nega
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self.get_users_rating(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_users_rating(self, batch_start, batch_stop):
        final_user_embeddings = self._model.L_u.mm(self._model.user_embed.weight)
        final_item_embeddings = self._model.L_i.mm(self._model.item_embed.weight)
        batch_user_embeddings = final_user_embeddings[batch_start:batch_stop]
        predictions = (batch_user_embeddings.mm(final_item_embeddings.t())).sigmoid()
        return predictions - self.rate_matrix[batch_start:batch_stop] * 1000

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))
