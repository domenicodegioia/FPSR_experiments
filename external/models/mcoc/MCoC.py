"""
Module description:
This module contains the implementation of the Multiclass Co-Clustering (MCoC) model.
MCoC is a clustering-based collaborative filtering approach that improves recommendations
by identifying user-item subgroups. It allows users and items to belong to multiple subgroups
simultaneously, capturing diverse user interests. Recommendations are then generated within
these more coherent and dense subgroups.
This implementation uses SVD as the prediction model within each subgroup.

Paper: An Exploration of Improving Collaborative Recommender Systems via User-Item Subgroups
Authors: Xu, B., Bu, J., Chen, C., & Cai, D.
Conference: WWW 2012
"""

import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, svds
from tqdm import tqdm

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class MCoC(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Initialize the MCoC model.

        Args:
            data: The processed data object.
            config: The configuration dictionary.
            params: The model parameters dictionary.
        """
        self._params_list = [
            ("_num_subgroups", "c", "c", 15, int, None), #"Number of subgroups (clusters)."
            ("_embedding_dim", "r", "r", 3, int, None), #"Dimensionality of the shared embedding space."),
            ("_user_item_k", "k", "k", 2, int, None), # "Number of subgroups each user/item can belong to."),
            ("_svd_k", "svd_k", "svd_k", 20, int, None), # "Number of latent factors for SVD within subgroups."),
            ("_fuzzy_m", "m", "m", 2.0, float, None), # "Fuzziness parameter for Fuzzy C-Means."),
            ("_max_iter_fuzzy", "max_iter_fuzzy", "max_iter_fuzzy", 100, int, None), # "Max iterations for Fuzzy C-Means."),
            ("_min_subgroup_size", "min_subgroup_size", "min_subgroup_size", 20, int, None), # "Minimum number of ratings to train a subgroup predictor.")
        ]
        self.autoset_params()

        # These will be populated during training
        self._subgroup_predictors = {}
        self._subgroup_maps = {}
        self._user_subgroups = {}  # {user_id: [(subgroup_idx, membership_score), ...]}
        self._item_subgroups = {}  # {item_id: [subgroup_idx, ...]}

    @property
    def name(self):
        return f"MCoC_{self.get_params_shortcut()}"

    def _fuzzy_c_means(self, X):
        """
        A simplified implementation of Fuzzy C-Means clustering.
        Args:
            X: The data matrix (n_samples, n_features) to be clustered.

        Returns:
            The partition matrix P (n_samples, n_clusters) with membership scores.
        """
        n_samples = X.shape[0]

        # Initialize partition matrix P randomly
        P = np.random.rand(n_samples, self._num_subgroups)
        P = P / np.sum(P, axis=1, keepdims=True)

        for _ in (range(self._max_iter_fuzzy)):
            # Calculate cluster centers (centroids)
            P_m = P ** self._fuzzy_m
            centers = (P_m.T @ X) / np.sum(P_m, axis=0, keepdims=True).T

            # Update partition matrix P
            dist = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            dist = np.where(dist == 0, 1e-8, dist)  # Avoid division by zero

            inv_dist = dist ** (-2 / (self._fuzzy_m - 1))
            P_new = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)

            if np.linalg.norm(P - P_new) < 1e-5:
                break
            P = P_new

        # Enforce top-k constraint
        top_k_indices = np.argsort(P, axis=1)[:, -self._user_item_k:]
        P_top_k = np.zeros_like(P)
        np.put_along_axis(P_top_k, top_k_indices, np.take_along_axis(P, top_k_indices, axis=1), axis=1)

        # Re-normalize rows to sum to 1
        row_sums = np.sum(P_top_k, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero for rows with no membership
        P_final = P_top_k / row_sums

        return P_final

    def _compute_subgroup_predictor(self, sub_matrix: sp.csr_matrix):
        """
        Computes the SVD components for a given subgroup matrix.

        Args:
            sub_matrix: The sparse user-item rating matrix for the subgroup.

        Returns:
            A tuple (U, s, Vt) representing the SVD components. Returns None if SVD fails.
        """
        if sub_matrix.shape[0] < self._svd_k or sub_matrix.shape[1] < self._svd_k:
            self.logger.warning(f"Subgroup too small for SVD with k={self._svd_k}. Skipping.")
            return None
        try:
            U, s, Vt = svds(sub_matrix.asfptype(), k=self._svd_k)
            # Sort singular values in descending order
            s_desc_indices = np.argsort(-s)
            s = s[s_desc_indices]
            U = U[:, s_desc_indices]
            Vt = Vt[s_desc_indices, :]
            return U, s, Vt
        except Exception as e:
            self.logger.error(f"SVD failed for a subgroup: {e}")
            return None

    def train(self):
        if self._restore:
            return self.restore_weights()

        start_time = time.time()
        self.logger.info("MCoC training started...")
        T = self._data.sp_i_train

        # --- Step 1: Dimensionality Reduction ---
        self.logger.info("Step 1: Performing dimensionality reduction...")
        n_users, n_items = T.shape

        row_sum = np.array(T.sum(axis=1)).flatten()
        col_sum = np.array(T.sum(axis=0)).flatten()

        D_row_inv_sqrt = sp.diags(1 / np.sqrt(row_sum, where=row_sum > 0, out=np.zeros_like(row_sum)))
        D_col_inv_sqrt = sp.diags(1 / np.sqrt(col_sum, where=col_sum > 0, out=np.zeros_like(col_sum)))

        S = D_row_inv_sqrt @ T @ D_col_inv_sqrt

        I_n = sp.eye(n_users)
        I_m = sp.eye(n_items)

        M = sp.bmat([[I_n, -S], [-S.T, I_m]], format='csr')

        # Find the r smallest eigenvectors
        _, eigenvectors = eigs(M, k=self._embedding_dim, which='SR')
        X = eigenvectors.real

        # --- Step 2: Subgroup Discovery ---
        self.logger.info("Step 2: Discovering subgroups via Fuzzy C-Means...")
        P = self._fuzzy_c_means(X)
        Q = P[:n_users, :]  # User partitions
        R = P[n_users:, :]  # Item partitions

        # Pre-compute subgroup memberships for faster lookup
        for u_idx in range(n_users):
            user_id = self._data.private_users[u_idx]
            memberships = [(sg_idx, score) for sg_idx, score in enumerate(Q[u_idx]) if score > 0]
            self._user_subgroups[user_id] = sorted(memberships, key=lambda x: x[1], reverse=True)

        for i_idx in range(n_items):
            item_id = self._data.private_items[i_idx]
            self._item_subgroups[item_id] = [sg_idx for sg_idx, score in enumerate(R[i_idx]) if score > 0]

        # --- Step 3: Train Subgroup Predictors ---
        self.logger.info("Step 3: Training SVD predictors for each subgroup...")
        for s_idx in tqdm(range(self._num_subgroups), desc="Training Subgroup SVDs"):
            # Find users and items in the current subgroup
            user_indices_in_sg = np.where(Q[:, s_idx] > 0)[0]
            item_indices_in_sg = np.where(R[:, s_idx] > 0)[0]

            if len(user_indices_in_sg) == 0 or len(item_indices_in_sg) == 0:
                continue

            sub_matrix = T[user_indices_in_sg, :][:, item_indices_in_sg]

            if sub_matrix.nnz < self._min_subgroup_size:
                continue

            predictor = self._compute_subgroup_predictor(sub_matrix)
            if predictor:
                self._subgroup_predictors[s_idx] = predictor
                # Store mappings from local (subgroup) to global indices
                self._subgroup_maps[s_idx] = {
                    'users': {local_idx: global_idx for local_idx, global_idx in enumerate(user_indices_in_sg)},
                    'items': {local_idx: global_idx for local_idx, global_idx in enumerate(item_indices_in_sg)},
                    'users_rev': {global_idx: local_idx for local_idx, global_idx in enumerate(user_indices_in_sg)},
                    'items_rev': {global_idx: local_idx for local_idx, global_idx in enumerate(item_indices_in_sg)}
                }

        self.logger.info(f"MCoC training finished in {time.time() - start_time:.2f} seconds.")
        self.evaluate()

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k):
        recs = {}
        # Use a smaller batch size as MCoC prediction can be memory intensive
        batch_size = 256
        total_batches = len(self._data.train_dict.keys()) // batch_size + (
            1 if len(self._data.train_dict.keys()) % batch_size != 0 else 0)

        for i in tqdm(range(0, len(self._data.train_dict.keys()), batch_size), desc="Processing batches",
                      total=total_batches):
            batch = list(self._data.train_dict.keys())[i:i + batch_size]
            mat = self.get_user_recs_batch(batch, mask, k)
            proc_batch = dict(zip(batch, mat))
            recs.update(proc_batch)
        return recs

    def get_user_recs_batch(self, user_batch, mask, k):
        # user_batch: list of public user IDs

        batch_recs = []
        for public_user_id in user_batch:
            user_idx = self._data.public_users[public_user_id]

            # Initialize prediction scores for all items to a very low value
            user_preds = np.full(self._data.num_items, -np.inf, dtype='float32')

            user_sgs = self._user_subgroups.get(public_user_id, [])
            if not user_sgs:
                batch_recs.append([])
                continue

            # Use the most relevant subgroup for the user, as per the paper's strategy
            best_sg_idx = user_sgs[0][0]

            if best_sg_idx in self._subgroup_predictors:
                U, s, Vt = self._subgroup_predictors[best_sg_idx]
                sg_maps = self._subgroup_maps[best_sg_idx]

                # Check if user is in this subgroup's map
                if user_idx not in sg_maps['users_rev']:
                    continue

                local_user_idx = sg_maps['users_rev'][user_idx]

                # Reconstruct scores for all items WITHIN the subgroup
                subgroup_scores = U[local_user_idx, :] @ np.diag(s) @ Vt

                # Map local item indices and scores back to global item indices
                global_item_indices = np.array(list(sg_maps['items_rev'].keys()))
                user_preds[global_item_indices] = subgroup_scores

            # Apply mask to filter out already rated items
            # --- START CORRECTION ---
            # The original line was incorrect because it used public IDs from dict.keys()
            # on an array indexed by private indices.

            # 1. Get the public IDs of items the user has interacted with.
            rated_public_item_ids = self._data.train_dict[public_user_id].keys()
            # 2. Convert these public IDs to their corresponding private/internal indices.
            rated_private_item_indices = [self._data.public_items[item_id] for item_id in rated_public_item_ids]
            # 3. Use the list of private indices to correctly mask the predictions array.
            if rated_private_item_indices:
                user_preds[rated_private_item_indices] = -np.inf
            # --- END CORRECTION ---

            # Get top-k recommendations
            indices = np.argpartition(user_preds, -k)[-k:]
            top_k_indices = indices[np.argsort(-user_preds[indices])]

            top_k_scores = user_preds[top_k_indices]

            # Map private item indices back to public IDs
            public_item_ids = [self._data.private_items[i] for i in top_k_indices]

            batch_recs.append(list(zip(public_item_ids, top_k_scores)))

        return batch_recs