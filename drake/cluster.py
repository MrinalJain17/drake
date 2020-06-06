import bisect
from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.utils import check_random_state
from tqdm.auto import tqdm


class ConsensusClustering(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(
        self, n_clusters=8, iterations=1000, sample_size=0.8, random_state=None
    ):
        """TODO

        """
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.sample_size = sample_size
        self.random_state = random_state
        self.consensus_matrix = None
        self.hierarchical_clusters = None
        self.scores = {}
        self.best_k = None
        self.epsilon = 1e-8

    def fit(self, X, y=None, verbose=False):
        """TODO

        """
        random_state = check_random_state(self.random_state)
        if isinstance(self.n_clusters, int):
            self.n_clusters = [self.n_clusters]
        num_samples, _ = X.shape
        self.consensus_matrix = np.zeros(
            (len(self.n_clusters), num_samples, num_samples)
        )
        self.hierarchical_clusters = np.zeros((len(self.n_clusters), num_samples))
        total_matrix = np.zeros((num_samples, num_samples))

        for idx, k in enumerate(tqdm(self.n_clusters)):
            if verbose:
                print(f"Clustering using k={k}")
            for h in tqdm(range(self.iterations)):
                indices = random_state.choice(
                    num_samples, size=int(num_samples * self.sample_size), replace=True,
                )
                subset = X[indices, :]
                assigned_clusters = KMeans(
                    n_clusters=k, random_state=self.random_state
                ).fit_predict(subset)

                idx_clusters = np.argsort(assigned_clusters)
                sorted_clusters = assigned_clusters[idx_clusters]

                for i in range(k):
                    idx_left = bisect.bisect_left(sorted_clusters, i)
                    idx_right = bisect.bisect_right(sorted_clusters, i)
                    idx_cluster_i = idx_clusters[idx_left:idx_right]
                    same_cluster_entities = np.array(
                        list(combinations(idx_cluster_i, 2))
                    ).T
                    if same_cluster_entities.size != 0:
                        self.consensus_matrix[
                            idx, same_cluster_entities[0], same_cluster_entities[1]
                        ] += 1

                selected_entities = np.array(list(combinations(indices, 2))).T
                total_matrix[selected_entities[0], selected_entities[1]] += 1

            self.consensus_matrix[idx] /= total_matrix + self.epsilon
            self.consensus_matrix[idx] += self.consensus_matrix[idx].T
            np.fill_diagonal(self.consensus_matrix[idx], 1)
            total_matrix = np.zeros((num_samples, num_samples))

            self.hierarchical_clusters[idx] = AgglomerativeClustering(
                n_clusters=k, affinity="euclidean", linkage="average"
            ).fit_predict(self.consensus_matrix[idx])

            self.scores[k] = calinski_harabasz_score(
                self.consensus_matrix[idx], self.hierarchical_clusters[idx]
            )

            self.best_k = max(self.scores, key=self.scores.get)

        return self
