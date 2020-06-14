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
        self.epsilon = 1e-8
        self.consensus_matrix = None
        self.hierarchical_clusters = None
        self.scores = {}
        self.best_k = None
        self._num_samples = None
        self._num_clusters = None
        self._resampling_matrix = None
        self._resampling_co_occurence_matrix = None
        self._clustering_co_occurence_matrix = None

    def fit(self, X, y=None, show_progress=True):
        """TODO

        """
        random_state = check_random_state(self.random_state)

        self._num_samples, _ = X.shape
        if isinstance(self.n_clusters, int):
            self.n_clusters = [self.n_clusters]
        self._num_clusters = len(self.n_clusters)
        self._initialize_matrices()

        for idx, k in enumerate(tqdm(self.n_clusters, disable=(not show_progress))):
            self._fit_cluster(X, idx, random_state, show_progress)

        self.best_k = max(self.scores, key=self.scores.get)

        return self

    def _initialize_matrices(self):
        assert isinstance(self._num_samples, int)
        assert isinstance(self.n_clusters, list)
        assert isinstance(self._num_clusters, int)

        self._resampling_matrix = np.zeros(
            (self._num_clusters, self.iterations, self._num_samples)
        )
        self._clustering_co_occurence_matrix = np.zeros(
            (self._num_clusters, self._num_samples, self._num_samples,)
        )
        self.consensus_matrix = np.zeros(
            (self._num_clusters, self._num_samples, self._num_samples)
        )
        self.hierarchical_clusters = np.zeros((self._num_clusters, self._num_samples))

    def _fit_cluster(self, X, cluster_idx, random_state, show_progress):
        k = self.n_clusters[cluster_idx]
        if show_progress:
            print(f"Clustering using k={k}")

        for iteration in tqdm(range(self.iterations), disable=(not show_progress)):
            self._fit_subset(X, iteration, cluster_idx, random_state)

        self._resampling_co_occurence_matrix = np.dot(
            self._resampling_matrix[cluster_idx].T,
            self._resampling_matrix[cluster_idx],
        )

        self.consensus_matrix[cluster_idx] = self._clustering_co_occurence_matrix[
            cluster_idx
        ] / (self._resampling_co_occurence_matrix + self.epsilon)

        np.fill_diagonal(self.consensus_matrix[cluster_idx], 1)

        self.hierarchical_clusters[cluster_idx] = AgglomerativeClustering(
            n_clusters=k, affinity="euclidean", linkage="average"
        ).fit_predict(self.consensus_matrix[cluster_idx])

        self.scores[k] = calinski_harabasz_score(
            self.consensus_matrix[cluster_idx], self.hierarchical_clusters[cluster_idx],
        )

    def _fit_subset(self, X, current_iteration, current_cluster_idx, random_state):
        indices, subset = self._resample(
            X, current_iteration, current_cluster_idx, random_state
        )
        assigned_clusters = KMeans(
            n_clusters=self.n_clusters[current_cluster_idx],
            random_state=self.random_state,
        ).fit_predict(subset)
        clustering_matrix = np.zeros((self._num_clusters + 1, self._num_samples))
        clustering_matrix[assigned_clusters, indices] = 1
        self._clustering_co_occurence_matrix[current_cluster_idx] += np.dot(
            clustering_matrix.T, clustering_matrix,
        )

    def _resample(self, X, current_iteration, current_cluster_idx, random_state):
        indices = random_state.choice(
            self._num_samples,
            size=int(self._num_samples * self.sample_size),
            replace=True,
        )
        self._resampling_matrix[
            current_cluster_idx, current_iteration, np.unique(indices)
        ] = 1

        return indices, X[indices, :]
