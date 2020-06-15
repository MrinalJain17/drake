import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

try:
    # A much faster implementation of KMeans, backed by Intel's DAAL
    # GitHub: https://github.com/IntelPython/daal4py
    # Conda installation: https://anaconda.org/intel/daal4py
    from daal4py.sklearn.cluster import KMeans
except ImportError:
    # Fallback to use sklearn if DAAL is not installed
    from sklearn.cluster import KMeans


class ConsensusClustering(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        n_clusters=8,
        iterations=1000,
        sample_size=0.8,
        n_jobs=None,
        random_state=None,
    ):
        """TODO

        """
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.sample_size = sample_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.epsilon = 1e-8

    def fit(self, X, y=None, disable_progress=False):
        """TODO

        """
        self.num_samples_, _ = X.shape
        self.n_jobs = effective_n_jobs(self.n_jobs)
        self.consensus_matrix_ = self._fit(X, disable_progress)
        self.labels_ = self._fit_distance_matrix(self.consensus_matrix_)

        return self

    def fit_predict(self, X, y=None, disable_progress=False):
        """TODO

        """
        return self.fit(X, y, disable_progress).labels_

    def score(self, X, y=None):
        """TODO

        """
        check_is_fitted(self)
        return calinski_harabasz_score(X, self.labels_)

    def _fit(self, X, disable_progress=False):
        """TODO

        """
        resampled_indices, resampling_co_occurence_matrix = self._resample()
        co_occurence_matrix_list = None

        if self.n_jobs == 1:
            # With no multi-processing (hence n_jobs=1), using a simple list comprehension
            # is slightly faster than using joblib's Parallel
            co_occurence_matrix_list = [
                self._fit_subset(X, index)
                for index in tqdm(
                    resampled_indices,
                    desc=f"ConsensusClustering(n_clusters={self.n_clusters})",
                    disable=disable_progress,
                )
            ]
        else:
            co_occurence_matrix_list = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self._fit_subset)(X, index=index)
                for index in tqdm(
                    resampled_indices,
                    desc=f"ConsensusClustering(n_clusters={self.n_clusters})",
                    disable=disable_progress,
                )
            )

        clustering_co_occurence_matrix = np.sum(co_occurence_matrix_list, axis=0)

        return self._compute_consensus_matrix(
            clustering_co_occurence_matrix, resampling_co_occurence_matrix
        )

    def _fit_distance_matrix(self, distance_matrix):
        """TODO

        """
        return AgglomerativeClustering(
            n_clusters=self.n_clusters, affinity="euclidean", linkage="average"
        ).fit_predict(distance_matrix)

    def _fit_subset(self, X, index):
        """TODO

        """
        subset = X[index, :]
        assigned_clusters = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state,
        ).fit_predict(subset)

        clustering_matrix = np.zeros((self.n_clusters, self.num_samples_))
        clustering_matrix[assigned_clusters, index] = 1

        return np.dot(clustering_matrix.T, clustering_matrix)

    def _resample(self):
        """TODO

        """
        random_state = check_random_state(self.random_state)
        size = (self.iterations, int(self.num_samples_ * self.sample_size))
        idx_matrix = random_state.choice(self.num_samples_, size, replace=True)

        resampling_matrix = np.zeros((self.iterations, self.num_samples_))
        rows = np.arange(self.iterations).reshape(self.iterations, -1)
        resampling_matrix[rows, idx_matrix] = 1
        resampling_co_occurence_matrix = np.dot(resampling_matrix.T, resampling_matrix)

        return idx_matrix, resampling_co_occurence_matrix

    def _compute_consensus_matrix(
        self, clustering_co_occurence_matrix, resampling_co_occurence_matrix
    ):
        """TODO

        """
        consensus_matrix = clustering_co_occurence_matrix / (
            resampling_co_occurence_matrix + self.epsilon
        )
        np.fill_diagonal(consensus_matrix, 1)

        return consensus_matrix
