import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import check_random_state
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
    """
    Consensus Clustering

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to find.

    iterations : int, default=1000
        The number of times to execute K-Means on random subsets of the data,
        selected based `sample_size`.

    sample_size : float, default=0.8
        The fraction of the data to randomly sample (with replacement) for each
        iteration.

    n_jobs : int or None, default=None
        The number of parallel jobs to run. If `None`, then there is essentially
        no parallelism. `-1` means use all available CPU cores.

    random_state : int or None, default=None
        Determines random number generation for the sampling procedure and the
        initialization of subsequent K-Means algorithm. Pass an integer to make
        the randomness deterministic.

    Attributes
    ----------
    num_samples_ : int
        The number of samples/objects in the dataset.

    consensus_matrix_ : np.ndarray of shape (n_samples, n_samples)
        Each element of the matrix represents the proportion of times that the two
        objects were included in the same cluster, i.e., the ratio of the number of
        times a given pair of objects were included in the same cluster to the number
        of times both of the objects were selected in the random subset. Therefore,
        each element of the matrix can be interpreted as a probability that the two
        objects belong to the same cluster.

    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each sample/object.

    """

    def __init__(
        self,
        n_clusters=8,
        iterations=1000,
        sample_size=0.8,
        n_jobs=None,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.sample_size = sample_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.epsilon = 1e-8

    def fit(self, X, y=None, disable_progress=False):
        """Fit the consensus clustering from features

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training instances/objects to cluster

        y : Ignored
            Not used, present here for consistency with the sklearn API.

        disable_progress : bool, default=False
            Whether to show the progress bar or not, when fitting multiple iterations
            of K-Means on random subsets of the data. Set `True` to disable it.

        Returns
        -------
        self

        """
        self.num_samples_, _ = X.shape
        self.n_jobs = effective_n_jobs(self.n_jobs)
        self.consensus_matrix_ = self._fit(X, disable_progress)
        self.labels_ = self._fit_distance_matrix(self.consensus_matrix_)

        return self

    def fit_predict(self, X, y=None, disable_progress=False):
        """Fit the consensus clustering from features, and return cluster labels

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training instances/objects to cluster

        y : Ignored
            Not used, present here for consistency with the sklearn API.

        disable_progress : bool, default=False
            Whether to show the progress bar or not, when fitting multiple iterations
            of K-Means on random subsets of the data. Set `True` to disable it.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels for each sample/object.

        """
        return self.fit(X, y, disable_progress).labels_

    def _fit(self, X, disable_progress=False):
        """Internally used to compute the consensus matrix

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training instances/objects to cluster

        disable_progress : bool, default=False
            Whether to show the progress bar or not, when fitting multiple iterations
            of K-Means on random subsets of the data. Set `True` to disable it.

        Returns
        -------
        consensus_matrix : np.ndarray of shape (n_samples, n_samples)
            Each element of the matrix can be interpreted as a probability that two
            objects belong to the same cluster.

        """
        resampled_indices, resampling_co_occurence_matrix = self._resample()
        co_occurence_matrix_list = None
        iterator = tqdm(
            resampled_indices,
            desc=f"ConsensusClustering(n_clusters={self.n_clusters})",
            disable=disable_progress,
        )  # Wrapped the list of indices in tqdm for the progress bar

        if self.n_jobs == 1:
            # With no multi-processing (hence n_jobs=1), using a list comprehension
            # is slightly faster than using joblib's Parallel
            co_occurence_matrix_list = [
                self._fit_subset(X, index) for index in iterator
            ]
        else:
            co_occurence_matrix_list = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self._fit_subset)(X, index=index) for index in iterator
            )

        clustering_co_occurence_matrix = np.sum(co_occurence_matrix_list, axis=0)

        return self._compute_consensus_matrix(
            clustering_co_occurence_matrix, resampling_co_occurence_matrix
        )

    def _fit_distance_matrix(self, distance_matrix):
        """Internally used to fit a hierarchical clustering model using the elements
        of the consensus matrix as the distance measure between objects.

        Parameters
        ----------
        distance_matrix : np.ndarray of shape (n_samples, n_samples)
            Matrix representing the distance between the objects (intended to be the
            consensus matrix).

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            The final cluster labels assigned to each object.

        """
        return AgglomerativeClustering(
            n_clusters=self.n_clusters, affinity="euclidean", linkage="average"
        ).fit_predict(distance_matrix)

    def _fit_subset(self, X, index):
        """Internally used to fit a K-Means model on a random subset of the data and
        return the corresponding co-occurence matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training instances/objects to cluster

        index : array-like, of length int(n_samples * sample_size)
            Represents the list of indices to select the subset of samples from `X`.
            Since sampling is done with replacement, the indices are not unique.

        Returns
        -------
        co_occurence_matrix : np.ndarray of shape (n_samples, n_samples)
            If two objects, i and j are predicted to be in the same cluster, then
            co_occurence_matrix[i, j] = 1, otherwise it's 0.

            Note: The matrix is composed of 1s and 0s only because an object cannot
            be assigned to multiple clusters.

        """
        subset = X[index, :]
        assigned_clusters = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state,
        ).fit_predict(subset)

        clustering_matrix = np.zeros((self.n_clusters, self.num_samples_))
        clustering_matrix[assigned_clusters, index] = 1

        return np.dot(clustering_matrix.T, clustering_matrix)

    def _resample(self):
        """Generates the indices that determines which samples are to be considered
        to create the random subset in each iteration of K-Means.

        Returns
        -------
        idx_matrix : np.ndarray of shape (iterations, int(n_samples * sample_size))
            Each row i of the matrix is the list of indices to select the subset of
            samples from `X` in iteration i. Since sampling is done with replacement,
            the indices are not unique.

        resampling_co_occurence_matrix : np.ndarray of shape (n_samples, n_samples)
            Each element [i, j] is the number of times the pair of objects i and j
            were selected together in the same (random) subset, out of the total
            iterations performed.

            Note: Although the diagonal terms are not important, they essentially
            represent how many times each individual object i was selected when
            generating the subsets.

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
        """Computes the consensus matrix

        Parameters
        ----------
        clustering_co_occurence_matrix : np.ndarray of shape (n_samples, n_samples)
            Each element [i, j] is the number of times the pair of objects i and j
            were clustered together in the same cluster out of the total iterations
            performed.

        resampling_co_occurence_matrix : np.ndarray of shape (n_samples, n_samples)
            Each element [i, j] is the number of times the pair of objects i and j
            were selected together in the same (random) subset, out of the total
            iterations performed.

        Returns
        -------
        consensus_matrix : np.ndarray of shape (n_samples, n_samples)
            Each element of the matrix can be interpreted as a probability that two
            objects belong to the same cluster.

        """
        consensus_matrix = clustering_co_occurence_matrix / (
            resampling_co_occurence_matrix + self.epsilon
        )
        np.fill_diagonal(consensus_matrix, 1)

        return consensus_matrix
