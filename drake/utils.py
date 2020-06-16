import numpy as np


class Entity(object):
    """A utility to create (and store) the snake and dragon vectors

    This class is intended to store the relevant information corresponding to a
    single entity/object.

    Parameters
    ----------
    name : str
        The name of the entity/object

    correlation : np.ndarray of shape (n_features, n_features)
        The correlation matrix of the features

    mean : np.ndarray of shape (n_features,)
        The mean value of each feature

    variance : np.ndarray of shape (n_features,)
        The variance of each feature

    Attributes
    ----------
    name : str
        The name of the entity/object

    correlation : np.ndarray of shape (n_features, n_features)
        The correlation matrix of the features

    dragon : np.ndarray
        Concatenation of mean, variance and snake vector, as described in the paper.

    """

    def __init__(self, name, correlation, mean, variance):
        self._name = name
        self._mean = mean
        self._variance = variance
        self._correlation = correlation
        self._snake = create_snake(self._correlation)
        self._dragon = create_dragon(self._mean, self._variance, self._snake)

    @property
    def name(self):
        return self._name

    @property
    def correlation(self):
        return self._correlation

    @property
    def dragon(self):
        return self._dragon


def create_snake(arr: np.ndarray) -> np.ndarray:
    """Creates the sanke vector as described in the paper from a 2-D square matrix

    Parameters
    ----------
    arr : np.ndarray of shape (n_features, n_features)
        A 2-D square matrix (according to the paper, it should be the correlation
        matrix of an entity/object).

    Returns
    -------
    snake_vector : np.ndarray
        1-D snake vector created from the 2-D matrix.

    """
    # Sanity check for the array
    is_2d_square_numpy_array(arr)

    dim = arr.shape[0]
    diagonals = []
    for diag_idx in range(1, dim):
        temp = np.diagonal(arr, offset=diag_idx).copy()
        if (diag_idx % 2) == 0:
            # Reversing every alternate diagonal to generate the "snake"
            temp = temp[::-1]
        diagonals.append(temp)

    return np.concatenate(diagonals)


def create_dragon(mean, variance, snake):
    """Concatenates the arrays required to create the dragon vector"""
    return np.concatenate((mean, variance, snake))


def is_2d_square_numpy_array(arr: np.ndarray) -> bool:
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape[0] == arr.shape[1]
