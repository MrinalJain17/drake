import numpy as np


class Entity(object):
    def __init__(self, name, correlation, mean, variance):
        """TODO

        """
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
    def dragon(self):
        return self._dragon


def create_snake(arr: np.ndarray) -> np.ndarray:
    """TODO

    """
    # Sanity check for the array
    assert is_2d_square_numpy_array(arr)

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
    """TODO

    """
    return np.concatenate((mean, variance, snake))


def is_2d_square_numpy_array(arr: np.ndarray) -> bool:
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape[0] == arr.shape[1]
    return True
