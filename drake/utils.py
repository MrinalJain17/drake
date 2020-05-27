import numpy as np
import pandas as pd


def create_snake(arr: np.ndarray) -> np.ndarray:
    """TODO

    """
    # Sanity check if a pandas dataframe is passed instead
    if isinstance(arr, pd.DataFrame):
        arr = arr.to_numpy()

    # Takes as input only 2-d square arrays
    assert arr.ndim == 2
    assert arr.shape[0] == arr.shape[1]

    dim = arr.shape[0]
    diagonals = []
    for diag_idx in range(1, dim):
        temp = np.diagonal(arr, offset=diag_idx).copy()
        if (diag_idx % 2) == 0:
            # Reversing every alternate diagonal to generate the "snake"
            temp = temp[::-1]
        diagonals.append(temp)

    return np.concatenate(diagonals)
