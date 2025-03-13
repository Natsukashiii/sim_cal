import numpy as np


def combine_multi_dimension(*matrices, common_repos):
    """
    Combine multiple aligned distance matrices into a single matrix of paired values.

    Args:
        *matrices (np.ndarray): A variable number of aligned distance matrices.
        common_repos (List[str]): The repository list associated with all matrices.

    Returns:
        np.ndarray: Combined matrix with multiple values (tuples).
    """
    shapes = [matrix.shape for matrix in matrices]
    if len(set(shapes)) != 1:
        raise ValueError("All distance matrices must have the same dimensions.")

    size = len(common_repos)
    combined_matrix = np.empty((size, size), dtype=object)  # dtype=object for tuples

    for i in range(size):
        for j in range(size):
            combined_matrix[i][j] = tuple(matrix[i][j] for matrix in matrices)

    return combined_matrix