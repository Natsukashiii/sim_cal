import numpy as np

WEIGHT1 = 1
WEIGHT2 = 1


def combine_multiple_matrices(matrix_list, common_repos, weights=None):
    """
    Combine multiple aligned distance matrices into a single matrix of paired values.

    Args:
        matrix_list (List[np.ndarray]): A list of aligned distance matrices to combine.
        common_repos (List[str]): The repository list associated with the matrices.
        weights (List[float], optional): A list of weights for each matrix. If None, equal weights are used.

    Returns:
        np.ndarray: Combined matrix with tuples of weighted values for each pair of repositories.
    """
    num_matrices = len(matrix_list)
    size = len(common_repos)

    # Ensure all matrices have the same dimensions
    for matrix in matrix_list:
        if matrix.shape != (size, size):
            raise ValueError("All distance matrices must have the same dimensions as common_repos.")

    # Use equal weights if none are provided
    if weights is None:
        weights = [1 / num_matrices] * num_matrices

    if len(weights) != num_matrices:
        raise ValueError("Number of weights must match the number of matrices.")

    combined_matrix = np.empty((size, size), dtype=object)  # Using dtype=object for tuples

    # Combine the values into tuples (weighted values)
    for i in range(size):
        for j in range(size):
            combined_matrix[i][j] = tuple(
                weights[k] * matrix_list[k][i][j] for k in range(num_matrices)
            )

    return combined_matrix





def combine(matrix1, matrix2, common_repos):
    """
    Combine two aligned distance matrices into a single matrix of paired values.

    Args:
        matrix1 (np.ndarray): The first aligned distance matrix.
        matrix2 (np.ndarray): The second aligned distance matrix.
        common_repos (List[str]): The repository list associated with both matrices.

    Returns:
        np.ndarray: Combined matrix with paired values (tuples).
    """
    # Ensure matrices have the same dimensions
    if matrix1.shape != matrix2.shape:
        raise ValueError("Distance matrices must have the same dimensions.")

    size = len(common_repos)
    combined_matrix = np.empty((size, size), dtype=object)  # Using dtype=object for tuples

    print("Matrix1 shape:", matrix1.shape)
    print("Matrix2 shape:", matrix2.shape)

    # Combine the values into pairs (weighted values as a tuple)
    for i in range(size):
        for j in range(size):
            weighted_value1 = WEIGHT1 * matrix1[i][j]
            weighted_value2 = WEIGHT2 * matrix2[i][j]
            combined_matrix[i][j] = (weighted_value1, weighted_value2)  # Store as a tuple

    return combined_matrix