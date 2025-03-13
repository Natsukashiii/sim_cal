import numpy as np
from matplotlib import pyplot as plt


def cal_distance(distance_matrix, i, j):
    """
    Calculate the distance between two points (i, j) in a given distance matrix.

    Args:
        distance_matrix (np.ndarray): The distance matrix.
        i (int): Index of the first point (Repo i).
        j (int): Index of the second point (Repo j).

    Returns:
        float: The calculated distance.
    """
    # Step 1: Check the format of the distance matrix
    distance_matrix = np.array(distance_matrix)


    if distance_matrix.ndim == 2 and not isinstance(distance_matrix[0, 0], (tuple, list)):
        # Case 1: One-dimensional distance (scalar values in the matrix)
        distance_matrix = np.round(np.array(distance_matrix), decimals=8)

        return distance_matrix[i, j]

    elif distance_matrix.ndim == 2 and isinstance(distance_matrix[0, 0], (tuple, list)):
        # Case 2: Two-dimensional distance (tuples as elements in the matrix)
        Da_ij, Db_ij = distance_matrix[i, j]  # Extract (Da_ij, Db_ij)
        weights = [1,1]
        w_a, w_b = weights
        distance = np.sqrt(w_a * Da_ij ** 2 + w_b * Db_ij ** 2)

        return distance

    else:
        # Handle unsupported formats
        raise ValueError("Unsupported distance matrix format. Matrix must be 2D or 3D with shape (N, N, 2).")


def cal_distance_multi_dimension(distance_matrix, i, j, weights=None):
    """
    Efficiently calculate multi-dimensional distance between two points (i, j).

    Args:
        distance_matrix (np.ndarray): The distance matrix.
        i (int): Index of the first point.
        j (int): Index of the second point.
        weights (list or np.ndarray, optional): Weights for each dimension.

    Returns:
        float: Computed distance.
    """
    distance_matrix = np.asarray(distance_matrix)  # Convert to NumPy array if not already
    weights = np.ones(distance_matrix.shape[-1]) if weights is None else np.asarray(weights)
    values = np.array(distance_matrix[i, j])

    if distance_matrix.ndim == 2 and not isinstance(distance_matrix[0, 0], (tuple, list)):
        # Case 1: One-dimensional distance (scalar values in the matrix)
        return distance_matrix[i, j]

    elif len(values) == 2 and isinstance(distance_matrix[0, 0], (tuple, list)):
        Da_ij, Db_ij = distance_matrix[i, j]  # Extract (Da_ij, Db_ij)
        weights = [1, 1]
        w_a, w_b = weights
        distance = np.sqrt(w_a * Da_ij ** 2 + w_b * Db_ij ** 2)

        return distance
    elif len(values) == 3 and isinstance(distance_matrix[0, 0], (tuple, list)):
        num_dims = len(values)

        Da_ij, Db_ij,Dc_ij = distance_matrix[i, j]  # Extract (Da_ij, Db_ij)
        weights = [1, 1,1]
        w_a, w_b,w_c = weights
        distance = np.sqrt(w_a * Da_ij ** 2 + w_b * Db_ij ** 2+w_c * Dc_ij ** 2)

        return distance


    else:
        raise ValueError("Unsupported distance matrix format. ")


def analysis_tuple_matrix_multi_dimension(combine_distance_matrix):
    """
    Analyze a combined distance matrix (supporting 1D, 2D, 3D, and N-dimensional data).

    Args:
        combine_distance_matrix (np.ndarray): A matrix where each entry is an N-dimensional tuple or scalar.

    Returns:
        tuple: (min_distance, mean_distance, max_distance)
    """
    n = combine_distance_matrix.shape[0]
    distances = []

    for i in range(n):
        for j in range(i + 1, n):  # Exclude diagonal
            values = combine_distance_matrix[i, j]  # Extract value (could be tuple or scalar)

            # Handle single-dimensional case
            if isinstance(values, (int, float, np.float32, np.float64)):
                euclidean_distance = abs(values)  # 1D case
            else:
                euclidean_distance = np.sqrt(sum(x ** 2 for x in values))  # Multi-dimensional case

            distances.append(euclidean_distance)

    # Convert list to NumPy array for statistical operations
    distances = np.array(distances)

    # Compute min, mean, and max
    combine_min = np.min(distances)
    combine_mean = np.mean(distances)
    combine_max = np.max(distances)

    return combine_min, combine_mean, combine_max

