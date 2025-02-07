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
        # 控制精度
        distance_matrix = np.round(np.array(distance_matrix), decimals=8)

        return distance_matrix[i, j]

    elif distance_matrix.ndim == 2 and isinstance(distance_matrix[0, 0], (tuple, list)):
        # Case 2: Two-dimensional distance (tuples as elements in the matrix)
        Da_ij, Db_ij = distance_matrix[i, j]  # Extract (Da_ij, Db_ij)
        weights = [1,1]
        w_a, w_b = weights

        # if w_a==1:
        #     distance = Da_ij
        # elif w_b==1:
        #     distance = Db_ij
        # else:
        # distance = w_a * abs(Da_ij) + w_b * abs(Db_ij)
        distance = np.sqrt(w_a * Da_ij ** 2 + w_b * Db_ij ** 2)

        # distance = np.sqrt(w_a * Da_ij ** 2 + w_b * Db_ij ** 2)

        #---
        # magnitude_a = np.sqrt(Da_ij ** 2 + Db_ij ** 2)
        # cos_theta = Da_ij / magnitude_a if magnitude_a != 0 else 0
        # distance = magnitude_a * (1 - cos_theta)

        return distance

    else:
        # Handle unsupported formats
        raise ValueError("Unsupported distance matrix format. Matrix must be 2D or 3D with shape (N, N, 2).")


def analysis_tuple_matrix(combine_distance_matrix):
    # Extract upper triangle indices (excluding diagonal)
    print(f"Type =  {type(combine_distance_matrix)} ")
    n = combine_distance_matrix.shape[0]
    distances = []

    for i in range(n):
        for j in range(i + 1, n):  # Exclude diagonal
            a, b = combine_distance_matrix[i, j]  # Extract tuple (a, b)
            euclidean_distance = np.sqrt(a**2 + b**2)  # Compute Euclidean distance
            distances.append(euclidean_distance)

    # Convert list to NumPy array for statistical operations
    distances = np.array(distances)

    # Compute min, mean, and max
    combine_min = np.min(distances)
    combine_mean = np.mean(distances)
    combine_max = np.max(distances)
    return combine_min,combine_mean,combine_max


def print_value_ratios(distance_matrix, intervals=10):
    flattened_values = distance_matrix.flatten()

    min_value = np.min(flattened_values)
    max_value = np.max(flattened_values)
    interval_bounds = np.linspace(min_value, max_value, intervals + 1)

    counts = np.zeros(intervals, dtype=int)
    for i in range(intervals):
        lower_bound = interval_bounds[i]
        upper_bound = interval_bounds[i + 1]
        if i == intervals - 1:
            counts[i] = np.sum((flattened_values >= lower_bound) & (flattened_values <= upper_bound))
        else:
            counts[i] = np.sum((flattened_values >= lower_bound) & (flattened_values < upper_bound))

    total_count = len(flattened_values)
    ratios = counts / total_count

    print(f"Total values: {total_count}")
    for i in range(intervals):
        lower_bound = interval_bounds[i]
        upper_bound = interval_bounds[i + 1]
        print(f"Interval {i + 1} [{lower_bound:.4f}, {upper_bound:.4f}]: Count = {counts[i]}, Ratio = {ratios[i]:.4%}")

