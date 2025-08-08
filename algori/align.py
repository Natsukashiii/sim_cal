import random
from typing import List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   StandardScaler)


def stretch(original_matrix, source_name):
    if isinstance(original_matrix, pd.DataFrame):
        original_matrix = original_matrix.to_numpy()

    original_data = original_matrix[np.triu_indices(original_matrix.shape[0], k=1)]
    stretch_data, min_val, max_val, crop_ratio = stretch_to_zscore(original_data)
    print(f" original_matrix.shape: {original_data.shape}, stretch_data.shape: {stretch_data.shape}")
    print(f" original_matrix.shape[0]: {original_data.shape[0]}, stretch_data.shape[0]: {stretch_data.shape[0]}")
    reshaped_original_matrix = reshape_matrix(stretch_data, original_matrix.shape[0])

    return reshaped_original_matrix, min_val, max_val, crop_ratio


def reshape_matrix(original_data, matrix_size):
    """
    Restore the expanded upper triangular data to a symmetric matrix form.

    Parameters:
        original_data (numpy.ndarray): Flattened one-dimensional original data.
        transformed_data (numpy.ndarray): Flattened one-dimensional transformed data.
        matrix_size (int): The original dimension of the matrix (e.g., 562).

    Returns:
        reshaped_original_matrix (numpy.ndarray): Reconstructed original matrix.
        reshaped_transformed_matrix (numpy.ndarray): Reconstructed transformed matrix.
    """
    # init
    reshaped_original_matrix = np.zeros((matrix_size, matrix_size))

    # upper side
    indices = np.triu_indices(matrix_size, k=1)
    reshaped_original_matrix[indices] = original_data
    reshaped_original_matrix += reshaped_original_matrix.T

    return reshaped_original_matrix


def stretch_to_zscore(data, use_log=False, iqr_multiplier=1.5, non_linear_mapping=None, verbose=False):
    """
    Normalize data using Z-Score normalization and force the result into [0, 1] using Min-Max normalization.
    Outliers are not removed; instead, they are clipped to the nearest boundary.

    Parameters:
        data (numpy.ndarray): Input data (1D array).
        use_log (bool): Whether to apply log transformation.
        iqr_multiplier (float): Multiplier for IQR range. Default is 1.5.
        non_linear_mapping (callable): Custom non-linear mapping function to transform data.
        verbose (bool): Whether to print intermediate statistics and distributions. Default is False.

    Returns:
        normalized_data (numpy.ndarray): Data after Z-Score and Min-Max normalization.
        original_mean (float): Mean value after cropping or transformation.
        original_std (float): Standard deviation after cropping or transformation.
        crop_ratio_actual (float): Actual crop ratio applied.
    """
    # Step 1: Filter out non-positive values for analysis
    # non_zero_data = data[data > 0]
    # non_zero_data = data - data.min() + 1e-6

    if verbose:
        print_stats("Before Cropping", data)

    # Step 2: Calculate IQR bounds
    Q1, Q3 = np.percentile(data, [10, 90])
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    # Step 3: Clip data to IQR bounds (instead of removing outliers, clip them to bounds)
    num_cropped = np.sum((data < lower_bound) | (data > upper_bound))
    crop_ratio_actual = num_cropped / len(data)
    cropped_data = np.clip(data, lower_bound, upper_bound)  # Clip to bounds
    # cropped_data = non_zero_data  # Clip to bounds

    if verbose:
        print_stats("After Clipping (to IQR Bounds)", cropped_data, crop_ratio_actual)

    # Step 4: Apply optional log transformation
    if use_log:
        cropped_data = np.log(cropped_data + 1e-6)  # Avoid log(0)
        if verbose:
            print_stats("After Log Transformation", cropped_data)

    # Step 5: Apply custom non-linear mapping if provided
    if non_linear_mapping is not None:
        cropped_data = non_linear_mapping(cropped_data)

    # Step 6: Z-Score normalization
    original_mean = np.mean(cropped_data)
    original_std = np.std(cropped_data)
    zscore_data = (cropped_data - original_mean) / (original_std + 1e-6)  # Avoid division by zero

    if verbose:
        print_stats("After Z-Score Normalization", zscore_data)

    # Step 7: Min-Max normalization (to force Z-Score results into [0, 1])
    min_val = zscore_data.min()
    max_val = zscore_data.max()
    normalized_data = (zscore_data - min_val) / (max_val - min_val + 1e-6)  # Normalize to [0, 1]

    if verbose:
        print_stats("After Min-Max Normalization", normalized_data)

    # Plot normalized data distribution
    # plot_distribution(normalized_data, title="Z-Score + Min-Max Normalization (0-1 Range)")

    return normalized_data, min_val, max_val, crop_ratio_actual


def print_stats(step, data, crop_ratio=None):
    """
    Helper function to print statistical summary of the data.

    Parameters:
        step (str): Step name to identify the output stage.
        data (numpy.ndarray): Input data to analyze.
        crop_ratio (float, optional): Ratio of cropped data if applicable.
    """
    print(f"\n>>> {step}: Data Statistics <<<")
    print(
        f"Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}, Std: {data.std():.4f}, Median: {np.median(data):.4f}")
    print(f"Data Distribution (Histogram): {np.histogram(data, bins=10)}")
    if crop_ratio is not None:
        print(f"Crop Ratio: {crop_ratio:.4f}")






def align_list_eva(
    data: List[Tuple[np.ndarray, List[str], str]],
    repo_num: Optional[int] = None,
    deterministic: bool = True,
    seed: int = 42
) -> Tuple[List[Tuple[np.ndarray, List[str], str]], List[str]]:
    # Step 1: Find common repos across all sources (for default behavior)
    all_sets = [set(repos) for _, repos, source in data if source != 'repopal']
    if all_sets:
        common_repos = set.intersection(*all_sets)
    else:
        common_repos = set()

    # Step 2: Extend common repos with all repos from repopal source
    repopal_repos = set()
    for _, repos, source in data:
        if source == 'repopal':
            repopal_repos.update(repos)
    full_repo_list = sorted(common_repos.union(repopal_repos))

    # Step 3: Subset sampling
    if repo_num is not None and repo_num < len(full_repo_list):
        if deterministic:
            random.seed(seed)
        full_repo_list = sorted(random.sample(full_repo_list, repo_num))

    # Step 4: Print mismatch report
    print("🔍 Repo stats per source:")
    for _, repos, source in data:
        missing = sorted(set(full_repo_list) - set(repos))
        print(f"  [{source}] total: {len(repos)} | missing in alignment: {len(missing)}")

    # Step 5: Build aligned matrix
    repo_index_map = {repo: i for i, repo in enumerate(full_repo_list)}
    n = len(full_repo_list)
    aligned_data = []

    for sim_matrix, repos, source in data:
        print(f"\n🔧 Aligning source: {source}")
        if source == 'repopal':
            # Expand to full_repo_list, fill 0 if missing
            full_matrix = np.zeros((n, n))
            old_index = {repo: i for i, repo in enumerate(repos)}
            for i_old, repo_i in enumerate(repos):
                for j_old, repo_j in enumerate(repos):
                    i_new = repo_index_map[repo_i]
                    j_new = repo_index_map[repo_j]
                    full_matrix[i_new, j_new] = sim_matrix[i_old, j_old]
            aligned_data.append((full_matrix, full_repo_list, source))
        else:
            # Trim to only common subset
            kept_indices = [i for i, r in enumerate(repos) if r in full_repo_list]
            name_to_index = {name: i for i, name in enumerate(repos)}
            m = len(full_repo_list)
            trimmed_matrix = np.zeros((m, m))
            for i, repo_i in enumerate(full_repo_list):
                for j, repo_j in enumerate(full_repo_list):
                    if repo_i in name_to_index and repo_j in name_to_index:
                        trimmed_matrix[i, j] = sim_matrix[name_to_index[repo_i], name_to_index[repo_j]]
            aligned_data.append((trimmed_matrix, full_repo_list, source))

    print(f"\n✅ Final aligned repo count: {len(full_repo_list)}")
    return aligned_data, full_repo_list





def align_list(
    data: List[Tuple[np.ndarray, List[str]]],
    repo_num: Optional[int] = None,
    deterministic: bool = True,
    seed: int = 42
) -> List[Tuple[np.ndarray, List[str]]]:
    """
    Aligns a list of similarity matrices and their associated repositories based on common repositories.
    If `repo_num` is provided, selects a subset of common repositories.

    Args:
        data: A list of tuples where each tuple contains a similarity matrix and a list of repositories.
        repo_num: The number of repositories to align to (optional).
        deterministic: If True, ensures consistent random subset selection by fixing the random seed.
        seed: The seed value for random subset selection when `deterministic` is True.

    Returns:
        aligned_data: A list of tuples where each tuple contains an aligned similarity matrix and a list of repositories.
        common_repos: The list of common repositories after alignment.
    """
    # Step 1: Find common repositories
    common_repos = find_common_repos(data)

    # Step 2: Randomly select a subset of repositories if repo_num is specified
    if repo_num is not None and repo_num < len(common_repos):
        common_repos = random_subset(common_repos, repo_num, seed=seed)

    # Step 3: Align each similarity matrix and repository list
    aligned_data = []
    for similarity_matrix, repos,source in data:
        aligned_matrix, aligned_repos = align_single(similarity_matrix, repos, common_repos)

    
        aligned_data.append((aligned_matrix, aligned_repos,source))

    print(f"Common Repositories: {len(common_repos)}" )
    return aligned_data, common_repos

def find_common_repos(data: List[Tuple[np.ndarray, List[str]]]) -> Set[str]:
    common_repos = set(data[0][1])
    for _, repos,_ in data:
        common_repos.intersection_update(repos)  # 求交集

    common_repos = sorted(common_repos)
    return common_repos


def align_single(similarity_matrix: np.ndarray, repos: List[str], common_repos: Set[str]) -> Tuple[
    np.ndarray, List[str]]:

    aligned_repos = [repo for repo in repos if repo in common_repos]
    repo_to_index = {repo: idx for idx, repo in enumerate(repos)}

    size = len(aligned_repos)
    aligned_matrix = np.zeros((size, size))

    for i, repo_i in enumerate(aligned_repos):
        for j, repo_j in enumerate(aligned_repos):
            index_i = repo_to_index[repo_i]
            index_j = repo_to_index[repo_j]
            # Check if indices are within bounds
            if index_i < similarity_matrix.shape[0] and index_j < similarity_matrix.shape[1]:
                aligned_matrix[i, j] = similarity_matrix[index_i, index_j]
            else:
                pass
                # print(f"Skipping out-of-bounds indices: {index_i}, {index_j}")

    return aligned_matrix, aligned_repos


def get_min_max(data):
    """
    Returns the minimum, maximum, and mean values from the given data,
    excluding zero values.

    Args:
        data (numpy.ndarray): The input numerical data (e.g., similarity or distance matrix).

    Returns:
        tuple: (min_value, max_value, mean_value)
    """
    non_zero_data = data[data > 0]

    if non_zero_data.size == 0:
        return None, None, None

    return non_zero_data.min(), non_zero_data.max(), non_zero_data.mean()


def random_subset(repos: Set[str], repo_num: int, seed: int = None) -> Set[str]:
    """
    Selects a random subset of repositories with an optional fixed seed.

    Args:
        repos (Set[str]): The set of repositories to select from.
        repo_num (int): The number of repositories to select.
        seed (int, optional): A random seed for reproducibility. If None, randomness is not fixed.

    Returns:
        Set[str]: A random subset of repositories.
    """
    repo_list = list(repos)

    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(repo_list)
    return set(repo_list[:repo_num])