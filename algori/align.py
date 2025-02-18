from typing import List, Tuple, Optional, Set
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import numpy as np
import pandas as pd


def normalize(data, log_transform=False, feature_range=(0, 1)):
    """
    Scale input data to ensure balanced and uniform transformation.

    Args:
        data: np.ndarray or pd.DataFrame
            Input data to be scaled.
        log_transform: bool, optional
            If True, apply logarithmic transformation to handle skewed data.
        feature_range: tuple, optional
            Desired range for MinMaxScaler.

    Returns:
        pd.DataFrame: Scaled data.
    """
    # Step 1: Ensure input is a DataFrame
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # Step 2: Replace NaN values with 0
    data = data.fillna(0)

    # Step 3: Adaptive clipping to retain broader range
    lower_bound = np.percentile(data, 2)  # Use 1st percentile for lower bound
    upper_bound = np.percentile(data, 99)  # Use 99th percentile for upper bound
    data_clipped = np.clip(data, lower_bound, upper_bound)

    # Step 4: Optional logarithmic transformation
    if log_transform:
        data_clipped = np.log1p(data_clipped - data_clipped.min().min() + 1e-6)

    # Step 5: Scale data to [0, 1] using MinMaxScaler
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_data = scaler.fit_transform(data_clipped)

    return scaled_data


def stretch(original_matrix, source_name, crop_ratio=0):
    if isinstance(original_matrix, pd.DataFrame):
        original_matrix = original_matrix.to_numpy()

    original_data = original_matrix[np.triu_indices(original_matrix.shape[0], k=1)]
    stretch_data, min_val, max_val, crop_ratio = stretch_to_zscore(original_data)
    print(f" original_matrix.shape: {original_data.shape}, stretch_data.shape: {stretch_data.shape}")
    print(f" original_matrix.shape[0]: {original_data.shape[0]}, stretch_data.shape[0]: {stretch_data.shape[0]}")
    reshaped_original_matrix = reshape_matrix(stretch_data, original_matrix.shape[0])

    return reshaped_original_matrix, min_val, max_val, crop_ratio


def distance_stretch(original_matrix, source_name):
    """
    y = (log(f(x) + 1) - 1) * 10
    f(x) = original_matrix + 1，
    """
    if isinstance(original_matrix, pd.DataFrame):
        original_matrix = original_matrix.to_numpy()

    adjusted_matrix = original_matrix +(np.e)
    stretched_matrix = (np.log(adjusted_matrix)-1) * 10

    # adjusted_matrix = original_matrix +1
    # stretched_matrix = (np.log(adjusted_matrix)) * 10

    print(f"[{source_name}] Distance Stretch Applied: Min={np.min(stretched_matrix)}, Max={np.max(stretched_matrix)}, Mean={np.mean(stretched_matrix)}")

    return stretched_matrix

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
    non_zero_data = data - data.min() + 1e-6

    if verbose:
        print_stats("Before Cropping", non_zero_data)

    # Step 2: Calculate IQR bounds
    Q1, Q3 = np.percentile(non_zero_data, [10, 90])
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    # Step 3: Clip data to IQR bounds (instead of removing outliers, clip them to bounds)
    num_cropped = np.sum((non_zero_data < lower_bound) | (non_zero_data > upper_bound))
    crop_ratio_actual = num_cropped / len(non_zero_data)
    cropped_data = np.clip(non_zero_data, lower_bound, upper_bound)  # Clip to bounds
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

def plot_distribution(data, title="Data Distribution"):
    """
    Plot the histogram and density plot of the data.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.6, color="blue", label="Histogram")
    plt.axvline(data.mean(), color="red", linestyle="--", label=f"Mean: {data.mean():.2f}")
    plt.axvline(data.std(), color="green", linestyle="--", label=f"Std: {data.std():.2f}")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

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


def align_list(
    data: List[Tuple[np.ndarray, List[str]]],
    repo_num: Optional[int] = None,
    deterministic: bool = True,  # 开关：是否固定随机性
    seed: int = 42  # 可选：固定的随机种子值
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
    # print(f"common_repos: {len(common_repos)}, repos: {len(repos)}")
    # print(f"aligned_repos: {len(aligned_repos)}")

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
    non_zero_data = data[data > 0]
    return non_zero_data.min()
    return 0


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