import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from utils.path import PLOTS_DIR
from scipy.stats import spearmanr, permutation_test


def cal_spearman(matrix1, matrix2,name, IS_SPARSE=False, num_permutations=3000):
    """
    Compute Spearman correlation between two matrices and perform a permutation test.
    Saves the null distribution plot.

    Parameters:
    - matrix1: First similarity matrix (numpy.ndarray or pandas.DataFrame)
    - matrix2: Second similarity matrix (numpy.ndarray or pandas.DataFrame)
    - IS_SPARSE: Boolean flag to indicate whether to filter zero values (default: False)
    - num_permutations: Number of permutations for permutation test (default: 200)

    Returns:
    - rho: Spearman correlation coefficient
    - p_value: Permutation test p-value
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError("The shape is different.")

    # Convert pandas DataFrame to numpy array if necessary
    if isinstance(matrix1, pd.DataFrame):
        matrix1 = matrix1.values
    if isinstance(matrix2, pd.DataFrame):
        matrix2 = matrix2.values

    # Extract upper triangular values (excluding diagonal)
    if IS_SPARSE:
        upper_m1, upper_m2 = upper_aligned(matrix1, matrix2, filter_zeros=IS_SPARSE)
    else:
        upper_m1, upper_m2 = matrix1[np.triu_indices_from(matrix1, k=1)], matrix2[np.triu_indices_from(matrix2, k=1)]

    # Flatten the upper triangle matrices
    upper_m1 = upper_m1.flatten()
    upper_m2 = upper_m2.flatten()

    # Compute Spearman correlation
    rho, origin_p_value = spearmanr(upper_m1, upper_m2)
    print(f"Spearman (rho): {rho}, origin_p_value: {origin_p_value}")

    # Perform permutation test using SciPy
    def statistic(x):  # Only shuffle `x`
        return spearmanr(x, upper_m2).statistic  # Ignore p-value, only return rho

    res_permutation = permutation_test(
        (upper_m1,), statistic,
        permutation_type='pairings',
        n_resamples=num_permutations
    )

    p_value = res_permutation.pvalue  # Get p-value from permutation test

    # Plot and save permutation test null distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(res_permutation.null_distribution, bins=30, density=True, alpha=0.6, color='orange',
            label=f'exact\n({num_permutations} permutations)')
    ax.axvline(x=rho, color='r', linestyle='--', label='Observed Statistic')

    ax.set_title("Spearman's Rho Test Null Distribution")
    ax.set_xlabel("Statistic")
    ax.set_ylabel("Probability Density")
    ax.legend()

    output_path = PLOTS_DIR / f"{name}_permutation_test.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Spearman (rho): {rho}, p_value: {p_value}")
    print(f"Permutation test null distribution plot saved at {output_path}")

    return rho, p_value


def cal_gromov(matrix1, matrix2, IS_SPARSE=False, block_size=200):
    print("cal gromv...")
    if isinstance(matrix1, pd.DataFrame):
        matrix1 = matrix1.values
    if isinstance(matrix2, pd.DataFrame):
        matrix2 = matrix2.values

    if matrix1.shape != matrix2.shape:
        raise ValueError("The shape is different")

    if IS_SPARSE:
        upper_tri1, upper_tri2 = upper_aligned(matrix1, matrix2, filter_zeros=True)
    else:
        upper_tri1, upper_tri2 = matrix1, matrix2

    def block_generator(upper_tri1, upper_tri2):
        num_rows1 = upper_tri1.shape[0]
        num_rows2 = upper_tri2.shape[0]
        for i in range(0, num_rows1, block_size):
            row_block1 = upper_tri1[i:i + block_size]
            for j in range(0, num_rows2, block_size):
                row_block2 = upper_tri2[j:j + block_size]
                yield row_block1, row_block2

    max_min_d1 = max_min_d2 = 0

    for row_block1, row_block2 in block_generator(upper_tri1, upper_tri2):
        dists_block = cdist(row_block1, row_block2, metric='euclidean')

        max_min_d1 = max(max_min_d1, np.max(np.min(dists_block, axis=1)))
        max_min_d2 = max(max_min_d2, np.max(np.min(dists_block, axis=0)))

        del dists_block

    gh_distance = max(max_min_d1, max_min_d2)

    print("Gromov-Hausdorff:", gh_distance)
    return gh_distance

def cal_frobenius(matrix1, matrix2, IS_SPARSE=False, normalization_method="log_scale"):
    if isinstance(matrix1, pd.DataFrame):
        matrix1 = matrix1.values
    if isinstance(matrix2, pd.DataFrame):
        matrix2 = matrix2.values

    if matrix1.shape != matrix2.shape:
        raise ValueError("The shape is different.")
    if IS_SPARSE:
        upper_tri1, upper_tri2 = upper_aligned(matrix1, matrix2)
        frobenius_norm = np.linalg.norm(upper_tri1 - upper_tri2)
    else:
        frobenius_norm = np.linalg.norm(matrix1 - matrix2, 'fro')

    print("Frobenius distance:", frobenius_norm)
    return frobenius_norm


def upper_aligned(matrix1, matrix2, filter_zeros=False):
    """
    Extracts the upper triangular parts of two matrices and aligns them by trimming to the minimum length.
    Optionally filters out zero values.

    Parameters:
    - matrix1: First input matrix.
    - matrix2: Second input matrix.
    - filter_zeros: If True, filters out zero values from the upper triangular parts.

    Returns:dan
    - Two aligned 2D arrays containing upper triangular values, with or without zeros, as specified.
    """
    # Extract the upper triangular parts without the diagonal (k=1) for both matrices

    upper_tri1 = matrix1[np.triu_indices_from(matrix1, k=1)]
    upper_tri2 = matrix2[np.triu_indices_from(matrix2, k=1)]

    if filter_zeros:
        # Filter out zero values from both matrices
        non_zero_indices1 = upper_tri1 != 0
        non_zero_indices2 = upper_tri2 != 0

        # Only keep indices where both matrices have non-zero values
        non_zero_indices = non_zero_indices1 & non_zero_indices2
        upper_tri1 = upper_tri1[non_zero_indices]
        upper_tri2 = upper_tri2[non_zero_indices]

    # Determine the minimum length to align both matrices
    min_len = min(len(upper_tri1), len(upper_tri2))

    # Return the trimmed, aligned upper triangular parts as 2D arrays with consistent shape
    return upper_tri1[:min_len].reshape(-1, 1), upper_tri2[:min_len].reshape(-1, 1)