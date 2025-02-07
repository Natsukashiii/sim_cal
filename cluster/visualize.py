import os.path

import numpy as np
from apprise.plugins import sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from cluster.distance import  cal_distance
from utils.path import RESULT_DIR



def visualize_one_matrix(matrix1, file_name,crop_ratio=None, min_val=0, max_val=1):
    """
    Visualize a single matrix as a histogram of its upper triangular values.

    Args:
        matrix1 (np.ndarray): Input square matrix to visualize.
        crop_ratio (float): Optional crop ratio, included in the title if provided.
        file_name (str): Output file name for the saved plot.
        min_val (float): Minimum value for x-axis scaling.
        max_val (float): Maximum value for x-axis scaling.
    """
    # Ensure inputs are square matrices
    assert matrix1.shape[0] == matrix1.shape[1], "matrix1 must be a square matrix."

    # Extract upper triangle data (excluding diagonal)
    original_data = matrix1[np.triu_indices(matrix1.shape[0], k=1)]

    # Create plot title
    if crop_ratio:
        original_title = f"Original({file_name}) - stretched {min_val:.4f} to {max_val:.4f} - (Total: {len(original_data)} - Crop ratio: {crop_ratio:.4f})"
    else:
        original_title = f"Original({file_name}) - stretched {min_val:.4f} to {max_val:.4f} - (Total: {len(original_data)})"

    # Generate the plot
    plt.figure(figsize=(8, 6))  # Adjust figure size for single plot
    # Plot original data distribution
    plt.hist(
        original_data,
        bins=80,
        range=(min_val, max_val),
        color="skyblue",
        edgecolor="black",
        alpha=0.7
    )
    plt.title(original_title, fontsize=12)
    plt.xlabel("Original Values", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, file_name))


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def visualize_comparison_multiple(
    original_matrices, normalized_matrices, labels, file_name,
    xlim_original=None, xlim_normalized=None, bins=50, use_log=False
):
    """
    Visualize the comparison between multiple original matrices and their corresponding normalized matrices
    using histograms (with dual y-axes for density).

    Parameters:
        original_matrices (list of numpy.ndarray): List of original matrices to compare.
        normalized_matrices (list of numpy.ndarray): List of normalized matrices to compare.
        labels (list of str): Labels for each dataset (e.g., institution names).
        file_name (str): File name to save the plot.
        xlim_original (tuple): Limits for the x-axis of the original data distribution plot (e.g., (0, 1)).
        xlim_normalized (tuple): Limits for the x-axis of the normalized data distribution plot (e.g., (0, 1)).
        bins (int): Number of bins for the histogram.
        use_log (bool): Whether to apply a log transformation to the data before visualization.
    """
    print(f"type(labels): {type(labels)}")
    assert len(original_matrices) == len(normalized_matrices) == len(labels), \
        "The number of original matrices, normalized matrices, and labels must be the same."

    # Prepare data for original distributions
    original_data = []
    for i, matrix in enumerate(original_matrices):
        if isinstance(matrix, tuple):
            matrix = matrix[0]
        upper_triangle = matrix[np.triu_indices(matrix.shape[0], k=1)]
        if use_log:
            upper_triangle = np.log1p(upper_triangle)  # Apply log transformation if specified
        original_data.extend([(x, labels[i]) for x in upper_triangle])

    # Prepare data for normalized distributions
    normalized_data = []
    for i, matrix in enumerate(normalized_matrices):
        if isinstance(matrix, tuple):
            matrix = matrix[0]
        upper_triangle = matrix[np.triu_indices(matrix.shape[0], k=1)]
        normalized_data.extend([(x, labels[i]) for x in upper_triangle])

    # Convert data to DataFrame for seaborn
    original_df = pd.DataFrame(original_data, columns=["Value", "Institution"])
    normalized_df = pd.DataFrame(normalized_data, columns=["Value", "Institution"])

    # Initialize figure
    plt.figure(figsize=(14, 6))

    # Original data distribution with dual y-axis
    ax1 = plt.subplot(1, 2, 1)
    for label in labels:
        subset = original_df[original_df["Institution"] == label]["Value"]
        sns.histplot(
            subset,
            bins=bins,
            kde=False,
            stat="count",  # Use count for histogram
            label=label,
            element="bars",
            alpha=0.4,
            edgecolor="white",
            ax=ax1
        )

    # Add secondary y-axis for KDE
    ax1_2 = ax1.twinx()
    for label in labels:
        subset = original_df[original_df["Institution"] == label]["Value"]
        sns.kdeplot(
            subset,
            ax=ax1_2,
            label=f"{label} (Density)",
            linestyle="--"
        )
    ax1.set_title("Original Similarity Distribution")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency (Count)")
    ax1_2.set_ylabel("Density")
    ax1.legend(loc="upper left")
    if xlim_original:
        ax1.set_xlim(xlim_original)

    # Normalized data distribution with dual y-axis
    ax2 = plt.subplot(1, 2, 2)
    for label in labels:
        subset = normalized_df[normalized_df["Institution"] == label]["Value"]
        sns.histplot(
            subset,
            bins=bins,
            kde=False,
            stat="count",  # Use count for histogram
            label=label,
            element="bars",
            alpha=0.4,
            edgecolor="white",
            ax=ax2
        )

    # Add secondary y-axis for KDE
    ax2_2 = ax2.twinx()
    for label in labels:
        subset = normalized_df[normalized_df["Institution"] == label]["Value"]
        sns.kdeplot(
            subset,
            ax=ax2_2,
            label=f"{label} (Density)",
            linestyle="--"
        )
    ax2.set_title("Normalized Similarity Distribution")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency (Count)")
    ax2_2.set_ylabel("Density")
    ax2.legend(loc="upper left")
    if xlim_normalized:
        ax2.set_xlim(xlim_normalized)
    plt.xlim(0, 1)
    # Save and show the plot
    plt.tight_layout()



    plt.savefig(os.path.join(RESULT_DIR, file_name), format='pdf')


def visualize_combined_matrices(
    distance_matrices, labels, file_name, bins=50
):
    """
    Visualize the combined distribution of multiple distance matrices with dual y-axes.

    Parameters:
        distance_matrices (list of numpy.ndarray): List of distance matrices to combine and visualize.
        labels (list of str): Labels for each distance matrix (e.g., source names).
        file_name (str): File name to save the visualization.
        bins (int): Number of bins to use for the histogram.
    """
    assert len(distance_matrices) == len(labels), \
        "The number of matrices and labels must match."

    # Prepare data for visualization
    combined_data = []
    for i, matrix in enumerate(distance_matrices):
        # Extract the upper triangle
        upper_triangle = matrix[np.triu_indices(matrix.shape[0], k=1)]

        # Validate data range
        min_val, max_val = upper_triangle.min(), upper_triangle.max()
        print(f"Matrix {labels[i]} - Min Value: {min_val:.6f}, Max Value: {max_val:.6f}")

        # Optionally normalize data to [0, 1] if needed
        if min_val < 0 or max_val > 1:
            print(f"Matrix {labels[i]} contains values outside [0, 1]. Normalizing...")
            upper_triangle = (upper_triangle - min_val) / (max_val - min_val + 1e-6)

        # Collect data for visualization
        combined_data.extend([(x, labels[i]) for x in upper_triangle])

    # Convert data to DataFrame
    combined_df = pd.DataFrame(combined_data, columns=["Value", "Attribute"])

    # Initialize figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot histograms
    for label in labels:
        subset = combined_df[combined_df["Attribute"] == label]["Value"]
        sns.histplot(
            subset,
            bins=bins,
            kde=False,
            stat="count",  # Use count for the left y-axis
            label=label,
            element="bars",
            alpha=0.4,
            edgecolor="white",
            ax=ax1
        )

    # Set labels and title for the histogram
    ax1.set_xlabel("Distance Value")
    ax1.set_ylabel("Frequency (Count)", color="black")
    ax1.set_title("Distance Distribution")
    ax1.legend(title="Attribute", loc="upper left")

    # Create secondary y-axis for KDE
    ax2 = ax1.twinx()

    # Plot KDE for each source
    for label in labels:
        subset = combined_df[combined_df["Attribute"] == label]["Value"]
        sns.kdeplot(
            subset,
            ax=ax2,
            label=f"{label} (Density)",
            linestyle="--"
        )

    # Set labels for KDE axis
    ax2.set_ylabel("Density", color="black")

    # Align y-axis scales
    ax1.set_ylim(0, ax1.get_ylim()[1])  # Match left y-axis scale
    ax2.set_ylim(0, ax2.get_ylim()[1])  # Match right y-axis scale

    # Save the plot
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, file_name), format='pdf')

def visualize_two_compare_matrix(matrix1, matrix2,  crop_ratio, file_name,min_val=0, max_val=1,):
    # Ensure inputs are square matrices
    assert matrix1.shape[0] == matrix1.shape[1], "matrix1 must be a square matrix."
    assert matrix2.shape[0] == matrix2.shape[1], "matrix2 must be a square matrix."

    # Extract upper triangle data (excluding diagonal)
    original_data = matrix1[np.triu_indices(matrix1.shape[0], k=1)]
    transformed_data = matrix2[np.triu_indices(matrix2.shape[0], k=1)]

    if crop_ratio:
        original_title=f"Original({file_name}) - stretched {min_val:.4f} to {max_val} - (Total: {len(original_data)} - Crop ratio: {crop_ratio:.4f} )"
    else:
        original_title = f"Original({file_name}) - stretched {min_val:.4f} to {max_val} - (Total: {len(original_data)} )"

    # generate the composed plots
    plt.figure(figsize=(12, 6))

    # Plot original data distribution
    plt.subplot(1, 2, 1)
    plot_data_distribution(
        original_data,
        bins=80,
        min_value_x=min_val,
        max_value_x=max_val,
        title=original_title,
        xlabel="Original Values"
    )

    # Plot transformed data distribution
    plt.subplot(1, 2, 2)
    plot_data_distribution(
        transformed_data,
        bins=80,
        min_value_x=min_val,
        max_value_x=max_val,
        title=f"Transformed Sim ({file_name})",
        xlabel="Transformed Values"
    )

    plt.subplots_adjust(wspace=0.6)  # Adjust spacing between plots
    plt.savefig(os.path.join(RESULT_DIR, file_name))  # Save the plot

def plot_data_distribution(data, bins, min_value_x, max_value_x, title, xlabel, ylabel="Frequency", color="skyblue", kde_color="orange", secondary_y_label="Density"):
    """
    Plot data distribution with histogram and KDE curve.

    Parameters:
        data (array-like): The data to plot.
        bins (int): Number of bins for the histogram.
        min_value_x (float): Minimum value of the x-axis.
        max_value_x (float): Maximum value of the x-axis.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    # Clean data: Remove NaNs and clip to range
    data = np.asarray(data)
    data = data[~np.isnan(data)]  # Remove NaNs
    data = np.clip(data, min_value_x, max_value_x)  # Clip to specified range

    # Compute histogram
    hist_values, bin_edges = np.histogram(data, bins=bins, range=(min_value_x, max_value_x))

    # Plot histogram
    plt.bar(
        (bin_edges[:-1] + bin_edges[1:]) / 2,
        hist_values,
        width=(bin_edges[1] - bin_edges[0]),
        color=color,
        alpha=0.7,
        label="Histogram"
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel, color="black")
    set_multiline_title(plt.gca(), title, max_line_length=40)

    plt.xlim(min_value_x, max_value_x)
    plt.legend(loc="upper left")

    # Plot KDE
    if kde_color is not None:
        ax2 = plt.gca().twinx()  # Create a single secondary axis
        sns.kdeplot(
            data,
            ax=ax2,
            color=kde_color,
            label="KDE (Density)",
            linewidth=2,
            bw_adjust=0.8  # Adjust bandwidth as needed
        )
        ax2.set_ylabel(secondary_y_label, color=kde_color)
        ax2.tick_params(axis='y', labelcolor=kde_color)
        ax2.legend(loc="upper right")
        ax2.set_xlim(min_value_x, max_value_x)  # Ensure consistent x-limits

    plt.tight_layout()

def set_multiline_title(ax, title, max_line_length=60):
    """
    Set a multiline title for a plot.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object to set the title on.
        title (str): The title text.
        max_line_length (int): The maximum length of each line.
    """
    if len(title) > max_line_length:
        # Split the title into multiple lines
        lines = []
        while len(title) > max_line_length:
            split_index = title[:max_line_length].rfind(' ')
            if split_index == -1:
                split_index = max_line_length
            lines.append(title[:split_index])
            title = title[split_index:].strip()
        lines.append(title)
        title = '\n'.join(lines)

    ax.set_title(title)

def visualize_cluster_results(distance_matrix, cluster_labels, repos, file_path):
    """
    Visualize clustering results with centroids displayed.

    Args:
        distance_matrix (np.ndarray): Precomputed distance.py matrix.
        cluster_labels (List[int]): Cluster labels for each repository.
        repos (List[str]): List of repository names corresponding to the matrix.
        file_path (str): Path to save the plot.
    """
    # Step 1: Compute composite distances using `cal_distance`
    num_repos = len(repos)
    composite_distances = np.zeros((num_repos, num_repos))

    for i in range(num_repos):
        for j in range(num_repos):
            composite_distances[i, j] = cal_distance(distance_matrix, i, j)

    # Step 2: Ensure the composite distances are symmetric
    composite_distances = (composite_distances + composite_distances.T) / 2

    # Step 3: Convert distance.py matrix to a feature matrix using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    feature_matrix = mds.fit_transform(composite_distances)

    # Step 4: Perform PCA on the feature matrix (optional, for further dimensionality reduction)
    pca = PCA(n_components=2)
    distance_matrix_2d = pca.fit_transform(feature_matrix)

    # Step 5: Create a color map for clusters
    unique_labels = set(cluster_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    # Step 6: Plot each cluster
    plt.figure(figsize=(10, 8))
    centroids = {}

    for cluster_id, color in zip(unique_labels, colors):
        if cluster_id == -1:
            # Noise points or unprocessed points
            cluster_points = [distance_matrix_2d[i] for i in range(len(cluster_labels)) if
                              cluster_labels[i] == cluster_id]
            plt.scatter(
                [p[0] for p in cluster_points],
                [p[1] for p in cluster_points],
                c=[color],
                label="Noise",
                alpha=0.5,
                marker="x",
                s=50
            )
        else:
            # Cluster points
            cluster_points = [distance_matrix_2d[i] for i in range(len(cluster_labels)) if
                              cluster_labels[i] == cluster_id]
            plt.scatter(
                [p[0] for p in cluster_points],
                [p[1] for p in cluster_points],
                c=[color],
                label=f"Cluster {cluster_id}",
                alpha=0.6,
                s=60
            )

            # Compute the centroid of the cluster
            indices = [i for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_id]
            cluster_repos = [repo for i, repo in enumerate(repos) if i in indices]
            submatrix = composite_distances[np.ix_(indices, indices)]
            centroid_index = submatrix.mean(axis=1).argmin()
            centroid_repo = cluster_repos[centroid_index]
            centroid_coords = distance_matrix_2d[indices][centroid_index]

            # Mark the centroid
            plt.scatter(
                centroid_coords[0],
                centroid_coords[1],
                c="black",
                marker="o",
                edgecolor="white",
                s=150,
                label=f"Centroid (Cluster {cluster_id})"
            )

            # Save centroid info
            centroids[cluster_id] = centroid_repo

    # Step 7: Add legend and title
    plt.legend(loc="best")
    plt.title("Clustering Results with Centroids")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(alpha=0.3)

    # Step 8: Save the plot
    plt.subplots_adjust(top=0.95, bottom=0.15)
    plt.savefig(file_path)



def visualize_silhouette_results(results, best_result, file_path):
    """
    Generate and save plots for silhouette scores and cluster counts.

    Args:
        results (list): List of dictionaries containing T1, silhouette, and cluster count.
    """
    best_t1 = best_result['best_t1']
    best_silhouette = best_result['best_silhouette']

    t1_values = [res['t1'] for res in results]
    silhouettes = [res['silhouette'] for res in results]
    cluster_counts = [res['n_clusters'] for res in results]

    fig, ax1 = plt.subplots(figsize=(8, 6))  # Adjust figure size

    # Plot silhouette scores with improved style
    ax1.set_xlabel('T1', fontsize=12)
    ax1.set_ylabel('Silhouette Score', color='tab:blue', fontsize=12)
    ax1.plot(
        t1_values,
        silhouettes,
        color='tab:blue',
        linewidth=2,
        linestyle='-',
        marker='o',
        markersize=6,
        label='Silhouette Score'
    )
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axvline(
        x=best_t1,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'Best T1 = {best_t1}'
    )

    # Create a second y-axis for cluster counts
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Clusters', color='tab:orange', fontsize=12)
    ax2.plot(
        t1_values,
        cluster_counts,
        color='tab:orange',
        linewidth=2,
        linestyle='--',
        marker='s',
        markersize=6,
        label='Cluster Count'
    )
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Add a legend with improved placement and style
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper right',  # Place legend in the upper right corner
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1
    )

    # Tight layout and improved title
    fig.tight_layout()
    plt.title(
        f'Silhouette Score = {best_silhouette} and Cluster Count vs. T1 = {best_t1}',
        fontsize=14,
        pad=15
    )
    plt.grid(alpha=0.5)  # Add a subtle grid for better readability

    # Save the figure
    plt.savefig(file_path, dpi=300, bbox_inches='tight')

def compute_t1_range_source(distance_matrix, num_steps=10):
    """
    Dynamically compute the T1 range based on the distance.py matrix.

    Args:
        distance_matrix (np.ndarray): The distance.py matrix.
        num_steps (int): Number of steps for T1 values in the range.

    Returns:
        np.ndarray: The range of T1 values.
    """
    flattened_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

    min_distance = max(0.01, np.min(flattened_distances))
    max_distance = np.mean(flattened_distances) + 3 * np.std(flattened_distances)  # Mean + 3*std deviation

    t1_range = np.linspace(min_distance, max_distance, num_steps)
    return t1_range

