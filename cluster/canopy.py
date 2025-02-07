import random
from collections import Counter

import numpy as np
from cluster.distance import cal_distance
import os
from utils.path import RESULT_DIR
from cluster.visualize import visualize_cluster_results, visualize_silhouette_results
from sklearn.cluster import KMeans

IS_DYNAMTIC = True
USE_KMEANS = False
USE_KMEANS_REFINEMENT = True


def canopy_clustering(distance_matrix, repos, file_name, isCombine=False):

    # Ensure result directory exists
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # Dynamically compute T1 range
    t1_range = compute_t1_range(distance_matrix, isCombine)

    print(
        f">>>>>>> {file_name} distance matrix min: {np.min(distance_matrix)},mean:{np.mean(distance_matrix)}  max: {np.max(distance_matrix)}")
    print(f"t1_range = {t1_range}")


    best_silhouette = -1
    best_t1 = None
    best_labels = None
    best_cluster_nums = None
    silhouette_plot_info = []
    plot_info = []
    canopy_centers = []
    for t1 in t1_range:
        t2 = t1
        processed = set()  # Tracks points that are already processed
        cluster_labels = [-1] * len(repos)  # Initialize all points as unprocessed
        cluster_id = 0  # Cluster ID starts from 0

        # Randomly shuffle the indices to ensure random selection of cluster seeds  try to fix the seed
        random.seed(42)
        indices = list(range(len(repos)))
        random.shuffle(indices)

        # Iterate over the shuffled indices
        for i in indices:
            if i in processed:
                continue
            # Create a new cluster
            cluster_id += 1
            cluster_labels[i] = cluster_id
            canopy_centers.append(i)

            # Iterate over all points to find those within the distance.py threshold
            for j in range(len(repos)):
                distance = cal_distance(distance_matrix, i, j)
                if j not in processed:
                    if distance <= t1:  # Add to the cluster if within T1
                        cluster_labels[j] = cluster_id
                    if distance <= t2:  # Mark as processed if within T2
                        processed.add(j)

        # Ensure valid clusters
        unique_labels = set(cluster_labels)
        if len(unique_labels) < 2 or -1 in unique_labels:
            continue

        # Compute silhouette score
        try:
            silhouette = manual_silhouette_score(distance_matrix, cluster_labels)
        except ValueError:
            silhouette = -1

        plot_info.append({'t1': t1, 'silhouette': silhouette, 'n_clusters': len(unique_labels)})

        # Update the best result
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_t1 = t1
            best_labels = cluster_labels
            best_cluster_nums = len(unique_labels)

     # single_point_ratio
    label_counts = Counter(label for label in best_labels if label != -1)
    single_point_clusters = sum(1 for count in label_counts.values() if count == 1)
    single_point_ratio = single_point_clusters / best_cluster_nums

    best_result = {
        'best_t1': best_t1,
        'best_silhouette': best_silhouette,
        'best_labels': best_labels,
        'best_cluster_nums': best_cluster_nums,
        'single_point_ratio': single_point_ratio,
    }

    print(f"{file_name} Best Cluster Result: T1: {best_result['best_t1']}, Silhouette Coefficient: {best_result['best_silhouette']:4f}, Number of Clusters: {best_result['best_cluster_nums'],}, Single Point Ratio: {best_result['single_point_ratio']:4f}")

    visualize_silhouette_results(plot_info, best_result, os.path.join(RESULT_DIR, f"rq2_silhouette_plot_{file_name}.pdf"))
    visualize_cluster_results(distance_matrix, best_labels, repos,
                              os.path.join(RESULT_DIR, f"cluster_plot_{file_name}.png"))


    save_cluster_info(best_result, os.path.join(RESULT_DIR, f"cluster_info_{file_name}.txt"))
    return best_result



def save_cluster_info(results, file_path):
    best_t1 = results.get('best_t1', None)
    best_silhouette = results.get('best_silhouette', None)
    best_labels = results.get('best_labels', [])
    best_cluster_nums = results.get('best_cluster_nums', 0)
    single_point_ratio = results.get('single_point_ratio', 0)

    with open(file_path, "w") as f:
        f.write(f"Best T1: {best_t1}\n")
        f.write(f"Best Silhouette Score: {best_silhouette}\n")
        f.write(f"Best Cluster Numbers: {best_cluster_nums}\n")
        f.write(f"Best single_point_ratio:{single_point_ratio}\n")
        label_counts = Counter(best_labels)
        sorted_label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        f.write("Cluster Label Counts (sorted by count):\n")
        for label, count in sorted_label_counts:
            f.write(f"Label {label}: {count}\n")


def compute_t1_range(distance_matrix, isCombine=False,  num_steps=20 ,isDynamtic=IS_DYNAMTIC):
    """
    Dynamically compute the T1 range based on the distance.py matrix using `cal_distance`.

    Args:
        distance_matrix (np.ndarray): The distance.py matrix. Each element can be:
            - A scalar (one-dimensional distances).
            - A tuple (x, y) representing two-dimensional distances.
        num_steps (int): Number of steps for T1 values in the range.

    Returns:
        np.ndarray: The range of T1 values.
    """
    t1_range = []
    if isDynamtic:
        # Step 1: Flatten the upper triangle of the matrix excluding diagonal
        n = distance_matrix.shape[0]
        flattened_distances = []

        for i in range(n):
            for j in range(i + 1, n):  # Only consider upper triangle
                distance = cal_distance(distance_matrix, i, j)
                flattened_distances.append(distance)
        # Convert list to numpy array
        flattened_distances = np.array(flattened_distances)
        t1_range = np.linspace(np.percentile(flattened_distances, 10), np.percentile(flattened_distances, 80),
                               num_steps)
    else:
        if isCombine:
            t1_range = [0.1, 0.2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 15, 16, 20, 30, 40, 50, 120]
        else:
            t1_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10, 11, 15, 16]

    return t1_range



def manual_silhouette_score(distance_matrix, labels):
    # Ensure the input is valid
    assert distance_matrix.shape[0] == distance_matrix.shape[1], "Distance matrix must be square."
    assert len(labels) == distance_matrix.shape[0], "Labels length must match number of points."

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        raise ValueError("Silhouette Score requires at least 2 clusters.")

    silhouette_scores = []

    # Iterate over each point
    for i in range(len(labels)):
        current_label = labels[i]

        # Calculate a(i): intra-cluster distance
        same_cluster = [j for j in range(len(labels)) if labels[j] == current_label and i != j]
        if same_cluster:
            a_i = np.mean([cal_distance(distance_matrix, i, j) for j in same_cluster])
        else:
            a_i = 0  # If no other points in the same cluster, set a(i) to 0

        # Calculate b(i): nearest cluster distance
        b_i = float('inf')
        for other_label in unique_labels:
            if other_label == current_label:
                continue
            other_cluster = [j for j in range(len(labels)) if labels[j] == other_label]
            if other_cluster:
                mean_distance = np.mean([cal_distance(distance_matrix, i, j) for j in other_cluster])
                b_i = min(b_i, mean_distance)

        # Silhouette score for point i
        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        silhouette_scores.append(s_i)

    # Average silhouette score for all points
    return np.mean(silhouette_scores)



