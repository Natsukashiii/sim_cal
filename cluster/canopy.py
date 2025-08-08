import os
import random
import time
import traceback
from collections import Counter

import numpy as np
from scipy.stats import skew

from cluster.distance import (analysis_tuple_matrix_multi_dimension,
                              cal_distance, cal_distance_multi_dimension)
from cluster.visualize import (visualize_cluster_results,
                               visualize_silhouette_results)
from utils.config import load_num_steps
from utils.path import RESULT_DIR

IS_DYNAMTIC = True
USE_KMEANS = False

def canopy_clustering_multi_dimension(distance_matrix, repos, file_name):
    """
    Perform Canopy Clustering (Multi-Dimensional) using a point-by-point iteration method.

    Args:
        distance_matrix (np.ndarray): The distance matrix.
        repos (List[str]): List of repository names.
        file_name (str): File name for saving results.

    Returns:
        dict: Best clustering results.
    """
    start_time = time.time()

    # Ensure result directory exists
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # Compute T1 range dynamically
    t1_range = compute_t1_range(distance_matrix)

    print(
        f">>>>>>> {file_name} distance matrix min: {np.min(distance_matrix)}, mean: {np.mean(distance_matrix)}, max: {np.max(distance_matrix)}")
    print(f"t1_range = {t1_range}")

    best_silhouette = -1
    best_t1 = None
    best_labels = None
    best_cluster_nums = None
    plot_info = []

    # Step 2: Initialize Clustering Variables
    n = len(repos)
    processed = set()
    cluster_labels = [-1] * n

    for t1 in t1_range:
        t2 = t1

        cluster_id = 0
        processed.clear()

        random.seed(42)
        indices = list(range(n))
        random.shuffle(indices)

        for i in indices:
            if i in processed:
                continue

            cluster_id += 1
            cluster_labels[i] = cluster_id

            for j in range(n):
                if j in processed:
                    continue

                distance = cal_distance_multi_dimension(distance_matrix, i, j)

                if distance <= t1:
                    cluster_labels[j] = cluster_id

                if distance <= t2:
                    processed.add(j)

        unique_labels = set(cluster_labels)
        if len(unique_labels) < 2 or -1 in unique_labels:
            continue

        try:
            silhouette = manual_silhouette_score(distance_matrix, cluster_labels)
        except ValueError:
            silhouette = -1

        plot_info.append({'t1': t1, 'silhouette': silhouette, 'n_clusters': len(unique_labels)})

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_t1 = t1
            best_labels = cluster_labels.copy()
            best_cluster_nums = len(unique_labels)

        # print(f"processed {t1} done, silhouette: {silhouette}, n_clusters: {len(unique_labels)}")

    label_counts = Counter(label for label in best_labels if label != -1)
    single_point_clusters = sum(1 for count in label_counts.values() if count == 1)
    single_point_ratio = single_point_clusters / best_cluster_nums

    best_result = {
        'best_t1': best_t1 if best_t1 is not None else 0,
        'best_silhouette': best_silhouette if best_silhouette is not None else -1,
        'best_labels': best_labels if best_labels is not None else [-1] * len(repos),
        'best_cluster_nums': best_cluster_nums if best_cluster_nums is not None else 1,
        'single_point_ratio': single_point_ratio if single_point_ratio is not None else 0,
    }

    print(
        f"{file_name} Best Cluster Result: T1: {best_result['best_t1']}, Silhouette Coefficient: {best_result['best_silhouette']:.4f}, Number of Clusters: {best_result['best_cluster_nums']}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"!!!!!! Clustering for {file_name} completed in {elapsed_time:.2f} seconds.")

    # save_cluster_info(best_result, os.path.join(RESULT_DIR, f"cluster_info_{file_name}.txt"))

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


def compute_t1_range(distance_matrix, num_steps=load_num_steps()):
    """
    Compute the T1 range dynamically based on the distance matrix.
    Args:
        distance_matrix (np.ndarray): Distance matrix.
        num_steps (int): Number of steps for T1 values.

    Returns:
        np.ndarray: Adaptive T1 range.
    """
    if distance_matrix is None:
        raise ValueError("Error: distance_matrix is None.")

    n = distance_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)


    # Step 1: Compute Euclidean distances
    if distance_matrix.ndim == 2 and not isinstance(distance_matrix[0, 0], (tuple, list)):
        distances = np.array(distance_matrix[upper_tri_indices], dtype=np.float64)
        skewness = skew(distances)
        print(f"ske: {skewness}")
    else:
        converted_matrix = np.array([[np.array(x, dtype=np.float64) for x in row] for row in distance_matrix])
        distances = np.sqrt(np.sum(np.square(converted_matrix[upper_tri_indices]), axis=-1))

    mean_val = np.mean(distances)
    std_val = np.std(distances)
    t1_min = max(mean_val -3 * std_val, np.min(distances))
    t1_max = min(mean_val + 3 * std_val, np.max(distances))



    t1_range = np.linspace(t1_min, t1_max, num_steps)
    print(f"Computed T1 Range: Min={t1_min:.4f}, Max={t1_max:.4f}, Steps={num_steps}")


    return t1_range


def manual_silhouette_score(distance_matrix, labels):
    try:
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
                a_i = np.mean([cal_distance_multi_dimension(distance_matrix, i, j) for j in same_cluster])
            else:
                a_i = 0  # If no other points in the same cluster, set a(i) to 0

            # Calculate b(i): nearest cluster distance
            b_i = float('inf')
            for other_label in unique_labels:
                if other_label == current_label:
                    continue
                other_cluster = [j for j in range(len(labels)) if labels[j] == other_label]
                if other_cluster:
                    # mean -> median
                    mean_distance = np.median([cal_distance_multi_dimension(distance_matrix, i, j) for j in other_cluster])  # 改用median
                    b_i = min(b_i, mean_distance)

            # Silhouette score for point i
            s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
            silhouette_scores.append(s_i)

        # Average silhouette score for all points
        return np.mean(silhouette_scores)

    except Exception as e:
        print(f"Error in manual_silhouette_score: {e}")
        print(f"Labels: {labels}")
        print(f"Distance matrix shape: {distance_matrix.shape}")
        traceback.print_exc()

        return -1
