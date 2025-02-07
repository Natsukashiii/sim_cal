import os.path
from collections import defaultdict
import numpy as np
from utils.path import RESULT_DIR
from cluster.distance import cal_distance

COMPUTE_BASED_ON_ORIGIN = True

def cal_overlap(distance_matrix1,source1_cluster_result, combine_cluster_result, repos):
    """
    Calculate the overlap between source cluster results and combined cluster results.

    Args:
        source_cluster_result (dict): Cluster results for the source attribute.
        combine_cluster_result (dict): Cluster results for the combined attribute.
        repos (List[str]): List of repository names corresponding to the cluster labels.

    Returns:
        float: The overall overlap between source and combined attributes.
    """
    # Extract labels from clustering results
    repos = sorted(repos)
    labels_A = source1_cluster_result["best_labels"]
    labels_C = combine_cluster_result["best_labels"]

    # Step 1: Group repositories by labels
    cluster_A = defaultdict(set)  # Map of LabelA -> repos
    cluster_C = defaultdict(set)  # Map of LabelC -> repos
    for i, repo in enumerate(repos):
        cluster_A[labels_A[i]].add(repo)
        cluster_C[labels_C[i]].add(repo)

    # Step 2: Find centroids of each LabelC cluster

    # Compute distance.py matrix for repos
    distance_matrix = distance_matrix1


    # Find the centroid repo for each LabelC cluster
    labelA_to_centroid = {}
    for label_A, repos_in_A in cluster_A.items():
        if repos_in_A:
            labelA_to_centroid[label_A] = find_centroid(repos_in_A, distance_matrix, repos)

    labelC_to_centroid = {}
    for label_C, repos_in_C in cluster_C.items():
        if repos_in_C:
            labelC_to_centroid[label_C] = find_centroid(repos_in_C, distance_matrix, repos)

################################################################################################
    # Step 3: Map LabelC clusters to LabelA clusters using centroids
    # labelC_to_labelA = {}
    # for label_C, centroid_repo in labelC_to_centroid.items():
    #     for label_A, repos_in_A in cluster_A.items():
    #         if centroid_repo in repos_in_A:
    #             labelC_to_labelA[label_C] = label_A
    #             break
    ################################################################################################
    labelC_to_labelA = {}
    for label_C, centroid_C_repo in labelC_to_centroid.items():
        min_distance = float('inf')
        closest_label_A = None

        for label_A, centroid_A_repo in labelA_to_centroid.items():
            # Get indices of the centroids in the repos list
            centroid_C_index = repos.index(centroid_C_repo)
            centroid_A_index = repos.index(centroid_A_repo)

            # Calculate the distance between the two centroids
            distance = cal_distance(distance_matrix, centroid_C_index, centroid_A_index)
            if distance < min_distance:
                min_distance = distance
                closest_label_A = label_A

        if closest_label_A is not None:
            labelC_to_labelA[label_C] = closest_label_A
    ################################################################################################
    # labelC_to_labelA = {}
    # for label_C, centroid_repo in labelC_to_centroid.items():
    #     min_distance = float('inf')
    #     closest_label_A = None
    #     repos_list = list(repos)
    #
    #     for label_A, repos_in_A in cluster_A.items():
    #         for repo in sorted(repos_in_A):
    #             centroid_index = repos_list.index(centroid_repo)
    #             repo_index = repos_list.index(repo)
    #             distance = cal_distance(distance_matrix, centroid_index, repo_index)
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 closest_label_A = label_A
    #     if closest_label_A is not None:
    #         labelC_to_labelA[label_C] = closest_label_A
    ################################################################################################

    # Step 4: Calculate overlap for each LabelA cluster
    all_overlap_details = []
    labelA_overlap_scores = []
    filtered_overlap_scores = []
    labelA_to_labelC_mapping = {}
    for label_A in cluster_A.keys():
        # Get all LabelC clusters mapped to LabelA
        labelC_list = [k for k, v in labelC_to_labelA.items() if v == label_A]
        labelA_to_labelC_mapping[label_A] = labelC_list

        if not labelC_list:
            continue  # Skip if no LabelC clusters are mapped to this LabelA

        labelC_scores = []
        for label_C in labelC_list:
            repos_in_A = cluster_A[label_A]
            repos_in_C = cluster_C[label_C]
            overlap_count = len(repos_in_A & repos_in_C)  # Intersection of repos in A and C
            labelC_score = overlap_count / len(repos_in_C)  # Overlap score for this LabelC
            labelC_scores.append(labelC_score)

            if not (overlap_count == 1 and len(repos_in_A) == 1 and len(repos_in_C) == 1):
                filtered_overlap_scores.append(labelC_score)

            # Save intermediate overlap details
            all_overlap_details.append(
                f"LabelA: {label_A}, LabelC: {label_C}, Overlap Count: {overlap_count}, "
                f"Repos in A: {len(repos_in_A)}, Repos in C: {len(repos_in_C)}, Overlap Score: {labelC_score:.4f}"
            )

        # Average overlap scores of all LabelC clusters mapped to this LabelA
        if labelC_scores:
            labelA_overlap = sum(labelC_scores) / len(labelC_scores)
            labelA_overlap_scores.append(labelA_overlap)



    # Step 5: Calculate the final overlap
    if labelA_overlap_scores:
        overall_overlap = sum(labelA_overlap_scores) / len(labelA_overlap_scores)
    else:
        overall_overlap = 0.0

    if filtered_overlap_scores:
        overall_filtered_overlap = sum(filtered_overlap_scores) / len(filtered_overlap_scores)
    else:
        overall_filtered_overlap = 0.0

    with open(os.path.join(RESULT_DIR, "overlap_score.csv"), "w") as f:
        f.write("Detailed Overlap Scores:\n")
        f.write(
            f"Overall Overlap: {overall_overlap:.4f}\n"
            f"Filtered Overall Overlap: {overall_filtered_overlap:.4f}\n"
        )
        f.write("\n".join(all_overlap_details))
        f.write("\n\n")
        f.write("LabelA to LabelC Mapping:\n")
        for label_A, labelC_list in labelA_to_labelC_mapping.items():
            f.write(f"LabelA: {label_A}, LabelC List: {labelC_list}\n")
        f.write("\n")


    return overall_overlap,overall_filtered_overlap



def find_centroid(cluster, distance_matrix, repos):
    """
    Find the centroid of a cluster using the distance.py matrix.

    Args:
        cluster (set): Repositories in the cluster.
        distance_matrix (np.ndarray): Distance matrix of all repos.
        repos (List[str] or Set[str]): List or set of repository names corresponding to the distance.py matrix.

    Returns:
        str: Repository that is closest to the centroid.
    """
    # Ensure repos is a list
    if isinstance(repos, set):
        repos = list(repos)
        repos = sorted(repos)

    # ensure the cluster is sorted so evertime the centroid is the same
    cluster = sorted(cluster)
    # Ensure cluster is a list to use indices
    cluster = list(cluster)

    # Get the indices of the repos in the cluster
    indices = [repos.index(repo) for repo in cluster if repo in repos]

    # If indices are empty, return None (no valid repos found in cluster)
    if not indices:
        return None

    # Extract the submatrix for the cluster
    submatrix = distance_matrix[np.ix_(indices, indices)]

    #### ----

    # Compute the mean distance to all other points for each repo in the cluster
    mean_distances = np.round(submatrix.mean(axis=1), decimals=8)
    min_distance = mean_distances.min()

    # Handle ties (multiple points with the same minimum mean distance)
    # Find all candidates with the minimum mean distance
    candidates = [
        (cluster[i], mean_distances[i])  # Store both repo and its mean distance
        for i, d in enumerate(mean_distances) if abs(d - min_distance) < 1e-9
    ]

    # Sort candidates by repo name (to ensure consistent selection)
    candidates = sorted(candidates, key=lambda x: x[0])  # Sort by repo name (alphabetically)

    # Return the first candidate's repo
    return candidates[0][0]



    # # Compute the mean distance.py to all other points for each repo in the cluster
    # centroid_index = submatrix.mean(axis=1).argmin()
    #
    # # Return the repo corresponding to the centroid
    # return cluster[centroid_index]