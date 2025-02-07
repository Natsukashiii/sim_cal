import itertools

import numpy as np
from data_process import extract
import cluster.canopy
import algori
from algori import align, distance, combine
from compute import overlap
from cluster.distance import analysis_tuple_matrix
from cluster.visualize import visualize_two_compare_matrix, visualize_combined_matrices, visualize_comparison_multiple
from utils.file import save_all_results_to_csv
from utils.config import load_attributes,load_normalize,load_integrate_level

# This number is used to pick the top N repositories from the similarity matrix for testing the function(to reduce the computation time)
PICK_REPO_NUM = 10

def compare_multiple_attributes():
    """
    Compare multiple attributes
    """
    print("-------------------------------------- 0. Load Config--------------------------------------")
    attributes = load_attributes()
    integrate_level = load_integrate_level()
    print(f"1. Attributes: {attributes}, integrate_level: {integrate_level}, PICK_REPO_NUM: {PICK_REPO_NUM}")

    print("-------------------------------------- 1. Load Similarity Data --------------------------------------")
    origin_sim_matrix_list =[]
    for source in attributes:
        matrix, repos = extract.load_data(source)
        origin_sim_matrix_list.append((matrix, repos,source))
    print(f"2. Similarity Data for attributes: {attributes} loaded")

    print("-------------------------------------- 2. Data preprocess --------------------------------------")
    aligned_sim_matrix_list, common_repos = algori.align.align_list(origin_sim_matrix_list, PICK_REPO_NUM)
    print(f"3. Picked {PICK_REPO_NUM} repositories for comparison, align repos to {len(common_repos)}")

    print("-------------------------------------- 3. Data Normalization --------------------------------------")
    normalize_sim_matrix_list = []
    for aligned_sim_matrix, repos,source in aligned_sim_matrix_list:
        normalized_sim_matrix, min, max, crop_ratio = align.stretch(aligned_sim_matrix, source)
        normalize_sim_matrix_list.append((normalized_sim_matrix,repos, source))
    visualize_comparison_multiple(aligned_sim_matrix_list,
                                  normalize_sim_matrix_list,
                                  labels=attributes,
                                  file_name="normalize_similarity.pdf")

    print("-------------------------------------- 4. Build Distance --------------------------------------")
    distance_matrix_map = {}
    for normalized_sim_matrix,reops, source in normalize_sim_matrix_list:
        distance_matrix = algori.distance.build_distance_matrix(normalized_sim_matrix)
        distance_matrix_map[source] = distance_matrix

    for source, distance_matrix in distance_matrix_map.items():
        if isinstance(distance_matrix, tuple):
            distance_matrix = distance_matrix[0]
        min_val = align.get_min_max(distance_matrix)
        mean_val = np.mean(distance_matrix)
        print(f"Source: {source}, Min: {min_val}, Max: {min_val}, Mean: {mean_val}")

    visualize_combined_matrices(
        list(distance_matrix_map.values()),
        attributes,
        # list(distance_matrix_map.keys()),
        "normalize_distance.pdf"
    )
    print("-------------------------------------- 2. Generate Combinations --------------------------------------")
    combinations = list(itertools.combinations(attributes, integrate_level))
    print(f"Generated combinations: {combinations}")
    results = []

    for combination in combinations:
        print("-------------------------------------- 2. Calculation --------------------------------------")
        print(f"Processing combination: {combination}")
        selected_matrices = [distance_matrix_map[attr] for attr in combination]

        combine_distance_matrix = combine.combine(selected_matrices[0],selected_matrices[1], common_repos)
        combine_min, combine_mean, combine_max = analysis_tuple_matrix(combine_distance_matrix)
        print(f"Combine distance:np.min={combine_min}, np.mean = {combine_mean}, np.max = {combine_max}")
        print(f"Combining attributes {combination} completed.")

        for source in combination:
            print(f"Comparing combined matrix with attribute: {source}")

            single_distance_matrix = distance_matrix_map[source]

            single_cluster_result = cluster.canopy.canopy_clustering(single_distance_matrix, common_repos, source, False)
            combined_cluster_result = cluster.canopy.canopy_clustering(combine_distance_matrix, common_repos, "Combined", True)

            overlap_ratio, filtered_overlap_ratio = overlap.cal_overlap(single_distance_matrix, single_cluster_result,
                                                                         combined_cluster_result, common_repos)

            print(f">>>> Overlap ratio (Combined vs {source}): {overlap_ratio}")

            results.append({
                "combination": combination,
                "compared_with": source,
                "overlap_ratio": overlap_ratio
                # "filtered_overlap_ratio": filtered_overlap_ratio
            })

    print("-------------------------------------- Finished --------------------------------------")
    save_all_results_to_csv(results)
    return results



def compare_two_attributes(source1,source2):
    print("-------------------------------------- 0. Load Config--------------------------------------")
    normalize = load_normalize()
    # This number is used to pick the top N repositories from the similarity matrix for testing the function(to reduce the computation time)
    print(f"1. Normalize: {normalize}")


    print("-------------------------------------- 1. Load Similarity Data --------------------------------------")
    matrix1, repos1 = extract.load_data(source1)
    matrix2, repos2 = extract.load_data(source2)
    origin_sim_matrix_list = [
        (matrix1, repos1,source1),
        (matrix2, repos2,source2),
    ]
    print(f"2. Similarity Data for {source1} and {source2} loaded")

    print("-------------------------------------- 2. Data preprocess --------------------------------------")
    aligned_sim_matrix_list, common_repos = algori.align.align_list(origin_sim_matrix_list, PICK_REPO_NUM)
    aligned_sim_matrix_1, repo_list_1,source1= aligned_sim_matrix_list[0]
    aligned_sim_matrix_2, repo_list_2,source2 = aligned_sim_matrix_list[1]
    print(f"3. Picked {PICK_REPO_NUM} repositories for comparison, align repos to {len(common_repos)}")
    # Todo Is there a better way to do this?
    processed_sim_matrix1 = aligned_sim_matrix_1
    processed_sim_matrix1 = aligned_sim_matrix_2

    print("-------------------------------------- 3. Data Normalization --------------------------------------")
    if normalize:
        normalized_sim_matrix_1, min_1, max_1, crop_ratio1 = align.stretch(aligned_sim_matrix_1, source1)
        normalized_sim_matrix_2, min_2, max_2, crop_ratio2 = align.stretch(aligned_sim_matrix_2, source2)
        visualize_comparison_multiple(
            original_matrices=[
                aligned_sim_matrix_1,
                aligned_sim_matrix_2,
            ],
            normalized_matrices=[
                normalized_sim_matrix_1,
                normalized_sim_matrix_2,
            ],
            labels=[f"Attribute-1{source1}", f"Attribute-2{source2}"],
            file_name="normalize_similarity.pdf"
        )

        # Todo
        processed_sim_matrix1 = normalized_sim_matrix_1
        processed_sim_matrix2 = normalized_sim_matrix_2


    print("-------------------------------------- 4. Build Distance --------------------------------------")

    distance_matrix_1 = algori.distance.build_distance_matrix(processed_sim_matrix1)
    distance_matrix_2 = algori.distance.build_distance_matrix(processed_sim_matrix2)
    print(
        f"Origin distance1:np.min besides 0: {align.get_min_max(distance_matrix_1)}, np.mean = {np.mean((distance_matrix_1))}, np.max = {np.max(distance_matrix_1)}")
    print(
        f"Origin distance2:np.min besides 0: {align.get_min_max(distance_matrix_2)}, np.mean = {np.mean((distance_matrix_2))}, np.max = {np.max(distance_matrix_2)}")
    visualize_combined_matrices(
        distance_matrices=[distance_matrix_1, distance_matrix_2],
        labels=[f"Attribute-1{source1}", f"Attribute-2{source2}"],
        file_name="normalize_distance.pdf"
    )


    print("-------------------------------------- 5. Combine --------------------------------------")
    combine_distance_matrix = combine.combine(distance_matrix_1, distance_matrix_2, common_repos)
    combine_min, combine_mean, combine_max = analysis_tuple_matrix(combine_distance_matrix)
    print(f"Combine distance:np.min={combine_min}, np.mean = {combine_mean}, np.max = {combine_max}")


    print("-------------------------------------- 6. Cluster --------------------------------------")

    source1_cluster_result = cluster.canopy.canopy_clustering(distance_matrix_1, common_repos, source1, False)
    source2_cluster_result = cluster.canopy.canopy_clustering(distance_matrix_2, common_repos, source2, False)
    combine_cluster_result = cluster.canopy.canopy_clustering(combine_distance_matrix, common_repos, "combine", True)

    print("-------------------------------------- 7. Computing --------------------------------------")
    # compute
    overlap_ratio, filtered_overlap_ratio1 = overlap.cal_overlap(distance_matrix_1, source1_cluster_result,
                                                                 combine_cluster_result,
                                                                 common_repos)
    # overlap_ratio2 = overlap.cal_overlap(distance_matrix_2, combine_cluster_result, distance_matrix_1,
    #                                      source_cluster_result, common_repos)
    print(f"overlap_ratio={overlap_ratio}")
    print(f"Filtered overlap_ratio={filtered_overlap_ratio1}")
    # print(f"overlap_ratio2={overlap_ratio2}")
    print("-------------------------------------- Finished --------------------------------------")
    return overlap_ratio


# compare_two_attributes("mudablue","crosssim")
compare_multiple_attributes()

