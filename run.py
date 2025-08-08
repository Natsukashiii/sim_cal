import itertools

import algori
import cluster.canopy
from algori import align, combine, distance
from cluster.distance import analysis_tuple_matrix_multi_dimension
from cluster.visualize import (visualize_combined_matrices,
                               visualize_comparison_multiple,
                               visualize_heatmap)
from compute import overlap
from data_process import extract
from utils.config import (load_attributes, load_eva_attributes,
                          load_integrate_level_high, load_integrate_level_low,
                          load_pick_repo_number)
from utils.file import save_all_results_to_csv, save_combined_distance_matrix

#############
# Following variables are edited in config.py
PICK_REPO_NUM = load_pick_repo_number() #This number is used to pick the top N repositories from the similarity matrix for testing the function(to reduce the computation time)
DIMENSION_LEVEL_LOW =load_integrate_level_low()
DIMENSION_LEVEL_HIGH = load_integrate_level_high()
#############

GEN_HEATMAP = True
# save the esm res
SAVE_DISTANCE_MATRIX = True
# evaluation if using EVA dataset, also need to provided the eva results.
EXEC_EVA= False

def compare_multiple_attributes():
    """
    Compare multiple attributes
    """
    print("-------------------------------------- 0. Load Config--------------------------------------")
    if EXEC_EVA:
        attributes = load_eva_attributes()
    else:
        attributes = load_attributes()

    print("-------------------------------------- 1. Load Similarity Data --------------------------------------")
    origin_sim_matrix_list =[]
    for source in attributes:
        if EXEC_EVA:
            matrix, repos = extract.load_eva_data(source)
        else:
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
    if GEN_HEATMAP:
        visualize_heatmap(normalize_sim_matrix_list, attributes, "rq1_matrix_compare.pdf")

    print("-------------------------------------- 4. Build Distance --------------------------------------")
    distance_matrix_map = {}
    for normalized_sim_matrix,reops, source in normalize_sim_matrix_list:
        distance_matrix = algori.distance.build_distance_matrix(normalized_sim_matrix)
        distance_matrix_map[source] = distance_matrix

    for source, distance_matrix in distance_matrix_map.items():
        if isinstance(distance_matrix, tuple):
            distance_matrix = distance_matrix[0]
        min,max,mean = align.get_min_max(distance_matrix)
        print(f"Source: {source}, Min: {min}, Max: {max}, Mean: {mean}")


    visualize_combined_matrices(
        list(distance_matrix_map.values()),
        attributes,
        "normalize_distance.pdf"
    )

    print("-------------------------------------- 6. Generate Combinations --------------------------------------")

    combinations = []
    for level in range(DIMENSION_LEVEL_LOW, DIMENSION_LEVEL_HIGH+1):  #enable multi dimension
        combinations.extend(list(itertools.combinations(attributes, level)))
    print(f"Generated combinations: {combinations}")


    print("-------------------------------------- 6. Generate Combinations Matrix --------------------------------------")
    combinations_distance_dict = {}

    for combination in combinations:
        if len(combination) == 1:
            single_distance_matrix = distance_matrix_map[combination[0]]
            combinations_distance_dict[combination] = single_distance_matrix
            print(f"Single attribute {combination} added directly.")
            if SAVE_DISTANCE_MATRIX:
                save_combined_distance_matrix(single_distance_matrix, common_repos, combination)
        else:
            selected_matrices = [distance_matrix_map[attr] for attr in combination]
            combine_distance_matrix = combine.combine_multi_dimension(*selected_matrices, common_repos=common_repos)
            combine_min, combine_mean, combine_max = analysis_tuple_matrix_multi_dimension(combine_distance_matrix)
            print(f"Combine {combination} completed: distance: np.min={combine_min}, np.mean = {combine_mean}, np.max = {combine_max}")
            combinations_distance_dict[combination] = combine_distance_matrix
            if SAVE_DISTANCE_MATRIX:
                save_combined_distance_matrix(combine_distance_matrix, common_repos, combination)



    
    print("-------------------------------------- 6. Generate Cluster Result --------------------------------------")
    combinations_cluster_result_dict = {}

    for combination in combinations:
        combine_name = "_".join(combination) if len(combination) > 1 else combination[0]
        selected_distance_matrix = combinations_distance_dict.get(combination)
        if combine_distance_matrix is None:
            print(f"Warning: No distance matrix found for {combination}, skipping...")
            continue
        cluster_result = cluster.canopy.canopy_clustering_multi_dimension(selected_distance_matrix, common_repos,
                                                                          combine_name)
        combinations_cluster_result_dict[combination] = cluster_result
    print("-------------------------------------- 6. Calculate overlap Ratio --------------------------------------")

    for base in combinations:
        base_distance_matrix = combinations_distance_dict[base]
        base_cluster_result = combinations_cluster_result_dict[base]

        for other in combinations:
            if base == other:
                continue

            if not (set(base).issubset(set(other)) or set(other).issubset(set(base))):
                continue

            other_distance_matrix = combinations_distance_dict[other]
            other_cluster_result = combinations_cluster_result_dict[other]

            overlap_ratio, filtered_overlap_ratio = overlap.cal_overlap(
                base_distance_matrix, base_cluster_result,
                other_cluster_result, common_repos
            )

            print(f">>>> Overlap ratio ({base} vs {other}): {overlap_ratio}")

            result = {
                "base": base,
                "base_silhouette": base_cluster_result["best_silhouette"],
                "base_cluster_size": base_cluster_result["best_cluster_nums"],
                "compared_with": other,
                "compared_with_silhouette": other_cluster_result["best_silhouette"],
                "compared_cluster_size": other_cluster_result["best_cluster_nums"],
                "overlap_ratio": overlap_ratio
            }

            print(f">>>>>>>>>>>>>>>>>>>>>>>>  {base} + {other} overlap ratio: {overlap_ratio}")

            save_all_results_to_csv(result)
    print("-------------------------------------- Finished --------------------------------------")


compare_multiple_attributes()

