import pandas as pd

from utils.path import RESULT_DIR
import os
import csv

def save_results_to_csv(results,filename="results.csv"):
    """
    Save the results to a CSV file.

    Args:
        results (list of dict): Each dictionary contains overlap ratios and attribute mappings.
        file_name (str): The name of the CSV file to save.
    """
    file_path = os.path.join(RESULT_DIR, filename)

    headers = ["Run ID", "Overlap Ratio", "Attribute1", "Attribute2", "Attribute3"]
    with open(file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for i, result in enumerate(results, 1):
            writer.writerow({
                "Run ID": f"Run {i}",
                "Overlap Ratio": result["overlap_ratio"],
                "Attribute1": result["Attribute1"],
                "Attribute2": result["Attribute2"],
                "Attribute3": result["Attribute3"],
            })


def save_all_results_to_csv(results, filename="results.csv"):
    """
    Save results to a CSV file.

    Args:
        results (list of dict) or (dict): List containing dictionaries with results data.
        filename (str): Name of the output CSV file.
    """
    if not results:
        print("No results to save.")
        return

    if isinstance(results, dict):
        results = [results]

    file_path = os.path.join(RESULT_DIR, filename)

    is_new_file = not os.path.exists(file_path)

    fieldnames = results[0].keys()

    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if is_new_file:
            writer.writeheader()

        writer.writerows(results)

    print(f"Results saved to {filename}")

def save_cluster_results_to_csv(result, filename):
    """
    Save clustering results to a CSV file to avoid redundant computation.

    Args:
        result (dict): Clustering results dictionary.
        filename (str): Output file name.
    """
    result_df = pd.DataFrame([result])
    result_df.to_csv(filename, index=False)
    print(f"Clustering results saved to {filename}")