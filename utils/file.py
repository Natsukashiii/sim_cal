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
        results (list of dict): List containing dictionaries with results data.
        filename (str): Name of the output CSV file.
    """
    if not results:
        print("No results to save.")
        return

    file_path = os.path.join(RESULT_DIR, filename)

    # Define CSV column headers based on keys in the dictionary
    fieldnames = results[0].keys()

    # Write to CSV file
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Write column headers
        writer.writerows(results)  # Write data rows

    print(f"Results saved to {filename}")
