import pandas as pd


def analyze_repo_attributes(csv_file, attributes):
    """
    Analyze specified attributes in a CSV file and output a summary table.

    Args:
        csv_file (str): Path to the CSV file.
        attributes (list): List of attributes to analyze.

    Returns:
        pd.DataFrame: A summary table with attribute values and their counts.
    """
    data = pd.read_csv(csv_file)

    summary = {}

    for attribute in attributes:
        if attribute in data.columns:
            # Count the occurrences of each unique value in the attribute
            summary[attribute] = data[attribute].value_counts().reset_index()
            summary[attribute].columns = [attribute, 'Count']
        else:
            print(f"Attribute '{attribute}' not found in the dataset.")

    for attribute, result in summary.items():
        print(f"\nAnalysis for '{attribute}':")
        print(result)

    return summary


csv_file = "metadata.csv"

attributes_to_analyze = ["mainLanguage", "CITool"]
summary = analyze_repo_attributes(csv_file, attributes_to_analyze)