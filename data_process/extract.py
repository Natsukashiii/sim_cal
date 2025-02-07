
import os
import pickle
from typing import re
import re
import numpy as np
import pandas as pd
from utils.path import BASELINE_CROSSSIM,BASELINE_REPOPAL,BASELINE_MUDABLUE,INPUT_DIR
from data_process.mock import mock_sim1,mock_sim2,mock_sim3

REPOPAL_RESULT = os.path.join(BASELINE_REPOPAL, "result_readme")
CROSSSIM_RESULT = os.path.join(BASELINE_CROSSSIM, "result")
MUDABLUE_RESULT = BASELINE_MUDABLUE


def load_data(source_name, random_seed=42):
    similarity_file_path = os.path.join(INPUT_DIR, f"{source_name}.csv")
    repos_file_path = os.path.join(INPUT_DIR, "repos.txt")

    if os.path.exists(similarity_file_path):
        print(f"Loading {source_name} data from saved file.")
        combined_df = pd.read_csv(similarity_file_path)
        repos = combined_df.iloc[:, 0].tolist()
        matrix = combined_df.iloc[:, 1:].values
    else:
        print(f"Data file for {source_name} not found. Extracting data...")
        if source_name == 'crosssim':
            df = extract_crosssim()
        elif source_name == "repopal":
            df = extract_repopal()
        elif source_name == "mudablue":
            df = extract_mudablue()
        elif source_name == "mock1":
            df = mock_sim1()
        elif source_name == "mock2":
            df = mock_sim2()
        elif source_name == "mock3":
            df = mock_sim3()
        else:
            print("Not valid datasoure")

        matrix, repos = build_similarity_matrix(df)

        combined_df = pd.DataFrame(matrix, index=repos, columns=repos)
        combined_df.insert(0, "Repo", repos)

        os.makedirs(INPUT_DIR, exist_ok=True)
        combined_df.to_csv(similarity_file_path, index=False)
        print(f"Data for {source_name} saved to {similarity_file_path}.")


    if not os.path.exists(repos_file_path):
        with open(repos_file_path, 'w') as f:
            for repo in repos:
                f.write(repo + "\n")
        print(f"All repos saved to {repos_file_path}.")
    else:
        print(f"Repos file already exists at {repos_file_path}. Skipping save.")

    embeddings = None
    matrix = clean_matrix(matrix)
    return matrix, repos


def clean_matrix(matrix):
    # Ensure the input numpy array
    matrix = np.array(matrix, dtype=float)
    # replace NaN and inf to 0
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix

def build_similarity_matrix(df):

    repos = sorted(set(df['repo1']).union(set(df['repo2'])))
    repo_to_index = {repo: idx for idx, repo in enumerate(repos)}

    size = len(repos)
    similarity_matrix = np.zeros((size, size))

    for _, row in df.iterrows():
        repo1 = row['repo1']
        repo2 = row['repo2']
        similarity = row['similarity']

        if repo1 not in repo_to_index or repo2 not in repo_to_index:
            continue
        if not isinstance(similarity, (int, float)) or np.isnan(similarity):
            continue

        idx1 = repo_to_index[repo1]
        idx2 = repo_to_index[repo2]

        similarity_matrix[idx1][idx2] = similarity
        similarity_matrix[idx2][idx1] = similarity

    return similarity_matrix, repos

def extract_repopal():
    """
    Parse Repopal similarity data files.
    File format: {owner}__{repo}.txt
    Each line: {owner1}/{repo1}\t{owner2}/{repo2}\t1.0
    """
    all_data = []

    if not os.path.exists(REPOPAL_RESULT):
        raise FileNotFoundError(f"Directory not found: {REPOPAL_RESULT}")

    for file_name in os.listdir(REPOPAL_RESULT):
        if file_name.endswith('.txt') and '|' in file_name:
            file_path = os.path.join(REPOPAL_RESULT, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        repo1, repo2, similarity = parts
                        try:
                            similarity = float(similarity)
                            all_data.append({'repo1': repo1.strip(), 'repo2': repo2.strip(), 'similarity': similarity})
                        except ValueError:
                            print(f"Invalid similarity value: {similarity}")

    return pd.DataFrame(all_data)


def extract_owner_repo(url):
    """
    Extract {owner}/{repo} from a URL.
    Example: git://github.com/{owner}/{repo}.git
    """
    dot_index = url.rfind('.')
    last_slash_index = url.rfind('/')
    second_last_slash_index = url.rfind('/', 0, last_slash_index)
    owner_repo = url[second_last_slash_index + 1:dot_index]
    return owner_repo


def extract_crosssim():
    """
    Parse CrossSim similarity data files.
    File format: {owner}_{repo}.txt
    Each line: git://github.com/{owner1}/{repo1}.git\tgit://github.com/{owner2}/{repo2}.git\t0.005533854477107525
    """
    all_data = []

    if not os.path.exists(CROSSSIM_RESULT):
        raise FileNotFoundError(f"Directory not found: {CROSSSIM_RESULT}")

    for file_name in os.listdir(CROSSSIM_RESULT):
        if file_name.endswith('.txt') and '|' in file_name:
            file_path = os.path.join(CROSSSIM_RESULT, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        repo1, repo2, similarity = parts
                        try:
                            repo1_name = extract_owner_repo(repo1).strip()
                            repo2_name = extract_owner_repo(repo2).strip()
                            similarity = float(similarity)
                            all_data.append({'repo1': repo1_name, 'repo2': repo2_name, 'similarity': similarity})
                        except ValueError:
                            print(f"Invalid similarity value: {similarity}")

    return pd.DataFrame(all_data)


def extract_mudablue():
    index_file = os.path.join(MUDABLUE_RESULT, "index.txt")
    result_file = os.path.join(MUDABLUE_RESULT, "result.txt")

    method_to_repo = {}
    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(": ", 1)
                if len(parts) == 2:
                    index = int(parts[0].strip())
                    owner_repo = parts[1].strip()
                    method_to_repo[index] = owner_repo

    repos = list(method_to_repo.values())
    size = len(method_to_repo)
    similarity_matrix = np.zeros((size, size))

    with open(result_file, 'r') as f:
        row_index = -1
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = re.search(r"BlockRealMatrix\{\{(.*?)\}\}", line)
            if match:
                row_index += 1
                matrix_values = match.group(1)
                values = re.split(r"[,\s]+", matrix_values)

                for col_index in range(min(len(values), size)):
                    try:
                        similarity_matrix[row_index][col_index] = float(values[col_index])
                    except ValueError:
                        similarity_matrix[row_index][col_index] = 0.0
                continue

    all_data = []
    for i in range(size):
        for j in range(size):
            all_data.append({
                'repo1': repos[i],
                'repo2': repos[j],
                'similarity': similarity_matrix[i][j]
            })

    df = pd.DataFrame(all_data)
    return df


# matrix1, repos1, _ = load_data("crosssim")
# matrix2, repos2, _ = load_data("repopal")
# matrix3, repos3, _ = load_data("mock3")
