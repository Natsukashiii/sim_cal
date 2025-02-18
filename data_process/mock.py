import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score

REPOS_FILE_PATH = "input/repos.txt"

# todo hashcode with 3number, tuple geometric space

def mock_sim1():
    """
     Levenshtein Distance
    """
    repos = load_repos(REPOS_FILE_PATH)

    data = []
    for i in range(len(repos)):
        for j in range(i + 1, len(repos)):
            repo1 = repos[i]
            repo2 = repos[j]

            edit_distance = levenshtein_distance(repo1, repo2)

            max_length = max(len(repo1), len(repo2))
            similarity = 1 - (edit_distance / max_length)

            data.append({'repo1': repo1, 'repo2': repo2, 'similarity': similarity})

    return pd.DataFrame(data)


def mock_sim2():
    repos = load_repos(REPOS_FILE_PATH)

    vectorizer = CountVectorizer(analyzer='char')
    char_matrix = vectorizer.fit_transform(repos)

    similarity_matrix = cosine_similarity(char_matrix)

    data = []
    for i in range(len(repos)):
        for j in range(i + 1, len(repos)):
            data.append({'repo1': repos[i], 'repo2': repos[j], 'similarity': similarity_matrix[i][j]})

    return pd.DataFrame(data)

def mock_sim4(random_seed):
    """
    Generate a structured similarity matrix with clear clustering patterns.

    Args:
        random_seed (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Structured similarity matrix in long format (repo1, repo2, similarity).
    """
    # Load the repository names
    repos = load_repos(REPOS_FILE_PATH)

    # Determine number of clusters dynamically
    num_clusters = random_seed // 2
    if num_clusters < 2:
        num_clusters = random_seed - 1

    np.random.seed(random_seed)

    num_repos = len(repos)

    cluster_sizes = [num_repos // num_clusters] * num_clusters
    cluster_sizes[-1] += num_repos % num_clusters

    similarity_matrix = np.zeros((num_repos, num_repos))

    start = 0
    for size in cluster_sizes:
        end = start + size
        similarity_matrix[start:end, start:end] = np.random.uniform(0.8, 1.0, (size, size))
        start = end

    similarity_matrix += np.random.uniform(0.1, 0.4, (num_repos, num_repos))

    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

    np.fill_diagonal(similarity_matrix, 1)

    data = []
    for i in range(num_repos):
        for j in range(i + 1, num_repos):
            data.append({
                'repo1': repos[i],
                'repo2': repos[j],
                'similarity': round(similarity_matrix[i, j], 4)
            })

    return pd.DataFrame(data)

def calculate_vowel_count(repo_name):
    vowels = set("aeiouAEIOU")
    return sum(1 for char in repo_name if char in vowels)

def calculate_letter_count(repo_name):
    return sum(1 for char in repo_name if char.isalpha())

def mock_sim3(random_seed=42):
    """
    Generate a similarity matrix based on the number of letters and vowels in project names.

    Args:
        random_seed (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Similarity matrix in long format (repo1, repo2, similarity).
    """
    repos = load_repos(REPOS_FILE_PATH)

    random_seed=42
    np.random.seed(random_seed)

    repo_stats = {
        repo: {
            "letter_count": calculate_letter_count(repo),
            "vowel_count": calculate_vowel_count(repo),
        }
        for repo in repos
    }

    data = []
    for i in range(len(repos)):
        for j in range(i + 1, len(repos)):
            repo1 = repos[i]
            repo2 = repos[j]

            repo1_stats = repo_stats[repo1]
            repo2_stats = repo_stats[repo2]

            letter_diff = abs(repo1_stats["letter_count"] - repo2_stats["letter_count"])
            vowel_diff = abs(repo1_stats["vowel_count"] - repo2_stats["vowel_count"])

            max_letter_diff = max(repo_stats[repo]["letter_count"] for repo in repos)
            max_vowel_diff = max(repo_stats[repo]["vowel_count"] for repo in repos)

            normalized_letter_diff = letter_diff / max_letter_diff if max_letter_diff > 0 else 0
            normalized_vowel_diff = vowel_diff / max_vowel_diff if max_vowel_diff > 0 else 0

            similarity = 1 - 0.5 * (normalized_letter_diff + normalized_vowel_diff)  # Weighted average

            similarity = max(0, min(1, similarity))

            data.append({'repo1': repo1, 'repo2': repo2, 'similarity': round(similarity, 4)})

    return pd.DataFrame(data)


def load_repos(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as f:
        repos = [line.strip() for line in f.readlines() if line.strip()]
    return repos