import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import CountVectorizer
from hashlib import sha256
from utils.path import REPOS_FILE_PATH
from sklearn.metrics.pairwise import manhattan_distances


# todo hashcode with 3number, tuple geometric space

def hash_to_3d(repo_name):
    """
    Map a repository name to a 3D coordinate using a hash function.

    Args:
        repo_name (str): The repository name.

    Returns:
        np.array: A 3D coordinate (x, y, z).
    """
    hash_digest = sha256(repo_name.encode()).hexdigest()
    x = int(hash_digest[:8], 16) % 1000 / 1000  # Normalize to [0,1]
    y = int(hash_digest[8:16], 16) % 1000 / 1000
    z = int(hash_digest[16:24], 16) % 1000 / 1000
    return np.array([x, y, z], dtype=np.float32)

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


def mock_sim3(p=3):
    """
    Generate a similarity matrix using a 3D Minkowski distance metric for repository names.

    Args:
        p (int): The Minkowski distance parameter. p=1 (Manhattan), p=2 (Euclidean), p=3+ (higher order).

    Returns:
        pd.DataFrame: Similarity matrix with repo1, repo2, and similarity scores.
    """
    repos = load_repos(REPOS_FILE_PATH)

    # Map repositories to 3D coordinates
    repo_coords = {repo: hash_to_3d(repo) for repo in repos}

    # Calculate all distances first to normalize them
    all_distances = []
    for i in range(len(repos)):
        for j in range(i + 1, len(repos)):
            dist = minkowski_distance(repo_coords[repos[i]], repo_coords[repos[j]], p)
            all_distances.append(dist)

    max_dist = max(all_distances) if all_distances else 1  # Avoid division by zero

    data = []
    for i in range(len(repos)):
        for j in range(i + 1, len(repos)):
            repo1, repo2 = repos[i], repos[j]
            similarity = calculate_similarity_minkowski(repo_coords[repo1], repo_coords[repo2], max_dist, p)
            data.append({'repo1': repo1, 'repo2': repo2, 'similarity': round(similarity, 4)})

    return pd.DataFrame(data)



def minkowski_distance(coord1, coord2, p):
    """
    Compute Minkowski distance between two points.

    Args:
        coord1 (np.array): First 3D coordinate.
        coord2 (np.array): Second 3D coordinate.
        p (int): The order of the Minkowski metric.

    Returns:
        float: Minkowski distance.
    """
    return np.power(np.sum(np.abs(coord1 - coord2) ** p), 1 / p)


def calculate_similarity_minkowski(coord1, coord2, max_dist, p):
    """
    Convert Minkowski distances to similarity scores.

    Args:
        coord1 (np.array): First 3D coordinate.
        coord2 (np.array): Second 3D coordinate.
        max_dist (float): Maximum distance for normalization.
        p (int): Minkowski distance order.

    Returns:
        float: Similarity score between 0 and 1.
    """
    distance = minkowski_distance(coord1, coord2, p)
    normalized_distance = distance / max_dist  # Normalize distance
    return np.exp(-normalized_distance * 5)

def mock_sim4():
    """
    Generate a similarity matrix using cosine similarity in 3D space.
    """
    repos = load_repos(REPOS_FILE_PATH)

    # Convert repo names to 3D vectors
    repo_coords = np.array([hash_to_3d(repo) for repo in repos])

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(repo_coords)

    # Format results into a DataFrame
    data = []
    for i in range(len(repos)):
        for j in range(i + 1, len(repos)):
            data.append({
                'repo1': repos[i],
                'repo2': repos[j],
                'similarity': round(similarity_matrix[i, j], 4)
            })

    return pd.DataFrame(data)
def jaccard_similarity(set1, set2):
    """
    Compute Jaccard similarity between two sets.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def mock_sim5():
    """
    Generate a similarity matrix using Jaccard similarity based on character sets in repo names.
    """
    repos = load_repos(REPOS_FILE_PATH)

    # Convert repo names to sets of characters
    repo_sets = [set(repo) for repo in repos]

    data = []
    for i in range(len(repos)):
        for j in range(i + 1, len(repos)):
            similarity = jaccard_similarity(repo_sets[i], repo_sets[j])
            data.append({
                'repo1': repos[i],
                'repo2': repos[j],
                'similarity': round(similarity, 4)
            })

    return pd.DataFrame(data)
def load_repos(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as f:
        repos = [line.strip() for line in f.readlines() if line.strip()]
    return repos