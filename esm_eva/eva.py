import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
                             normalized_mutual_info_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algori.align import align_list, align_list_
from data_process import extract
from utils.path import ESM_RESULT

ESM_EVA_PROJECTS = "eva_projects.csv"
ESM_CSV_FILE = ESM_RESULT / "esm.csv"
FILTER_OUT_DEMO = False 


def load_esm_from_csv(path):
    df = pd.read_csv(path, index_col=0)
    repos = df.index.tolist()
    vectors = []

    for row in df.itertuples(index=False):
        vec = []
        for val in row:
            numbers = eval(val) if isinstance(val, str) else val
            if isinstance(numbers, (float, int)):
                vec.append(numbers)
            else:
                vec.extend(list(numbers))
        vectors.append(np.array(vec))

    n = len(repos)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = euclidean(vectors[i], vectors[j])
            matrix[i, j] = 1 / (1 + dist)
    return matrix, repos

def evaluate_topk_accuracy(sim_df, project_types, topk=5):
    hits, total = 0, 0
    for repo in sim_df.index:
        if repo not in project_types:
            continue
        true_type = project_types[repo]
        sims = sim_df.loc[repo].drop(repo).sort_values(ascending=False)
        top_k_neighbors = sims.head(topk).index
        match_count = sum(project_types.get(r) == true_type for r in top_k_neighbors)
        hits += match_count
        total += topk
    return hits / total if total > 0 else 0

def evaluate_clustering(sim_df, project_types, n_clusters=None):
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(sim_df)
    k = n_clusters or len(set(project_types.values()))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(embedding)
    cluster_labels = kmeans.labels_
    type_labels = [project_types[r] for r in sim_df.index]
    return normalized_mutual_info_score(type_labels, cluster_labels)

def evaluate_clustering_classification(sim_df, project_types, n_clusters=None):
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(sim_df)
    k = n_clusters or len(set(project_types.values()))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(embedding)
    cluster_labels = kmeans.labels_
    true_labels = [project_types[r] for r in sim_df.index]
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)
    ari = adjusted_rand_score(true_encoded, cluster_labels)
    f1 = f1_score(true_encoded, cluster_labels, average='macro')
    acc = accuracy_score(true_encoded, cluster_labels)
    return ari, f1, acc

def eva():
    print("---- 0. Load ESM Evaluation Config ----")
    eva_df = pd.read_csv(ESM_EVA_PROJECTS)
    target_repos = eva_df['repo'].tolist()
    project_types = dict(zip(eva_df['repo'], eva_df['type']))

    print("---- 1. Load Multi-dimensional ESM Matrix ----")
    esm_matrix, esm_repos = load_esm_from_csv(ESM_CSV_FILE)
    esm_df = pd.DataFrame(esm_matrix, index=esm_repos, columns=esm_repos)

    print("---- 2. Load and Align Baseline Attributes ----")
    attributes = ["repopal", "crosssim", "mudablue", "build_eva"]
    raw_matrix_list = [(*extract.load_eva_data(src), src) for src in attributes]
    aligned_matrix_list, common_repos = align_list_(raw_matrix_list)
    print(f"Common Repositories: {len(common_repos)}")

    # Align all repos to intersection of esm, common baseline repos, and evaluation repos
    final_repos = list(set(esm_repos) & set(common_repos) & set(target_repos))
    final_repos.sort()  # sort for consistency

    if FILTER_OUT_DEMO:
        before = len(final_repos)
        final_repos = [r for r in final_repos if project_types.get(r) != "demo"]
        after = len(final_repos)
        print(f"Removed 'demo' type repos: {before - after} removed, {after} remain.")

    # Filter and reindex ESM
    filtered_df = esm_df.loc[final_repos, final_repos]
    filtered_df.to_csv("esm_eval_from_file.csv")
    print("Saved processed similarity matrix to esm_eval_from_file.csv")

    print("✅ ESM and Baseline repo sets unified.")

    print("---- 3. Evaluate ESM ----")
    esm_acc = evaluate_topk_accuracy(filtered_df, project_types, topk=5)
    esm_nmi = evaluate_clustering(filtered_df, project_types)
    esm_ari, esm_f1, esm_cls_acc = evaluate_clustering_classification(filtered_df, project_types)
    print(f"[ESM from CSV] Top-5 acc: {esm_acc:.3f}, NMI: {esm_nmi:.3f}, ARI: {esm_ari:.3f}, "
          f"F1: {esm_f1:.3f}, Cls-Acc: {esm_cls_acc:.3f}")

    print("---- 4. Evaluate Baseline Attributes ----")
    for attr, (matrix, repos, _) in zip(attributes, aligned_matrix_list):
        df = pd.DataFrame(matrix, index=repos, columns=repos)
        df = df.loc[final_repos, final_repos]
        acc = evaluate_topk_accuracy(df, project_types, topk=5)
        nmi = evaluate_clustering(df, project_types)
        ari, f1, cls_acc = evaluate_clustering_classification(df, project_types)
        print(f"[{attr}] Top-5 acc: {acc:.3f}, NMI: {nmi:.3f}, ARI: {ari:.3f}, F1: {f1:.3f}, Cls-Acc: {cls_acc:.3f}")
    print("---- 5. Save Per-Project Top-5 Matching Report ----")
    rows = []
    for repo in filtered_df.index:
        if repo not in project_types:
            continue
        true_type = project_types[repo]
        sims = filtered_df.loc[repo].drop(repo).sort_values(ascending=False)
        top_k = sims.head(5).index
        top_k_types = [project_types.get(r, "N/A") for r in top_k]
        match_count = sum(t == true_type for t in top_k_types)
        acc_per_project = match_count / 5.0

        row = {
            "repo": repo,
            "true_type": true_type,
            "top1": top_k[0], "top1_type": top_k_types[0],
            "top2": top_k[1], "top2_type": top_k_types[1],
            "top3": top_k[2], "top3_type": top_k_types[2],
            "top4": top_k[3], "top4_type": top_k_types[3],
            "top5": top_k[4], "top5_type": top_k_types[4],
            "match_count": match_count,
            "acc@5": acc_per_project
        }
        rows.append(row)

    output_df = pd.DataFrame(rows)
    acc_str = f"{esm_acc:.3f}".replace(".", "_")
    output_file = f"esm_top5_report_{acc_str}.csv"
    output_df.to_csv(output_file, index=False)
    print(f"Saved detailed matching results to: {output_file}")
    
    
    print("---- Label Distribution in Evaluation Set ----")
    from collections import Counter
    label_counts = Counter([project_types[r] for r in final_repos if r in project_types])
    for label, count in label_counts.items():
        print(f"  {label}: {count} projects")

if __name__ == "__main__":
    eva()