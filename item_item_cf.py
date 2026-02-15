import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from implicit.als import AlternatingLeastSquares
from sklearn.metrics.pairwise import cosine_similarity

def precision_at_k(recommended, ground_truth, k=10):
    hits = 0
    for rec in recommended[:k]:
        if rec in ground_truth:
            hits += 1
    return hits / k

def jaccard_similarity(matrix):
    # matrix: items x users (binary)
    intersection = matrix.dot(matrix.T).toarray()
    row_sums = matrix.sum(axis=1).A1
    union = row_sums[:, None] + row_sums[None, :] - intersection
    sim = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    return sim

def pearson_similarity(matrix):
    # matrix: items x users
    matrix = matrix.toarray()
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    sim = np.corrcoef(matrix)
    np.fill_diagonal(sim, 0)
    return sim

def evaluate_item_item_all_methods():
    df = pd.read_csv('user_movie_interactions.csv')
    user_ids = df['userid'].astype('category').cat.codes.values
    movie_ids = df['movieid'].astype('category').cat.codes.values
    n_users = len(np.unique(user_ids))
    n_movies = len(np.unique(movie_ids))
    matrix = csr_matrix((df['score'], (user_ids, movie_ids)), shape=(n_users, n_movies))
    matrix_bin = (matrix > 0).astype(int)

    # ALS expects item-user matrix (items as rows)
    als = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=10)
    als.fit(matrix.T)
    als_precisions = []
    n_items_als = als.item_factors.shape[0]
    for m in range(n_items_als):
        recs, _ = als.similar_items(m, N=10)
        users_who_watched = matrix[:, m].nonzero()[0]
        for u in users_who_watched:
            watched = set(matrix[u].nonzero()[1])
            als_precisions.append(precision_at_k(recs, watched, k=10))
    print(f'Item-Item ALS Precision@10: {np.mean(als_precisions):.4f}')

    # KNN on item-item similarity
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(matrix.T)
    knn_precisions = []
    for m in range(n_movies):
        dists, neighbors = knn.kneighbors(matrix.T[m], n_neighbors=11)
        neighbors = neighbors[0][1:]  # Exclude self
        users_who_watched = matrix[:, m].nonzero()[0]
        for u in users_who_watched:
            watched = set(matrix[u].nonzero()[1])
            knn_precisions.append(precision_at_k(neighbors, watched, k=10))
    print(f'Item-Item KNN Precision@10: {np.mean(knn_precisions):.4f}')

    # Cosine Similarity (manual)
    cos_sim = cosine_similarity(matrix.T)
    cos_precisions = []
    for m in range(n_movies):
        recs = np.argsort(-cos_sim[m])[:11]
        recs = recs[recs != m][:10]  # Exclude self
        users_who_watched = matrix[:, m].nonzero()[0]
        for u in users_who_watched:
            watched = set(matrix[u].nonzero()[1])
            cos_precisions.append(precision_at_k(recs, watched, k=10))
    print(f'Item-Item Cosine Similarity Precision@10: {np.mean(cos_precisions):.4f}')

    # Pearson Correlation
    pearson_sim = pearson_similarity(matrix.T)
    pearson_precisions = []
    for m in range(n_movies):
        recs = np.argsort(-pearson_sim[m])[:11]
        recs = recs[recs != m][:10]
        users_who_watched = matrix[:, m].nonzero()[0]
        for u in users_who_watched:
            watched = set(matrix[u].nonzero()[1])
            pearson_precisions.append(precision_at_k(recs, watched, k=10))
    print(f'Item-Item Pearson Correlation Precision@10: {np.mean(pearson_precisions):.4f}')

    # Jaccard Similarity (binary)
    jac_sim = jaccard_similarity(matrix_bin.T)
    jac_precisions = []
    for m in range(n_movies):
        recs = np.argsort(-jac_sim[m])[:11]
        recs = recs[recs != m][:10]
        users_who_watched = matrix[:, m].nonzero()[0]
        for u in users_who_watched:
            watched = set(matrix[u].nonzero()[1])
            jac_precisions.append(precision_at_k(recs, watched, k=10))
    print(f'Item-Item Jaccard Similarity Precision@10: {np.mean(jac_precisions):.4f}')

if __name__ == '__main__':
    evaluate_item_item_all_methods()
