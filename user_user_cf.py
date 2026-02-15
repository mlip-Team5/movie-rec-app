import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

def precision_at_k(recommended, ground_truth, k=10):
    hits = 0
    for rec in recommended[:k]:
        if rec in ground_truth:
            hits += 1
    return hits / k

def jaccard_similarity(matrix):
    intersection = matrix.dot(matrix.T).toarray()
    row_sums = matrix.sum(axis=1).A1
    union = row_sums[:, None] + row_sums[None, :] - intersection
    sim = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    return sim

def pearson_similarity(matrix):
    matrix = matrix.toarray()
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    sim = np.corrcoef(matrix)
    np.fill_diagonal(sim, 0)
    return sim

def evaluate_user_user_all_methods():
    df = pd.read_csv('user_movie_interactions.csv')
    user_ids = df['userid'].astype('category').cat.codes.values
    movie_ids = df['movieid'].astype('category').cat.codes.values
    n_users = len(np.unique(user_ids))
    n_movies = len(np.unique(movie_ids))
    matrix = csr_matrix((df['score'], (user_ids, movie_ids)), shape=(n_users, n_movies))
    matrix_bin = (matrix > 0).astype(int)

    # Train/test split (hold out 1 movie per user for testing)
    train_matrix = matrix.copy().tolil()
    test_items = {}
    for u in range(n_users):
        items = matrix[u].nonzero()[1]
        if len(items) > 1:
            test_item = np.random.choice(items)
            train_matrix[u, test_item] = 0
            test_items[u] = test_item
    train_matrix = train_matrix.tocsr()

    # KNN
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(train_matrix)
    knn_precisions = []
    for u, test_item in test_items.items():
        dists, neighbors = knn.kneighbors(train_matrix[u], n_neighbors=11)
        neighbors = neighbors[0][1:]  # Exclude self
        neighbor_movies = set()
        for n in neighbors:
            neighbor_movies.update(train_matrix[n].nonzero()[1])
        already_watched = set(train_matrix[u].nonzero()[1])
        recs = [m for m in neighbor_movies if m not in already_watched][:10]
        knn_precisions.append(precision_at_k(recs, [test_item], k=10))
    print(f'User-User KNN Precision@10: {np.mean(knn_precisions):.4f}')

    # Cosine Similarity (manual)
    cos_sim = cosine_similarity(train_matrix)
    cos_precisions = []
    for u, test_item in test_items.items():
        recs = np.argsort(-cos_sim[u])[:11]
        recs = recs[recs != u][:10]  # Exclude self
        neighbor_movies = set()
        for n in recs:
            neighbor_movies.update(train_matrix[n].nonzero()[1])
        already_watched = set(train_matrix[u].nonzero()[1])
        recs_movies = [m for m in neighbor_movies if m not in already_watched][:10]
        cos_precisions.append(precision_at_k(recs_movies, [test_item], k=10))
    print(f'User-User Cosine Similarity Precision@10: {np.mean(cos_precisions):.4f}')

    # Pearson Correlation
    pearson_sim = pearson_similarity(train_matrix)
    pearson_precisions = []
    for u, test_item in test_items.items():
        recs = np.argsort(-pearson_sim[u])[:11]
        recs = recs[recs != u][:10]
        neighbor_movies = set()
        for n in recs:
            neighbor_movies.update(train_matrix[n].nonzero()[1])
        already_watched = set(train_matrix[u].nonzero()[1])
        recs_movies = [m for m in neighbor_movies if m not in already_watched][:10]
        pearson_precisions.append(precision_at_k(recs_movies, [test_item], k=10))
    print(f'User-User Pearson Correlation Precision@10: {np.mean(pearson_precisions):.4f}')

    # Jaccard Similarity (binary)
    jac_sim = jaccard_similarity(matrix_bin)
    jac_precisions = []
    for u, test_item in test_items.items():
        recs = np.argsort(-jac_sim[u])[:11]
        recs = recs[recs != u][:10]
        neighbor_movies = set()
        for n in recs:
            neighbor_movies.update(train_matrix[n].nonzero()[1])
        already_watched = set(train_matrix[u].nonzero()[1])
        recs_movies = [m for m in neighbor_movies if m not in already_watched][:10]
        jac_precisions.append(precision_at_k(recs_movies, [test_item], k=10))
    print(f'User-User Jaccard Similarity Precision@10: {np.mean(jac_precisions):.4f}')

if __name__ == '__main__':
    evaluate_user_user_all_methods()
