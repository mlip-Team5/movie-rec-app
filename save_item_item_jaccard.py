import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle

def jaccard_similarity(matrix):
    intersection = matrix.dot(matrix.T).toarray()
    row_sums = matrix.sum(axis=1).A1
    union = row_sums[:, None] + row_sums[None, :] - intersection
    sim = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    return sim

def main():
    df = pd.read_csv('user_movie_interactions.csv')
    movie_cats = df['movieid'].astype('category')
    movie_ids = movie_cats.cat.codes.values
    user_ids = df['userid'].astype('category').cat.codes.values
    n_users = len(np.unique(user_ids))
    n_movies = len(np.unique(movie_ids))
    matrix = csr_matrix((df['score'], (user_ids, movie_ids)), shape=(n_users, n_movies))
    matrix_bin = (matrix > 0).astype(int)
    jac_sim = jaccard_similarity(matrix_bin.T)
    # Save the similarity matrix and the mapping from index to movieid
    with open('item_item_jaccard.pkl', 'wb') as f:
        pickle.dump({'similarity': jac_sim, 'movieid_categories': movie_cats.cat.categories.tolist()}, f)
    print('Jaccard similarity matrix and movieid mapping saved to item_item_jaccard.pkl')

if __name__ == '__main__':
    main()
