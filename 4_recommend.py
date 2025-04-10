import pandas as pd
import numpy as np
import os
# ------------------------------
# 0. Load movie data and user rating data
# ------------------------------
MOVIES_FILEPATH = './IMDB-Dataset/movies.csv'
RATINGS_FILEPATH = './IMDB-Dataset/ratings.csv'
movies = pd.read_csv(MOVIES_FILEPATH)
ratings = pd.read_csv(RATINGS_FILEPATH)


# ------------------------------
# 1. Load TFIDF and svd result
# ------------------------------
RESULTS_PATH = "./results"
RATING_MATRIX_PATH = os.path.join(RESULTS_PATH, "rating_matrix.pkl")
COS_SIM_PATH = os.path.join(RESULTS_PATH, "cos_sim_df.pkl")
SVD_PRED_PATH = os.path.join(RESULTS_PATH, "svd_pred_df.pkl")

rating_matrix = pd.read_pickle(RATING_MATRIX_PATH)
cos_sim_df = pd.read_pickle(COS_SIM_PATH)
svd_pred_df = pd.read_pickle(SVD_PRED_PATH)

# ------------------------------
# 2. Get rated and unrated movies
# ------------------------------
user_id = 1
user_ratings = rating_matrix.loc[user_id]
rated_movie_ids = user_ratings[user_ratings > 0].index
unrated_movies = user_ratings[user_ratings == 0].index


# ------------------------------
# 3. Get Content-based (TF-IDF) scores for unrated movies
# ------------------------------
# Score function
def content_score(movie_id):
    sim_scores = cos_sim_df.loc[movie_id, rated_movie_ids]
    user_scores = user_ratings.loc[rated_movie_ids]
    return np.dot(sim_scores, user_scores)

# Generate content-based scores
content_scores = {movie_id: content_score(movie_id) for movie_id in unrated_movies}
content_scores_series = pd.Series(content_scores)

# ------------------------------
# 4. Get SVD predicted scores for unrated movies
# ------------------------------
svd_scores_series = svd_pred_df.loc[user_id, unrated_movies]

# ------------------------------
# 5. Compute scores
# ------------------------------
hybrid_scores = 0.5 * svd_scores_series + 0.5 * content_scores_series
top_hybrid = hybrid_scores.sort_values(ascending=False).head(10)
top_hybrid.columns = ['movieId', 'score', 'title']

# ------------------------------
# 6. Display Results
# ------------------------------
movies_subset = movies.set_index('movieId')
top_hybrid = top_hybrid.rename_axis('movieId').reset_index()
top_hybrid['title'] = top_hybrid['movieId'].map(movies_subset['title'])
print("SVD score range:", svd_scores_series.min(), "to", svd_scores_series.max())
print("Content score range:", content_scores_series.min(), "to", content_scores_series.max())

print("\n🎬 Top 10 Hybrid Recommendations for User", user_id)
top_hybrid.columns = ['movieId', 'score', 'title']
print(top_hybrid[['movieId', 'title', 'score']])
