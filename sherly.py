import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load movie data and user rating data
# ------------------------------
MOVIES_FILEPATH = './IMDB-Dataset/movies.csv'
RATINGS_FILEPATH = './IMDB-Dataset/ratings.csv'
movies = pd.read_csv(MOVIES_FILEPATH)
ratings = pd.read_csv(RATINGS_FILEPATH)

# ------------------------------
# 2. TF-IDF on genres
# ------------------------------
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cos_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
cos_sim_df = pd.DataFrame(cos_sim_matrix, index=movies['movieId'], columns=movies['movieId'])

# ------------------------------
# 3. Build user-movie rating matrix
# ------------------------------
rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print("Rating matrix shape:", rating_matrix.shape)

# ------------------------------
# 4. SVD Decomposition
# ------------------------------
# Get matrix values and perform SVD
R = rating_matrix.values
U, sigma, Vt = svds(R, k=50)  # k can be tuned
sigma = np.diag(sigma)

# Predicted full rating matrix from SVD
svd_pred = np.dot(np.dot(U, sigma), Vt)
svd_pred_df = pd.DataFrame(svd_pred, index=rating_matrix.index, columns=rating_matrix.columns)

# ------------------------------
# 5. Generate Recommendations (Hybrid)
# ------------------------------
user_id = 1
user_ratings = rating_matrix.loc[user_id]
rated_movie_ids = user_ratings[user_ratings > 0].index
unrated_movies = user_ratings[user_ratings == 0].index

# Content-based (TF-IDF) score function
def content_score(movie_id):
    sim_scores = cos_sim_df.loc[movie_id, rated_movie_ids]
    user_scores = user_ratings.loc[rated_movie_ids]
    return np.dot(sim_scores, user_scores)

# Generate content-based scores
content_scores = {movie_id: content_score(movie_id) for movie_id in unrated_movies}
content_scores_series = pd.Series(content_scores)

# Get SVD predicted scores for unrated movies
svd_scores_series = svd_pred_df.loc[user_id, unrated_movies]

# Combine both scores (50% each for hybrid)
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
print(top_hybrid[['movieId', 'title', 0]])

