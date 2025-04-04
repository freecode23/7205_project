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
# Assume movies.csv contains movieId, title, genres
MOVIES_FILEPATH='./IMDB-Dataset/movies.csv'
RATINGS_FILEPATH='./IMDB-Dataset/ratings.csv'
movies = pd.read_csv(MOVIES_FILEPATH)
ratings = pd.read_csv(RATINGS_FILEPATH)

# ------------------------------
# 2. Construct movie feature vectors based on TF-IDF
# ------------------------------
# Here we use the genres of the movies as text features, assuming genres are separated by "|"
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
tfidf_matrix = tfidf.fit_transform(movies['genres'])
print("Shape of TF-IDF matrix:", tfidf_matrix.shape)

# ------------------------------
# 3. Compute cosine similarity matrix between movies
# ------------------------------
cos_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Convert the cosine similarity matrix to a DataFrame, indexed by movieId
cos_sim_df = pd.DataFrame(cos_sim_matrix, index=movies['movieId'], columns=movies['movieId'])
print("Shape of movie similarity matrix:", cos_sim_df.shape)

# Visualize part of the similarity matrix as a heatmap (e.g., the top 20 movies)
top_items = 20
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_df.iloc[:top_items, :top_items], cmap="viridis", cbar=True)
plt.title("Heatmap of the first 20 movies' similarity")
plt.xlabel("Movie ID")
plt.ylabel("Movie ID")
plt.show()

# ------------------------------
# 4. Build user-movie rating matrix
# ------------------------------
rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print("Shape of user-movie rating matrix:", rating_matrix.shape)

# ------------------------------
# 5. Generate recommendations for a user
# ------------------------------
user_id = 1
user_ratings = rating_matrix.loc[user_id]
# Find movies the user has already rated
rated_movie_ids = user_ratings[user_ratings > 0].index

# Define a function to compute predicted score based on user ratings and movie similarity
def get_recommendation_score(movie_id):
    # Get similarity scores between this movie and the user's rated movies
    sim_scores = cos_sim_df.loc[movie_id, rated_movie_ids]
    # Get the user's ratings for those movies
    user_scores = user_ratings.loc[rated_movie_ids]
    # Weighted sum: similarity * rating
    return np.dot(sim_scores, user_scores)

# Compute predicted scores for movies the user hasn't rated yet
unrated_movies = user_ratings[user_ratings == 0].index
predicted_scores = {movie_id: get_recommendation_score(movie_id) for movie_id in unrated_movies}
predicted_scores_series = pd.Series(predicted_scores)

# Get the top 10 movies with highest predicted scores as recommendations
top_recommendations = predicted_scores_series.sort_values(ascending=False).head(10)
print("Recommended movies for user", user_id, ":")
print(top_recommendations)

# ------------------------------
# 6. (Optional) Join recommendations with movie titles for display
# ------------------------------
movies_subset = movies.set_index('movieId')
top_recommendations = top_recommendations.rename_axis('movieId').reset_index()
top_recommendations['title'] = top_recommendations['movieId'].map(movies_subset['title'])
print(top_recommendations[['movieId', 'title', 0]])
