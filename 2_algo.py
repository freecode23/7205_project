import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from helper.evaluate import evaluate_content_precision_at_k

# ------------------------------
# 0. Load movie data and user rating data
# ------------------------------
MOVIES_FILEPATH = './IMDB-Dataset/movies.csv'
RATINGS_FILEPATH = './IMDB-Dataset/ratings.csv'
movies = pd.read_csv(MOVIES_FILEPATH)
ratings = pd.read_csv(RATINGS_FILEPATH)

# ------------------------------
# 1. Define Path for results
# ------------------------------
RESULTS_PATH = "./results"
os.makedirs(RESULTS_PATH, exist_ok=True)

RATING_MATRIX_PATH = os.path.join(RESULTS_PATH, "rating_matrix.pkl")
COS_SIM_PATH = os.path.join(RESULTS_PATH, "cos_sim_df.pkl")
SVD_PRED_PATH = os.path.join(RESULTS_PATH, "svd_pred_df.pkl")

# Build user-movie rating matrix
rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
rating_matrix.to_pickle(RATING_MATRIX_PATH)

# ------------------------------
# 2. Content-Based Filtering (TF-IDF on Genres + Cosine Similarity)
# ------------------------------
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cos_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
cos_sim_df = pd.DataFrame(cos_sim_matrix, index=movies['movieId'], columns=movies['movieId'])
cos_sim_df.to_pickle(COS_SIM_PATH)


# ------------------------------
# 3. Evaluate Content-Based Filtering
# ------------------------------
# evaluate_content_precision_at_k(ratings, rating_matrix, cos_sim_df, k=10)

# ------------------------------
# 4. SVD Decomposition
# ------------------------------
rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Split ratings into train/test
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

# Build training rating matrix WITHOUT fillna(0)
train_matrix = train_df.pivot(index='userId', columns='movieId', values='rating')

# Prepare matrix for SVD (mean-center with NaN-safe logic)
R_train = train_matrix.values

# Compute user means while ignoring NaNs
user_means = np.nanmean(R_train, axis=1).reshape(-1, 1)

# Subtract user mean only from rated movies (leave NaNs untouched)
R_demeaned = np.where(np.isnan(R_train), 0, R_train - user_means)

# Apply SVD
U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)

# Reconstruct ratings and re-add user means
svd_pred_train = np.dot(np.dot(U, sigma), Vt) + user_means
svd_pred_df_train = pd.DataFrame(svd_pred_train, index=train_matrix.index, columns=train_matrix.columns)

# Predict ratings for test set
def predict_rating(row):
    try:
        return svd_pred_df_train.loc[row['userId'], row['movieId']]
    except KeyError:
        return np.nan  # Handle new users or movies not in training set

test_df['predicted'] = test_df.apply(predict_rating, axis=1)

# Drop rows where we couldn't predict
test_df_clean = test_df.dropna(subset=['predicted'])
test_df_clean['predicted'] = test_df_clean['predicted'].clip(0.5, 5.0)
test_df_clean['predicted'] = (2 * test_df_clean['predicted']).round() / 2

# Evaluate
rmse = np.sqrt(mean_squared_error(test_df_clean['rating'], test_df_clean['predicted']))
mae = mean_absolute_error(test_df_clean['rating'], test_df_clean['predicted'])

print("\n📊 SVD Evaluation Results:")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ MAE:  {mae:.4f}")


# Merge test_df_clean with movie titles
movies_subset = movies.set_index('movieId')

# Select relevant columns
preview_df = test_df_clean[['userId', 'movieId', 'rating', 'predicted']].copy()
preview_df['title'] = preview_df['movieId'].map(movies_subset['title'])

# Show first two rows
print(preview_df[['userId', 'title', 'rating', 'predicted']].head(50))
