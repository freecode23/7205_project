import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # optional, for multi-core use
tqdm.pandas()  # enables .progress_apply()

# For Content-Based filtering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from helper.evaluate import get_content_score

# For SVD
from surprise.model_selection import train_test_split
from surprise import Dataset, Reader


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
PRED_PATH = os.path.join(RESULTS_PATH, "content_pred_df.pkl")

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
# 3. Creating predicted ratings for the test set — based on content (genres) and what the user has rated before.
# ------------------------------
# Assuming 'data' is your surprise Dataset object
reader = Reader(rating_scale=(0.5, 5.0))

# Remove any rows that don't have valid ratings (e.g., NaN or 0)
ratings_clean = ratings.dropna(subset=['rating'])
ratings_clean = ratings_clean[ratings_clean['rating'] > 0]

# Then pass to Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_clean[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Convert trainset to a DataFrame for use in get_content_score
train_df = pd.DataFrame(trainset.build_testset(), columns=['userId', 'movieId', 'rating'])

# Create a DataFrame version of the testset
test_df = pd.DataFrame(testset, columns=['userId', 'movieId', 'rating'])

# Apply get_content_score to your test set
test_df['predicted_raw'] = test_df.progress_apply(
    lambda row: get_content_score(row['userId'], row['movieId'], train_df, cos_sim_df),
    axis=1
)

# Drop predictions that couldn't be made.
test_df_clean = test_df.dropna(subset=['predicted_raw'])

# ------------------------------
# 4. Rescales predictions to match 1.0–5.0 rating scale
# ------------------------------
min_pred = test_df_clean['predicted_raw'].min()
max_pred = test_df_clean['predicted_raw'].max()
test_df_clean['predicted'] = 4 * (test_df_clean['predicted_raw'] - min_pred) / (max_pred - min_pred) + 1


# Save the df with prediction column.
test_df_clean.to_pickle(PRED_PATH)
rmse_scaled = root_mean_squared_error(test_df_clean['rating'], test_df_clean['predicted'])
mae_scaled = mean_absolute_error(test_df_clean['rating'], test_df_clean['predicted'])

print(f"\n📊 Scaled Content-Based Evaluation:")
print(f"✅ RMSE (scaled): {rmse_scaled:.4f}")
print(f"✅ MAE (scaled):  {mae_scaled:.4f}")


# ------------------------------
# 4. Preview Predictions vs Actual
# ------------------------------
# Merge with movie titles for display
movies_subset = movies.set_index('movieId')
test_df_clean['title'] = test_df_clean['movieId'].map(movies_subset['title'])

# Round predicted rating to nearest 0.5 if you want to match SVD-style display
test_df_clean['predicted'] = (2 * test_df_clean['predicted']).round() / 2
test_df_clean['predicted'] = test_df_clean['predicted'].clip(0.5, 5.0)


# Preview first few predictions
preview = test_df_clean[['userId', 'title', 'rating', 'predicted']].head(10)

print("\n🎬 Preview of Real vs Predicted Ratings (Content-Based):")
print(preview.to_string(index=False))

