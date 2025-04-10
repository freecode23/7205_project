import os
import pandas as pd


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


# ------------------------------
# 3. Calculate the number of missing ratings
# ------------------------------
# First, create a full user-item matrix to identify missing ratings
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Find the total number of possible user-item interactions
total_user_item_pairs = user_item_matrix.shape[0] * user_item_matrix.shape[1]

# Find the number of ratings present (non-NaN values)
rated_count = user_item_matrix.count().sum()

# Calculate the number of missing ratings
missing_count = total_user_item_pairs - rated_count

print(f"Total possible user-item pairs: {total_user_item_pairs}")
print(f"Number of ratings present: {rated_count}")
print(f"Number of missing ratings: {missing_count}")
