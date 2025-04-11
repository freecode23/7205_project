import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
tqdm.pandas()  # enables .progress_apply()

# This program evaluate:
# model 3: Content-Based Filtering and model 
# model 4: SVD (Collaborative Filtering)
# ------------------------------
# 1. Define Path to load SVD or Content Filtering Result
# ------------------------------
RESULTS_PATH = "./results"
os.makedirs(RESULTS_PATH, exist_ok=True)

PRED_PATH = os.path.join(RESULTS_PATH, "content_pred_df.pkl")
PRED_PATH = os.path.join(RESULTS_PATH, "svd_pred_df.pkl")
test_df_clean = pd.read_pickle(PRED_PATH)

# ------------------------------
# 2. Make top 10 recommendation for a user and see the actual rating.
# ------------------------------
# 1. Pick a user with many ratings
user_id = test_df_clean['userId'].value_counts().idxmax()

# 2. Get all the movies rated by this user
user_data_test = test_df_clean[test_df_clean['userId'] == user_id]

# 3. From those, pick the ones they rated highly (e.g., ≥ 4.0)
high_rated = user_data_test[user_data_test['rating'] >= 4.0]
print("\nHigh_rated Movies\n", high_rated)


# 4. Select top-k recommendations for the user
top_k = 10  # Can change to other numbers based on your choice
top_recs = test_df_clean[test_df_clean['userId'] == user_id].sort_values(by='predicted', ascending=False).head(top_k)
print("\nTop-k Recommended Movies\n", top_recs)


# ------------------------------
# 5. Calculate Precision and Recall fpr a single user.
# ------------------------------
# 5.1. Precision: of the top-k recommendations, how many have rating ≥ 4.0 (i.e., are relevant)?
top_recs_vals = top_recs['movieId'].values
high_rated_vals = high_rated['movieId'].values
intersection = np.intersect1d(top_recs_vals, high_rated_vals)
print("\nTop rev vals\n", top_recs_vals)

# Precision = (Number of relevant recommended items) / (Total recommended items)
precision = len(intersection) / len(top_recs_vals)

# Print Precision and Recall
print(f"\n🔍 Precision: {precision:.4f}")


# ------------------------------
# 6. Calculate Precision and Recall for each user
# ------------------------------
user_ids = test_df_clean['userId'].unique()  # Get all unique users
precision_scores = []
recall_scores = []

# Loop over each user to compute precision and recall
for user_id in user_ids:
    # Get all the movies rated by this user
    user_data_test = test_df_clean[test_df_clean['userId'] == user_id]

    # Get movies with a rating ≥ 4.0 (highly rated)
    high_rated = user_data_test[user_data_test['rating'] >= 4.0]

    # Get top-k recommended movies for this user
    top_k = 10  # Can change this number based on your requirement
    top_recs = test_df_clean[test_df_clean['userId'] == user_id].sort_values(by='predicted', ascending=False).head(top_k)

    # Precision: of the top-k recommendations, how many have rating ≥ 4.0 (i.e., are relevant)?
    recommended_movies = top_recs['movieId'].values
    relevant_movies = high_rated['movieId'].values
    intersection = np.intersect1d(recommended_movies, relevant_movies)
    
    # Precision = (Number of relevant recommended items) / (Total recommended items)
    precision = len(intersection) / len(recommended_movies)


    # Append the precision and recall for this user
    precision_scores.append(precision)


# Calculate the average precision and recall over all users
avg_precision = np.mean(precision_scores)


# Print the results
print(f"\n🔍 Average Precision: {avg_precision:.4f}")

