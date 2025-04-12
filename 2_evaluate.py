import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import dump
from helper.evaluate import dcg_at_k, ndcg_at_k
from tqdm import tqdm
tqdm.pandas()

# This program evaluate:
# model 3: Content-Based Filtering and model 
# model 4: SVD (Collaborative Filtering)
# ------------------------------
# 1. Define Path to load SVD or Content Filtering Result
# ------------------------------
RESULTS_PATH = "./results"
RATINGS_FILEPATH = './IMDB-Dataset/ratings.csv'
PRED_PATH = os.path.join(RESULTS_PATH, "content_pred_df.pkl")
# PRED_PATH = os.path.join(RESULTS_PATH, "svd_pred_df.pkl")
K=10


# ------------------------------
# 2. Get train and test df with predicted ratings.
# ------------------------------
ratings = pd.read_csv(RATINGS_FILEPATH)
test_df_clean = pd.read_pickle(PRED_PATH)


# ------------------------------
# 3. Get train and test df with predicted ratings.
# ------------------------------
all_movie_ids = set(ratings['movieId'].unique())
user_ids = test_df_clean['userId'].unique()
# user_ids = [2]


# ------------------------------
# 4. Calculate HR@K with sampled ranking for each user.
# ------------------------------
hit_scores = []
ndcg_scores = []
print(f"Evaluating sampled HR@{K}...")
for user_id in tqdm(user_ids):

    # Get test data for this user
    user_data_test = test_df_clean[test_df_clean['userId'] == user_id]
    if user_data_test.empty:
        continue

    # Get "relevant" movies actual user ratings ≥ 4.0 (ground truth)
    high_rated = user_data_test[user_data_test['rating'] >= 4.0]
    ground_truth_high_rated = high_rated['movieId'].values
    if len(ground_truth_high_rated) == 0:
        continue
    
    # Get Top-K recommended movies based on predicted rating (model’s guesses)
    top_k = user_data_test.sort_values(by='predicted', ascending=False).head(K)
    recommended_movies = top_k['movieId'].values

    # Hit@K: was at least one relevant item in top-K?
    # Print detailed debug info
    # print(f"\n👤 User ID = {user_id}")
    # print("🎯 Ground-truth high-rated (actual rating ≥ 4.0):")
    # print(user_data_test[user_data_test['rating'] >= 4.0][['movieId', 'rating']].to_string(index=False))

    # print("\n🤖 Top-K Recommended Movies (sorted by predicted rating):")
    # print(top_k[['movieId', 'rating', 'predicted']].to_string(index=False))
    hit = 1 if np.intersect1d(recommended_movies, ground_truth_high_rated).size > 0 else 0
    hit_scores.append(hit)

    # NDCG@K: how well are the relevant items ranked?
    ndcg = ndcg_at_k(recommended_movies, ground_truth_high_rated, K)
    ndcg_scores.append(ndcg)

# Sampled HR@K
print(f"\n✅ Sampled HR@{K}: {np.mean(hit_scores):.4f}")
print(f"✅ Sampled NDCG@{K}: {np.mean(ndcg_scores):.4f}")
