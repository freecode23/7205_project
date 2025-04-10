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

from helper.evaluate import get_content_score, evaluate_scaled_predictions_precision

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
# 1. Define Path to load data frame.
# ------------------------------
RESULTS_PATH = "./results"
os.makedirs(RESULTS_PATH, exist_ok=True)
COS_SIM_PATH = os.path.join(RESULTS_PATH, "cos_sim_df.pkl")
PRED_PATH = os.path.join(RESULTS_PATH, "content_pred_df.pkl")

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

train_df = pd.DataFrame(trainset.build_testset(), columns=['userId', 'movieId', 'rating'])
cos_sim_df = pd.read_pickle(COS_SIM_PATH)
test_df_clean = pd.read_pickle(PRED_PATH)
print("\ntest_df_clean\n", test_df_clean[test_df_clean['rating'] >= 4.0])

# ------------------------------
# 2. Make top 10 recommendation and see the actual rating.
# ------------------------------
# 1. Pick a user with many ratings
user_id = test_df_clean['userId'].value_counts().idxmax()

# 2. Get all the movies rated by this user
user_data_test = test_df_clean[test_df_clean['userId'] == user_id]
print("\nuser_data\n", user_data_test)

# 3. From those, pick the ones they rated highly (e.g., ≥ 4.0)
high_rated = user_data_test[user_data_test['content_pred_scaled'] >= 4.0]
print("\nhigh_rated\n", high_rated)
