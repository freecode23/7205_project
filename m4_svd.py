import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For SVD
from surprise.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from surprise import SVDpp
from surprise.model_selection import GridSearchCV


class VerboseSVDpp(SVDpp):
    def __init__(self, **kwargs):
        kwargs['verbose'] = True  # Force verbosity
        super().__init__(**kwargs)


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

# ------------------------------
# 2. Prepare data for Surprise
# ------------------------------
# Prepare data for Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# ------------------------------
# 3. Grid Search for Best SVD Hyperparameters
# ------------------------------
# Define parameter grid
param_grid = {
    'n_factors': [10, 20, 30, 40],          # K - number of latent features (a.k.a. latent factors)
    'reg_all': [0.05, 0.1, 0.2, 0.4],       # Regularization Strength: Controls how much the model penalizes large weights (to prevent overfitting)
    'lr_all': [0.005, 0.007, 0.01, 0.015]   # Learning rate: This controls how fast the model updates weights during training.
}
print("Setting up Grid Search:")
# Set up GridSearchCV
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, joblib_verbose=2, n_jobs=-1) 

gs.fit(data)

# Best score and params
print("\n🔍 Best Grid Search Results:")
print(f"✅ Best RMSE: {gs.best_score['rmse']:.4f}")
print(f"✅ Best MAE:  {gs.best_score['mae']:.4f}")
print(f"📌 Best Params: {gs.best_params['rmse']}")

# ------------------------------
# 4. Preview Predictions vs Actual
# ------------------------------

# Use a reasonable value for k to preview predictions
best_params = gs.best_params['rmse']
algo = SVDpp(**best_params)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
algo.fit(trainset)
predictions = algo.test(testset)

# Convert predictions to DataFrame
pred_df = pd.DataFrame([{
    'userId': int(pred.uid),
    'movieId': int(pred.iid),
    'rating': pred.r_ui,
    'predicted': round(pred.est * 2) / 2  # Round to nearest 0.5

} for pred in predictions])
pred_df['predicted'] = pred_df['predicted'].clip(0.5, 5.0)

# Merge with movie titles
movies_subset = movies.set_index('movieId')
pred_df['title'] = pred_df['movieId'].map(movies_subset['title'])
pred_df.to_pickle(SVD_PRED_PATH)

# Show first few predictions
print("\n🎬 Preview of Real vs Predicted Ratings:")
print(pred_df[['userId', 'title', 'rating', 'predicted']].head(10))
