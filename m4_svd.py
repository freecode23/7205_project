import os
import pandas as pd

# For SVD
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
from surprise import dump


# ------------------------------
# 0. Load movie data and user rating data
# ------------------------------
MOVIES_FILEPATH = './IMDB-Dataset/movies.csv'
RATINGS_FILEPATH = './IMDB-Dataset/ratings.csv'
# MOVIES_FILEPATH = './MovieLens-Dataset/movies.csv'
# RATINGS_FILEPATH = './MovieLens-Dataset/ratings.csv'

movies = pd.read_csv(MOVIES_FILEPATH)
ratings = pd.read_csv(RATINGS_FILEPATH)

# ------------------------------
# 1. Define Path for results
# ------------------------------
RESULTS_PATH = "./results"
os.makedirs(RESULTS_PATH, exist_ok=True)
SVD_PRED_PATH = os.path.join(RESULTS_PATH, "svd_pred_df.pkl")
MODEL_PATH = os.path.join(RESULTS_PATH, "svd_model.pkl")

# ------------------------------
# 2. Prepare data for Surprise
# ------------------------------
# 1. Split trainset and testset.
print("Split trainset and testset..")
train_df = ratings.sample(frac=0.8, random_state=42)
test_df = ratings.drop(train_df.index)

# 2. Save it for all models
train_df.to_csv('./results/train.csv', index=False)
test_df.to_csv('./results/test.csv', index=False)

# 3. Convert to dataset for GridSearch
print("Convert to Dataset..")
reader = Reader(rating_scale=(0.5, 5.0))
train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
test_data = Dataset.load_from_df(test_df[['userId', 'movieId', 'rating']], reader)

# 4. Convert to Surprise train and testsest
print("Convert to Surprise train and testset..")
trainset = train_data.build_full_trainset()
testset = list(zip(test_df['userId'], test_df['movieId'], test_df['rating']))

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
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, joblib_verbose=3, n_jobs=-1) 
gs.fit(train_data)

# Best score and params
print("\n🔍 Best Grid Search Results:")
print(f"✅ Best RMSE: {gs.best_score['rmse']:.4f}")
print(f"✅ Best MAE:  {gs.best_score['mae']:.4f}")
print(f"📌 Best Params: {gs.best_params['rmse']}")

# ------------------------------
# 4. Preview Predictions vs Actual
# ------------------------------
# Use the best param to predict the test dataset.
best_params = gs.best_params['rmse']
algo = SVD(**best_params)
algo.fit(trainset)
dump.dump(MODEL_PATH, algo=algo)
predictions = algo.test(testset)

# Convert predictions to DataFrame
pred_df = pd.DataFrame([{
    'userId': int(pred.uid),
    'movieId': int(pred.iid),
    'rating': pred.r_ui,
    'predicted': round(pred.est * 2) / 2  # Round to nearest 0.5

} for pred in predictions])
pred_df['predicted'] = pred_df['predicted'].clip(0.5, 5.0)

# Merge prediction with titles and save the df.
movies_subset = movies.set_index('movieId')
pred_df['title'] = pred_df['movieId'].map(movies_subset['title'])
pred_df.to_pickle(SVD_PRED_PATH)

# Show first few predictions.
print("\n🎬 Preview of Real vs Predicted Ratings:")
print(pred_df[['userId', 'title', 'rating', 'predicted']].head(10))
previewUser = pred_df[pred_df['userId'] == 33]
print(previewUser)