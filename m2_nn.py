import os
import pandas as pd
import numpy as np
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the movie data and ratings data
MOVIES_FILEPATH = './IMDB-Dataset/movies.csv'
RATINGS_FILEPATH = './IMDB-Dataset/ratings.csv'
movies = pd.read_csv(MOVIES_FILEPATH)
ratings = pd.read_csv(RATINGS_FILEPATH)

# Map original userId and movieId to new continuous indices
user_mapper = {old: new for new, old in enumerate(ratings['userId'].unique())}
movie_mapper = {old: new for new, old in enumerate(ratings['movieId'].unique())}

# Add mapped indices to the ratings DataFrame
ratings['user_idx'] = ratings['userId'].map(user_mapper)
ratings['movie_idx'] = ratings['movieId'].map(movie_mapper)

# **Add mapped indices to the movies DataFrame as well**
movies['movie_idx'] = movies['movieId'].map(movie_mapper)

# Preprocessing the data using mapped indices
user_ids = ratings['user_idx'].values
movie_ids = ratings['movie_idx'].values
ratings_values = ratings['rating'].values

# Get number of unique users and movies after mapping
num_users = ratings['user_idx'].nunique()
num_movies = ratings['movie_idx'].nunique()

# Normalize the ratings to [0, 1]
ratings_values = ratings_values / 5.0

# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    list(zip(user_ids, movie_ids)), ratings_values, test_size=0.2, random_state=42
)

# Define the Neural Network model
embedding_size = 50  # Size of the embeddings for users and movies

# User embedding
user_input = Input(shape=(1,), name='user')
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
user_embedding = Flatten()(user_embedding)

# Movie embedding
movie_input = Input(shape=(1,), name='movie')
movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size)(movie_input)
movie_embedding = Flatten()(movie_embedding)

# Concatenate user and movie embeddings
concat = tf.keras.layers.concatenate([user_embedding, movie_embedding])

# Add fully connected layers
dense = Dense(128, activation='relu')(concat)
dense = Dropout(0.2)(dense)
dense = Dense(64, activation='relu')(dense)
output = Dense(1, activation='sigmoid')(dense)  # Use sigmoid for normalization to 0-1 range

# Compile the model (using the correct learning_rate parameter)
model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Train the model using mapped indices
model.fit(
    [np.array([user for user, movie in X_train]), np.array([movie for user, movie in X_train])],
    y_train, epochs=10, batch_size=256,
    validation_data=(
        [np.array([user for user, movie in X_test]), np.array([movie for user, movie in X_test])],
        y_test
    )
)

# ------------------------------
# 5. Make predictions for the test set
# ------------------------------
y_pred = model.predict([np.array([user for user, movie in X_test]), np.array([movie for user, movie in X_test])])

# ------------------------------
# 6. Calculate MAE for evaluation
# ------------------------------
y_pred_rescaled = y_pred * 5.0
y_test_rescaled = y_test * 5.0

mae = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))
print(f"Mean Absolute Error (MAE) on Test Set: {mae:.4f}")

# ------------------------------
# 7. Create predictions dataframe
# ------------------------------
predictions_df = pd.DataFrame({
    'userId': [user for user, movie in X_test],
    'movieId': [movie for user, movie in X_test],
    'actual_rating': y_test_rescaled,
    'predicted_rating': y_pred_rescaled.flatten(),
})

# If needed, merge with movies DataFrame.
# Note: The movies DataFrame still contains the original movieId.
# You can create a reverse mapping for movie ids if necessary.
# For simplicity, here we merge on the mapped movieId if movies DataFrame was updated accordingly.
predictions_df = predictions_df.merge(movies[['movie_idx', 'title']], left_on='movieId', right_on='movie_idx', how='left')

print("\n🎬 Preview of Real vs Predicted Ratings:")
print(predictions_df[['userId', 'title', 'actual_rating', 'predicted_rating']].head(10))

predictions_df.to_pickle("./results/nn_pred_df.pkl")