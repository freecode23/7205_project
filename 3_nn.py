import os
import pandas as pd
import numpy as np
import tensorflow as tf
print(tf.__version__)

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split

# Load the movie data and ratings data
# MOVIES_FILEPATH = './IMDB-Dataset/movies.csv'
# RATINGS_FILEPATH = './IMDB-Dataset/ratings.csv'
# movies = pd.read_csv(MOVIES_FILEPATH)
# ratings = pd.read_csv(RATINGS_FILEPATH)

# Preprocessing the data
# user_ids = ratings['userId'].values
# movie_ids = ratings['movieId'].values
# ratings_values = ratings['rating'].values

# # Number of unique users and movies
# num_users = ratings['userId'].nunique()
# num_movies = ratings['movieId'].nunique()

# # Normalizing the ratings (optional, depending on the dataset)
# ratings_values = ratings_values / 5.0  # Normalize to a scale of 0 to 1

# # Split data into train and test sets (80-20 split)
# X_train, X_test, y_train, y_test = train_test_split(
#     list(zip(user_ids, movie_ids)), ratings_values, test_size=0.2, random_state=42
# )

# # Define the Neural Network model
# embedding_size = 50  # Size of the embeddings for users and movies

# # User embedding
# user_input = Input(shape=(1,), name='user')
# user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, input_length=1)(user_input)
# user_embedding = Flatten()(user_embedding)

# # Movie embedding
# movie_input = Input(shape=(1,), name='movie')
# movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size, input_length=1)(movie_input)
# movie_embedding = Flatten()(movie_embedding)

# # Concatenate user and movie embeddings
# concat = tf.keras.layers.concatenate([user_embedding, movie_embedding])

# # Add fully connected layers
# dense = Dense(128, activation='relu')(concat)
# dense = Dropout(0.2)(dense)
# dense = Dense(64, activation='relu')(dense)
# output = Dense(1, activation='sigmoid')(dense)  # Use sigmoid for normalization to 0-1 range

# # Compile the model
# model = Model(inputs=[user_input, movie_input], outputs=output)
# model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mae'])

# # Train the model
# model.fit([np.array([user for user, movie in X_train]), np.array([movie for user, movie in X_train])],
#           y_train, epochs=10, batch_size=256, validation_data=([np.array([user for user, movie in X_test]), np.array([movie for user, movie in X_test])], y_test))

# # ------------------------------
# # 5. Make predictions for the test set
# # ------------------------------

# # Predict ratings for the test set
# y_pred = model.predict([np.array([user for user, movie in X_test]), np.array([movie for user, movie in X_test])])

# # ------------------------------
# # 6. Calculate MAE for evaluation
# # ------------------------------

# # Rescale predictions back to original rating scale (1 to 5)
# y_pred_rescaled = y_pred * 5.0
# y_test_rescaled = y_test * 5.0

# mae = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))  # Mean Absolute Error
# print(f"Mean Absolute Error (MAE) on Test Set: {mae:.4f}")

# # ------------------------------
# # 7. Create predictions dataframe
# # ------------------------------

# # Create a DataFrame with the predicted ratings
# predictions_df = pd.DataFrame({
#     'userId': [user for user, movie in X_test],
#     'movieId': [movie for user, movie in X_test],
#     'actual_rating': y_test_rescaled,
#     'predicted_rating': y_pred_rescaled.flatten(),
# })

# # Merge with movie titles
# predictions_df = predictions_df.merge(movies[['movieId', 'title']], on='movieId', how='left')

# # Show the first few predictions
# print("\n🎬 Preview of Real vs Predicted Ratings:")
# print(predictions_df[['userId', 'title', 'actual_rating', 'predicted_rating']].head(10))

# # Optionally, save the DataFrame
# predictions_df.to_pickle("./results/nn_pred_df.pkl")
