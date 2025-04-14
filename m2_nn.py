import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tqdm import tqdm
from helper.evaluate import ndcg_at_k
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

# Load the pre-saved split
train_df = pd.read_csv('./results/train.csv')
test_df = pd.read_csv('./results/test.csv')

# Apply same user/movie mapping
#train_df['user_idx'] = train_df['userId'].map(user_mapper)
#train_df['movie_idx'] = train_df['movieId'].map(movie_mapper)
#test_df['user_idx'] = test_df['userId'].map(user_mapper)
#test_df['movie_idx'] = test_df['movieId'].map(movie_mapper)

# Normalize ratings to 0–1
train_df['rating_norm'] = train_df['rating'] / 5.0
test_df['rating_norm'] = test_df['rating'] / 5.0

# Define training and test data
X_train = list(zip(train_df['user_idx'], train_df['movie_idx']))
y_train = train_df['rating_norm'].values

X_test = list(zip(test_df['user_idx'], test_df['movie_idx']))
y_test = test_df['rating_norm'].values

num_users = ratings['user_idx'].nunique()
num_movies = ratings['movie_idx'].nunique()

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
rmse = np.sqrt(np.mean((y_test_rescaled - y_pred_rescaled)**2))

print(f"Mean Absolute Error (MAE) on Test Set: {mae:.4f}")
print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse:.4f}")

# ------------------------------
# 7. Create predictions dataframe
# ------------------------------
#predictions_df = pd.DataFrame({
 #   'userId': [user for user, movie in X_test],
  #  'movieId': [movie for user, movie in X_test],
   # 'actual_rating': y_test_rescaled,
    #'predicted_rating': y_pred_rescaled.flatten(),
#})

# 创建预测结果 DataFrame（nn 模型）
predictions_df = pd.DataFrame({
    'user_idx': [user for user, movie in X_test],
    'movie_idx': [movie for user, movie in X_test],
    'actual_rating': y_test_rescaled,
    'predicted_rating': y_pred_rescaled.flatten(),
})


test_df = predictions_df.copy()

# ------------------------------
# 8. Evaluate HR@K and NDCG@K on the test set
# ------------------------------
# 这里假设我们已经生成了 predictions_df
# 如果你之前生成的 DataFrame 是 predictions_df，就可以直接使用；否则，可以重新命名，例如：
test_df = predictions_df.copy()

# 定义 Top-K 的值
K = 10

# 获取所有测试用户的ID
user_ids = test_df['user_idx'].unique()

hit_scores = []
ndcg_scores = []

print(f"Evaluating sampled HR@{K}...")
for user_id in tqdm(user_ids):
    # 获取该用户在测试集中的数据
    user_data_test = test_df[test_df['user_idx'] == user_id]
    if user_data_test.empty:
        continue

    # 定义 ground truth：取真实评分≥4.0的电影作为用户真正喜欢的电影
    high_rated = user_data_test[user_data_test['actual_rating'] >= 4.0]
    ground_truth_high_rated = high_rated['movie_idx'].values
    if len(ground_truth_high_rated) == 0:
        continue

    # 根据预测的评分排序，选取前 K 个推荐的电影
    top_k_recs = user_data_test.sort_values(by='predicted_rating', ascending=False).head(K)
    recommended_movies = top_k_recs['movie_idx'].values

    # 计算 Hit@K：如果推荐列表中至少包含一个真实喜欢的电影，则命中1，否则为0
    hit = 1 if np.intersect1d(recommended_movies, ground_truth_high_rated).size > 0 else 0
    hit_scores.append(hit)

    # 计算 NDCG@K：调用 ndcg_at_k 函数（需要在 helper.evaluate 中定义或导入）
    ndcg = ndcg_at_k(recommended_movies, ground_truth_high_rated, K)
    ndcg_scores.append(ndcg)

# 输出所有用户的平均 HR@K 和 NDCG@K 值
print(f"\n✅ Sampled HR@{K}: {np.mean(hit_scores):.4f}")
print(f"✅ Sampled NDCG@{K}: {np.mean(ndcg_scores):.4f}")

# If needed, merge with movies DataFrame.
# Note: The movies DataFrame still contains the original movieId.
# You can create a reverse mapping for movie ids if necessary.
# For simplicity, here we merge on the mapped movieId if movies DataFrame was updated accordingly.
predictions_df = predictions_df.merge(movies[['movie_idx', 'title']], left_on='movie_idx', right_on='movie_idx', how='left')

print("\n🎬 Preview of Real vs Predicted Ratings:")
print(predictions_df[['user_idx', 'title', 'actual_rating', 'predicted_rating']].head(10))

predictions_df.to_pickle("./results/nn_pred_df.pkl")