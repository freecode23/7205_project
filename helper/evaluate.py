import numpy as np
from tqdm import tqdm


def evaluate_scaled_predictions_precision(test_df, train_df, sim_df, k=10, min_rated=5):
    precision_scores = []
    users = test_df['userId'].unique()

    for user_id in tqdm(users, desc=f"Evaluating Scaled Rating-Based Recommender (Precision@{k})"):
        user_test_ratings = test_df[test_df['userId'] == user_id]
        high_rated = user_test_ratings[user_test_ratings['rating'] >= 4.0]

        if len(high_rated) < min_rated:
            continue

        test_movie = high_rated.sample(1, random_state=42)['movieId'].values[0]

        # Predict ratings for all other movies for this user
        unrated_movies = test_df[(test_df['userId'] == user_id) & (test_df['movieId'] != test_movie)]
        unrated_movies = unrated_movies.copy()

        unrated_movies['scaled_score'] = unrated_movies.progress_apply(
            lambda row: get_content_score(row['userId'], row['movieId'], train_df, sim_df),
            axis=1
        )

        # Scale those scores
        scaled = unrated_movies['scaled_score'].dropna()
        if len(scaled) == 0:
            continue

        min_val, max_val = scaled.min(), scaled.max()
        unrated_movies['scaled_score'] = 4 * (unrated_movies['scaled_score'] - min_val) / (max_val - min_val) + 1
        # 💡 NEW: Filter only those predicted to be ≥ 4.0
        filtered_recs = unrated_movies[unrated_movies['scaled_score'] >= 4.0]


        # If there are not enough, skip this user
        if len(filtered_recs) < k:
            continue

        top_k_recs = filtered_recs.sort_values(by='scaled_score', ascending=False)['movieId'].head(k).tolist()
        precision = 1 if test_movie in top_k_recs else 0
        precision_scores.append(precision)

    mean_precision = np.mean(precision_scores)
    print(f"\n🎯 Scaled Prediction-Based Precision@{k}: {mean_precision:.4f} over {len(precision_scores)} users")
    return mean_precision


def get_content_recommendations(user_id, rating_matrix, cos_sim_df, top_k=10):
    # Get the ratings given by the user
    user_ratings = rating_matrix.loc[user_id]

    # Find the movies the user has rated (watched)
    rated_movies = user_ratings[user_ratings > 0].index

    # Find the movies the user has NOT rated (unwatched)
    unrated_movies = user_ratings[user_ratings == 0].index

    # Define how to compute a content-based score for an unrated movie
    def content_score(movie_id):
        # Get similarity scores between this movie and all movies the user has rated
        sim_scores = cos_sim_df.loc[movie_id, rated_movies]

        # Get the actual ratings the user gave to those movies
        user_scores = user_ratings.loc[rated_movies]

        if sim_scores.sum() == 0:
            return 0

        # Compute the weighted score: similarity * user rating
        return np.dot(sim_scores, user_scores)

    # Compute scores for all unrated movies
    scores = {movie: content_score(movie) for movie in unrated_movies}

    # Sort the movies by score in descending order and select the top K
    top_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Return just the movie IDs of the top recommendations
    return [movie_id for movie_id, score in top_recs]


def evaluate_content_precision_at_k(ratings, rating_matrix, cos_sim_df, k=10, min_rated=5):
    """
    For each user, recommend k movies. Then check: did the movie they really liked appear in the top k.
    """
    precision_scores = []
    users = rating_matrix.index

    for user_id in tqdm(users, desc=f"Evaluating Content-Based Recommender (Precision@{k})"):
        user_data = ratings[ratings['userId'] == user_id]
        high_rated = user_data[user_data['rating'] >= 4.0]

        if len(high_rated) < min_rated:
            continue

        test_movie = high_rated.sample(1, random_state=42)['movieId'].values[0]

        temp_user_ratings = rating_matrix.loc[user_id].copy()
        temp_user_ratings.loc[test_movie] = 0
        temp_rating_matrix = rating_matrix.copy()
        temp_rating_matrix.loc[user_id] = temp_user_ratings

        recs = get_content_recommendations(user_id, temp_rating_matrix, cos_sim_df, top_k=k)
        precision = 1 if test_movie in recs else 0
        precision_scores.append(precision)

    mean_precision = np.mean(precision_scores)
    print(f"\n🎯 Content-Based Precision@{k}: {mean_precision:.4f} over {len(precision_scores)} users")
    return mean_precision


def get_content_score(user_id, target_movie_id, train_df, sim_df):
    """
    For a given (user_id, target_movie_id):
    Get all movies the user has rated

    For each of those movies:
        - Check its genre similarity with the target movie
        - Multiply that similarity score by the user’s actual rating of that movie
        - Take a weighted average of those scores — where more similar movies get more influence
    """
    # Get all the movies this user has rated
    user_rated = train_df[train_df['userId'] == user_id]
    
    scores = []
    similarities = []
    
    # Loop through each rated movie + its rating
    for _, row in user_rated.iterrows():
        
        rated_movie_id = row['movieId']
        rating = row['rating']

        # For each movie the user rated:
        # check if it has a genre similarity score with the target_movie_id
        if rated_movie_id in sim_df.columns and target_movie_id in sim_df.index:
            sim = sim_df.at[target_movie_id, rated_movie_id]

            # Build a weighted sum:
            scores.append(sim * rating)

            # If a movie is more similar to the target, its rating gets more influence
            similarities.append(sim)
    
    if not scores or sum(similarities) == 0:
        return np.nan
    
    return sum(scores) / sum(similarities)
