import numpy as np
from tqdm import tqdm

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
