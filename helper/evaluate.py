import numpy as np
from tqdm import tqdm



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
