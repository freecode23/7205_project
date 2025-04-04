import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import seaborn as sns
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# movie_data_read csv file
movie_data = pd.read_csv('./IMDB-Dataset/movies.csv')

#show information of the movie data
print(movie_data.describe(include='all'))

#show structure and basic info of the movie data
print(movie_data.info())

# show the first 6 lines of the movie data¨
print(movie_data.head(6))

#rating_data_read csv file
rating_data = pd.read_csv('./IMDB-Dataset/ratings.csv')

#show information of the rating data
print(rating_data.describe(include='all'))

#show structure and basic info of the rating data
print(rating_data.info())

# show the first 6 lines of the rating data
print(rating_data.head(6))


# Split genes column
movie_genre2 = movie_data['genres'].str.split('|', expand=True)

# Define all possible types
list_genre = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
              "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
              "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "IMAX"]

# Create a zero matrix (with rows representing the number of movies and columns representing the number of types)
genre_mat2 = pd.DataFrame(0, index=np.arange(len(movie_genre2)), columns=list_genre)

# Traverse movie_genre2 and fill in one hot encoding
for index, row in movie_genre2.iterrows():
    for genre in row.dropna():  # 去除 NaN 值
        genre_mat2.at[index, genre] = 1
                

# Output data 
print(genre_mat2.info())
print(genre_mat2.head(6))  

# Combine the first two columns of movie_data with genre_mat2
SearchMatrix = pd.concat([movie_data.iloc[:, :2], genre_mat2], axis=1)

# Output data
print(SearchMatrix.info())
print(SearchMatrix.head(6))

# Create rating matrix
rating_matrix = rating_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Convert to Sparse Matrix
sparse_rating_matrix = csr_matrix(rating_matrix.values)

# print(rating_matrix.info())
print(rating_matrix)
print(sparse_rating_matrix)

# Perform matrix decomposition using SVD
U, sigma, Vt = svds(sparse_rating_matrix)

sigma = np.diag(sigma)

# Take the top 2 singular values (for dimensionality reduction)
k = 2
U_k = U[:, :k]   # User feature vectors
Vt_k = Vt[:k, :] # Item feature vectors

# Compute cosine similarity between users
num_users = U_k.shape[0]
user_similarity = np.zeros((num_users, num_users))

for i in range(num_users):
    for j in range(num_users):
        user_similarity[i, j] = 1 - cosine(U_k[i], U_k[j])  # Cosine similarity

print("User similarity matrix (based on SVD with dimensionality reduction):\n", np.round(user_similarity, 2))


# Assume rating_matrix is a NumPy matrix or Pandas DataFrame (rows are users, columns are movies)
movie_views = np.sum(rating_matrix > 0, axis=0)  # Calculate number of views per movie (non-zero ratings)

# Create DataFrame
table_views = pd.DataFrame({
    "movie": rating_matrix.columns,  # Movie ID
    "views": movie_views              # View count
})

# Sort by number of views (descending)
table_views = table_views.sort_values(by="views", ascending=False).reset_index(drop=True)

# Add movie title column (from movie_data)
table_views["title"] = table_views["movie"].map(movie_data.set_index("movieId")["title"])

# Output top 6 rows
print(table_views.head(6))

# Select the top 6 most viewed movies
top_movies = table_views.head(6)

# Set chart size
plt.figure(figsize=(10, 6))

# Plot bar chart
sns.barplot(x="title", y="views", data=top_movies, color='steelblue')

# Add view count labels above bars
for index, row in enumerate(top_movies.itertuples()):
    plt.text(index, row.views + 5, str(row.views), ha='center', fontsize=12)

# Set title and axis labels
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.xlabel("Movie Title")
plt.ylabel("Total Views")
plt.title("Total Views of the Top Films")

# Show plot
plt.show()

# Select first 20 rows and 25 columns of the rating data
heatmap_data = rating_matrix.iloc[:20, :25]

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)

# Set title
plt.title("Heatmap of the First 25 Rows and 25 Columns")

# Show heatmap
plt.show()

# ------------------------------
# 1. Load movie data and user rating data
# ------------------------------
# Assume movies.csv contains movieId, title, genres
movies = pd.read_csv('./IMDB-Dataset/movies.csv')
ratings = pd.read_csv('./IMDB-Dataset/ratings.csv')

# ------------------------------
# 2. Build movie feature vectors using TF-IDF
# ------------------------------
# Here we use movie genres as text features, assuming each genre is separated by "|"
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
tfidf_matrix = tfidf.fit_transform(movies['genres'])
print("Shape of TF-IDF matrix:", tfidf_matrix.shape)

# ------------------------------
# 3. Compute cosine similarity matrix between movies
# ------------------------------
cos_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Convert cosine similarity matrix to DataFrame, indexed by movieId
cos_sim_df = pd.DataFrame(cos_sim_matrix, index=movies['movieId'], columns=movies['movieId'])
print("Shape of movie similarity matrix:", cos_sim_df.shape)

# Visualize part of the similarity matrix (top 20 movies)
top_items = 20
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_df.iloc[:top_items, :top_items], cmap="viridis", cbar=True)
plt.title("Heatmap of the First 20 Movies' Similarity")
plt.xlabel("Movie ID")
plt.ylabel("Movie ID")
plt.show()

# ------------------------------
# 4. Build user-movie rating matrix
# ------------------------------
rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print("Shape of user-movie rating matrix:", rating_matrix.shape)

# ------------------------------
# 5. Generate recommendations for a specific user
# ------------------------------
user_id = 1
user_ratings = rating_matrix.loc[user_id]
# Find movies that the user has already rated
rated_movie_ids = user_ratings[user_ratings > 0].index

# Define a function to calculate predicted score based on user's rated movies and similarity
def get_recommendation_score(movie_id):
    # Get similarity scores between this movie and the user's rated movies
    sim_scores = cos_sim_df.loc[movie_id, rated_movie_ids]
    # Get user's ratings for those movies
    user_scores = user_ratings.loc[rated_movie_ids]
    # Weighted sum: similarity × rating
    return np.dot(sim_scores, user_scores)

# Compute predicted scores for movies the user hasn't rated
unrated_movies = user_ratings[user_ratings == 0].index
predicted_scores = {movie_id: get_recommendation_score(movie_id) for movie_id in unrated_movies}
predicted_scores_series = pd.Series(predicted_scores)

# Select the top 10 movies with the highest predicted scores
top_recommendations = predicted_scores_series.sort_values(ascending=False).head(10)
print("Recommended movies for user", user_id, ":")
print(top_recommendations)

# ------------------------------
# 6. (Optional) Combine recommendations with movie titles for display
# ------------------------------
movies_subset = movies.set_index('movieId')
top_recommendations = top_recommendations.rename_axis('movieId').reset_index()
top_recommendations['title'] = top_recommendations['movieId'].map(movies_subset['title'])
print(top_recommendations[['movieId', 'title', 0]])
