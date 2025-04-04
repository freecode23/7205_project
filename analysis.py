import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import seaborn as sns
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

MOVIES_FILEPATH='./IMDB-Dataset/movies.csv'
RATINGS_FILEPATH='./IMDB-Dataset/ratings.csv'

# movie_data_read csv file
movie_data = pd.read_csv(MOVIES_FILEPATH)
rating_data = pd.read_csv(RATINGS_FILEPATH)

#show information of the movie data
print(movie_data.describe(include='all'))

#show structure and basic info of the movie data
print(movie_data.info())

# show the first 6 lines of the movie data¨
print(movie_data.head(6))

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

# Take the top 2 singular values (dimensionality reduction)
k = 2
U_k = U[:, :k]  # User feature vectors
Vt_k = Vt[:k, :]  # Item feature vectors

# Compute cosine similarity between users
num_users = U_k.shape[0]
user_similarity = np.zeros((num_users, num_users))

for i in range(num_users):
    for j in range(num_users):
        user_similarity[i, j] = 1 - cosine(U_k[i], U_k[j])  # Cosine similarity

print("User similarity matrix (calculated after SVD dimensionality reduction):\n", np.round(user_similarity, 2))


# Assume rating_matrix is a NumPy matrix or Pandas DataFrame (rows are users, columns are movies)
movie_views = np.sum(rating_matrix > 0, axis=0)  # Calculate the number of views for each movie (non-zero ratings)

# Create a DataFrame
table_views = pd.DataFrame({
    "movie": rating_matrix.columns,  # Movie ID
    "views": movie_views             # View count
})

# Sort by view count in descending order
table_views = table_views.sort_values(by="views", ascending=False).reset_index(drop=True)

# Add movie title column (from movie_data)
table_views["title"] = table_views["movie"].map(movie_data.set_index("movieId")["title"])

# Output the top 6 rows
print(table_views.head(6))

# Select the top 6 most-viewed movies
top_movies = table_views.head(6)

# Set the figure size
plt.figure(figsize=(10, 6))

# Plot a bar chart
sns.barplot(x="title", y="views", data=top_movies, color='steelblue')

# Add view counts above bars
for index, row in enumerate(top_movies.itertuples()):
    plt.text(index, row.views + 5, str(row.views), ha='center', fontsize=12)

# Set title and axis labels
plt.xticks(rotation=45, ha='right')  # Rotate X-axis labels
plt.xlabel("Movie Title")
plt.ylabel("Total Views")
plt.title("Total Views of the Top Films")

# Show the plot
plt.show()


# Select the first 20 rows and 25 columns of the rating data
heatmap_data = rating_matrix.iloc[:20, :25]

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)

# Set title
plt.title("Heatmap of the first 25 rows and 25 columns")

# Show the heatmap
plt.show()
