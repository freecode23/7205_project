# 🎬 Hybrid Movie Recommendation System

This project builds a hybrid movie recommender system using both **Content-Based Filtering** and **Collaborative Filtering (SVD)** on the MovieLens dataset. It combines genre-based cosine similarity and matrix factorization for more robust recommendations.

Inspired by the original [DataFlair article](https://data-flair.training/blogs/data-science-r-movie-recommendation-system/).

---

## 📁 Project Structure

```
.
├── 1_analysis.py         # Exploratory data analysis and visualization
├── 2_algo.py             # Core algorithm: builds and evaluates models
├── 3_recommend.py        # Generate hybrid recommendations for a user
├── helper/
│   └── evaluate.py       # Evaluation utilities (e.g., Precision@K)
├── IMDB-Dataset/         # Contains movies.csv and ratings.csv
├── results/              # Stores saved pickled matrices (e.g., SVD output)
├── .gitignore
└── README.md
```

---

## ✅ Features

- **Content-Based Filtering**  
  - Uses TF-IDF on movie genres  
  - Cosine similarity to find genre-similar movies

- **Collaborative Filtering (SVD)**  
  - Matrix factorization of user-movie rating matrix  
  - Predicts unseen ratings

- **Hybrid Recommender**  
  - Combines both methods using weighted average

---

## 📦 Prerequisites

Install the required Python packages using pip:

```bash
pip3 install pandas numpy scipy seaborn matplotlib scikit-learn tqdm
```

---

## 📊 Evaluation Metrics

- **Precision@10** for content-based filtering
- **RMSE / MAE** for SVD predictions using train-test split

Example:

```
🎯 Content-Based Precision@10: 0.0076 over 662 users

📊 SVD Evaluation Results:
✅ RMSE: 3.2248
✅ MAE:  3.0189
```

---

## 🚀 Future Improvements

- Experiment with different SVD `k` values
- Incorporate user/item metadata (e.g., director, year)
- Consider item-based collaborative filtering

---

## 📝 Acknowledgement

Based on the tutorial by DataFlair:  
https://data-flair.training/blogs/data-science-r-movie-recommendation-system/
