# 🎌 Anime Recommendation System

## 🎯 Objective
To build an anime recommendation engine using:

- **Content-Based Filtering** (genre-based similarity)
- **Collaborative Filtering** (user-user interaction)
- Evaluate models using appropriate metrics like **RMSE**, **Precision**, **Recall**, and **F1-score**

---

## 📁 Dataset Description

### 1. `anime.csv`
Contains anime metadata.

| Column     | Description                                           |
|------------|-------------------------------------------------------|
| `anime_id` | Unique ID for each anime                              |
| `name`     | Anime title                                           |
| `genre`    | Genre(s) of the anime (e.g., Action, Sci-Fi, etc.)    |
| `type`     | Format (TV, Movie, OVA)                               |
| `episodes` | Number of episodes                                    |
| `rating`   | Average community rating                              |
| `members`  | Number of community members who interacted with anime |

### 2. `rating.csv`
Contains user-anime rating interactions.

| Column     | Description                                     |
|------------|-------------------------------------------------|
| `user_id`  | Unique ID for each user                         |
| `anime_id` | Anime ID corresponding to anime in `anime.csv` |
| `rating`   | User rating (values from 1 to 10, -1 if unscored) |

A sample from `rating.csv`:

![rating.csv sample](./196cae71-fbdb-4c3d-bb06-3c4af27343b7.png)

---

## 📂 Files in This Repository

| File Name           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `anime.csv`         | Primary dataset with anime metadata                                         |
| `rating.csv`        | User-anime interactions with ratings                                        |
| `Recommendation System.txt` | Describes objectives, dataset, and task requirements                  |
| `Best attempt.ipynb`| Combines both content-based and collaborative filtering with evaluations    |
| `new try.ipynb`     | Focuses on collaborative filtering and evaluates it using RMSE             |
| `Untitled.ipynb`    | Focuses on content-based filtering and evaluates with precision/recall/F1   |

---

## 🛠️ Tools and Libraries

- `pandas`, `numpy` – Data manipulation
- `sklearn.feature_extraction.text.TfidfVectorizer` – Genre vectorization
- `sklearn.metrics.pairwise.cosine_similarity` – Content similarity
- `sklearn.model_selection.train_test_split` – Dataset splitting
- `scipy.sparse.csr_matrix` – Sparse matrix creation for user-item ratings
- `sklearn.neighbors.NearestNeighbors` – User similarity (collaborative filtering)
- `sklearn.metrics` – `mean_squared_error`, `precision_score`, `recall_score`, `f1_score`

---

## 🔄 Project Workflow

### 1. Data Preprocessing

- Loaded datasets into pandas DataFrames.
- Removed missing genres in `anime.csv`.
- Filtered out ratings = -1 from `rating.csv`.
- Merged on `anime_id` to create a unified dataset.

### 2. Feature Extraction (Content-Based)

- Used `TfidfVectorizer` on `genre` column to create TF-IDF matrix.
- Computed **cosine similarity** between animes based on genre vectors.

### 3. Recommendation Engines

#### A. 🎨 Content-Based Filtering
Function: `recommend_anime_by_features(anime_name, top_n=10)`

- Computes similarity scores with all other animes.
- Recommends top `n` most similar animes based on genres.

#### B. 👥 Collaborative Filtering (User-Based)
Function: `recommend_anime_cf(user_id, top_n=10)`

- Creates a user-item matrix from ratings.
- Uses `NearestNeighbors` with cosine similarity.
- Recommends top `n` anime based on ratings of similar users.

---

## 📊 Evaluation

### ✅ Content-Based Evaluation (from `Untitled.ipynb`)
- Used train-test split.
- Calculated:
  - `Precision`
  - `Recall`
  - `F1-score`

```python
precision_score(y_true, y_pred, average='macro')
recall_score(y_true, y_pred, average='macro')
f1_score(y_true, y_pred, average='macro')
```

### ✅ Collaborative Filtering Evaluation (from `new try.ipynb`)
- Predicted ratings for user-anime pairs in test set.
- Used RMSE:

```python
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(actual, predicted, squared=False)
```

---

## ▶️ How to Run

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/anime-recommendation.git
   cd anime-recommendation
   ```

2. **Install Dependencies**
   ```bash
   pip install pandas numpy scikit-learn scipy
   ```

3. **Launch Notebooks**
   ```bash
   jupyter notebook
   ```

4. **Explore the Following Notebooks**
   - `Best attempt.ipynb`: Both filtering techniques
   - `new try.ipynb`: Collaborative Filtering
   - `Untitled.ipynb`: Content-Based Evaluation

---

## ❓ Interview Questions

### Q1: What’s the difference between user-based and item-based CF?

| Aspect            | User-Based CF                           | Item-Based CF                           |
|-------------------|------------------------------------------|------------------------------------------|
| Similarity        | Between users                            | Between items                            |
| Use Case          | Find users like you                      | Find items similar to what you liked     |
| Performance       | Costly on large user bases               | More stable, often faster                |

---

### Q2: What is Collaborative Filtering?

**Collaborative Filtering** is a recommendation technique based on the past behavior of users. It works by:

- Finding **users** with similar preferences (user-based)
- Finding **items** that are similarly rated (item-based)
- Can be:
  - **Memory-based** (similarity matrix)
  - **Model-based** (matrix factorization, deep learning)

---

## ✅ Summary

This project demonstrates how to build and evaluate a hybrid recommendation engine that uses both **content-based** (genre similarity) and **collaborative filtering** (user similarity) techniques. The evaluation strategies ensure reliability, and techniques like TF-IDF and NearestNeighbors provide robust foundations for scalable recommender systems.

---
