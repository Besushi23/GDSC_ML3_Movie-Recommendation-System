# 🎬 CineMatch — Movie Recommendation System

> **GDSC UTD ML Sprint 2026** | Built with Python, Scikit-learn & Streamlit

---

## 🍿 What is CineMatch?

CineMatch is a full-stack movie recommendation web app built on the **MovieLens dataset**. It uses **item-based collaborative filtering** with **cosine similarity** to recommend movies based on user rating patterns — the same core technique used by Netflix and Spotify.

The app features a fully custom cinematic UI built with Streamlit, real movie posters via the OMDb API, Google search links, a watchlist system, trending movies, background music, and rich exploratory data analysis visualizations.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 Smart Recommendations | Item-based collaborative filtering using cosine similarity |
| 🖼️ Movie Posters | Auto-fetched from OMDb API |
| 🔍 Google Links | Every movie card links directly to Google search |
| 🔥 Trending Page | Most-rated movies from recent timestamps |
| 📋 Watchlist | Save and manage movies you want to watch |
| 🎲 Surprise Me | Random movie picker from popular titles |
| 🎭 Genre Filter | Filter recommendations by genre |
| ⭐ Rating Filter | Set minimum average star rating |
| 🌙 Dark / Light Mode | Toggle between themes |
| 🎵 Background Music | Cinematic ambient music with volume control |
| 💡 Fun Facts | Random cinema trivia on every load |
| 📊 EDA Dashboard | Interactive data visualizations |
| ✅ Model Evaluation | RMSE and MAE metrics on held-out test split |

---

## 🧠 Machine Learning

### Algorithm — Item-Based Collaborative Filtering

1. **Build user-item matrix** — rows are users, columns are movies, values are ratings
2. **Compute cosine similarity** — measures how similar two movies are based on shared user ratings
3. **Generate recommendations** — returns top N most similar movies to the selected one
4. **Filter results** — applies minimum ratings threshold, genre filters, and star rating filters

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error) — penalises large prediction errors
- **MAE** (Mean Absolute Error) — average prediction offset in stars
- Evaluated on a 20% held-out test split (2,000 sample for speed)

---

## 📁 Project Structure

```
GDSC_ML3_Movie-Recommendation-System/
│
├── app2.py              # Main Streamlit application
├── movies.csv           # MovieLens movies dataset
├── ratings.csv          # MovieLens ratings dataset
├── background.mp3       # Background music file
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Besushi23/GDSC_ML3_Movie-Recommendation-System.git
cd GDSC_ML3_Movie-Recommendation-System
```

**2. Install dependencies**
```bash
pip install streamlit pandas numpy scikit-learn matplotlib requests
```

**3. Add the datasets**

Download from Kaggle and place in the project folder:
- [movies.csv](https://www.kaggle.com/datasets/jneupane12/movielens?select=movies.csv)
- [ratings.csv](https://www.kaggle.com/datasets/jneupane12/movielens?select=ratings.csv)

**4. Run the app**
```bash
python -m streamlit run app2.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📊 Dataset

**MovieLens Dataset** — a classic benchmark dataset for recommender systems

| File | Records | Description |
|---|---|---|
| movies.csv | 9,742 movies | Movie titles, genres, year |
| ratings.csv | 100,836 ratings | User ratings (0.5 to 5.0 stars) |

- **Users:** 610 unique users
- **Rating scale:** 0.5 — 5.0 stars
- **Time period:** 1996 — 2018

---

## 📸 App Pages

- **🎬 Recommender** — Search any movie and get personalised recommendations with match percentages
- **🔥 Trending** — Top 20 most-rated movies from recent timestamps, ranked with cards and a full table
- **📊 EDA** — Movies per year, rating distribution, top genres, top 10 most rated movies
- **📋 Watchlist** — Save movies and manage your personal list
- **✅ Model Eval** — Live RMSE, MAE and accuracy estimate on test data

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, custom CSS, HTML
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn (cosine similarity, train-test split, RMSE, MAE)
- **Visualizations:** Matplotlib
- **Posters:** OMDb API
- **Fonts:** Google Fonts (Bebas Neue, Outfit)
- **Version Control:** Git & GitHub

---

---

## 📄 License

This project is for educational purposes as part of the GDSC UTD ML Sprint program.

---

<div align="center">
Made with ❤️ and 🍿 for GDSC UTD ML Sprint 2026
</div>
