import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="CineMatch • Movie Recommender", layout="wide")

# ----------------------------
# CINEMA THEME (CSS)
# ----------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* Page background */
        .stApp {
            background: radial-gradient(circle at top, #141414 0%, #0b0b0b 55%, #050505 100%);
            color: #f5f5f5;
        }

        /* Make headers pop */
        h1, h2, h3 {
            font-family: "Trebuchet MS", sans-serif;
            letter-spacing: 0.5px;
        }

        /* Remove extra top padding */
        .block-container { padding-top: 2rem; }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111111 0%, #0b0b0b 100%);
            border-right: 1px solid rgba(255,255,255,0.07);
        }

        /* Card style */
        .card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.35);
        }

        /* Small badge */
        .badge {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(229, 9, 20, 0.18);
            border: 1px solid rgba(229, 9, 20, 0.35);
            color: #ffb3b3;
            font-size: 12px;
            margin-right: 8px;
        }

        /* Poster card */
        .poster {
            background: linear-gradient(135deg, rgba(229,9,20,0.25), rgba(255,255,255,0.05));
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 14px;
            height: 190px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 10px 26px rgba(0,0,0,0.35);
        }

        .poster-title {
            font-size: 16px;
            font-weight: 700;
            line-height: 1.2;
        }

        .poster-meta {
            font-size: 12px;
            opacity: 0.85;
        }

        /* Buttons */
        .stButton>button {
            background: #e50914;
            color: white;
            border-radius: 12px;
            border: 0px;
            padding: 0.6rem 1rem;
            font-weight: 700;
        }
        .stButton>button:hover {
            background: #ff1c24;
            color: white;
        }

        /* Dataframe */
        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
        }

        /* Metric cards */
        div[data-testid="metric-container"]{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 14px;
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    return movies, ratings


movies, ratings = load_data()

# ----------------------------
# Header (Hero)
# ----------------------------
st.markdown(
    """
    <div class="card">
        <span class="badge">🍿 CineMatch</span>
        <span class="badge">Collaborative Filtering</span>
        <h1 style="margin: 10px 0 0 0;">Movie Recommendation System</h1>
        <p style="margin: 6px 0 0 0; opacity: 0.85;">
            Explore the dataset, view cinema-style insights, and get recommendations based on user ratings.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.markdown("## 🎞 Navigation")
page = st.sidebar.radio("Choose a page", ["EDA (Graphs)", "Recommender", "Model Evaluation"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙ Settings")

top_n = st.sidebar.slider("Recommendations", 5, 25, 10)
min_ratings = st.sidebar.slider("Min ratings per movie (quality filter)", 0, 300, 50, step=10)

# ----------------------------
# EDA
# ----------------------------
if page == "EDA (Graphs)":
    st.markdown("## 📊 EDA (Graphs)")

    c1, c2, c3 = st.columns(3)
    c1.metric("🎬 Movies", int(movies["movieId"].nunique()))
    c2.metric("⭐ Ratings", int(len(ratings)))
    c3.metric("👥 Users", int(ratings["userId"].nunique()))

    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>Movies Preview</h3></div>', unsafe_allow_html=True)
        st.dataframe(movies.head(10), use_container_width=True)

    with col2:
        st.markdown('<div class="card"><h3>Ratings Preview</h3></div>', unsafe_allow_html=True)
        st.dataframe(ratings.head(10), use_container_width=True)

    st.write("")

    # Movies per year
    st.markdown('<div class="card"><h3>Movies Released Per Year</h3></div>', unsafe_allow_html=True)
    movies_per_year = movies.dropna(subset=["year"]).groupby("year").size()

    fig = plt.figure(figsize=(10, 5))
    plt.scatter(movies_per_year.index, movies_per_year.values)
    plt.xlabel("Year")
    plt.ylabel("Number of Movies")
    plt.title("Number of Movies Released Over the Years")
    st.pyplot(fig)

    st.write("")

    # Ratings distribution
    st.markdown('<div class="card"><h3>Ratings Distribution</h3></div>', unsafe_allow_html=True)
    fig = plt.figure(figsize=(10, 5))
    plt.hist(ratings["rating"], bins=20)
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.title("Distribution of Ratings")
    st.pyplot(fig)

    st.write("")

    # Average rating over years
    st.markdown('<div class="card"><h3>Average Rating Over the Years</h3></div>', unsafe_allow_html=True)
    merged = ratings.merge(movies[["movieId", "year"]], on="movieId", how="left")
    avg_rating_year = merged.dropna(subset=["year"]).groupby("year")["rating"].mean()

    fig = plt.figure(figsize=(10, 5))
    plt.plot(avg_rating_year.index, avg_rating_year.values)
    plt.xlabel("Year")
    plt.ylabel("Average Rating")
    plt.title("Average Movie Rating Over the Years")
    st.pyplot(fig)

    st.write("")

    # Top genres
    st.markdown('<div class="card"><h3>Top Genres</h3></div>', unsafe_allow_html=True)
    genre_counts = (
        movies["genres"]
        .fillna("")
        .str.split("|")
        .explode()
        .value_counts()
        .head(15)
    )

    fig = plt.figure(figsize=(10, 6))
    plt.bar(genre_counts.index, genre_counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.title("Top 15 Genres")
    st.pyplot(fig)

# ----------------------------
# Recommender
# ----------------------------
elif page == "Recommender":
    st.markdown("## 🤝 Recommender")
    st.caption("Pick a movie you like — CineMatch will suggest similar movies based on user rating patterns.")

    # User-movie matrix
    user_movie_matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)

    # Similarity between movies
    movie_similarity = cosine_similarity(user_movie_matrix.T)
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )

    # Quality filter
    counts = ratings.groupby("movieId").size()
    popular_ids = counts[counts >= min_ratings].index

    titles = sorted(movies["title"].unique())
    selected_movie = st.selectbox("🎞 Choose a movie", titles)

    def recommend_movies(movie_title, n=10):
        movie_id = movies.loc[movies["title"] == movie_title, "movieId"].values[0]
        if movie_id not in movie_similarity_df.index:
            return pd.DataFrame()

        sims = movie_similarity_df[movie_id].sort_values(ascending=False).iloc[1:]
        sims = sims[sims.index.isin(popular_ids)] if len(popular_ids) > 0 else sims
        sims = sims.head(n)

        recs = movies[movies["movieId"].isin(sims.index)][["movieId", "title", "year", "genres"]].copy()
        recs["similarity"] = recs["movieId"].map(lambda mid: float(sims.get(mid, np.nan)))
        recs = recs.sort_values("similarity", ascending=False)
        return recs

    if st.button("🍿 Recommend"):
        recs = recommend_movies(selected_movie, n=top_n)
        st.markdown("### 🎬 Recommended Movies")

        if recs.empty:
            st.warning("No recommendations found. Try lowering the min ratings filter.")
        else:
            # Poster-style grid
            cols = st.columns(5)
            for i, row in recs.reset_index(drop=True).iterrows():
                c = cols[i % 5]
                year = "" if pd.isna(row["year"]) else int(row["year"])
                genres = row["genres"] if isinstance(row["genres"], str) else ""
                sim = row["similarity"]
                with c:
                    st.markdown(
                        f"""
                        <div class="poster">
                            <div>
                                <div class="poster-title">{row['title']}</div>
                                <div class="poster-meta">Year: {year}</div>
                                <div class="poster-meta">Genres: {genres}</div>
                            </div>
                            <div class="poster-meta">Similarity: {sim:.3f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.write("")
            st.markdown("#### Full Table")
            st.dataframe(recs[["title", "year", "genres", "similarity"]], use_container_width=True)

# ----------------------------
# Model Evaluation
# ----------------------------
else:
    st.markdown("## ✅ Model Evaluation")
    st.caption("Computes RMSE and MAE on a sample test split for speed.")

    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    train_matrix = train.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    sim = cosine_similarity(train_matrix.T)
    sim_df = pd.DataFrame(sim, index=train_matrix.columns, columns=train_matrix.columns)

    def predict_rating(user_id, movie_id):
        if user_id not in train_matrix.index or movie_id not in train_matrix.columns:
            return np.nan

        user_ratings = train_matrix.loc[user_id]
        sims = sim_df[movie_id]

        rated = user_ratings[user_ratings > 0]
        if rated.empty:
            return np.nan

        sims = sims[rated.index]
        denom = np.sum(np.abs(sims.values))
        if denom == 0:
            return np.nan

        return float(np.dot(rated.values, sims.values) / denom)

    sample = test.sample(min(2000, len(test)), random_state=42).copy()
    sample["pred"] = sample.apply(lambda r: predict_rating(r["userId"], r["movieId"]), axis=1)
    sample = sample.dropna(subset=["pred"])

    rmse = np.sqrt(mean_squared_error(sample["rating"], sample["pred"]))
    mae = mean_absolute_error(sample["rating"], sample["pred"])

    c1, c2 = st.columns(2)
    c1.metric("RMSE", f"{rmse:.4f}")
    c2.metric("MAE", f"{mae:.4f}")

    st.markdown(
        '<div class="card"><h3>Notes</h3><p style="opacity:0.85;">'
        "Evaluation is on a sample for speed. Increase the sample size if your laptop can handle it."
        "</p></div>",
        unsafe_allow_html=True
    )
    