import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # Extract year from title e.g. "Toy Story (1995)"
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")

    return movies, ratings


movies, ratings = load_data()

# ----------------------------
# Sidebar Navigation
# ----------------------------
page = st.sidebar.radio("Navigate", ["EDA (Graphs)", "Recommender", "Model Evaluation"])

# ----------------------------
# EDA (Graphs)
# ----------------------------
if page == "EDA (Graphs)":
    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Movies Preview")
        st.dataframe(movies.head(10), use_container_width=True)

    with col2:
        st.subheader("Ratings Preview")
        st.dataframe(ratings.head(10), use_container_width=True)

    st.divider()

    # Basic Stats
    st.subheader("Quick Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Movies", int(movies["movieId"].nunique()))
    c2.metric("Total Ratings", int(len(ratings)))
    c3.metric("Unique Users", int(ratings["userId"].nunique()))

    st.divider()

    # Missing Values
    st.subheader("Missing Values")
    missing_movies = movies.isna().sum()
    missing_ratings = ratings.isna().sum()
    st.write("Movies missing values:")
    st.write(missing_movies)
    st.write("Ratings missing values:")
    st.write(missing_ratings)

    st.divider()

    # Graph 1: Movies per year
    st.subheader("Movies Released Per Year")
    movies_per_year = movies.dropna(subset=["year"]).groupby("year").size()

    fig = plt.figure(figsize=(10, 5))
    plt.scatter(movies_per_year.index, movies_per_year.values)
    plt.xlabel("Year")
    plt.ylabel("Number of Movies")
    plt.title("Number of Movies Released Over the Years")
    st.pyplot(fig)

    st.divider()

    # Graph 2: Ratings distribution
    st.subheader("Ratings Distribution")
    fig = plt.figure(figsize=(10, 5))
    plt.hist(ratings["rating"], bins=20)
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.title("Distribution of Ratings")
    st.pyplot(fig)

    st.divider()

    # Graph 3: Average rating by year (requires merge)
    st.subheader("Average Rating Over the Years")
    merged = ratings.merge(movies[["movieId", "year"]], on="movieId", how="left")
    avg_rating_year = merged.dropna(subset=["year"]).groupby("year")["rating"].mean()

    fig = plt.figure(figsize=(10, 5))
    plt.plot(avg_rating_year.index, avg_rating_year.values)
    plt.xlabel("Year")
    plt.ylabel("Average Rating")
    plt.title("Average Movie Rating Over the Years")
    st.pyplot(fig)

    st.divider()

    # Graph 4: Top genres
    st.subheader("Top Genres (Most Common)")
    # Genres are like "Adventure|Animation|Children|Comedy|Fantasy"
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
    st.header("🤝 Movie Recommender (Collaborative Filtering)")

    # Create user-movie matrix
    user_movie_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    # Similarity between movies
    movie_similarity = cosine_similarity(user_movie_matrix.T)
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )

    # Select movie
    selected_movie = st.selectbox("Choose a movie you like", sorted(movies["title"].unique()))

    def recommend_movies(movie_title, n=10):
        movie_id = movies.loc[movies["title"] == movie_title, "movieId"].values[0]
        if movie_id not in movie_similarity_df.index:
            return pd.DataFrame()

        sims = movie_similarity_df[movie_id].sort_values(ascending=False).iloc[1:n+1]
        recs = movies[movies["movieId"].isin(sims.index)][["title", "year", "genres"]]
        recs = recs.copy()
        recs["similarity"] = recs["title"].map(lambda t: float(sims[movies.loc[movies["title"] == t, "movieId"].values[0]]))
        return recs.sort_values("similarity", ascending=False)

    if st.button("Recommend Movies"):
        recs = recommend_movies(selected_movie, n=10)
        st.subheader("Recommended Movies")
        if recs.empty:
            st.warning("No recommendations found.")
        else:
            st.dataframe(recs, use_container_width=True)

# ----------------------------
# Model Evaluation
# ----------------------------
else:
    st.header("✅ Model Evaluation (RMSE / MAE)")

    # Train-test split
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    # Build train matrix
    train_matrix = train.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    sim = cosine_similarity(train_matrix.T)
    sim_df = pd.DataFrame(sim, index=train_matrix.columns, columns=train_matrix.columns)

    # Predict rating using weighted similarity (simple baseline)
    def predict_rating(user_id, movie_id):
        if user_id not in train_matrix.index or movie_id not in train_matrix.columns:
            return np.nan

        user_ratings = train_matrix.loc[user_id]
        sims = sim_df[movie_id]

        # Only consider movies the user has rated
        rated = user_ratings[user_ratings > 0]
        if rated.empty:
            return np.nan

        sims = sims[rated.index]
        if sims.abs().sum() == 0:
            return np.nan

        return np.dot(rated.values, sims.values) / np.sum(np.abs(sims.values))

    # Evaluate on a sample for speed
    sample = test.sample(min(2000, len(test)), random_state=42).copy()
    sample["pred"] = sample.apply(lambda r: predict_rating(r["userId"], r["movieId"]), axis=1)
    sample = sample.dropna(subset=["pred"])

    rmse = np.sqrt(mean_squared_error(sample["rating"], sample["pred"]))
    mae = mean_absolute_error(sample["rating"], sample["pred"])

    st.write(f"RMSE: **{rmse:.4f}**")
    st.write(f"MAE: **{mae:.4f}**")

    st.caption("Evaluation is done on a sample for speed. Increase sample size if needed.")