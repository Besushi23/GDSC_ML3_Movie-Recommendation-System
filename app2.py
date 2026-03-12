import streamlit as st
import pandas as pd
import numpy as np
import random
import requests

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(page_title="CineMatch • Movie Recommender", layout="wide")

# -----------------------------------
# CINEMA THEME
# -----------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg,#0b0b0b,#050505);
            color: #f5f5f5;
        }
        h1 {
            background: linear-gradient(90deg,#ff004c,#ff7b00,#ffd000);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h2,h3{ color:#ffd000; }
        section[data-testid="stSidebar"]{
            background:#111;
            border-right:1px solid #222;
        }
        .card{
            background:linear-gradient(145deg,#111,#1a1a1a);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:18px;
            padding:20px;
            box-shadow:0 10px 30px rgba(0,0,0,0.6);
        }
        .badge {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255, 0, 76, 0.18);
            border: 1px solid rgba(255, 0, 76, 0.35);
            color: #ffb3c6;
            font-size: 12px;
            margin-right: 8px;
        }
        .funstat {
            background: linear-gradient(135deg, #1a0a00, #2a1000);
            border: 1px solid #ff7b00;
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 16px;
            font-size: 15px;
            color: #ffd000;
        }
        .poster{
            background:linear-gradient(135deg,#1f1f1f,#121212);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:16px;
            padding:16px;
            height:200px;
            box-shadow:0 10px 25px rgba(0,0,0,0.5);
            transition:0.3s;
            display:flex;
            flex-direction:column;
            justify-content:space-between;
        }
        .poster:hover{
            transform:scale(1.03);
            border:1px solid #ff004c;
            box-shadow:0 0 15px #ff004c;
        }
        .poster-title{
            font-weight:700;
            font-size:15px;
            color:#ffffff;
            line-height:1.2;
        }
        .poster-meta{ font-size:12px; opacity:0.85; }
        .trend-rank{
            font-size:28px;
            font-weight:900;
            color:#ff004c;
            line-height:1;
        }
        .stButton>button{
            background:linear-gradient(90deg,#ff004c,#ff7b00);
            color:white;
            border-radius:10px;
            border:none;
            font-weight:700;
            padding:0.6rem 1rem;
        }
        .stButton>button:hover{
            background:linear-gradient(90deg,#ff7b00,#ffd000);
            color:black;
        }
        div[data-testid="metric-container"]{
            background:linear-gradient(145deg,#121212,#1a1a1a);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:14px;
            padding:12px;
        }
        div[data-testid="stDataFrame"]{
            border-radius:12px;
            border:1px solid rgba(255,255,255,0.08);
            overflow:hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# -----------------------------------
# Load Data
# -----------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    links = pd.read_csv("links.csv")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    return movies, ratings

movies, ratings = load_data()

# -----------------------------------
# TMDB Poster Fetcher
# -----------------------------------
@st.cache_data(show_spinner=False)
def get_poster_url(tmdb_id, api_key):
    if pd.isna(tmdb_id) or not api_key:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}"
        r = requests.get(url, timeout=3)
        data = r.json()
        path = data.get("poster_path")
        if path:
            return f"https://image.tmdb.org/t/p/w300{path}"
    except:
        pass
    return None

# -----------------------------------
# Hero header
# -----------------------------------
st.markdown(
    """
    <div class="card">
        <span class="badge">🍿 CineMatch</span>
        <span class="badge">Collaborative Filtering</span>
        <span class="badge">Streamlit UI</span>
        <h1 style="margin: 10px 0 0 0;">Movie Recommendation System</h1>
        <p style="margin: 6px 0 0 0; opacity: 0.85;">
            Explore the dataset, view cinema-style insights, and get recommendations based on user ratings.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.markdown("## 🎞 Navigation")
page = st.sidebar.radio("Choose a page", ["EDA (Graphs)", "Recommender", "Trending", "Model Evaluation"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙ Settings")
top_n = st.sidebar.slider("Recommendations", 5, 25, 10)
min_ratings = st.sidebar.slider("Min ratings per movie (quality filter)", 0, 300, 50, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖼 Movie Posters")
tmdb_api_key = st.sidebar.text_input("TMDB API Key (optional)", type="password", help="Get a free key at themoviedb.org → Settings → API")
show_posters = bool(tmdb_api_key)

# -----------------------------------
# EDA Page
# -----------------------------------
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

    st.markdown('<div class="card"><h3>Movies Released Per Year</h3></div>', unsafe_allow_html=True)
    movies_per_year = movies.dropna(subset=["year"]).groupby("year").size()
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(movies_per_year.index, movies_per_year.values)
    plt.xlabel("Year"); plt.ylabel("Number of Movies")
    plt.title("Number of Movies Released Over the Years")
    st.pyplot(fig)

    st.write("")

    st.markdown('<div class="card"><h3>Ratings Distribution</h3></div>', unsafe_allow_html=True)
    fig = plt.figure(figsize=(10, 5))
    plt.hist(ratings["rating"], bins=20)
    plt.xlabel("Rating"); plt.ylabel("Frequency")
    plt.title("Distribution of Ratings")
    st.pyplot(fig)

    st.write("")

    st.markdown('<div class="card"><h3>Average Rating Over the Years</h3></div>', unsafe_allow_html=True)
    merged = ratings.merge(movies[["movieId", "year"]], on="movieId", how="left")
    avg_rating_year = merged.dropna(subset=["year"]).groupby("year")["rating"].mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(avg_rating_year.index, avg_rating_year.values)
    plt.xlabel("Year"); plt.ylabel("Average Rating")
    plt.title("Average Movie Rating Over the Years")
    st.pyplot(fig)

    st.write("")

    st.markdown('<div class="card"><h3>Top Genres</h3></div>', unsafe_allow_html=True)
    genre_counts = movies["genres"].fillna("").str.split("|").explode().value_counts().head(15)
    fig = plt.figure(figsize=(10, 6))
    plt.bar(genre_counts.index, genre_counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Genre"); plt.ylabel("Count")
    plt.title("Top 15 Genres")
    st.pyplot(fig)

# -----------------------------------
# Trending Page
# -----------------------------------
elif page == "Trending":
    st.markdown("## 🔥 Trending Movies")
    st.caption("Movies with the most ratings from the latest timestamps in the dataset.")

    cutoff = ratings["timestamp"].quantile(0.80)
    recent_ratings = ratings[ratings["timestamp"] >= cutoff]

    trend_counts = recent_ratings.groupby("movieId").size().reset_index(name="recent_count")
    trend_avg = recent_ratings.groupby("movieId")["rating"].mean().reset_index(name="avg_rating")
    trending = trend_counts.merge(trend_avg, on="movieId")
    trending = trending[trending["recent_count"] >= 10]
    trending = trending.sort_values("recent_count", ascending=False).head(20)
    trending = trending.merge(movies[["movieId", "title", "year", "genres", "tmdbId"]], on="movieId", how="left")

    st.write("")
    st.markdown(f"### 🏆 Top {len(trending)} Trending Right Now")
    st.write("")

    cols = st.columns(5)
    for i, row in trending.reset_index(drop=True).iterrows():
        c = cols[i % 5]
        year = "" if pd.isna(row["year"]) else int(row["year"])
        genres = row["genres"] if isinstance(row["genres"], str) else ""
        avg_r = f"{row['avg_rating']:.1f} ⭐"
        count = int(row["recent_count"])
        rank = i + 1

        with c:
            if show_posters:
                poster_url = get_poster_url(row.get("tmdbId"), tmdb_api_key)
                if poster_url:
                    st.image(poster_url, use_container_width=True)

            st.markdown(
                f"""
                <div class="poster">
                    <div>
                        <div class="trend-rank">#{rank}</div>
                        <div class="poster-title">{row['title']}</div>
                        <div class="poster-meta">Year: {year}</div>
                        <div class="poster-meta">{genres}</div>
                    </div>
                    <div class="poster-meta">{avg_r} &nbsp;|&nbsp; {count} recent ratings</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.write("")

    st.write("")
    st.markdown("#### Full Trending Table")
    st.dataframe(
        trending[["title", "year", "genres", "avg_rating", "recent_count"]].reset_index(drop=True),
        use_container_width=True
    )

# -----------------------------------
# Recommender Page
# -----------------------------------
elif page == "Recommender":
    st.markdown("## 🤝 Recommender")
    st.caption("Pick a movie you like — CineMatch will suggest similar movies based on user rating patterns.")

    genre_counts_all = movies["genres"].fillna("").str.split("|").explode().value_counts()
    top_genre = genre_counts_all.index[0]
    top_genre_count = genre_counts_all.iloc[0]
    avg_rating_overall = ratings["rating"].mean()
    most_rated_movie = ratings["movieId"].value_counts().idxmax()
    most_rated_title = movies.loc[movies["movieId"] == most_rated_movie, "title"].values
    most_rated_title = most_rated_title[0] if len(most_rated_title) > 0 else "Unknown"

    fun_facts = [
        f"🎭 Did you know? <b>{top_genre}</b> is the most common genre with <b>{top_genre_count:,}</b> movies!",
        f"⭐ The average rating across all movies in this dataset is <b>{avg_rating_overall:.2f} / 5.0</b>",
        f"🏆 The most rated movie in this dataset is <b>{most_rated_title}</b>",
        f"🎬 This dataset contains <b>{movies['movieId'].nunique():,}</b> unique movies from across the decades!",
        f"👥 Over <b>{ratings['userId'].nunique():,}</b> users contributed ratings to this dataset!",
    ]

    if "fun_fact_index" not in st.session_state:
        st.session_state.fun_fact_index = 0

    st.markdown(
        f'<div class="funstat">💡 Fun Stat: {fun_facts[st.session_state.fun_fact_index]}</div>',
        unsafe_allow_html=True
    )
    if st.button("🔄 Next Fun Stat"):
        st.session_state.fun_fact_index = (st.session_state.fun_fact_index + 1) % len(fun_facts)
        st.rerun()

    st.write("")

    user_movie_matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)
    movie_similarity = cosine_similarity(user_movie_matrix.T)
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )

    counts = ratings.groupby("movieId").size()
    popular_ids = counts[counts >= min_ratings].index

    all_genres = sorted(set(
        g for genres in movies["genres"].fillna("").str.split("|")
        for g in genres if g and g != "(no genres listed)"
    ))
    selected_genre = st.selectbox("🎭 Filter by Genre (optional)", ["All Genres"] + all_genres)
    min_avg_rating = st.slider("⭐ Minimum average rating for recommendations", 0.0, 5.0, 0.0, step=0.5)
    avg_ratings_per_movie = ratings.groupby("movieId")["rating"].mean()

    titles = sorted(movies["title"].unique())

    col_select, col_surprise = st.columns([4, 1])
    with col_select:
        selected_movie = st.selectbox("🎞 Choose a movie", titles)
    with col_surprise:
        st.write("")
        st.write("")
        if st.button("🎲 Surprise Me!"):
            st.session_state.surprise_movie = random.choice(titles)

    if "surprise_movie" in st.session_state:
        selected_movie = st.session_state.surprise_movie
        st.info(f"🎲 Randomly picked: **{selected_movie}**")

    def recommend_movies(movie_title, n=10):
        movie_id = movies.loc[movies["title"] == movie_title, "movieId"].values[0]
        if movie_id not in movie_similarity_df.index:
            return pd.DataFrame()
        sims = movie_similarity_df[movie_id].sort_values(ascending=False).iloc[1:]
        if len(popular_ids) > 0:
            sims = sims[sims.index.isin(popular_ids)]
        sims = sims.head(n * 3)
        recs = movies[movies["movieId"].isin(sims.index)][["movieId", "title", "year", "genres", "tmdbId"]].copy()
        recs["similarity"] = recs["movieId"].map(lambda mid: float(sims.get(mid, np.nan)))
        recs["avg_rating"] = recs["movieId"].map(lambda mid: avg_ratings_per_movie.get(mid, np.nan))
        if selected_genre != "All Genres":
            recs = recs[recs["genres"].str.contains(selected_genre, na=False)]
        if min_avg_rating > 0:
            recs = recs[recs["avg_rating"] >= min_avg_rating]
        return recs.sort_values("similarity", ascending=False).head(n)

    if st.button("🍿 Recommend"):
        if "surprise_movie" in st.session_state:
            del st.session_state.surprise_movie

        recs = recommend_movies(selected_movie, n=top_n)
        st.markdown("### 🎬 Recommended Movies")

        if recs.empty:
            st.warning("No recommendations found. Try lowering the min ratings filter or changing the genre.")
        else:
            cols = st.columns(5)
            for i, row in recs.reset_index(drop=True).iterrows():
                c = cols[i % 5]
                year = "" if pd.isna(row["year"]) else int(row["year"])
                genres = row["genres"] if isinstance(row["genres"], str) else ""
                sim = row["similarity"]
                avg_r = row["avg_rating"]
                avg_r_str = f"{avg_r:.1f} ⭐" if not pd.isna(avg_r) else "N/A"

                with c:
                    if show_posters:
                        poster_url = get_poster_url(row.get("tmdbId"), tmdb_api_key)
                        if poster_url:
                            st.image(poster_url, use_container_width=True)

                    st.markdown(
                        f"""
                        <div class="poster">
                            <div>
                                <div class="poster-title">{row['title']}</div>
                                <div class="poster-meta">Year: {year}</div>
                                <div class="poster-meta">Genres: {genres}</div>
                                <div class="poster-meta">Avg Rating: {avg_r_str}</div>
                            </div>
                            <div class="poster-meta">Similarity: {sim:.3f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.write("")
            st.markdown("#### Full Table")
            st.dataframe(recs[["title", "year", "genres", "avg_rating", "similarity"]], use_container_width=True)

# -----------------------------------
# Model Evaluation Page
# -----------------------------------
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