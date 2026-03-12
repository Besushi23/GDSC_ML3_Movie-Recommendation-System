import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from urllib.parse import quote_plus
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(page_title="CineMatch • Movie Recommender", layout="wide")

# -----------------------------------
# Session State Init
# -----------------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# -----------------------------------
# Google Search Link Helper
# -----------------------------------
def get_google_url(title, year=None):
    clean_title = title.split("(")[0].strip()
    query = f"{clean_title} {int(year)} movie" if year and not pd.isna(year) else f"{clean_title} movie"
    return f"https://www.google.com/search?q={quote_plus(query)}"

# -----------------------------------
# Theme CSS
# -----------------------------------
def inject_css(dark_mode):
    if dark_mode:
        bg = "linear-gradient(180deg,#0b0b0b,#050505)"
        text = "#f5f5f5"
        card_bg = "linear-gradient(145deg,#111,#1a1a1a)"
        card_border = "rgba(255,255,255,0.08)"
        sidebar_bg = "#111"
        table_border = "rgba(255,255,255,0.08)"
        google_btn = "rgba(255,255,255,0.08)"
        google_btn_text = "#aaa"
    else:
        bg = "linear-gradient(180deg,#f0f0f0,#ffffff)"
        text = "#111111"
        card_bg = "linear-gradient(145deg,#ffffff,#f5f5f5)"
        card_border = "rgba(0,0,0,0.1)"
        sidebar_bg = "#e8e8e8"
        table_border = "rgba(0,0,0,0.1)"
        google_btn = "rgba(0,0,0,0.06)"
        google_btn_text = "#555"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');

        .stApp {{
            background: {bg};
            color: {text};
            font-family: 'Inter', sans-serif;
        }}
        h1 {{
            font-family: 'Bebas Neue', sans-serif;
            background: linear-gradient(90deg,#ff004c,#ff7b00,#ffd000);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            letter-spacing: 2px;
        }}
        h2,h3 {{ color: #ffd000; }}
        section[data-testid="stSidebar"] {{
            background: {sidebar_bg};
            border-right: 1px solid {card_border};
        }}
        .card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 12px;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 999px;
            background: rgba(255,0,76,0.18);
            border: 1px solid rgba(255,0,76,0.35);
            color: #ffb3c6;
            font-size: 12px;
            margin-right: 8px;
        }}
        .movie-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            transition: 0.3s;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 220px;
            margin-bottom: 4px;
        }}
        .movie-card:hover {{
            transform: scale(1.02);
            border: 1px solid #ff004c;
            box-shadow: 0 0 20px rgba(255,0,76,0.3);
        }}
        .movie-icon {{
            font-size: 36px;
            margin-bottom: 8px;
        }}
        .movie-title {{
            font-weight: 700;
            font-size: 14px;
            color: {text};
            line-height: 1.3;
            margin-bottom: 4px;
        }}
        .movie-meta {{
            font-size: 11px;
            opacity: 0.7;
            margin-top: 3px;
        }}
        .star-rating {{
            color: #ffd000;
            font-size: 13px;
            margin-top: 4px;
        }}
        .match-bar-wrap {{
            background: rgba(255,255,255,0.1);
            border-radius: 999px;
            height: 6px;
            margin-top: 6px;
            overflow: hidden;
        }}
        .match-bar {{
            height: 6px;
            border-radius: 999px;
            background: linear-gradient(90deg,#ff004c,#ffd000);
        }}
        .match-label {{
            font-size: 11px;
            color: #ffd000;
            font-weight: 700;
            margin-top: 6px;
        }}
        .google-link {{
            display: inline-block;
            margin-top: 10px;
            padding: 5px 12px;
            border-radius: 8px;
            background: {google_btn};
            color: {google_btn_text} !important;
            font-size: 11px;
            font-weight: 600;
            text-decoration: none !important;
            border: 1px solid {card_border};
            transition: 0.2s;
        }}
        .google-link:hover {{
            background: rgba(255,0,76,0.15);
            color: #ff6b8a !important;
            border-color: rgba(255,0,76,0.3);
        }}
        .stButton>button {{
            background: linear-gradient(90deg,#ff004c,#ff7b00);
            color: white;
            border-radius: 10px;
            border: none;
            font-weight: 700;
            padding: 0.6rem 1.2rem;
            transition: 0.2s;
        }}
        .stButton>button:hover {{
            background: linear-gradient(90deg,#ff7b00,#ffd000);
            color: black;
        }}
        .watchlist-item {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 10px 16px;
            margin-bottom: 8px;
        }}
        .fun-fact-box {{
            background: linear-gradient(135deg,rgba(255,0,76,0.15),rgba(255,123,0,0.15));
            border: 1px solid rgba(255,123,0,0.3);
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }}
        div[data-testid="metric-container"] {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 14px;
            padding: 12px;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            border: 1px solid {table_border};
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css(st.session_state.dark_mode)

# -----------------------------------
# Load Data
# -----------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    return movies, ratings

movies, ratings = load_data()

avg_ratings = ratings.groupby("movieId")["rating"].mean()
rating_counts = ratings.groupby("movieId")["rating"].count()

# -----------------------------------
# Helpers
# -----------------------------------
def stars(avg, max_stars=5):
    filled = int(round(avg))
    return "⭐" * filled + "☆" * (max_stars - filled)

GENRE_ICONS = {
    "Action": "💥", "Adventure": "🗺️", "Animation": "🎨", "Comedy": "😂",
    "Crime": "🔫", "Documentary": "📽️", "Drama": "🎭", "Fantasy": "🧙",
    "Horror": "👻", "Musical": "🎵", "Mystery": "🔍", "Romance": "❤️",
    "Sci-Fi": "🚀", "Thriller": "😱", "War": "⚔️", "Western": "🤠",
}

def genre_icon(genres_str):
    if not isinstance(genres_str, str):
        return "🎬"
    for g, icon in GENRE_ICONS.items():
        if g in genres_str:
            return icon
    return "🎬"

FUN_FACTS = [
    "🎬 The first movie ever made was 'Roundhay Garden Scene' (1888) — just 2 seconds long!",
    "🍿 Americans eat about 17 billion quarts of popcorn every year.",
    "🎥 It cost $63 million to make Titanic, but $200 million to promote it.",
    "🎭 The Wilhelm Scream sound effect has been used in over 400 movies.",
    "🦁 The MGM lion has had 7 different lions play the role over the years.",
    "🎞 The Lord of the Rings was filmed entirely in New Zealand over 438 days.",
    "💡 Alfred Hitchcock never won an Oscar despite 5 nominations.",
    "🤖 The T-800's famous 'I'll be back' line was almost 'I'll come back'.",
    "🎬 Clue (1985) had 3 different endings shown in different theaters.",
    "🌙 Stanley Kubrick burned The Shining sets after filming to protect secrecy.",
]

# -----------------------------------
# Hero Header
# -----------------------------------
col_hero, col_toggle = st.columns([5, 1])
with col_hero:
    st.markdown(
        """
        <div class="card">
            <span class="badge">🍿 CineMatch</span>
            <span class="badge">AI Powered</span>
            <span class="badge">Collaborative Filtering</span>
            <h1 style="margin: 10px 0 0 0;">CineMatch</h1>
            <p style="margin: 4px 0 0 0; opacity: 0.75; font-size:15px;">
                Your personal cinema — discover movies you'll love.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_toggle:
    st.write("")
    st.write("")
    mode_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
    if st.button(mode_label):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

fact = random.choice(FUN_FACTS)
st.markdown(f'<div class="fun-fact-box">💡 <b>Did you know?</b> {fact}</div>', unsafe_allow_html=True)

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.markdown("## 🎞 CineMatch")
page = st.sidebar.radio("Navigate", ["🎬 Recommender", "📊 EDA", "📋 Watchlist", "✅ Model Eval"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Filters")
top_n = st.sidebar.slider("# Recommendations", 5, 25, 10)
min_ratings = st.sidebar.slider("Min ratings per movie", 0, 300, 50, step=10)
min_star_filter = st.sidebar.slider("⭐ Min average star rating", 1.0, 5.0, 1.0, step=0.5)

st.sidebar.markdown("### 🎭 Genre Filter")
all_genres = sorted(set(
    g for sublist in movies["genres"].fillna("").str.split("|") for g in sublist
    if g and g != "(no genres listed)"
))
selected_genres = st.sidebar.multiselect("Filter by genre", all_genres, default=[])

st.sidebar.markdown("### 📅 Year Range")
min_year = int(movies["year"].min()) if not movies["year"].isna().all() else 1900
max_year = int(movies["year"].max()) if not movies["year"].isna().all() else 2024
year_range = st.sidebar.slider("Release year", min_year, max_year, (1990, max_year))

# -----------------------------------
# Build similarity matrix
# -----------------------------------
@st.cache_data(show_spinner=False)
def build_similarity(min_rat):
    counts = ratings.groupby("movieId").size()
    popular_ids = counts[counts >= min_rat].index
    filtered = ratings[ratings["movieId"].isin(popular_ids)]
    matrix = filtered.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    sim = cosine_similarity(matrix.T)
    sim_df = pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)
    return sim_df, popular_ids

sim_df, popular_ids = build_similarity(min_ratings)

# -----------------------------------
# Recommend function
# -----------------------------------
def recommend_movies(movie_title, n=10):
    row = movies.loc[movies["title"] == movie_title]
    if row.empty:
        return pd.DataFrame()
    movie_id = row["movieId"].values[0]
    if movie_id not in sim_df.index:
        return pd.DataFrame()

    sims = sim_df[movie_id].sort_values(ascending=False).iloc[1:]
    if len(popular_ids) > 0:
        sims = sims[sims.index.isin(popular_ids)]

    recs = movies[movies["movieId"].isin(sims.index)].copy()
    recs["similarity"] = recs["movieId"].map(lambda mid: float(sims.get(mid, np.nan)))
    recs["avg_rating"] = recs["movieId"].map(avg_ratings)
    recs["num_ratings"] = recs["movieId"].map(rating_counts)

    if selected_genres:
        recs = recs[recs["genres"].apply(lambda g: any(genre in str(g) for genre in selected_genres))]
    recs = recs[recs["avg_rating"] >= min_star_filter]
    recs = recs[
        (recs["year"].isna()) |
        ((recs["year"] >= year_range[0]) & (recs["year"] <= year_range[1]))
    ]

    return recs.sort_values("similarity", ascending=False).head(n)

# -----------------------------------
# Render movie card  (with Google link)
# -----------------------------------
def render_movie_card(row, show_watchlist_btn=True):
    title = row["title"]
    year = row.get("year", np.nan)
    year_display = "" if pd.isna(year) else int(year)
    genres = row.get("genres", "")
    avg = row.get("avg_rating", np.nan)
    sim = row.get("similarity", np.nan)

    icon = genre_icon(genres)
    star_str = stars(avg) if not pd.isna(avg) else "☆☆☆☆☆"
    avg_str = f"{avg:.1f}" if not pd.isna(avg) else "N/A"
    genre_display = " · ".join(genres.split("|")[:2]) if isinstance(genres, str) else ""
    google_url = get_google_url(title, year)

    match_html = ""
    if not pd.isna(sim):
        match_pct = f"{sim * 100:.0f}%"
        match_bar_width = f"{sim * 100:.0f}"
        match_html = f"""
            <div class="match-label">🎯 {match_pct} Match</div>
            <div class="match-bar-wrap">
                <div class="match-bar" style="width:{match_bar_width}%"></div>
            </div>
        """

    card_html = f"""
    <div class="movie-card">
        <div>
            <div class="movie-icon">{icon}</div>
            <div class="movie-title">{title}</div>
            <div class="movie-meta">📅 {year_display} &nbsp;|&nbsp; {genre_display}</div>
            <div class="star-rating">{star_str} <span style="font-size:11px;opacity:0.8;">({avg_str})</span></div>
            {match_html}
        </div>
        <div>
            <a href="{google_url}" target="_blank" class="google-link">
                🔍 Search on Google
            </a>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    if show_watchlist_btn:
        in_list = title in st.session_state.watchlist
        btn_label = "✅ In Watchlist" if in_list else "➕ Watchlist"
        if st.button(btn_label, key=f"wl_{title}_{hash(title)}"):
            if in_list:
                st.session_state.watchlist.remove(title)
            else:
                st.session_state.watchlist.append(title)
            st.rerun()

# -----------------------------------
# RECOMMENDER PAGE
# -----------------------------------
if page == "🎬 Recommender":
    st.markdown("## 🎬 Find Your Next Favourite")
    st.caption("Search for a movie, hit Recommend to discover similar films.")

    titles = sorted(movies["title"].unique())

    col_search, col_surprise = st.columns([4, 1])
    with col_search:
        search_query = st.text_input("🔎 Search for a movie...", placeholder="e.g. Inception, Toy Story...")
    with col_surprise:
        st.write("")
        st.write("")
        surprise_clicked = st.button("🎲 Surprise Me!")

    filtered_titles = [t for t in titles if search_query.lower() in t.lower()] if search_query else titles

    if surprise_clicked:
        valid_titles = [
            movies.loc[movies["movieId"] == mid, "title"].values[0]
            for mid in popular_ids
            if mid in movies["movieId"].values
        ]
        selected_movie = random.choice(valid_titles) if valid_titles else titles[0]
    elif filtered_titles:
        selected_movie = st.selectbox("Select a movie", filtered_titles)
    else:
        st.warning("No movies found matching your search.")
        selected_movie = None

    # Trending
    st.markdown("---")
    st.markdown("### 🔥 Trending Right Now")
    st.caption("Most rated movies in the dataset")

    trending_ids = rating_counts.sort_values(ascending=False).head(5).index
    trending_movies = movies[movies["movieId"].isin(trending_ids)].copy()
    trending_movies["avg_rating"] = trending_movies["movieId"].map(avg_ratings)
    trending_movies["num_ratings"] = trending_movies["movieId"].map(rating_counts)
    trending_movies["similarity"] = np.nan

    tcols = st.columns(5)
    for i, (_, row) in enumerate(trending_movies.iterrows()):
        with tcols[i % 5]:
            render_movie_card(row, show_watchlist_btn=True)

    # Recommendations
    st.markdown("---")
    if selected_movie and st.button("🍿 Get Recommendations"):
        with st.spinner("Finding movies you'll love..."):
            recs = recommend_movies(selected_movie, n=top_n)

        st.markdown(f"### 🎬 Because you liked *{selected_movie}*")

        if recs.empty:
            st.warning("No recommendations found. Try adjusting the filters in the sidebar.")
        else:
            cols = st.columns(5)
            for i, row in recs.reset_index(drop=True).iterrows():
                with cols[i % 5]:
                    render_movie_card(row)

            st.markdown("#### 📋 Full Table")
            display_df = recs[["title", "year", "genres", "avg_rating", "similarity"]].copy()
            display_df["similarity"] = (display_df["similarity"] * 100).round(1).astype(str) + "%"
            display_df["avg_rating"] = display_df["avg_rating"].round(2)
            display_df["google"] = display_df.apply(
                lambda r: get_google_url(r["title"], r["year"]), axis=1
            )
            display_df.columns = ["Title", "Year", "Genres", "Avg Rating", "Match %", "Google Link"]
            st.dataframe(display_df, use_container_width=True)

# -----------------------------------
# EDA PAGE
# -----------------------------------
elif page == "📊 EDA":
    st.markdown("## 📊 Dataset Insights")

    c1, c2, c3 = st.columns(3)
    c1.metric("🎬 Movies", int(movies["movieId"].nunique()))
    c2.metric("⭐ Ratings", int(len(ratings)))
    c3.metric("👥 Users", int(ratings["userId"].nunique()))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>Movies Preview</h3></div>', unsafe_allow_html=True)
        st.dataframe(movies.head(10), use_container_width=True)
    with col2:
        st.markdown('<div class="card"><h3>Ratings Preview</h3></div>', unsafe_allow_html=True)
        st.dataframe(ratings.head(10), use_container_width=True)

    bg_color = "#0b0b0b" if st.session_state.dark_mode else "#ffffff"
    text_color = "#f5f5f5" if st.session_state.dark_mode else "#111111"

    def dark_fig():
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        return fig, ax

    st.markdown('<div class="card"><h3>🎬 Movies Released Per Year</h3></div>', unsafe_allow_html=True)
    movies_per_year = movies.dropna(subset=["year"]).groupby("year").size()
    fig, ax = dark_fig()
    ax.scatter(movies_per_year.index, movies_per_year.values, color="#ff004c", alpha=0.7)
    ax.set_xlabel("Year"); ax.set_ylabel("Number of Movies")
    st.pyplot(fig)

    st.markdown('<div class="card"><h3>⭐ Ratings Distribution</h3></div>', unsafe_allow_html=True)
    fig, ax = dark_fig()
    ax.hist(ratings["rating"], bins=20, color="#ff7b00", alpha=0.85)
    ax.set_xlabel("Rating"); ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.markdown('<div class="card"><h3>📈 Average Rating Over the Years</h3></div>', unsafe_allow_html=True)
    merged = ratings.merge(movies[["movieId", "year"]], on="movieId", how="left")
    avg_rating_year = merged.dropna(subset=["year"]).groupby("year")["rating"].mean()
    fig, ax = dark_fig()
    ax.plot(avg_rating_year.index, avg_rating_year.values, color="#ffd000", linewidth=2)
    ax.set_xlabel("Year"); ax.set_ylabel("Average Rating")
    st.pyplot(fig)

    st.markdown('<div class="card"><h3>🎭 Top Genres</h3></div>', unsafe_allow_html=True)
    genre_counts = movies["genres"].fillna("").str.split("|").explode().value_counts().head(15)
    fig, ax = dark_fig()
    ax.bar(genre_counts.index, genre_counts.values, color="#ff004c")
    ax.set_xticklabels(genre_counts.index, rotation=45, ha="right")
    st.pyplot(fig)

    st.markdown('<div class="card"><h3>🏆 Top 10 Most Rated Movies</h3></div>', unsafe_allow_html=True)
    top10 = rating_counts.sort_values(ascending=False).head(10)
    top10_movies = movies[movies["movieId"].isin(top10.index)].copy()
    top10_movies["num_ratings"] = top10_movies["movieId"].map(rating_counts)
    top10_movies["avg_rating"] = top10_movies["movieId"].map(avg_ratings).round(2)
    st.dataframe(top10_movies[["title", "year", "genres", "num_ratings", "avg_rating"]], use_container_width=True)

# -----------------------------------
# WATCHLIST PAGE
# -----------------------------------
elif page == "📋 Watchlist":
    st.markdown("## 📋 Your Watchlist")

    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add movies from the Recommender page!")
    else:
        st.markdown(f"**{len(st.session_state.watchlist)} movies saved**")
        for i, title in enumerate(st.session_state.watchlist):
            row_data = movies[movies["title"] == title]
            if row_data.empty:
                continue
            row = row_data.iloc[0]
            year = row.get("year", np.nan)
            year_display = "" if pd.isna(year) else int(year)
            avg = avg_ratings.get(row["movieId"], np.nan)
            star_str = stars(avg) if not pd.isna(avg) else "☆☆☆☆☆"
            google_url = get_google_url(title, year)

            col_info, col_btn = st.columns([5, 1])
            with col_info:
                st.markdown(
                    f'<div class="watchlist-item">'
                    f'<div><b>{title}</b> ({year_display})<br>'
                    f'<span style="font-size:12px;opacity:0.7;">{row["genres"]}</span><br>'
                    f'<span style="color:#ffd000;">{star_str}</span>&nbsp;&nbsp;'
                    f'<a href="{google_url}" target="_blank" class="google-link">🔍 Google</a>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
            with col_btn:
                if st.button("🗑 Remove", key=f"rm_{i}"):
                    st.session_state.watchlist.remove(title)
                    st.rerun()

        if st.button("🗑 Clear All"):
            st.session_state.watchlist = []
            st.rerun()

# -----------------------------------
# MODEL EVALUATION PAGE
# -----------------------------------
else:
    st.markdown("## ✅ Model Evaluation")
    st.caption("RMSE and MAE on a 20% test split.")

    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    train_matrix = train.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    sim = cosine_similarity(train_matrix.T)
    sim_df_eval = pd.DataFrame(sim, index=train_matrix.columns, columns=train_matrix.columns)

    def predict_rating(user_id, movie_id):
        if user_id not in train_matrix.index or movie_id not in train_matrix.columns:
            return np.nan
        user_ratings = train_matrix.loc[user_id]
        sims = sim_df_eval[movie_id]
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
    c1.metric("📉 RMSE", f"{rmse:.4f}")
    c2.metric("📏 MAE", f"{mae:.4f}")

    st.markdown(
        '<div class="card"><h3>📝 Notes</h3><p style="opacity:0.85;">'
        "Evaluated on a 2000-sample subset for speed. "
        "Match scores on the Recommender page are shown as percentages (e.g. 87% Match)."
        "</p></div>",
        unsafe_allow_html=True
    )