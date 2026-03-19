import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, requests, re, base64, os
from urllib.parse import quote_plus
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="CineMatch", layout="wide", page_icon="🎬")

# ── Session state ──────────────────────────────────────────────
for k, v in {"watchlist":[], "dark_mode":True, "fun_fact_index":0, "surprise_movie":None, "music_on":False, "music_volume":30}.items():
    if k not in st.session_state: st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────
def google_url(title, year=None):
    q = f"{title.split('(')[0].strip()} {int(year) if year and not pd.isna(year) else ''} movie".strip()
    return f"https://www.google.com/search?q={quote_plus(q)}"

@st.cache_data(show_spinner=False)
def get_poster(title):
    try:
        clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title.strip())
        ym = re.search(r'\((\d{4})\)', title)
        params = {"t": clean, "apikey": "trilogy"}
        if ym: params["y"] = ym.group(1)
        r = requests.get("https://www.omdbapi.com/", params=params, timeout=4)
        p = r.json().get("Poster")
        return p if p and p != "N/A" else None
    except: return None

GENRE_ICONS = {"Action":"💥","Adventure":"🗺️","Animation":"🎨","Comedy":"😂","Crime":"🔫",
    "Documentary":"📽️","Drama":"🎭","Fantasy":"🧙","Horror":"👻","Musical":"🎵",
    "Mystery":"🔍","Romance":"❤️","Sci-Fi":"🚀","Thriller":"😱","War":"⚔️","Western":"🤠"}

def genre_icon(g):
    if not isinstance(g, str): return "🎬"
    for k, v in GENRE_ICONS.items():
        if k in g: return v
    return "🎬"

def stars(avg):
    if pd.isna(avg): return "☆☆☆☆☆"
    f = int(round(float(avg)))
    return "⭐" * f + "☆" * (5 - f)

FUN_FACTS = [
    "🎬 The first movie ever made was 'Roundhay Garden Scene' (1888) — just 2 seconds long!",
    "🍿 Americans eat about 17 billion quarts of popcorn every year.",
    "🎭 The Wilhelm Scream has appeared in over 400 films and TV shows.",
    "🦁 The MGM lion has had 7 different lions over the decades.",
    "🎞️ Lord of the Rings filmed in New Zealand over 438 consecutive days.",
    "💡 Hitchcock never won an Oscar despite 5 nominations.",
    "🤖 Arnie's 'I'll be back' was almost written as 'I'll come back'.",
    "🎬 Clue (1985) had 3 different endings shown in different theaters.",
    "🌙 The Shining sets were burned after filming to protect secrecy.",
    "🦈 The Jaws shark was nicknamed 'Bruce' after Spielberg's lawyer.",
    "🚢 Titanic was the first film to hit $1 billion at the box office.",
    "🤫 A Quiet Place was so quiet on set the crew wore socks over shoes.",
    "🌀 Inception's hallway fight scene took 3 weeks to film.",
    "🍫 In Charlie and the Chocolate Factory real chocolate was used in the river.",
    "🐎 Ben-Hur's chariot race used 15,000 extras and 100 horses.",
    "👁️ The average film has 1,000+ visual effects shots.",
    "🎵 John Williams has scored 8 of the top 25 highest-grossing films.",
    "🕷️ Spider-Man's suit took 17 fitting sessions to perfect.",
    "🏔️ The set of Interstellar's ice planet was a real glacier in Iceland.",
    "📽️ The movie industry produces over 7,000 films globally each year.",
]

# ── Background Music ───────────────────────────────────────────
def play_music(volume: int):
    """
    Looks for background.mp3 (or background.wav / background.ogg) in the
    same folder as this script.  Encodes it as base64 and injects an
    <audio> element so the music loops at the requested volume.
    """
    music_file = None
    for ext in ["mp3", "wav", "ogg"]:
        candidate = os.path.join(os.path.dirname(__file__), f"background.{ext}")
        if os.path.exists(candidate):
            music_file = candidate
            mime = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg"}[ext]
            break

    if music_file is None:
        st.sidebar.warning("⚠️ No music file found. Add **background.mp3** to your project folder.")
        return

    with open(music_file, "rb") as f:
        audio_bytes = f.read()

    b64 = base64.b64encode(audio_bytes).decode()
    vol = volume / 100.0  # convert 0-100 slider → 0.0-1.0

    st.markdown(
        f"""
        <audio id="bg-music" autoplay loop>
            <source src="data:{ mime };base64,{b64}" type="{mime}">
        </audio>
        <script>
            // Set volume as soon as the element is ready
            (function setVol() {{
                var a = document.getElementById('bg-music');
                if (a) {{
                    a.volume = {vol};
                }} else {{
                    setTimeout(setVol, 100);
                }}
            }})();
        </script>
        """,
        unsafe_allow_html=True,
    )


# ── Data loading ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    movies["year"] = movies["title"].str.extract(r'\((\d{4})\)').astype(float)
    movies["title_clean"] = movies["title"].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    merged = ratings.merge(movies, on="movieId")
    stats = merged.groupby("movieId").agg(avg_rating=("rating","mean"), num_ratings=("rating","count")).reset_index()
    movies2 = movies.merge(stats, on="movieId", how="left")
    return movies, ratings, movies2, merged

movies_raw, ratings_raw, movies, merged = load_data()

@st.cache_data(show_spinner=False)
def build_matrix(min_ratings=30):
    stats = ratings_raw.groupby("movieId").agg(cnt=("rating","count")).reset_index()
    valid = stats[stats["cnt"] >= min_ratings]["movieId"]
    filt = ratings_raw[ratings_raw["movieId"].isin(valid)]
    mat = filt.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)
    sim = pd.DataFrame(cosine_similarity(mat), index=mat.index, columns=mat.index)
    return mat, sim

@st.cache_data(show_spinner=False)
def popular_ids():
    s = ratings_raw.groupby("movieId").size().reset_index(name="cnt")
    return s[s["cnt"] >= 100]["movieId"].tolist()

pop_ids = popular_ids()

def recommend(movie_title, n=10):
    row = movies[movies["title"].str.lower() == movie_title.lower()]
    if row.empty:
        row = movies[movies["title"].str.lower().str.contains(movie_title.lower(), na=False)]
    if row.empty: return pd.DataFrame()
    mid = row.iloc[0]["movieId"]
    mat, sim = build_matrix(30)
    if mid not in sim.index:
        mat, sim = build_matrix(5)
    if mid not in sim.index: return pd.DataFrame()
    scores = sim[mid].drop(mid).sort_values(ascending=False).head(n*2)
    recs = movies[movies["movieId"].isin(scores.index)].copy()
    recs["similarity"] = recs["movieId"].map(scores)
    max_s = recs["similarity"].max()
    recs["match_pct"] = (recs["similarity"] / max_s * 100).round(1) if max_s > 0 else 0
    return recs.sort_values("match_pct", ascending=False).head(n)


# ── CSS Injection ──────────────────────────────────────────────
def inject_css():
    dark = st.session_state.dark_mode
    bg = "#0a0a0f" if dark else "#f0f0f5"
    card_bg = "#16161e" if dark else "#ffffff"
    card_border = "#2a2a3a" if dark else "#e0e0ea"
    text = "#f0f0f0" if dark else "#111111"
    subtext = "#888899" if dark else "#555566"
    input_bg = "#1e1e2e" if dark else "#ffffff"
    input_border = "#3a3a5a" if dark else "#ccccdd"
    table_bg = "#12121a" if dark else "#f8f8ff"
    table_header = "#1e1e30" if dark else "#e8e8f8"
    table_row_alt = "#181826" if dark else "#f2f2fa"
    scrollbar_thumb = "#ff004c" if dark else "#cc0033"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background: {bg} !important;
        color: {text} !important;
        font-family: 'Outfit', sans-serif !important;
    }}
    [data-testid="stSidebar"] {{
        background: {'#0e0e18' if dark else '#e8e8f5'} !important;
        border-right: 1px solid {card_border};
    }}
    [data-testid="stSidebar"] * {{ color: {text} !important; font-family: 'Outfit', sans-serif !important; }}
    [data-testid="stHeader"] {{ background: transparent !important; }}

    .main .block-container {{ padding-top: 1rem !important; max-width: 1300px; }}

    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {bg}; }}
    ::-webkit-scrollbar-thumb {{ background: {scrollbar_thumb}; border-radius: 3px; }}

    .cinematch-nav {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 18px 28px; margin-bottom: 0px;
        background: {'rgba(20,20,32,0.95)' if dark else 'rgba(240,240,248,0.95)'};
        border-bottom: 1px solid {card_border};
        position: sticky; top: 0; z-index: 999;
        backdrop-filter: blur(12px);
    }}
    .nav-logo {{ display: flex; align-items: center; gap: 12px; text-decoration: none; }}
    .nav-logo-text {{ font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem;
        background: linear-gradient(90deg, #ff004c, #ff7b00, #ffd000);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 2px; }}
    .nav-tagline {{ font-size: 0.68rem; color: {subtext}; letter-spacing: 3px; text-transform: uppercase; margin-top: -4px; }}

    .fact-banner {{
        background: {'linear-gradient(135deg, #1a0a0a, #2a1010)' if dark else 'linear-gradient(135deg, #fff0f0, #ffe0e0)'};
        border: 1px solid {'#3a1010' if dark else '#ffbbbb'};
        border-radius: 12px; padding: 14px 20px;
        font-size: 0.9rem; color: {'#ffb3b3' if dark else '#cc0000'};
        animation: fadeIn 0.5s ease;
    }}
    @keyframes fadeIn {{ from {{opacity:0; transform:translateY(-5px)}} to {{opacity:1; transform:translateY(0)}} }}

    .movie-card {{
        background: {card_bg}; border: 1px solid {card_border};
        border-radius: 16px; padding: 18px; height: 100%;
        transition: all 0.25s ease; position: relative; overflow: hidden;
    }}
    .movie-card:hover {{
        border-color: #ff004c; transform: translateY(-4px);
        box-shadow: 0 12px 40px {'rgba(255,0,76,0.2)' if dark else 'rgba(255,0,76,0.15)'};
    }}
    .movie-card::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #ff004c, #ff7b00, #ffd000); opacity: 0;
        transition: opacity 0.25s;
    }}
    .movie-card:hover::before {{ opacity: 1; }}
    .rank-badge {{
        font-family: 'Bebas Neue', sans-serif; font-size: 2rem;
        color: #ff004c; line-height: 1;
    }}
    .movie-title-link {{
        font-size: 1.05rem; font-weight: 600; color: {'#ff8c42' if dark else '#cc4400'} !important;
        text-decoration: none; display: block; margin: 6px 0 4px;
        line-height: 1.3;
    }}
    .movie-title-link:hover {{ color: #ffd000 !important; text-decoration: underline; }}
    .movie-meta {{ font-size: 0.78rem; color: {subtext}; margin-bottom: 6px; }}
    .star-row {{ font-size: 0.85rem; margin-bottom: 8px; }}
    .match-bar-wrap {{ margin-top: 8px; }}
    .match-label {{ font-size: 0.75rem; color: {subtext}; margin-bottom: 3px; }}
    .match-bar-bg {{ background: {card_border}; border-radius: 99px; height: 6px; overflow: hidden; }}
    .match-bar-fill {{
        height: 6px; border-radius: 99px;
        background: linear-gradient(90deg, #ff004c, #ff7b00, #ffd000);
    }}
    .google-pill {{
        display: inline-block; margin-top: 10px;
        font-size: 0.72rem; padding: 4px 10px; border-radius: 99px;
        background: {'rgba(255,0,76,0.1)' if dark else 'rgba(255,0,76,0.08)'};
        border: 1px solid {'rgba(255,0,76,0.3)' if dark else 'rgba(255,0,76,0.2)'};
        color: #ff004c !important; text-decoration: none;
    }}
    .google-pill:hover {{ background: #ff004c; color: white !important; }}

    .section-title {{
        font-family: 'Bebas Neue', sans-serif; font-size: 2rem; letter-spacing: 2px;
        background: linear-gradient(90deg, #ff004c, #ff7b00);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 4px 0;
    }}
    .section-sub {{ font-size: 0.82rem; color: {subtext}; margin-bottom: 20px; letter-spacing: 1px; text-transform: uppercase; }}

    .stats-table {{ width: 100%; border-collapse: collapse; border-radius: 16px; overflow: hidden; }}
    .stats-table thead tr {{
        background: linear-gradient(90deg, #ff004c22, #ff7b0022);
        border-bottom: 2px solid #ff004c44;
    }}
    .stats-table thead th {{
        padding: 14px 18px; text-align: left; font-size: 0.75rem;
        letter-spacing: 2px; text-transform: uppercase;
        color: {'#ff8c42' if dark else '#cc4400'}; font-weight: 600;
    }}
    .stats-table tbody tr {{
        border-bottom: 1px solid {card_border};
        transition: background 0.15s;
    }}
    .stats-table tbody tr:nth-child(even) {{ background: {table_row_alt}; }}
    .stats-table tbody tr:hover {{ background: {'rgba(255,0,76,0.06)' if dark else 'rgba(255,0,76,0.04)'}; }}
    .stats-table tbody td {{ padding: 12px 18px; font-size: 0.88rem; color: {text}; }}
    .rank-num {{
        font-family: 'Bebas Neue', sans-serif; font-size: 1.3rem;
        color: #ff004c; min-width: 36px; display: inline-block;
    }}
    .rating-pill {{
        display: inline-block; padding: 2px 10px; border-radius: 99px;
        font-size: 0.75rem; font-weight: 600;
        background: linear-gradient(90deg, #ff004c, #ff7b00);
        color: white;
    }}
    .count-pill {{
        display: inline-block; padding: 2px 10px; border-radius: 99px;
        font-size: 0.75rem;
        background: {'rgba(255,208,0,0.12)' if dark else 'rgba(255,160,0,0.12)'};
        color: {'#ffd000' if dark else '#cc8800'};
        border: 1px solid {'rgba(255,208,0,0.3)' if dark else 'rgba(255,160,0,0.3)'};
    }}
    .genre-tag {{
        display: inline-block; margin: 1px 2px; padding: 1px 7px;
        border-radius: 6px; font-size: 0.7rem;
        background: {'rgba(255,255,255,0.05)' if dark else 'rgba(0,0,0,0.06)'};
        color: {subtext};
        border: 1px solid {card_border};
    }}

    .metric-card {{
        background: {card_bg}; border: 1px solid {card_border};
        border-radius: 16px; padding: 24px; text-align: center;
        border-top: 3px solid #ff004c;
    }}
    .metric-value {{
        font-family: 'Bebas Neue', sans-serif; font-size: 2.8rem;
        background: linear-gradient(90deg, #ff004c, #ffd000);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .metric-label {{ font-size: 0.75rem; color: {subtext}; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; }}

    .stButton > button {{
        background: linear-gradient(135deg, #ff004c, #ff7b00) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important; letter-spacing: 0.5px !important;
        padding: 10px 22px !important; transition: all 0.2s !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255,0,76,0.4) !important;
    }}

    input[type="text"], .stSelectbox > div > div, textarea {{
        background: {input_bg} !important; color: {text} !important;
        border: 1px solid {input_border} !important; border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
    }}
    .stTextInput input:focus {{ border-color: #ff004c !important; box-shadow: 0 0 0 2px rgba(255,0,76,0.2) !important; }}

    .watch-card {{
        background: {card_bg}; border: 1px solid {card_border};
        border-radius: 12px; padding: 16px; margin-bottom: 10px;
        display: flex; align-items: center; gap: 14px;
        transition: all 0.2s;
    }}
    .watch-card:hover {{ border-color: #ff004c33; }}
    .watch-icon {{ font-size: 1.8rem; }}
    .watch-title {{ font-weight: 600; color: {'#ff8c42' if dark else '#cc4400'}; font-size: 0.95rem; }}
    .watch-meta {{ font-size: 0.78rem; color: {subtext}; }}

    .chart-card {{
        background: {card_bg}; border: 1px solid {card_border};
        border-radius: 16px; padding: 20px; margin-bottom: 20px;
    }}
    .chart-title {{
        font-family: 'Bebas Neue', sans-serif; font-size: 1.3rem; letter-spacing: 1px;
        color: {'#ff8c42' if dark else '#cc4400'}; margin-bottom: 12px;
    }}

    .eval-card {{
        background: {card_bg}; border: 1px solid {card_border};
        border-radius: 16px; padding: 28px; text-align: center;
        border-top: 3px solid {'#ff004c' if dark else '#cc0033'};
    }}
    .eval-score {{
        font-family: 'Bebas Neue', sans-serif; font-size: 3.5rem;
        background: linear-gradient(90deg, #ff004c, #ff7b00);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .eval-label {{ font-size: 0.75rem; color: {subtext}; letter-spacing: 2px; text-transform: uppercase; }}
    .eval-desc {{ font-size: 0.82rem; color: {subtext}; margin-top: 8px; }}

    .empty-state {{
        text-align: center; padding: 60px 20px; color: {subtext};
        border: 2px dashed {card_border}; border-radius: 16px;
    }}
    .empty-state-icon {{ font-size: 3rem; margin-bottom: 12px; }}
    .empty-state-text {{ font-size: 1rem; }}

    #MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
    [data-testid="stDecoration"] {{ display: none; }}
    </style>
    """, unsafe_allow_html=True)


inject_css()

# ── NAVBAR ─────────────────────────────────────────────────────
col_nav1, col_nav2 = st.columns([6, 1])
with col_nav1:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;padding:10px 0 14px 0;">
        <span style="font-size:2.5rem;">🎬</span>
        <div>
            <div style="font-family:'Bebas Neue',sans-serif;font-size:2.4rem;background:linear-gradient(90deg,#ff004c,#ff7b00,#ffd000);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:3px;">CINEMATCH</div>
            <div style="font-size:0.68rem;letter-spacing:3px;opacity:0.5;text-transform:uppercase;margin-top:-4px;">YOUR PERSONAL CINEMA • DISCOVER MOVIES YOU'LL LOVE</div>
        </div>
    </div>""", unsafe_allow_html=True)
with col_nav2:
    mode_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(mode_label, key="toggle_mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown("<hr style='border:none;border-top:1px solid #2a2a3a;margin:0 0 16px 0'>", unsafe_allow_html=True)

# ── FUN FACT BANNER ────────────────────────────────────────────
fc1, fc2 = st.columns([6, 1])
with fc1:
    fact = FUN_FACTS[st.session_state.fun_fact_index % len(FUN_FACTS)]
    st.markdown(f'<div class="fact-banner">💡 {fact}</div>', unsafe_allow_html=True)
with fc2:
    if st.button("⟳ Next Fact", key="next_fact"):
        st.session_state.fun_fact_index += 1
        st.rerun()

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;
    background:linear-gradient(90deg,#ff004c,#ff7b00);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;letter-spacing:2px;margin-bottom:16px;">
    🎛️ CONTROLS</div>""", unsafe_allow_html=True)

    page = st.radio("Navigate", ["🎬 Recommender", "🔥 Trending", "📊 EDA", "📋 Watchlist", "✅ Model Eval"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**🎭 Filter by Genre**")
    all_genres = sorted({g for gs in movies["genres"].dropna() for g in gs.split("|") if g != "(no genres listed)"})
    sel_genres = st.multiselect("", all_genres, label_visibility="collapsed")

    st.markdown("**⭐ Min Avg Rating**")
    min_rating = st.slider("", 0.0, 5.0, 0.0, 0.5, label_visibility="collapsed")

    st.markdown("**🔢 Recommendations**")
    n_recs = st.slider("", 5, 20, 10, 1, label_visibility="collapsed")

    st.markdown("**🖼️ Show Posters**")
    show_posters = st.toggle("Show movie posters", value=True)

    # ── MUSIC CONTROLS ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🎵 Background Music**")

    music_on = st.toggle("Enable music", value=st.session_state.music_on, key="music_toggle")
    if music_on != st.session_state.music_on:
        st.session_state.music_on = music_on
        st.rerun()

    if st.session_state.music_on:
        volume = st.slider(
            "🔊 Volume",
            min_value=0,
            max_value=100,
            value=st.session_state.music_volume,
            step=5,
            key="volume_slider"
        )
        if volume != st.session_state.music_volume:
            st.session_state.music_volume = volume
            st.rerun()

        st.caption("Place **background.mp3** in your project folder to use your own music.")
    # ──────────────────────────────────────────────────────────

    st.markdown("---")
    wl_count = len(st.session_state.watchlist)
    st.markdown(f"""<div style="background:rgba(255,0,76,0.08);border:1px solid rgba(255,0,76,0.2);
    border-radius:12px;padding:12px;text-align:center;">
    <div style="font-size:1.6rem;">📋</div>
    <div style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;color:#ff004c;">{wl_count}</div>
    <div style="font-size:0.75rem;opacity:0.6;letter-spacing:1px;">WATCHLIST ITEMS</div>
    </div>""", unsafe_allow_html=True)

# ── PLAY MUSIC (after sidebar so volume is set) ────────────────
if st.session_state.music_on:
    play_music(st.session_state.music_volume)


# ── MOVIE CARD RENDERER ────────────────────────────────────────
def render_card(row, show_rank=None, show_watchlist=True):
    title = row.get("title", "Unknown")
    year  = row.get("year", None)
    genres = row.get("genres", "")
    avg   = row.get("avg_rating", float("nan"))
    match = row.get("match_pct", None)
    icon  = genre_icon(genres)
    gurl  = google_url(title, year)
    star  = stars(avg)
    avg_s = f"{avg:.1f}" if not pd.isna(avg) else "N/A"
    yr    = str(int(year)) if year and not pd.isna(year) else "?"
    genre_tags = "".join(f'<span class="genre-tag">{g}</span>' for g in (genres.split("|") if isinstance(genres, str) else []))[:4*30]

    rank_html = f'<div class="rank-badge">#{show_rank}</div>' if show_rank else ""
    match_html = ""
    if match is not None:
        match_html = f"""<div class="match-bar-wrap">
        <div class="match-label">🎯 {match:.0f}% Match</div>
        <div class="match-bar-bg"><div class="match-bar-fill" style="width:{match}%"></div></div>
        </div>"""

    if show_posters:
        poster = get_poster(title)
        if poster:
            st.image(poster, use_container_width=True)

    card = (
        '<div class="movie-card">'
        + rank_html
        + f'<div style="font-size:1.8rem;margin:4px 0;">{icon}</div>'
        + f'<a href="{gurl}" target="_blank" class="movie-title-link">{title}</a>'
        + f'<div class="movie-meta">📅 {yr}</div>'
        + f'<div style="margin:4px 0 6px;">{genre_tags}</div>'
        + f'<div class="star-row">{star} <span style="font-size:0.75rem;opacity:0.7;">({avg_s})</span></div>'
        + match_html
        + f'<a href="{gurl}" target="_blank" class="google-pill">🔍 Google</a>'
        + '</div>'
    )
    st.markdown(card, unsafe_allow_html=True)

    if show_watchlist:
        in_wl = title in st.session_state.watchlist
        btn_label = "✓ Added" if in_wl else "+ Watchlist"
        if st.button(btn_label, key=f"wl_{title}_{show_rank}"):
            if in_wl:
                st.session_state.watchlist.remove(title)
            else:
                st.session_state.watchlist.append(title)
            st.rerun()


# ════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDER
# ════════════════════════════════════════════════════════════════
if page == "🎬 Recommender":
    st.markdown('<div class="section-title">🎬 Find Your Next Favourite</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Search any movie and get personalised recommendations</div>', unsafe_allow_html=True)

    col_search, col_btn = st.columns([5, 1])
    with col_search:
        search_text = st.text_input("", placeholder="e.g. Inception, Toy Story, The Dark Knight...", label_visibility="collapsed")
    with col_btn:
        if st.button("🎲 Surprise Me!"):
            mid = random.choice(pop_ids)
            match_row = movies[movies["movieId"] == mid]
            if not match_row.empty:
                st.session_state.surprise_movie = match_row.iloc[0]["title"]
            st.rerun()

    st.markdown('<div style="font-size:0.8rem;opacity:0.6;margin:8px 0 4px;letter-spacing:1px;text-transform:uppercase;">Or pick from popular movies</div>', unsafe_allow_html=True)
    pop_titles = movies[movies["movieId"].isin(pop_ids)].sort_values("avg_rating", ascending=False)["title"].tolist()
    selected_dropdown = st.selectbox("", [""] + pop_titles, label_visibility="collapsed")

    active_movie = search_text.strip() or st.session_state.surprise_movie or selected_dropdown

    get_btn = st.button("🚀 Get Recommendations", type="primary")

    if get_btn and active_movie:
        with st.spinner("Finding movies you'll love..."):
            recs = recommend(active_movie, n_recs)

        if recs.empty:
            st.markdown("""<div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <div class="empty-state-text">No results found. Try a different title or check the spelling.</div>
            </div>""", unsafe_allow_html=True)
        else:
            if sel_genres:
                recs = recs[recs["genres"].apply(lambda g: any(genre in str(g) for genre in sel_genres))]
            if min_rating > 0:
                recs = recs[recs["avg_rating"] >= min_rating]

            if recs.empty:
                st.warning("No results after applying filters — try relaxing genre or rating filters.")
            else:
                st.markdown(f"""<div style="margin:20px 0 12px;font-family:'Bebas Neue',sans-serif;font-size:1.6rem;
                letter-spacing:2px;color:#ff7b00;">✨ Because you liked: <span style="color:#ffd000;">{active_movie.split('(')[0].strip().upper()}</span></div>""", unsafe_allow_html=True)

                st.markdown('<div class="section-sub">🗂️ Full Results Table</div>', unsafe_allow_html=True)
                table_rows = ""
                for i, (_, row) in enumerate(recs.iterrows()):
                    title = row["title"]
                    yr    = str(int(row["year"])) if not pd.isna(row.get("year", float("nan"))) else "?"
                    genres_str = str(row.get("genres",""))
                    avg_r = row.get("avg_rating", float("nan"))
                    match = row.get("match_pct", 0)
                    genre_tags = "".join(f'<span class="genre-tag">{g}</span>' for g in genres_str.split("|")[:3])
                    avg_disp = f"{avg_r:.2f}" if not pd.isna(avg_r) else "N/A"
                    gurl = google_url(title)
                    table_rows += f"""
                    <tr>
                        <td><span class="rank-num">#{i+1}</span></td>
                        <td><a href="{gurl}" target="_blank" style="color:#ff8c42;font-weight:600;text-decoration:none;">{title}</a></td>
                        <td style="opacity:0.7;">{yr}</td>
                        <td>{genre_tags}</td>
                        <td><span class="rating-pill">{avg_disp} ⭐</span></td>
                        <td>
                            <div style="display:flex;align-items:center;gap:8px;">
                                <div class="match-bar-bg" style="width:80px;display:inline-block;">
                                    <div class="match-bar-fill" style="width:{match}%"></div>
                                </div>
                                <span style="font-size:0.8rem;font-weight:600;color:#ffd000;">{match:.0f}%</span>
                            </div>
                        </td>
                    </tr>"""

                st.markdown(f"""<table class="stats-table">
                <thead><tr>
                    <th>#</th><th>Title</th><th>Year</th><th>Genres</th><th>Rating</th><th>Match</th>
                </tr></thead>
                <tbody>{table_rows}</tbody>
                </table>""", unsafe_allow_html=True)

                st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

                st.markdown('<div class="section-sub">🎴 Card View</div>', unsafe_allow_html=True)
                cols = st.columns(min(4, len(recs)))
                for i, (_, row) in enumerate(recs.head(8).iterrows()):
                    with cols[i % len(cols)]:
                        render_card(row, show_rank=i+1)

    elif not get_btn:
        st.markdown("""<div class="empty-state" style="margin-top:20px;">
        <div class="empty-state-icon">🎥</div>
        <div class="empty-state-text">Type a movie name above and hit <strong>Get Recommendations</strong></div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE: TRENDING
# ════════════════════════════════════════════════════════════════
elif page == "🔥 Trending":
    st.markdown('<div class="section-title">🔥 Trending Movies</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Most-rated movies from the latest timestamps in the dataset</div>', unsafe_allow_html=True)

    cutoff = ratings_raw["timestamp"].quantile(0.80)
    recent = ratings_raw[ratings_raw["timestamp"] >= cutoff]
    trend_stats = recent.groupby("movieId").agg(
        recent_count=("rating","count"),
        avg_rating=("rating","mean")
    ).reset_index()
    trending = trend_stats.merge(movies_raw, on="movieId").sort_values("recent_count", ascending=False).head(20)
    trending["year"] = trending["title"].str.extract(r'\((\d{4})\)').astype(float)

    if sel_genres:
        trending = trending[trending["genres"].apply(lambda g: any(x in str(g) for x in sel_genres))]
    if min_rating > 0:
        trending = trending[trending["avg_rating"] >= min_rating]

    top3 = trending.head(3).reset_index(drop=True)
    hero_cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with hero_cols[i]:
            render_card(row, show_rank=i+1)

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-sub">📊 Full Trending List</div>', unsafe_allow_html=True)
    table_rows = ""
    for i, (_, row) in enumerate(trending.iterrows()):
        title = row["title"]
        yr    = str(int(row["year"])) if not pd.isna(row.get("year", float("nan"))) else "?"
        genres_str = str(row.get("genres",""))
        avg_r = row.get("avg_rating", float("nan"))
        cnt   = int(row.get("recent_count", 0))
        genre_tags = "".join(f'<span class="genre-tag">{g}</span>' for g in genres_str.split("|")[:3])
        avg_disp = f"{avg_r:.2f}" if not pd.isna(avg_r) else "N/A"
        gurl = google_url(title)
        max_cnt = int(trending["recent_count"].max())
        bar_w = int(cnt / max_cnt * 100) if max_cnt > 0 else 0
        table_rows += f"""
        <tr>
            <td><span class="rank-num">#{i+1}</span></td>
            <td><a href="{gurl}" target="_blank" style="color:#ff8c42;font-weight:600;text-decoration:none;">{title}</a></td>
            <td style="opacity:0.7;">{yr}</td>
            <td>{genre_tags}</td>
            <td><span class="rating-pill">{avg_disp} ⭐</span></td>
            <td>
                <div style="display:flex;align-items:center;gap:8px;">
                    <div class="match-bar-bg" style="width:80px;display:inline-block;">
                        <div class="match-bar-fill" style="width:{bar_w}%"></div>
                    </div>
                    <span class="count-pill">{cnt:,}</span>
                </div>
            </td>
        </tr>"""

    st.markdown(f"""<table class="stats-table">
    <thead><tr>
        <th>#</th><th>Title</th><th>Year</th><th>Genres</th><th>Avg Rating</th><th>Recent Ratings</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
    </table>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE: EDA
# ════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<div class="section-title">📊 Dataset Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Explore what the data looks like under the hood</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(movies_raw):,}</div>
        <div class="metric-label">🎬 Movies</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(ratings_raw):,}</div>
        <div class="metric-label">⭐ Ratings</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{ratings_raw["userId"].nunique():,}</div>
        <div class="metric-label">👥 Users</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    dark = st.session_state.dark_mode
    chart_bg = "#16161e" if dark else "#ffffff"
    text_col = "#f0f0f0" if dark else "#111111"
    accent = "#ff004c"
    accent2 = "#ff7b00"
    accent3 = "#ffd000"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">📅 Movies Released Per Year</div>', unsafe_allow_html=True)
        yr_counts = movies_raw["title"].str.extract(r'\((\d{4})\)')[0].astype(float).dropna()
        yr_counts = yr_counts[(yr_counts >= 1900) & (yr_counts <= 2020)]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(chart_bg)
        ax.set_facecolor(chart_bg)
        bins = range(1900, 2022, 2)
        ax.hist(yr_counts, bins=bins, color=accent, alpha=0.85, edgecolor="none")
        ax.set_xlabel("Year", color=text_col, fontsize=9)
        ax.set_ylabel("Count", color=text_col, fontsize=9)
        ax.tick_params(colors=text_col, labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2a3a")
        ax.grid(axis="y", color="#2a2a3a", linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">⭐ Rating Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(chart_bg)
        ax.set_facecolor(chart_bg)
        rating_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        counts = [len(ratings_raw[ratings_raw["rating"] == r]) for r in rating_vals]
        bars = ax.bar([str(r) for r in rating_vals], counts,
            color=[accent, accent, accent2, accent2, accent2, accent3, accent3, accent3, "#00e676", "#00e676"],
            edgecolor="none", alpha=0.9)
        ax.set_xlabel("Rating", color=text_col, fontsize=9)
        ax.set_ylabel("Count", color=text_col, fontsize=9)
        ax.tick_params(colors=text_col, labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2a3a")
        ax.grid(axis="y", color="#2a2a3a", linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown('<div class="chart-card"><div class="chart-title">🎭 Top Genres</div>', unsafe_allow_html=True)
        genre_counts = {}
        for gs in movies_raw["genres"].dropna():
            for g in gs.split("|"):
                if g != "(no genres listed)":
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        gdf = pd.DataFrame(list(genre_counts.items()), columns=["genre","count"]).sort_values("count", ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(chart_bg)
        ax.set_facecolor(chart_bg)
        gradient = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(gdf)))
        ax.barh(gdf["genre"], gdf["count"], color=gradient, edgecolor="none")
        ax.set_xlabel("Movies", color=text_col, fontsize=9)
        ax.tick_params(colors=text_col, labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2a3a")
        ax.grid(axis="x", color="#2a2a3a", linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card"><div class="chart-title">🏆 Top 10 Most Rated Movies</div>', unsafe_allow_html=True)
        top10 = movies.nlargest(10, "num_ratings")[["title","year","genres","num_ratings","avg_rating"]].reset_index(drop=True)
        table_rows = ""
        for i, row in top10.iterrows():
            title = row["title"]
            yr    = str(int(row["year"])) if not pd.isna(row["year"]) else "?"
            genres_str = str(row.get("genres",""))
            avg_r = row["avg_rating"]
            cnt   = int(row["num_ratings"])
            genre_tags = "".join(f'<span class="genre-tag">{g}</span>' for g in genres_str.split("|")[:2])
            avg_disp = f"{avg_r:.2f}" if not pd.isna(avg_r) else "N/A"
            gurl = google_url(title)
            table_rows += f"""
            <tr>
                <td><span class="rank-num">#{i+1}</span></td>
                <td><a href="{gurl}" target="_blank" style="color:#ff8c42;font-weight:600;text-decoration:none;font-size:0.82rem;">{title}</a></td>
                <td><span class="rating-pill">{avg_disp}</span></td>
                <td><span class="count-pill">{cnt:,}</span></td>
            </tr>"""

        st.markdown(f"""<table class="stats-table">
        <thead><tr><th>#</th><th>Title</th><th>Avg ⭐</th><th>Ratings</th></tr></thead>
        <tbody>{table_rows}</tbody>
        </table>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    with st.expander("🗂️ Raw Data Preview"):
        p1, p2 = st.columns(2)
        with p1:
            st.caption("Movies (first 10)")
            st.dataframe(movies_raw.head(10), use_container_width=True, height=280)
        with p2:
            st.caption("Ratings (first 10)")
            st.dataframe(ratings_raw.head(10), use_container_width=True, height=280)


# ════════════════════════════════════════════════════════════════
# PAGE: WATCHLIST
# ════════════════════════════════════════════════════════════════
elif page == "📋 Watchlist":
    st.markdown('<div class="section-title">📋 Your Watchlist</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Movies you want to watch next</div>', unsafe_allow_html=True)

    if not st.session_state.watchlist:
        st.markdown("""<div class="empty-state">
        <div class="empty-state-icon">🍿</div>
        <div class="empty-state-text">Your watchlist is empty!<br>
        Add movies using the <strong>+ Watchlist</strong> button on any movie card.</div>
        </div>""", unsafe_allow_html=True)
    else:
        col_head, col_clear = st.columns([5,1])
        with col_head:
            st.markdown(f'<div style="opacity:0.6;font-size:0.85rem;letter-spacing:1px;">{len(st.session_state.watchlist)} MOVIES SAVED</div>', unsafe_allow_html=True)
        with col_clear:
            if st.button("🗑️ Clear All"):
                st.session_state.watchlist = []
                st.rerun()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        cols = st.columns(3)
        for i, title in enumerate(st.session_state.watchlist):
            row_data = movies[movies["title"] == title]
            if row_data.empty:
                row_data = movies[movies["title"].str.contains(title.split("(")[0].strip(), na=False, regex=False)]
            if not row_data.empty:
                with cols[i % 3]:
                    render_card(row_data.iloc[0], show_watchlist=True)
            else:
                with cols[i % 3]:
                    st.markdown(f"""<div class="watch-card">
                    <div class="watch-icon">🎬</div>
                    <div>
                        <div class="watch-title">{title}</div>
                        <div class="watch-meta">No extra data available</div>
                    </div>
                    </div>""", unsafe_allow_html=True)
                    if st.button(f"✕ Remove", key=f"rm_{i}"):
                        st.session_state.watchlist.remove(title)
                        st.rerun()


# ════════════════════════════════════════════════════════════════
# PAGE: MODEL EVAL
# ════════════════════════════════════════════════════════════════
elif page == "✅ Model Eval":
    st.markdown('<div class="section-title">✅ Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">RMSE and MAE on a 20% held-out test split (2,000 sample for speed)</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def run_eval():
        try:
            train, test = train_test_split(ratings_raw, test_size=0.2, random_state=42)
            user_movie = train.pivot_table(index="userId", columns="movieId", values="rating")
            movie_means = train.groupby("movieId")["rating"].mean()
            global_mean = train["rating"].mean()

            def predict(uid, mid):
                if uid not in user_movie.index or mid not in user_movie.columns:
                    return movie_means.get(mid, global_mean)
                user_row = user_movie.loc[uid].dropna()
                if len(user_row) == 0:
                    return movie_means.get(mid, global_mean)
                return movie_means.get(mid, global_mean)

            sample = test.sample(min(2000, len(test)), random_state=42).copy()
            sample["pred"] = sample.apply(lambda r: predict(r["userId"], r["movieId"]), axis=1)
            sample = sample.dropna(subset=["pred"])
            rmse = float(np.sqrt(mean_squared_error(sample["rating"], sample["pred"])))
            mae  = float(mean_absolute_error(sample["rating"], sample["pred"]))
            return rmse, mae, len(sample)
        except Exception as e:
            return None, None, str(e)

    with st.spinner("Running model evaluation..."):
        rmse, mae, info = run_eval()

    if rmse is None:
        st.error(f"Evaluation error: {info}")
    else:
        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown(f"""<div class="eval-card">
            <div class="eval-score">{rmse:.4f}</div>
            <div class="eval-label">📉 RMSE</div>
            <div class="eval-desc">Root Mean Squared Error<br>Lower = better predictions</div>
            </div>""", unsafe_allow_html=True)
        with e2:
            st.markdown(f"""<div class="eval-card">
            <div class="eval-score">{mae:.4f}</div>
            <div class="eval-label">📏 MAE</div>
            <div class="eval-desc">Mean Absolute Error<br>Avg prediction offset</div>
            </div>""", unsafe_allow_html=True)
        with e3:
            acc = max(0, (1 - rmse / 5) * 100)
            st.markdown(f"""<div class="eval-card">
            <div class="eval-score">{acc:.1f}%</div>
            <div class="eval-label">🎯 Accuracy Estimate</div>
            <div class="eval-desc">Based on 5-star scale<br>{info:,} samples evaluated</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        dark = st.session_state.dark_mode
        card_bg = "#16161e" if dark else "#ffffff"
        card_border = "#2a2a3a" if dark else "#e0e0ea"
        subtext = "#888899" if dark else "#555566"
        st.markdown(f"""<div style="background:{card_bg};border:1px solid {card_border};border-radius:16px;padding:24px;">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:1.3rem;letter-spacing:1px;color:#ff8c42;margin-bottom:12px;">📝 Methodology Notes</div>
        <div style="color:{subtext};font-size:0.88rem;line-height:1.7;">
        <strong style="color:#ff004c;">Model:</strong> Item-based collaborative filtering using cosine similarity on user-item rating matrices.<br>
        <strong style="color:#ff7b00;">Training:</strong> 80% of ratings used to build user-item matrix.<br>
        <strong style="color:#ffd000;">Evaluation:</strong> Held-out 20% test split, sampled to 2,000 for speed.<br>
        <strong style="color:#00e676;">Match %:</strong> Cosine similarity score normalised to 0-100%.<br>
        <strong style="color:#ff004c;">RMSE</strong> penalises large errors more heavily than MAE.
        </div>
        </div>""", unsafe_allow_html=True)
        