import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("./data/clean_openlibrary_books.csv")
    df = df.dropna(subset=["title"]).reset_index(drop=True)
    return df

df = load_data()

# ----------------------------
# Build Content Model (TF-IDF)
# ----------------------------
@st.cache_resource
def build_content_model(df):
    df["text"] = (
        df["title"].fillna("") + " " +
        df["main_author"].fillna("") + " " +
        df["subjects"].fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english", min_df=2)
    X = tfidf.fit_transform(df["text"])
    sim = cosine_similarity(X)
    return sim

content_sim = build_content_model(df)

# ----------------------------
# Collaborative Filtering
# ----------------------------
if "interactions" not in st.session_state:
    st.session_state["interactions"] = defaultdict(set)

def add_like(user, book_idx):
    st.session_state["interactions"][user].add(book_idx)

def get_cf_recommendations(user, k=10):
    if user not in st.session_state["interactions"] or len(st.session_state["interactions"][user]) == 0:
        return []
    liked = list(st.session_state["interactions"][user])
    sims = content_sim[liked].mean(axis=0)
    for i in liked:
        sims[i] = -1e9
    recs = np.argsort(-sims)[:k]
    return recs

# ----------------------------
# Recommendation Functions
# ----------------------------
def recommend_content(fav_titles, k=10):
    if not fav_titles:
        return []
    indices = df[df["title"].isin(fav_titles)].index.tolist()
    sims = content_sim[indices].mean(axis=0)
    for i in indices:
        sims[i] = -1e9
    recs = np.argsort(-sims)[:k]
    return recs

def recommend_hybrid(user, fav_titles, hybrid_type="Balanced", k=10):
    if hybrid_type == "Mostly Content-Based":
        alpha = 0.8
    elif hybrid_type == "Mostly Collaborative":
        alpha = 0.2
    else:
        alpha = 0.5

    content_recs = recommend_content(fav_titles, k*2)
    cf_recs = get_cf_recommendations(user, k*2)

    scores = np.zeros(len(df))
    if len(content_recs) > 0:
        scores[content_recs] += alpha
    if len(cf_recs) > 0:
        scores[cf_recs] += (1-alpha)
    recs = np.argsort(-scores)[:k]
    return recs

# ----------------------------
# Page Styling (library look)
# ----------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f4f1ea; /* warm parchment color */
    background-image: url("https://www.transparenttextures.com/patterns/old-wall.png");
    background-size: cover;
}
[data-testid="stSidebar"] {
    background-color: #e0dacd; /* sidebar softer beige */
}
h1, h2, h3 {
    color: #4b2e2e; /* dark brown titles */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📚 Book Recommendation Engine")
st.write("A prototype recommender system using **Content-Based**, **Collaborative Filtering**, and **Hybrid** methods.")

user = st.sidebar.text_input("👤 Enter your user name:", "guest")
k = st.sidebar.slider("🔢 Number of recommendations", 1, 10, 5)

mode = st.sidebar.radio(
    "Recommendation Type:",
    ["Content-Based (Pick Favorites)", "Collaborative Filtering (My Likes)", "Hybrid"]
)

if mode == "Content-Based (Pick Favorites)" or mode == "Hybrid":
    fav_books = st.multiselect("⭐ Pick your favorite books:", df["title"].unique())
else:
    fav_books = []

if mode == "Hybrid":
    hybrid_type = st.sidebar.radio(
        "Mixing strategy:",
        ["Mostly Content-Based", "Balanced", "Mostly Collaborative"]
    )
else:
    hybrid_type = "Balanced"

if st.button("🔍 Get Recommendations"):
    if mode == "Content-Based (Pick Favorites)":
        recs = recommend_content(fav_books, k)
    elif mode == "Collaborative Filtering (My Likes)":
        recs = get_cf_recommendations(user, k)
    else:
        recs = recommend_hybrid(user, fav_books, hybrid_type, k)

    if recs is None or len(recs) == 0:
        st.warning("No recommendations available. Please pick some favorites or like some books first.")
    else:
        st.subheader("📖 Recommendations for you:")
        for idx in recs:
            row = df.iloc[idx]
            st.markdown(f"**{row['title']}** — *{row['main_author']}*")
            st.caption(f"Subjects: {row['subjects']}")
            if "cover_url" in row and pd.notna(row["cover_url"]):
                st.image(row["cover_url"], width=100)
            if st.button(f"👍 Like {row['title']}", key=f"like_{idx}"):
                add_like(user, idx)
                st.success(f"You liked {row['title']}!")

    # 📷 Add your books photo below results
    st.image("Books.jpg", use_column_width=True, caption="So many books, so little time 📚")
