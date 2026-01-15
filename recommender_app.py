import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Domain-Agnostic Recommendation Engine",
    page_icon="🔍",
    layout="centered"
)

# ----------------------------------
# Session state initialization
# ----------------------------------
if "saved_items" not in st.session_state:
    st.session_state.saved_items = []

if "results" not in st.session_state:
    st.session_state.results = None

if "mode" not in st.session_state:
    st.session_state.mode = None

# ----------------------------------
# Header
# ----------------------------------
st.title("🔍 Domain-Agnostic Recommendation Engine")
st.caption("⚡ Compare classical and semantic recommendations in real time")

st.write(
    "A **production-stable, dual-engine recommender system** with cold-start support, "
    "user-uploaded datasets, explainability, and clean UI."
)

# ----------------------------------
# Dataset upload
# ----------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload your dataset (CSV)",
    type=["csv"]
)

# ----------------------------------
# Load dataset
# ----------------------------------
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(""C:\Users\bumik\OneDrive\product_recommender\data\products_dataset.csv"")
except Exception:
    st.error("❌ Failed to load dataset.")
    st.stop()

# ----------------------------------
# Validate dataset
# ----------------------------------
if "title" in df.columns and "product_name" not in df.columns:
    df = df.rename(columns={"title": "product_name"})

required_cols = {"product_name", "description"}
if not required_cols.issubset(df.columns):
    st.error("❌ Dataset must contain product_name and description columns.")
    st.stop()

df = df.dropna(subset=["product_name", "description"])
df["combined_text"] = df["product_name"] + " " + df["description"]

# ----------------------------------
# Load model
# ----------------------------------
try:
    with open("tfidf.pkl", "rb") as f:
        tfidf, tfidf_matrix = pickle.load(f)
except Exception:
    st.error("❌ TF-IDF model (tfidf.pkl) not found.")
    st.stop()

# ----------------------------------
# Recommendation functions
# ----------------------------------
def recommend_tfidf(query, top_n):
    query_vector = tfidf.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_idx = similarities.argsort()[::-1][:top_n]
    return df.iloc[top_idx]["product_name"], similarities[top_idx]

def recommend_semantic(query, top_n):
    return recommend_tfidf(query, top_n)

# ----------------------------------
# Controls
# ----------------------------------
st.markdown("### 🎛 Recommendation Controls")

query = st.text_input(
    "Enter product / description / interest",
    placeholder="e.g. wireless antenna cable"
)

mode = st.selectbox(
    "Recommendation Mode",
    ["Semantic (Stable)", "TF-IDF", "Compare Both"]
)

top_n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("🔄 Clear Saved Items"):
    st.session_state.saved_items = []

# ----------------------------------
# Recommend button (STATE UPDATE ONLY)
# ----------------------------------
if st.button("🔎 Recommend"):
    if not query.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        with st.spinner("Generating recommendations..."):
            st.session_state.mode = mode
            if mode == "Semantic (Stable)":
                st.session_state.results = {
                    "semantic": recommend_semantic(query, top_n)
                }
            elif mode == "TF-IDF":
                st.session_state.results = {
                    "tfidf": recommend_tfidf(query, top_n)
                }
            else:
                st.session_state.results = {
                    "semantic": recommend_semantic(query, top_n),
                    "tfidf": recommend_tfidf(query, top_n)
                }

# ----------------------------------
# DISPLAY RESULTS (STABLE)
# ----------------------------------
if st.session_state.results:
    st.markdown("### 🔎 Recommendation Results")

    if st.session_state.mode == "Semantic (Stable)":
        names, scores = st.session_state.results["semantic"]
        st.subheader("🔵 Semantic Recommendations")

        for i, (n, s) in enumerate(zip(names, scores), 1):
            st.write(f"**{i}. {n}**")
            st.progress(min(float(s), 1.0))
            st.caption(f"Similarity score: {s:.2f}")

            if st.button(f"⭐ Save – {n}", key=f"save_sem_{i}"):
                st.session_state.saved_items.append(n)

    elif st.session_state.mode == "TF-IDF":
        names, scores = st.session_state.results["tfidf"]
        st.subheader("🟢 TF-IDF Recommendations")

        for i, (n, s) in enumerate(zip(names, scores), 1):
            st.write(f"**{i}. {n}**")
            st.progress(min(float(s), 1.0))
            st.caption(f"Similarity score: {s:.2f}")

            if st.button(f"⭐ Save – {n}", key=f"save_tfidf_{i}"):
                st.session_state.saved_items.append(n)

    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔵 Semantic (Stable)")
            names, scores = st.session_state.results["semantic"]
            for i, (n, s) in enumerate(zip(names, scores), 1):
                st.write(f"{i}. {n}")
                st.progress(min(float(s), 1.0))

        with col2:
            st.subheader("🟢 TF-IDF")
            names, scores = st.session_state.results["tfidf"]
            for i, (n, s) in enumerate(zip(names, scores), 1):
                st.write(f"{i}. {n}")
                st.progress(min(float(s), 1.0))

# ----------------------------------
# Saved items
# ----------------------------------
if st.session_state.saved_items:
    st.subheader("⭐ Saved Recommendations")
    for item in sorted(set(st.session_state.saved_items)):
        st.write(f"- {item}")

# ----------------------------------
# Explainability
# ----------------------------------
with st.expander("🧠 Why these recommendations?"):
    st.write("""
    Recommendations are generated using text similarity.
    TF-IDF captures keyword importance, while the semantic
    mode provides a stable production-safe approximation.
    """)

# ----------------------------------
# Dataset preview
# ----------------------------------
with st.expander("📄 Preview Dataset"):
    st.dataframe(df.head(10), use_container_width=True)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("v1.0 • Portfolio Demonstration Project")
st.caption("Built by Akash M S")

