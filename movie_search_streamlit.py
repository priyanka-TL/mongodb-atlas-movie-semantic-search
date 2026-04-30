import html
import os
import ssl
from typing import Any

import pymongo
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DB = "sample_mflix"
DEFAULT_COLLECTION = "movies"
DEFAULT_INDEX = "plotSemanticSearch"
PLACEHOLDER_POSTER = "https://placehold.co/360x540/111827/e5e7eb?text=No+Poster"


def get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise ValueError(f"{name} not set. Add it to your .env file.")
    return value


@st.cache_resource
def get_collection() -> pymongo.collection.Collection:
    mongo_uri = get_env("MONGO_URI")
    
    # Configure MongoDB client with proper TLS settings for Atlas
    client = pymongo.MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=False,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
    )
    
    database_name = os.getenv("MONGO_DB", DEFAULT_DB)
    collection_name = os.getenv("MONGO_COLLECTION", DEFAULT_COLLECTION)
    return client[database_name][collection_name]


@st.cache_resource
def get_hf_client() -> InferenceClient:
    hf_token = get_env("HUGGING_FACE_TOKEN")
    return InferenceClient(token=hf_token)


def generate_embedding(text: str) -> list[float]:
    model_name = os.getenv("HUGGING_FACE_MODEL", DEFAULT_MODEL)
    embedding: Any = get_hf_client().feature_extraction(text, model=model_name)

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
        return embedding[0]

    if isinstance(embedding, list):
        return embedding

    raise ValueError("Unexpected embedding format from Hugging Face API")


def search_movies(query: str, limit: int = 5) -> list[dict[str, Any]]:
    vector = generate_embedding(query)
    index_name = os.getenv("ATLAS_VECTOR_INDEX", DEFAULT_INDEX)

    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": vector,
                "path": "plot_embedding_hf",
                "numCandidates": max(100, limit * 20),
                "limit": limit,
                "index": index_name,
            }
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "plot": 1,
                "fullplot": 1,
                "year": 1,
                "poster": 1,
                "genres": 1,
                "runtime": 1,
                "cast": 1,
                "directors": 1,
                "languages": 1,
                "countries": 1,
                "rated": 1,
                "awards": 1,
                "imdb": 1,
                "tomatoes": 1,
                "type": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(get_collection().aggregate(pipeline))


def join_list(values: list[Any] | None, fallback: str = "Unknown") -> str:
    if not values:
        return fallback
    return ", ".join(html.escape(str(value)) for value in values if value)


def format_runtime(minutes: int | None) -> str:
    if not minutes:
        return "Runtime unknown"
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def render_metric(label: str, value: Any) -> None:
    st.markdown(
        f"""
        <div class="metric-tile">
            <span>{html.escape(str(label))}</span>
            <strong>{html.escape(str(value))}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score(score: float) -> None:
    score_percent = max(0, min(score, 1)) * 100
    st.markdown(
        f"""
        <div class="score-wrap">
            <div class="score-row">
                <span>Search score</span>
                <strong>{score:.4f}</strong>
            </div>
            <div class="score-track">
                <div class="score-fill" style="width: {score_percent:.1f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 12% 8%, rgba(20, 184, 166, 0.18), transparent 28rem),
                    radial-gradient(circle at 88% 0%, rgba(249, 115, 22, 0.12), transparent 24rem),
                    linear-gradient(135deg, #0f172a 0%, #111827 44%, #18181b 100%);
                color: #f8fafc;
            }

            .block-container {
                max-width: 1180px;
                padding-top: 2.4rem;
                padding-bottom: 4rem;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            .hero {
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 8px;
                padding: 1.4rem 1.5rem 1.5rem;
                background: rgba(15, 23, 42, 0.78);
                box-shadow: 0 22px 70px rgba(0, 0, 0, 0.30);
                margin-bottom: 1.4rem;
            }

            .hero h1 {
                font-size: clamp(2.25rem, 5vw, 4.75rem);
                line-height: 0.95;
                margin: 0 0 0.75rem;
                letter-spacing: 0;
            }

            .hero p {
                max-width: 760px;
                color: #cbd5e1;
                font-size: 1.05rem;
                margin: 0;
            }

            /* Style the search form */
            div[data-testid="stForm"] {
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 12px;
                padding: 1.35rem 1.5rem 1.45rem;
                background: rgba(15, 23, 42, 0.88);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.20);
                margin-bottom: 1.2rem;
            }

            .search-label {
                color: #f8fafc;
                font-size: 1.05rem;
                font-weight: 600;
                margin: 0 0 0.85rem;
            }

            div[data-testid="stTextInput"] input {
                border-radius: 10px;
                border: 1.5px solid rgba(148, 163, 184, 0.38);
                background: rgba(2, 6, 23, 0.75);
                color: #f8fafc;
                font-size: 1.05rem;
                padding: 0.85rem 1rem;
                transition: all 0.2s ease;
            }

            div[data-testid="stTextInput"] input:focus {
                border-color: rgba(94, 234, 212, 0.65);
                box-shadow: 0 0 0 3px rgba(94, 234, 212, 0.15);
                background: rgba(2, 6, 23, 0.95);
            }

            div[data-testid="stNumberInput"] input {
                border-radius: 8px;
                border: 1px solid rgba(148, 163, 184, 0.34);
                background: rgba(15, 23, 42, 0.86);
                color: #f8fafc;
            }

            div[data-testid="stButton"] button[kind="primary"] {
                border-radius: 10px;
                min-height: 3.2rem;
                font-weight: 700;
                font-size: 1.05rem;
                background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
                border: none;
                box-shadow: 0 8px 24px rgba(20, 184, 166, 0.30);
                transition: all 0.2s ease;
            }

            div[data-testid="stButton"] button[kind="primary"]:hover {
                background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
                box-shadow: 0 12px 32px rgba(20, 184, 166, 0.40);
                transform: translateY(-1px);
            }

            .search-hint {
                color: #94a3b8;
                font-size: 0.88rem;
                margin-top: 0.65rem;
            }

            .search-hint strong {
                color: #cbd5e1;
            }

            .result-card {
                border: 1px solid rgba(148, 163, 184, 0.24);
                border-radius: 8px;
                padding: 1rem;
                background: rgba(15, 23, 42, 0.80);
                box-shadow: 0 18px 52px rgba(0, 0, 0, 0.26);
                margin: 0.9rem 0 1.1rem;
            }

            .result-card h2 {
                margin: 0 0 0.35rem;
                font-size: 1.65rem;
                line-height: 1.12;
                letter-spacing: 0;
            }

            .muted {
                color: #94a3b8;
            }

            .chips {
                display: flex;
                flex-wrap: wrap;
                gap: 0.42rem;
                margin: 0.75rem 0 0.9rem;
            }

            .chip {
                border: 1px solid rgba(148, 163, 184, 0.28);
                border-radius: 999px;
                color: #dbeafe;
                background: rgba(30, 41, 59, 0.80);
                padding: 0.22rem 0.56rem;
                font-size: 0.78rem;
                line-height: 1.25;
            }

            .plot {
                color: #e2e8f0;
                font-size: 0.98rem;
                line-height: 1.55;
                margin: 0.5rem 0 1rem;
            }

            .meta-line {
                color: #cbd5e1;
                font-size: 0.9rem;
                margin: 0.28rem 0;
            }

            .meta-line strong {
                color: #f8fafc;
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.55rem;
                margin: 0.85rem 0;
            }

            .metric-tile {
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 8px;
                padding: 0.55rem 0.65rem;
                background: rgba(2, 6, 23, 0.34);
                min-height: 4.2rem;
            }

            .metric-tile span {
                display: block;
                color: #94a3b8;
                font-size: 0.76rem;
                margin-bottom: 0.2rem;
            }

            .metric-tile strong {
                display: block;
                color: #f8fafc;
                font-size: 1rem;
                line-height: 1.2;
            }

            .score-wrap {
                margin-top: 0.75rem;
            }

            .score-row {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                color: #cbd5e1;
                font-size: 0.88rem;
                margin-bottom: 0.32rem;
            }

            .score-row strong {
                color: #5eead4;
            }

            .score-track {
                height: 0.42rem;
                overflow: hidden;
                border-radius: 999px;
                background: rgba(148, 163, 184, 0.22);
            }

            .score-fill {
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, #14b8a6, #f97316);
            }

            [data-testid="stImage"] img {
                border-radius: 8px;
                border: 1px solid rgba(148, 163, 184, 0.24);
                box-shadow: 0 18px 44px rgba(0, 0, 0, 0.34);
            }

            @media (max-width: 760px) {
                .block-container {
                    padding-top: 1.2rem;
                }

                .hero {
                    padding: 1.05rem;
                }

                .metric-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_movie_card(item: dict[str, Any]) -> None:
    title = html.escape(str(item.get("title", "Untitled")))
    year = item.get("year")
    poster = item.get("poster") or PLACEHOLDER_POSTER
    score = float(item.get("score", 0))
    imdb = item.get("imdb") or {}
    tomatoes = item.get("tomatoes") or {}
    awards = item.get("awards") or {}
    critic = tomatoes.get("critic") or {}
    viewer = tomatoes.get("viewer") or {}
    description = html.escape(str(item.get("fullplot") or item.get("plot") or "No plot available."))

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    poster_col, content_col = st.columns([1, 2.3], gap="large")

    with poster_col:
        st.image(poster, use_container_width=True)

    with content_col:
        title_bits = [str(year)] if year else []
        if item.get("rated"):
            title_bits.append(str(item["rated"]))
        if item.get("runtime"):
            title_bits.append(format_runtime(item["runtime"]))

        st.markdown(f"<h2>{title}</h2>", unsafe_allow_html=True)
        if title_bits:
            st.markdown(f'<div class="muted">{" &bull; ".join(title_bits)}</div>', unsafe_allow_html=True)

        genres = item.get("genres") or []
        if genres:
            chips = "".join(f'<span class="chip">{html.escape(str(genre))}</span>' for genre in genres)
            st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

        st.markdown(f'<p class="plot">{description}</p>', unsafe_allow_html=True)

        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        metric_cols = st.columns(3)
        with metric_cols[0]:
            rating = imdb.get("rating")
            votes = imdb.get("votes")
            render_metric("IMDb", f"{rating}/10" if rating else "N/A")
            if votes:
                st.caption(f"{votes:,} votes")
        with metric_cols[1]:
            meter = critic.get("meter")
            render_metric("Critics", f"{meter}%" if meter is not None else "N/A")
            if critic.get("rating"):
                st.caption(f"{critic['rating']}/10 avg")
        with metric_cols[2]:
            viewer_meter = viewer.get("meter")
            render_metric("Audience", f"{viewer_meter}%" if viewer_meter is not None else "N/A")
            if viewer.get("rating"):
                st.caption(f"{viewer['rating']}/5 avg")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="meta-line"><strong>Director:</strong> {join_list(item.get("directors"))}</div>
            <div class="meta-line"><strong>Cast:</strong> {join_list((item.get("cast") or [])[:5])}</div>
            <div class="meta-line"><strong>Languages:</strong> {join_list(item.get("languages"))}</div>
            <div class="meta-line"><strong>Countries:</strong> {join_list(item.get("countries"))}</div>
            <div class="meta-line"><strong>Awards:</strong> {html.escape(str(awards.get("text", "No award data")))}</div>
            """,
            unsafe_allow_html=True,
        )

        render_score(score)

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Movie Finder", page_icon="🎬", layout="wide")
    apply_styles()

    st.markdown(
        """
        <section class="hero">
            <h1>Movie Finder</h1>
            <p>Find movies by story, mood, or scene using semantic search powered by MongoDB Atlas and Hugging Face embeddings.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.form("search_form", clear_on_submit=False):
        st.markdown('<div class="search-label">What kind of movie are you looking for?</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input(
                "Search query",
                placeholder="e.g., a suspenseful thriller on a train, romantic comedy in Paris, space adventure with aliens...",
                label_visibility="collapsed",
            )
        with col2:
            top_k = st.number_input("Results", min_value=1, max_value=20, value=5, step=1, label_visibility="collapsed")
        
        submitted = st.form_submit_button("🔍 Search Movies", type="primary", use_container_width=True)
        
        st.markdown(
            '<div class="search-hint"><strong>Search tips:</strong> Use natural language to describe plot, genre, mood, setting, or themes</div>',
            unsafe_allow_html=True,
        )

    if submitted:
        if not query.strip():
            st.warning("Please enter a query.")
            return

        try:
            results = search_movies(query, limit=top_k)
        except Exception as error:
            st.error(f"Search failed: {error}")
            return

        if not results:
            st.info("No movies found.")
            return

        st.success(f'Found {len(results)} result(s) for "{query.strip()}"')
        for item in results:
            render_movie_card(item)


if __name__ == "__main__":
    main()
