"""Movie Connection & Discovery Graph — Streamlit app.

Run with:
    streamlit run app.py

The LLM connection-story feature is optional and runs entirely locally via
Ollama.  All other features work without any external services or API keys.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Discovery Graph",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports (after page config)
# ---------------------------------------------------------------------------
from src import algorithms as alg
from src import llm as llm_mod
from src import visualization as viz
from src.data_loader import load_all
from src.graph_builder import get_or_build_graph
from src.models import Movie

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .tag-pill {
        display: inline-block;
        background: #2c3e50;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.82em;
    }
    .section-divider { border-top: 1px solid #2c3e50; margin: 16px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data & graph (cached across reruns)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading movie data from ml-32m…")
def _load_movies() -> Dict[int, Movie]:
    return load_all(min_ratings=100)


@st.cache_resource(
    show_spinner="Building movie graph (first run: ~2–5 min for ml-32m)…"
)
def _load_graph(movies: Dict[int, Movie]) -> nx.Graph:
    return get_or_build_graph(movies)


# ---------------------------------------------------------------------------
# Sidebar navigation + LLM settings
# ---------------------------------------------------------------------------

PAGES = [
    "Search & Neighbourhood",
    "Path Finder",
    "Rankings",
    "Recommendations",
    "Explore & Filter",
    "Graph Statistics",
]

st.sidebar.title("🎬 Movie Discovery")
st.sidebar.markdown("Explore cinema through a connection graph.")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.divider()

# --- LLM settings ---
st.sidebar.subheader("Connection Story (LLM)")

llm_mode = st.sidebar.selectbox(
    "Mode",
    ["Off (no LLM)", "Ollama (local)"],
    help="Ollama must be running locally.  All other features work without it.",
)

_ollama_model: Optional[str] = None
_ollama_host = "http://localhost:11434"

if llm_mode == "Ollama (local)":
    _base_model = st.sidebar.text_input("Model name", value=llm_mod.BASE_MODEL)
    _quant = st.sidebar.selectbox(
        "Quantisation",
        ["int4 (q4_K_M)", "int8 (q8_0)"],
        help="int4 uses less RAM; int8 is slightly more accurate.",
    )
    _quant_key = {"int4 (q4_K_M)": "int4", "int8 (q8_0)": "int8"}[_quant]
    _ollama_model = llm_mod.resolve_model_name(_base_model, _quant_key)
    _ollama_host = st.sidebar.text_input("Ollama host", value="http://localhost:11434")

    _running = llm_mod.ollama_is_running(_ollama_host)
    if _running:
        _local = llm_mod.list_local_models(_ollama_host)
        _model_pulled = any(_ollama_model in m for m in _local)

        if _model_pulled:
            st.sidebar.success(f"Ready — `{_ollama_model}`")
        else:
            st.sidebar.error(
                f"Model **`{_ollama_model}`** is not downloaded yet."
            )
            st.sidebar.code(f"ollama pull {_ollama_model}", language="bash")
            st.sidebar.caption("Run the command above in your terminal, then refresh.")
            _ollama_model = None  # degrade until model is pulled

        if _local:
            with st.sidebar.expander("Pulled models on this machine"):
                st.write(", ".join(_local))
    else:
        st.sidebar.warning("Ollama server is not running.")
        st.sidebar.code("ollama serve", language="bash")
        st.sidebar.caption(
            "Run the command above in a separate terminal, "
            "then pull the model:\n"
        )
        st.sidebar.code(f"ollama pull {_ollama_model}", language="bash")
        _ollama_model = None  # degrade gracefully

st.sidebar.divider()
st.sidebar.caption(
    "Data: MovieLens ml-32m (GroupLens) + IMDb Ratings\n\n"
    "Graph edges encode genome content similarity, audience overlap, and genre kinship."
)

# ---------------------------------------------------------------------------
# Load data (happens once per session)
# ---------------------------------------------------------------------------
movies = _load_movies()
G = _load_graph(movies)

options: List[str] = sorted(m.display_title() for m in movies.values())


def _find_movie(display: str) -> Optional[Movie]:
    for m in movies.values():
        if m.display_title() == display:
            return m
    return None


# ---------------------------------------------------------------------------
# Helper: movie info card
# ---------------------------------------------------------------------------


def _movie_card(movie: Movie) -> None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(movie.display_title())
        st.markdown(f"**Genres:** {movie.genre_str()}")
        if movie.tags:
            tag_html = " ".join(
                f'<span class="tag-pill">{t}</span>' for t in sorted(movie.tags)[:12]
            )
            st.markdown(f"**Tags:** {tag_html}", unsafe_allow_html=True)
    with col2:
        st.metric("MovieLens Rating", f"{movie.avg_rating:.2f} ⭐")
        st.metric("# Ratings", f"{movie.rating_count:,}")
        if movie.imdb_rating:
            st.metric("IMDb Rating", f"{movie.imdb_rating:.1f}")
        if movie.imdb_votes:
            st.metric("IMDb Votes", f"{movie.imdb_votes:,}")


# ===========================================================================
# PAGE 1 — Search & Neighbourhood
# ===========================================================================
if page == "Search & Neighbourhood":
    st.title("Search & Neighbourhood")
    st.markdown(
        "Search for a movie and explore its nearest connections in the graph."
    )

    query = st.text_input("Search movies", placeholder="e.g. Inception")
    depth = st.slider("Neighbourhood depth", 1, 2, 1)

    if query:
        hits = alg.search_movies(movies, query, max_results=15)
        if not hits:
            st.warning("No movies found. Try a different search term.")
        else:
            selected_title = st.selectbox(
                "Select a movie",
                [m.display_title() for m in hits],
            )
            selected = next((m for m in hits if m.display_title() == selected_title), None)

            if selected:
                st.divider()
                _movie_card(selected)
                st.divider()

                if selected.movie_id not in G:
                    st.info("This movie has no graph connections (insufficient data).")
                else:
                    sub = alg.get_neighborhood(G, selected.movie_id, depth=depth, max_nodes=40)
                    n_neighbours = sub.number_of_nodes() - 1
                    st.markdown(f"**{n_neighbours}** connected movies (depth {depth})")
                    st.plotly_chart(
                        viz.draw_neighborhood(sub, selected.movie_id, height=480),
                        use_container_width=True,
                    )

                    st.subheader("Direct neighbours")
                    neighbours = []
                    for nb in G.neighbors(selected.movie_id):
                        nb_movie: Movie = G.nodes[nb]["movie"]
                        rel = G[selected.movie_id][nb].get("relationship")
                        neighbours.append(
                            {
                                "Movie": nb_movie.display_title(),
                                "Genres": nb_movie.genre_str(),
                                "Similarity": f"{G[selected.movie_id][nb]['weight']:.3f}",
                                "Connection": rel.describe() if rel else "—",
                            }
                        )
                    neighbours.sort(key=lambda r: r["Similarity"], reverse=True)
                    st.dataframe(neighbours, use_container_width=True, height=280)
    else:
        st.info("Type a movie title above to begin exploring.")


# ===========================================================================
# PAGE 2 — Path Finder
# ===========================================================================
elif page == "Path Finder":
    st.title("Path Finder")
    st.markdown(
        "Discover how any two movies are connected through the graph"
        + (" — then let the local LLM narrate the story." if _ollama_model else ".")
    )

    col_a, col_b = st.columns(2)
    with col_a:
        title_a = st.selectbox("From", options, index=0, key="pf_a")
    with col_b:
        title_b = st.selectbox("To", options, index=1, key="pf_b")

    if st.button("Find Connection", type="primary"):
        movie_a = _find_movie(title_a)
        movie_b = _find_movie(title_b)

        if not movie_a or not movie_b:
            st.error("Could not resolve selected movies.")
        elif movie_a.movie_id == movie_b.movie_id:
            st.warning("Please choose two different movies.")
        else:
            path = alg.find_path(G, movie_a.movie_id, movie_b.movie_id)

            if path is None:
                st.error(
                    f"No path found between **{title_a}** and **{title_b}**. "
                    "They may be in disconnected parts of the graph."
                )
            else:
                st.success(f"Path length: **{len(path) - 1}** hop(s)")
                path_data = alg.path_relationships(G, path)

                st.plotly_chart(
                    viz.draw_path(G, path, height=300),
                    use_container_width=True,
                )

                st.subheader("Step-by-step connections")
                for i, (m_a, m_b, rel) in enumerate(path_data, 1):
                    with st.expander(
                        f"Step {i}: {m_a.display_title()} → {m_b.display_title()}"
                    ):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**{m_a.display_title()}**")
                            st.caption(m_a.genre_str())
                        with c2:
                            st.markdown(f"**{m_b.display_title()}**")
                            st.caption(m_b.genre_str())
                        st.markdown(f"**Why connected:** {rel.describe()}")
                        if rel.shared_tags:
                            tag_html = " ".join(
                                f'<span class="tag-pill">{t}</span>'
                                for t in rel.shared_tags
                            )
                            st.markdown(tag_html, unsafe_allow_html=True)

                st.divider()
                st.subheader("Connection Story")

                if _ollama_model:
                    with st.spinner(f"Generating story with {_ollama_model} …"):
                        story_container = st.empty()
                        story_text = ""
                        for chunk in llm_mod.stream_story(
                            path_data, model=_ollama_model, host=_ollama_host
                        ):
                            story_text += chunk
                            story_container.markdown(story_text)
                else:
                    story = llm_mod.generate_story(path_data, model=None)
                    st.markdown(story)
                    if llm_mode == "Off (no LLM)":
                        st.info(
                            "Enable Ollama in the sidebar to get an AI-generated "
                            "narrative.  Run `ollama pull qwen3.5:4b` first."
                        )


# ===========================================================================
# PAGE 3 — Rankings
# ===========================================================================
elif page == "Rankings":
    st.title("Movie Rankings")
    st.markdown(
        "Rank movies by their structural importance in the graph. "
        "Centrality reveals which films are the true hubs of cinema."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        metric = st.selectbox(
            "Centrality metric",
            list(alg._CENTRALITY_FNS.keys()),
            format_func=lambda k: alg.CENTRALITY_LABELS[k],
        )
    with col2:
        all_genres = sorted({g for m in movies.values() for g in m.genres})
        genre_filter = st.selectbox("Genre filter", ["All"] + all_genres)
    with col3:
        years = [m.year for m in movies.values() if m.year]
        y_min, y_max = min(years), max(years)
        year_range = st.slider("Year range", y_min, y_max, (y_min, y_max))

    with st.spinner(f"Computing {alg.CENTRALITY_LABELS[metric]}…"):
        ranked = alg.top_by_centrality(
            G,
            metric=metric,
            n=25,
            genre_filter=None if genre_filter == "All" else genre_filter,
            year_range=year_range if year_range != (y_min, y_max) else None,
        )

    if not ranked:
        st.warning("No movies match the current filters.")
    else:
        st.plotly_chart(
            viz.draw_rankings(ranked, alg.CENTRALITY_LABELS[metric], height=600),
            use_container_width=True,
        )

        st.subheader("Details")
        table = [
            {
                "Rank": i + 1,
                "Movie": m.display_title(),
                "Genres": m.genre_str(),
                "Score": f"{score:.5f}",
                "Avg Rating": f"{m.avg_rating:.2f}",
                "# Ratings": m.rating_count,
            }
            for i, (m, score) in enumerate(ranked)
        ]
        st.dataframe(table, use_container_width=True)

    with st.expander("What do these metrics mean?"):
        st.markdown(
            """
| Metric | What it measures |
|---|---|
| **Degree** | How many direct connections a movie has — the most-connected films |
| **Betweenness** | How often a movie lies on shortest paths — the "bridge" films |
| **PageRank** | Influence weighted by the importance of neighbours — prestige hubs |
| **Closeness** | How quickly a movie can reach all others — the most central titles |
            """
        )


# ===========================================================================
# PAGE 4 — Recommendations
# ===========================================================================
elif page == "Recommendations":
    st.title("Graph-Based Recommendations")
    st.markdown(
        "Get recommendations based on your movie's position in the graph — "
        "not just ratings, but genuine structural proximity."
    )

    seed_title = st.selectbox("Pick a movie you like", options)
    n_recs = st.slider("Number of recommendations", 5, 20, 10)

    seed = _find_movie(seed_title)
    if seed:
        if seed.movie_id not in G:
            st.warning("This movie has no graph connections.")
        else:
            recs = alg.get_recommendations(G, seed.movie_id, n=n_recs)
            if not recs:
                st.info("No recommendations found. Try a more popular movie.")
            else:
                st.divider()
                st.markdown(f"**Because you like {seed.display_title()}…**")
                for i, (movie, score, reason) in enumerate(recs, 1):
                    c1, c2, c3 = st.columns([3, 1, 2])
                    with c1:
                        st.markdown(f"**{i}. {movie.display_title()}**")
                        st.caption(movie.genre_str())
                    with c2:
                        st.metric("Score", f"{score:.3f}")
                    with c3:
                        st.caption(f"Why: {reason}")
                    st.markdown(
                        '<div class="section-divider"></div>', unsafe_allow_html=True
                    )


# ===========================================================================
# PAGE 5 — Explore & Filter
# ===========================================================================
elif page == "Explore & Filter":
    st.title("Explore & Filter")
    st.markdown("Browse the movie catalogue with precision filters.")

    col1, col2 = st.columns(2)
    with col1:
        all_genres = sorted({g for m in movies.values() for g in m.genres})
        selected_genres = st.multiselect("Genres (all must match)", all_genres)
        tag_query = st.text_input("Tag keyword", placeholder="e.g. dark")
    with col2:
        years = [m.year for m in movies.values() if m.year]
        y_min, y_max = min(years), max(years)
        year_range = st.slider("Release year", y_min, y_max, (y_min, y_max), key="ef_year")
        min_rating = st.slider("Minimum MovieLens rating", 0.0, 5.0, 3.5, step=0.1)

    results = alg.filter_movies(
        movies,
        genres=selected_genres or None,
        year_range=tuple(year_range),
        min_avg_rating=min_rating,
        tag=tag_query or None,
    )

    st.markdown(f"**{len(results)}** movies match your filters.")

    if results:
        table = [
            {
                "Title": m.display_title(),
                "Year": m.year or "?",
                "Genres": m.genre_str(),
                "Avg Rating": f"{m.avg_rating:.2f}",
                "# Ratings": m.rating_count,
                "IMDb Rating": m.imdb_rating or "—",
                "Tags (sample)": ", ".join(sorted(m.tags)[:4]) or "—",
            }
            for m in results[:200]
        ]
        st.dataframe(table, use_container_width=True, height=500)

        if len(results) >= 3:
            st.subheader("Connection map (top 30 results)")
            top_ids = {m.movie_id for m in results[:30]}
            sub = G.subgraph(top_ids & set(G.nodes())).copy()
            if sub.number_of_edges() > 0:
                st.plotly_chart(
                    viz.draw_neighborhood(sub, results[0].movie_id, height=420),
                    use_container_width=True,
                )
            else:
                st.info("Filtered movies share no graph edges.")


# ===========================================================================
# PAGE 6 — Graph Statistics
# ===========================================================================
elif page == "Graph Statistics":
    st.title("Graph Statistics")
    st.markdown("A structural overview of the entire movie graph.")

    stats = alg.graph_stats(G)

    c1, c2, c3 = st.columns(3)
    c1.metric("Movies (nodes)", f"{stats['nodes']:,}")
    c2.metric("Connections (edges)", f"{stats['edges']:,}")
    c3.metric("Graph density", f"{stats['density']:.5f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Connected components", stats["connected_components"])
    c5.metric("Largest component", f"{stats['largest_component_size']:,}")
    c6.metric("Average degree", stats["average_degree"])

    st.divider()

    st.subheader("Degree distribution")
    import plotly.express as px
    import pandas as pd

    degrees = [d for _, d in G.degree()]
    fig = px.histogram(
        pd.DataFrame({"degree": degrees}),
        x="degree",
        nbins=50,
        title="Number of connections per movie",
        labels={"degree": "Connections", "count": "Movies"},
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Genre breakdown in graph")
    genre_counts: Dict[str, int] = {}
    for mid in G.nodes():
        movie: Movie = G.nodes[mid]["movie"]
        for g in movie.genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
    genre_df = pd.DataFrame(
        sorted(genre_counts.items(), key=lambda x: x[1], reverse=True),
        columns=["Genre", "Count"],
    )
    fig2 = px.bar(
        genre_df,
        x="Genre",
        y="Count",
        title="Movies per genre in the graph",
        template="plotly_dark",
        color="Genre",
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Raw stats JSON"):
        st.json(stats)
